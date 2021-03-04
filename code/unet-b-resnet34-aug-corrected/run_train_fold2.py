import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

from common  import *
from model   import *
from dataset import *


from lib.net.lookahead import *
from lib.net.radam import *


import torch.cuda.amp as amp
class AmpNet(Net):
    @torch.cuda.amp.autocast()
    def forward(self,*args):
        return super(AmpNet, self).forward(*args)


is_mixed_precision = True  # False #True #


#################################################################################################

image_size = 256

def train_augment(record):
    image = record['image']
    mask  = record['mask']

    for fn in np.random.choice([
        lambda image, mask : do_random_rotate_crop(image, mask, size=image_size, mag=45),
        lambda image, mask : do_random_scale_crop(image, mask, size=image_size, mag=0.075),
        lambda image, mask : do_random_crop(image, mask, size=image_size),
    ],1): image, mask = fn(image, mask)

    for fn in np.random.choice([
        lambda image, mask : (image, mask),
        lambda image, mask : do_random_contast(image, mask, mag=0.8),
        lambda image, mask : do_random_gain(image, mask, mag=0.9),
        #lambda image, mask : do_random_hsv(image, mask, mag=[0.1, 0.2, 0]),
        lambda image, mask : do_random_noise(image, mask, mag=0.1),
    ],2): image, mask =  fn(image, mask)

    image, mask = do_random_hsv(image, mask, mag=[0.1, 0.2, 0])
    image, mask = do_random_flip_transpose(image, mask)

    record['mask'] = mask
    record['image'] = image
    return record


#------------------------------------
def do_valid(net, valid_loader):

    valid_num = 0
    valid_probability = []
    valid_mask = []

    net = net.eval()
    start_timer = timer()
    with torch.no_grad():
        for t, batch in enumerate(valid_loader):
            batch_size = len(batch['index'])
            mask  = batch['mask']
            image = batch['image'].cuda()

            logit = data_parallel(net, image) #net(input)#
            probability = torch.sigmoid(logit)

            valid_probability.append(probability.data.cpu().numpy())
            valid_mask.append(mask.data.cpu().numpy())
            valid_num += batch_size

            #debug
            if 0 :
                pass
                image = image.data.cpu().numpy()
                mask  = mask.data.cpu().numpy().squeeze(1)
                probability  = probability.data.cpu().numpy().squeeze(1)

                for b in range(batch_size):
                    m = image[b].transpose(1,2,0).copy()
                    t = mask[b]
                    p = probability[b]
                    image_show_norm('valid m',m,min=0,max=1)
                    image_show_norm('valid t',t,min=0,max=1)
                    image_show_norm('valid p',p,min=0,max=1)
                    cv2.waitKey(0)


            #---
            print('\r %8d / %d  %s'%(valid_num, len(valid_loader.dataset),time_to_str(timer() - start_timer,'sec')),end='',flush=True)
            #if valid_num==200*4: break

    assert(valid_num == len(valid_loader.dataset))
    #print('')
    #------
    probability = np.concatenate(valid_probability)
    mask = np.concatenate(valid_mask)

    loss = np_binary_cross_entropy_loss(probability, mask)
    dice = np_dice_score(probability, mask)
    tp, tn = np_accuracy(probability, mask)
    return [dice, loss,  tp, tn]



 ##----------------

def run_train():

    fold = 2
    out_dir = \
        '/root/share1/kaggle/2020/hubmap/result/en-resnet34-256-aug-corrected/fold%d'%fold
    initial_checkpoint = \
        out_dir+'/checkpoint/00009500_model.pth' #None #'/root/share1/kaggle/2020/hubmap/result/en-resnet34-256-2-aug3/fold2/checkpoint/00006500_model.pth'
        #

    start_lr   = 0.001
    batch_size = 32 #32 #32


    ## setup  ----------------------------------------
    for f in ['checkpoint','train','valid','backup'] : os.makedirs(out_dir +'/'+f, exist_ok=True)
    backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.train.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.train.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\t__file__ = %s\n' % __file__)
    log.write('\tout_dir  = %s\n' % out_dir)
    log.write('\n')


    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')


    train_dataset = HuDataset(
        image_id=[
            make_image_id('train-%d'%fold),
        ],
        image_dir=[
            '0.25_480_240_train_corrected',
        ],
        augment = train_augment,#
    )
    train_loader  = DataLoader(
        train_dataset,
        sampler = RandomSampler(train_dataset),
        batch_size  = batch_size,
        drop_last   = True,
        num_workers = 8,
        pin_memory  = True,
        collate_fn  = null_collate
    )

    valid_dataset = HuDataset(
        image_id  = [make_image_id ('valid-%d'%fold)],
        image_dir = ['0.25_480_240_train_corrected'],
    )
    valid_loader = DataLoader(
        valid_dataset,
        sampler = SequentialSampler(valid_dataset),
        batch_size  = 8,
        drop_last   = False,
        num_workers = 4,
        pin_memory  = True,
        collate_fn  = null_collate
    )


    log.write('fold = %s\n'%str(fold))
    log.write('train_dataset : \n%s\n'%(train_dataset))
    log.write('valid_dataset : \n%s\n'%(valid_dataset))
    log.write('\n')

    ## net ----------------------------------------
    log.write('** net setting **\n')

    if is_mixed_precision:
        scaler = amp.GradScaler()
        net = AmpNet().cuda()
    else:
        net = Net().cuda()


    if initial_checkpoint is not None:
        f = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
        start_iteration = f['iteration']
        start_epoch = f['epoch']
        state_dict  = f['state_dict']
        net.load_state_dict(state_dict,strict=False)  #True
    else:
        start_iteration = 0
        start_epoch = 0
        #net.load_pretrain(is_print=False)


    log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
    log.write('\n')


    ## optimiser ----------------------------------
    if 0: ##freeze
        for p in net.stem.parameters():   p.requires_grad = False
        pass

    def freeze_bn(net):
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
    #freeze_bn(net)

    #-----------------------------------------------

    #optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()),lr=schduler(0))
    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=start_lr)
    ##optimizer = torch.optim.RMSprop(net.parameters(), lr =0.0005, alpha = 0.95)
    #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=start_lr, momentum=0.5, weight_decay=0.0)
    #optimizer = Over9000(filter(lambda p: p.requires_grad, net.parameters()), lr=0.001, )
    #optimizer = Lookahead(torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=start_lr, momentum=0.0, weight_decay=0.0))

    #optimizer = Lookahead(torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=start_lr))
    optimizer = Lookahead(RAdam(filter(lambda p: p.requires_grad, net.parameters()),lr=start_lr), alpha=0.5, k=5)

    num_iteration = 8000*1000
    iter_log    = 250
    iter_valid  = 250
    iter_save   = list(range(0, num_iteration, 500))#1*1000

    log.write('optimizer\n  %s\n'%(optimizer))
    #log.write('schduler\n  %s\n'%(schduler))
    log.write('\n')

    ## start training here! ##############################################
    #array([0.57142857, 0.42857143])
    log.write('** start training here! **\n')
    log.write('   is_mixed_precision = %s \n'%str(is_mixed_precision))
    log.write('   batch_size = %d \n'%(batch_size))
    log.write('   experiment = %s\n' % str(__file__.split('/')[-2:]))
    log.write('                     |-------------- VALID---------|---- TRAIN/BATCH ----------------\n')
    log.write('rate     iter  epoch | dice   loss   tp     tn     | loss           | time           \n')
    log.write('-------------------------------------------------------------------------------------\n')
              #0.00100   0.50  0.80 | 0.891  0.020  0.000  0.000  | 0.000  0.000   |  0 hr 02 min

    def message(mode='print'):
        if mode==('print'):
            asterisk = ' '
            loss = batch_loss
        if mode==('log'):
            asterisk = '*' if iteration in iter_save else ' '
            loss = train_loss

        text = \
            '%0.5f  %5.2f%s %4.2f | '%(rate, iteration/1000, asterisk, epoch,) +\
            '%4.3f  %4.3f  %4.3f  %4.3f  | '%(*valid_loss,) +\
            '%4.3f  %4.3f   | '%(*loss,) +\
            '%s' % (time_to_str(timer() - start_timer,'min'))

        return text

    #----
    valid_loss = np.zeros(4,np.float32)
    train_loss = np.zeros(2,np.float32)
    batch_loss = np.zeros_like(train_loss)
    sum_train_loss = np.zeros_like(train_loss)
    sum_train = 0
    loss = torch.FloatTensor([0]).sum()


    start_timer = timer()
    iteration = start_iteration
    epoch = start_epoch
    rate = 0
    while  iteration < num_iteration:

        for t, batch in enumerate(train_loader):

            if (iteration % iter_valid==0):
                valid_loss = do_valid(net, valid_loader) #
                pass

            if (iteration % iter_log==0):
                print('\r',end='',flush=True)
                log.write(message(mode='log')+'\n')

            if iteration in iter_save:
                if iteration != start_iteration:
                    torch.save({
                        'state_dict': net.state_dict(),
                        'iteration': iteration,
                        'epoch': epoch,
                    }, out_dir + '/checkpoint/%08d_model.pth' % (iteration))
                    pass



            # learning rate schduler -------------
            #adjust_learning_rate(optimizer, schduler(iteration))
            rate = get_learning_rate(optimizer)

            # one iteration update  -------------
            batch_size = len(batch['index'])
            mask  = batch['mask'].cuda()
            image = batch['image'].cuda()

            net.train()
            optimizer.zero_grad()

            if is_mixed_precision:
                #assert (False)
                image = image.half()
                with amp.autocast():
                    logit = data_parallel(net, image)
                    loss = criterion_binary_cross_entropy(logit, mask)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            else :

                #logit = data_parallel(net, image)
                logit = net(image)
                loss = criterion_binary_cross_entropy(logit, mask)

                loss.backward()
                optimizer.step()


            # print statistics  --------
            epoch += 1 / len(train_loader)
            iteration += 1


            batch_loss = np.array([ loss.item(), 0 ])
            sum_train_loss += batch_loss
            sum_train += 1
            if iteration%100 == 0:
                train_loss = sum_train_loss/(sum_train+1e-12)
                sum_train_loss[...] = 0
                sum_train = 0

            print('\r',end='',flush=True)
            print(message(mode='print'), end='',flush=True)

            #debug
            if 0 :
            #if iteration%50==1:
                pass
                # buggy code ????

                probability = torch.sigmoid(logit)
                image = image.data.cpu().float().numpy()
                mask  = mask.data.cpu().float().numpy().squeeze(1)
                probability  = probability.data.cpu().float().numpy().squeeze(1)
                image = np.ascontiguousarray(image.transpose(0,2,3,1))
                batch_size, h, w, _ = image.shape


                for b in range(batch_size):
                    m = image[b]
                    t = mask[b]
                    p = probability[b]

                    #contour = mask_to_inner_contour(p)
                    m = draw_contour_overlay(m, t, color=(0, 0, 1), thickness=3)
                    m = draw_contour_overlay(m, p, color=(0, 1, 0), thickness=3)

                    overlay = np.hstack([
                        m,
                        np.tile(t.reshape(h,w,1),(1,1,3)),
                        np.tile(p.reshape(h,w,1),(1,1,3)),
                        np.stack([np.zeros_like(p),p,t],2),
                    ])
                    image_show_norm('overlay',overlay,min=0,max=1)
                    # image_show_norm('m',m,min=0,max=1)
                    # image_show_norm('t',t,min=0,max=1)
                    # image_show_norm('p',p,min=0,max=1)
                    cv2.waitKey(1)
                    cv2.imwrite(out_dir+'/train/%05d.png'%(b), (overlay*255).astype(np.uint8))

    log.write('\n')




# main #################################################################
if __name__ == '__main__':
    run_train()

'''
 

'''