import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
#os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = pow(2,40).__str__()
#import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from common  import *
from dataset import *
from model   import *

import torch.cuda.amp as amp
is_mixed_precision = False  # True #True #


def mask_to_csv(image_id, submit_dir):

    predicted = []
    for id in image_id:
        image_file = data_dir + '/test/%s.tiff' % id
        image = read_tiff(image_file)

        height, width = image.shape[:2]
        predict_file = submit_dir + '/%s.predict.png' % id
        # predict = cv2.imread(predict_file, cv2.IMREAD_GRAYSCALE)
        predict = np.array(PIL.Image.open(predict_file))
        predict = cv2.resize(predict, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
        predict = (predict > 128).astype(np.uint8) * 255

        p = rle_encode(predict)
        predicted.append(p)

    df = pd.DataFrame()
    df['id'] = image_id
    df['predicted'] = predicted
    return df


def run_submit():

    fold = 2
    out_dir = \
        '/root/share1/kaggle/2020/hubmap/result/en-resnet34-256-aug-corrected/fold%d'%fold
    #out_dir = '/root/share1/kaggle/2020/hubmap/result/resnet34/fold-all'
    initial_checkpoint = \
        out_dir + '/checkpoint/00008500_model.pth' #
        #out_dir + '/checkpoint/00011500_model.pth' #
        #out_dir + '/checkpoint/00008500_model.pth' #

    #server = 'local' # local or kaggle
    server = 'kaggle'

    #---
    submit_dir = out_dir + '/valid/%s-%s-mean'%(server, initial_checkpoint[-18:-4])
    os.makedirs(submit_dir,exist_ok=True)

    log = Logger()
    log.open(out_dir+'/log.submit.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))

    net = Net().cuda()
    state_dict = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)['state_dict']
    net.load_state_dict(state_dict,strict=True)  #True
    net = net.eval()

    #---
    if server == 'local':
        #valid_image_id = make_image_id('valid-%d' % fold)
        valid_image_id = make_image_id('train-all')
    if server == 'kaggle':
        #valid_image_id = ['c68fe75ea','afa5e8098'] #make_image_id('test-all')
        valid_image_id = make_image_id('test-all')


    tile_size = 640 #320
    tile_average_step = 320 #192
    tile_scale = 0.25
    tile_min_score = 0.25

    log.write('tile_size = %d \n'%tile_size)
    log.write('tile_average_step = %d \n'%tile_average_step)
    log.write('tile_scale = %f \n'%tile_scale)
    log.write('tile_min_score = %f \n'%tile_min_score)
    log.write('\n')


    start_timer = timer()
    for id in valid_image_id:
        if server == 'local':
            image_file = data_dir + '/train/%s.tiff' % id
            image = read_tiff(image_file)
            height, width = image.shape[:2]

            json_file  = data_dir + '/train/%s-anatomical-structure.json' % id
            structure = draw_strcuture(read_json_as_df(json_file), height, width, structure=['Cortex'])

            try:
                mask_file = data_dir + '/train/%s.corrected_shift_mask.png' % id
                mask  = read_mask(mask_file)
            except:
                mask_file  = data_dir + '/train/%s.corrected_mask.png' % id
                mask  = read_mask(mask_file)

        if server == 'kaggle':
            image_file = data_dir + '/test/%s.tiff' % id
            json_file  = data_dir + '/test/%s-anatomical-structure.json' % id

            image = read_tiff(image_file)
            height, width = image.shape[:2]
            structure = draw_strcuture(read_json_as_df(json_file), height, width, structure=['Cortex'])

            mask = None


        #--- predict here!  ---
        tile = to_tile(image, mask, structure, tile_scale, tile_size, tile_average_step, tile_min_score)

        tile_image = tile['tile_image']
        tile_image = np.stack(tile_image)[..., ::-1]
        tile_image = np.ascontiguousarray(tile_image.transpose(0,3,1,2))
        tile_image = tile_image.astype(np.float32)/255

        tile_probability = []
        batch = np.array_split(tile_image, len(tile_image)//4)
        for t,m in enumerate(batch):
            print('\r %s  %d / %d   %s'%(id, t, len(batch), time_to_str(timer() - start_timer, 'sec')), end='',flush=True)
            m = torch.from_numpy(m).cuda()

            p = []
            with torch.no_grad():
                logit = data_parallel(net, m)
                p.append(torch.sigmoid(logit))

                #---
                if 1: #tta here
                    logit = data_parallel(net, m.flip(dims=(2,)))
                    p.append(torch.sigmoid(logit.flip(dims=(2,))))

                    logit = data_parallel(net, m.flip(dims=(3,)))
                    p.append(torch.sigmoid(logit.flip(dims=(3,))))
                #---

            p = torch.stack(p).mean(0)
            tile_probability.append(p.data.cpu().numpy())

        print('\r' , end='',flush=True)
        log.write('%s  %d / %d   %s\n'%(id, t, len(batch), time_to_str(timer() - start_timer, 'sec')))

        tile_probability = np.concatenate(tile_probability).squeeze(1)
        height, width = tile['image_small'].shape[:2]
        probability = to_mask(tile_probability, tile['coord'], height, width,
                              tile_scale, tile_size, tile_average_step, tile_min_score,
                              aggregate='mean')

        #--- show results ---
        if server == 'local':
            truth = tile['mask_small'].astype(np.float32)/255
        if server == 'kaggle':
            truth = np.zeros((height, width), np.float32)

        overlay = np.dstack([
            np.zeros_like(truth),
            probability, #green
            truth, #red
        ])
        image_small = tile['image_small'].astype(np.float32)/255
        predict = (probability>0.5).astype(np.float32)
        overlay1 = 1-(1-image_small)*(1-overlay)
        overlay2 = image_small.copy()
        overlay2 = draw_contour_overlay(overlay2, tile['structure_small'], color=(1, 1, 1), thickness=3)
        overlay2 = draw_contour_overlay(overlay2, truth, color=(0, 0, 1), thickness=8)
        overlay2 = draw_contour_overlay(overlay2, probability, color=(0, 1, 0), thickness=3)


        if 1:
            image_show_norm('image_small', image_small, min=0, max=1, resize=0.1)
            image_show_norm('probability', probability, min=0, max=1, resize=0.1)
            image_show_norm('predict',     predict, min=0, max=1, resize=0.1)
            image_show_norm('overlay',     overlay,     min=0, max=1, resize=0.1)
            image_show_norm('overlay1',    overlay1,    min=0, max=1, resize=0.1)
            image_show_norm('overlay2',    overlay2,    min=0, max=1, resize=0.1)
            cv2.waitKey(1)

        if 1:
            cv2.imwrite(submit_dir+'/%s.image_small.png'%id, (image_small*255).astype(np.uint8))
            cv2.imwrite(submit_dir+'/%s.probability.png'%id, (probability*255).astype(np.uint8))
            cv2.imwrite(submit_dir+'/%s.predict.png'%id, (predict*255).astype(np.uint8))
            cv2.imwrite(submit_dir+'/%s.overlay.png'%id, (overlay*255).astype(np.uint8))
            cv2.imwrite(submit_dir+'/%s.overlay1.png'%id, (overlay1*255).astype(np.uint8))
            cv2.imwrite(submit_dir+'/%s.overlay2.png'%id, (overlay2*255).astype(np.uint8))

        #---

        if server == 'local':

            loss = np_binary_cross_entropy_loss(probability, truth)
            dice = np_dice_score(probability, truth)
            tp, tn = np_accuracy(probability, truth)
            log.write('submit_dir = %s \n'%submit_dir)
            log.write('initial_checkpoint = %s \n'%initial_checkpoint)
            log.write('loss   = %0.8f \n'%loss)
            log.write('dice   = %0.8f \n'%dice)
            log.write('tp, tn = %0.8f, %0.8f \n'%(tp, tn))
            log.write('\n')
            #cv2.waitKey(0)

    #-----
    if server == 'kaggle':
        csv_file = submit_dir +'/submission-%s-%s.csv'%(out_dir.split('/')[-1], initial_checkpoint[-18:-4])
        df = mask_to_csv(valid_image_id, submit_dir)
        df.to_csv(csv_file, index=False)
        print(df)

    zz=0





def run_make_csv():

    submit_dir = \
        '/root/share1/kaggle/2020/hubmap/result/resnet34-256/fold2/valid/kaggle-00004000_model-fix'
        #'/root/share1/kaggle/2020/hubmap/result/en-resnet34-256-2-aug3/fold2/valid/kaggle-00006500_model-fix1'

    csv_file = \
        submit_dir+'/kaggle-00004000_model-top1.csv'
        #submit_dir+'/00006500_model-fix1.csv'

    #-----
    image_id = make_image_id('test-all')
    predicted = []

    for id in image_id:
        print(id)
        image_file = data_dir + '/test/%s.tiff' % id
        image = read_tiff(image_file)
        height, width = image.shape[:2]
        try:
            predict_file = submit_dir+'/%s.top.png'%id
            #predict_file = submit_dir+'/%s.fix.png'%id
            predict = np.array(PIL.Image.open(predict_file))
        except:
            predict_file = submit_dir+'/%s.predict.png'%id
            predict = np.array(PIL.Image.open(predict_file))


        predict = cv2.resize(predict, dsize=(width,height), interpolation=cv2.INTER_LINEAR)
        predict = (predict>128).astype(np.uint8)*255

        p = rle_encode(predict)
        predicted.append(p)

    df = pd.DataFrame()
    df['id'] = image_id
    df['predicted'] = predicted

    df.to_csv(csv_file, index=False)
    print(df)


# main #################################################################
if __name__ == '__main__':
    #run_submit()
    run_make_csv()

'''
 

'''