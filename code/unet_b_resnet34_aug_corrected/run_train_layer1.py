import argparse
import os
from pathlib import Path

import git

from code.common import COMMON_STRING, DataLoader, RandomSampler, SequentialSampler
from code.data_preprocessing.dataset_v2020_11_12 import HuDataset, make_image_id, null_collate, train_augment, \
    CenteredHuDataset, train_albu_augment, get_data_path, val_albu_augment, train_albu_augment_layer1
from code.lib.utility.draw import image_show_norm
from code.lib.utility.file import Logger, time_to_str
from code.lib.training.checkpoint_bookeeping import CheckpointUpdate

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from code.unet_b_resnet34_aug_corrected.model import Net, np_binary_cross_entropy_loss, np_dice_score, \
    np_accuracy, criterion_binary_cross_entropy
from code.unet_b_resnet34_aug_corrected.image_preprocessing import do_random_crop, do_random_rotate_crop, \
    do_random_scale_crop, do_random_hsv, do_random_contast, do_random_gain, do_random_noise, \
    do_random_flip_transpose

from code.lib.net.lookahead import *
from code.lib.net.radam import *
from code.lib.net.rate import *
from code.lib.net.lovasz_loss import *
from code.lib.net.other_loss import *
from code.lib.net.segmentation_losses import *

from timeit import default_timer as timer
from torch.nn.parallel.data_parallel import data_parallel

import torch.nn as nn
import torch.cuda.amp as amp
from torch import optim


class AmpNet(Net):
    @torch.cuda.amp.autocast()
    def forward(self, *args):
        return super(AmpNet, self).forward(*args)


is_mixed_precision = False
image_size = 256


# ------------------------------------
def do_valid(net, valid_loader):
    valid_num = 0
    valid_probability = []
    valid_mask = []

    net = net.eval()
    start_timer = timer()
    with torch.no_grad():
        for t, batch in enumerate(valid_loader):
            batch_size = len(batch['index'])
            mask = batch['mask']
            image = batch['image'].cuda()

            logit = data_parallel(net, image)  # net(input)#
            probability = torch.sigmoid(logit)

            valid_probability.append(probability.data.cpu().numpy())
            valid_mask.append(mask.data.cpu().numpy())
            valid_num += batch_size

            # ---
            print('\r %8d / %d  %s' % (valid_num, len(valid_loader.dataset), time_to_str(timer() - start_timer, 'sec')),
                  end='', flush=True)
            # if valid_num==200*4: break

    assert (valid_num == len(valid_loader.dataset))
    # print('')
    # ------
    probability = np.concatenate(valid_probability)
    mask = np.concatenate(valid_mask)

    # print('\n1', timer() - start_timer)
    loss = np_binary_cross_entropy_loss(probability, mask)
    # print(timer() - start_timer)

    # print()
    # print(probability.shape)
    # print(type(probability), type(mask))
    # _tmp = torch.from_numpy(probability)
    #
    # print()
    #
    # loss = lovasz_loss(torch.logit(torch.from_numpy(probability)), mask)

    # print('2', timer() - start_timer)
    dice = np_dice_score(probability, mask)

    # print('3', timer() - start_timer)
    tp, tn, fp, fn = np_accuracy(probability, mask)
    return [dice, loss, tp, tn, fp, fn]


###################################################################################
### Train
###################################################################################

def get_loss(loss_type, logit, mask):
    if loss_type == 'bce':
        loss = criterion_binary_cross_entropy(logit, mask)
    elif loss_type == 'dice':
        criterion = DiceLoss()
        loss = criterion(logit, mask)
    elif loss_type == 'dice_bce':
        criterion = DiceBCELoss()
        loss = criterion(logit, mask)
    elif loss_type == 'focal':
        criterion = FocalLoss()
        loss = criterion(logit, mask)
    elif loss_type == 'tversky':
        criterion = TverskyLoss()
        loss = criterion(logit, mask, alpha=1, beta=1)
    elif loss_type == 'focal_tversky':
        criterion = FocalTverskyLoss()
        loss = criterion(logit, mask, alpha=1, beta=1, gamma=2)
    elif loss_type == 'weighted_bce':
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.5))
        loss = criterion(logit, mask)
    return loss


def split_dataset(sha, train_image_id, true_positives_dir, false_positives_dir):

    images = {k: 0 for k in train_image_id.values()}
    true_positives = {k: 0 for k in train_image_id.values()}
    false_positives = {k: 0 for k in train_image_id.values()}
    train_set = {k: [] for k in train_image_id.values()}
    val_set = {k: [] for k in train_image_id.values()}

    project_repo, raw_data_dir, data_dir = get_data_path('local')

    # -----------------------------------------
    ### Liste les chemins des diff??rentes image
    # -----------------------------------------

    for id in train_image_id.values():

        for i, _dataset in enumerate(true_positives_dir + false_positives_dir):
            image_dir = f"/tile/{_dataset}/{id}/"
            current_images = [
                image_dir + f.strip('.mask.png')
                for f in os.listdir(data_dir + image_dir)
                if 'mask' in f
            ]
            if i == 0:
                true_positives[id] += len(current_images)
            else:
                false_positives[id] += len(current_images)

            images[id] += len(current_images)
            val_size = int(len(current_images) * 0.2)
            val_set[id] += random.sample(current_images, val_size)
            train_set[id] += [c for c in current_images if c not in val_set[id]]

        print(f"{id}: train/val = {len(train_set[id])} / {len(val_set[id])}")

    print('\n True positives:', true_positives)
    print('\n False positives:', false_positives)

    for image_id, images in val_set.items():
        outdir = data_dir + f"/tile/val_aug_{sha}/"
        Path(outdir + f'/{image_id}').mkdir(parents=True, exist_ok=True)
        additional_data = []
        for image_path in images:
            # print(data_dir + image_path)
            image = cv2.imread(data_dir + f'{image_path}.png', cv2.IMREAD_COLOR)
            mask  = cv2.imread(data_dir + f'{image_path}.mask.png', cv2.IMREAD_GRAYSCALE)
            result = val_albu_augment({
                'image_size': image_size,
                'image': image.copy(),
                'mask': mask.copy(),
                'verbose': False
            })

            # image_show_norm('overlay1', result['image'])
            # cv2.waitKey(1)

            _image = f"/tile/val_aug_{sha}/{image_id}/{image_path.split('/')[-1]}"
            aug_data_path = data_dir + _image
            cv2.imwrite(f"{aug_data_path}.png", result['image'])
            cv2.imwrite(f"{aug_data_path}.mask.png", result["mask"])
            additional_data.append(_image)

        val_set[image_id].extend(additional_data)
            # print(aug_data_path)
        # sys.exit()

    return train_set, val_set


###########################################################################
# Training of the model ---------------------------------------------------
###########################################################################
def run_train(show_valid_images=False,
              sha='',
              fold=None,
              loss_type='bce',
              tile_size=320,       # overall size of the input images
              image_size=320,      # overall size of the input images
              tile_scale=1,
              backbone='resnet34',
              *args,
              **kwargs
              ):

    out_dir = f"result/Layer_1/fold{'_'.join(map(str, fold))}"
    initial_checkpoint = None

    start_lr = kwargs.get('start_lr', 0.001)
    batch_size = kwargs.get('batch_size', 16)

    ##################################################
    ## setup  ----------------------------------------
    ##################################################
    for f in [
        f'checkpoint_{sha}',
        f'predictions_{sha}',
    ]:
        os.makedirs(out_dir + '/' + f, exist_ok=True)

    log = Logger()
    log.open(out_dir + f'/log.train_{sha}.txt', mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\t__file__ = %s\n' % __file__)
    log.write('\tout_dir  = %s\n' % out_dir)
    log.write('\n')

    ##################################################
    ## dataset ---------------------------------------
    ##################################################
    log.write(30*'-' + '\n' + '*** TRAIN dataset setting ***\n' + 30*'-' + '\n')

    # -----------------------------
    ### Create CV scheme ----------
    # -----------------------------
    train_image_id = {
        0: '0486052bb',
        1: '095bf7a1f',
        2: '1e2425f28',
        3: '26dc41664',
        4: '2f6ecfcdf',
        5: '4ef6695ce',
        6: '54f2eec69',
        7: '8242609fa',
        8: 'aaa6a05cc',
        9: 'afa5e8098',
        10: 'b2dc8411c',
        11: 'b9a3865fc',
        12: 'c68fe75ea',
        13: 'cb2d976f4',
        14: 'e79de561c',
    }
    image_dir = [
        f'{tile_scale}_{tile_size}_train',
    ]
    train_set, val_set = split_dataset(
        sha                 = sha,
        train_image_id      = train_image_id,
        true_positives_dir  = image_dir,
        false_positives_dir = []
    )
    # ------------
    ### TRAIN SET
    # ------------
    train_dataset = CenteredHuDataset(
        images      = train_set,
        image_size  = image_size,
        augment     = train_albu_augment_layer1,
        logger      = log
    )
    train_loader = DataLoader(
        train_dataset,
        sampler     = RandomSampler(train_dataset),
        batch_size  = batch_size,
        drop_last   = True,
        num_workers = 8,
        pin_memory  = True,
        collate_fn  = null_collate
    )
    # ------------
    ### VALID SET
    # ------------
    log.write(30*'-' + '\n' + '*** VALID dataset setting ***\n' + 30*'-' + '\n')
    valid_dataset = CenteredHuDataset(
        images      = val_set,
        image_size  = image_size,
        augment     = None,
        logger      = log
    )

    valid_loader = DataLoader(
        valid_dataset,
        sampler     = SequentialSampler(valid_dataset),
        batch_size  = 8,
        drop_last   = False,
        num_workers = 8,
        pin_memory  = True,
        collate_fn  = null_collate
    )

    log.write(30 * '-' + '\n' + '*** dataset setting SUMMARY***\n' + 30 * '-' + '\n')
    log.write('fold = %s\n' % ' '.join(map(str, fold)))
    log.write('train_dataset : \n%s\n' % train_dataset)
    log.write('valid_dataset : \n%s\n' % valid_dataset)
    log.write('\n')


    ##################################################
    ## net -------------------------------------------
    ##################################################
    log.write('** net setting **\n')

    if is_mixed_precision:
        scaler = amp.GradScaler()
        net = AmpNet().cuda()
    else:
        net = Net(backbone).cuda()

    if initial_checkpoint is not None:
        f = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
        start_iteration = f['iteration']
        start_epoch = f['epoch']
        state_dict = f['state_dict']
        net.load_state_dict(state_dict, strict=False)
    else:
        start_iteration = 0
        start_epoch = 0
        # net.load_pretrain(is_print=False)

    log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
    log.write('\n')

    ## optimiser ----------------------------------
    if 0:  ##freeze
        for p in net.stem.parameters():   p.requires_grad = False
        pass

    def freeze_bn(net):
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    # freeze_bn(net)

    # -----------------------------------------------

    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()),lr=schduler(0))
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=start_lr)
    ##optimizer = torch.optim.RMSprop(net.parameters(), lr =0.0005, alpha = 0.95)
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=start_lr, momentum=0.5, weight_decay=0.0)
    # optimizer = Over9000(filter(lambda p: p.requires_grad, net.parameters()), lr=0.001, )
    # optimizer = Lookahead(torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=start_lr, momentum=0.0, weight_decay=0.0))

    # optimizer = Lookahead(torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=start_lr))
    optimizer = Lookahead(RAdam(filter(lambda p: p.requires_grad, net.parameters()), lr=start_lr), alpha=0.5, k=5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

    num_iteration = kwargs.get('num_iteration', 5000)   # total nb. of batch used to train the net
    iter_log = kwargs.get('iter_log', 250)              # show results every iter_log
    first_iter_save = kwargs.get('first_iter_save', 0)  # first checkpoint kept
    iter_valid = iter_log                               # validate every iter_valid
    # iter_save = list(range(0, num_iteration + 1, iter_log))

    log.write('optimizer\n  %s\n' % optimizer)
    # log.write('schduler\n  %s\n'%(schduler))
    log.write('\n')

    ######################################################################
    ## start training here! ##############################################
    ######################################################################

    num_iteration = (num_iteration // len(train_loader) + 1) * len(train_loader)

    log.write('** start training here! **\n')
    log.write('   is_mixed_precision = %s \n' % str(is_mixed_precision))
    log.write('   loss_type = %s \n' % loss_type)
    log.write('   batch_size = %d \n' % batch_size)
    log.write('   num_iterations = %d \n' % num_iteration)
    log.write('   experiment = %s\n' % str(__file__.split('/')[-2:]))
    log.write('                       |-------------- VALID----------------------|---- TRAIN/BATCH ----------------\n')
    log.write('rate       iter  epoch | dice   loss   tp     tn     fp     fn    | loss           | time           \n')
    log.write('----------------------------------------------------------------------------------------------------\n')

    def message(mode='print'):
        if iteration % iter_valid == 0 and iteration > 0:
            iter_save = True
        if mode == 'print':
            asterisk = ' '
            loss = batch_loss
        if mode == 'log':
            asterisk = '*' if iter_save else ' '
            loss = train_loss

        text = \
            '%0.7f  %5.2f%s %4.2f | ' % (rate, iteration / 1000, asterisk, epoch,) + \
            '%4.3f  %4.3f  %4.3f  %4.3f  %4.3f  %4.3f | ' % (*valid_loss,) + \
            '%4.3f  %4.3f   | ' % (*loss,) + \
            '%s' % (time_to_str(timer() - start_timer, 'min'))

        return text

    # ----
    valid_loss = np.zeros(6, np.float32)
    train_loss = np.zeros(2, np.float32)
    batch_loss = np.zeros_like(train_loss)
    sum_train_loss = np.zeros_like(train_loss)
    sum_train = 0
    loss = torch.FloatTensor([0]).sum()
    start_timer = timer()
    iteration = start_iteration
    epoch = start_epoch
    rate = 0

    bookeeping = CheckpointUpdate(
        net=net,
        first_iter_save=first_iter_save,
        # iter_save=iter_save,
        out_dir=out_dir,
        sha=sha,
        nbest=5,
    )

    while iteration < num_iteration:

        for t, batch in enumerate(train_loader):

            if iteration % iter_valid == 0 and iteration > 0:
                valid_loss = do_valid(net, valid_loader)
                bookeeping.update(
                    iteration=iteration,
                    epoch=epoch,
                    score=valid_loss[5],
                    max=False
                )
                # sys.exit()

            if iteration % iter_log == 0 and iteration > 0:
                print('\r', end='', flush=True)
                log.write(message(mode='log') + '\n')


            rate = scheduler.get_last_lr()[0]
            # rate = get_learning_rate(optimizer)

            # one iteration update  -------------
            batch_size = len(batch['index'])
            mask = batch['mask'].cuda()
            image = batch['image'].cuda()

            net.train()
            optimizer.zero_grad()

            ################################################
            ### Compute the loss ---------------------------
            ################################################
            if is_mixed_precision:
                # assert (False)
                image = image.half()
                with amp.autocast():
                    logit = data_parallel(net, image)
                    loss = get_loss(loss_type, logit, mask)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # logit = data_parallel(net, image)

                logit = net(image)
                loss = get_loss(loss_type, logit, mask)
                loss.backward()
                optimizer.step()
                scheduler.step()

            ##############################
            # print statistics  ----------
            ##############################
            epoch += 1 / len(train_loader)
            iteration += 1

            batch_loss = np.array([loss.item(), 0])
            sum_train_loss += batch_loss
            sum_train += 1
            if iteration % 100 == 0:
                train_loss = sum_train_loss / (sum_train + 1e-12)
                sum_train_loss[...] = 0
                sum_train = 0

            print('\r', end='', flush=True)
            print(message(mode='print'), end='', flush=True)

            # debug
            if show_valid_images:
                # if iteration%50==1:
                pass
                # buggy code ????

                probability = torch.sigmoid(logit)
                image = image.data.cpu().float().numpy()
                mask = mask.data.cpu().float().numpy().squeeze(1)
                probability = probability.data.cpu().float().numpy().squeeze(1)
                image = np.ascontiguousarray(image.transpose(0, 2, 3, 1))
                batch_size, h, w, _ = image.shape

                for b in range(batch_size):
                    m = image[b]
                    t = mask[b]
                    p = probability[b]

                    # contour = mask_to_inner_contour(p)
                    m = draw_contour_overlay(m, t, color=(0, 0, 1), thickness=3)
                    m = draw_contour_overlay(m, p, color=(0, 1, 0), thickness=3)

                    overlay = np.hstack([
                        m,
                        np.tile(t.reshape(h, w, 1), (1, 1, 3)),
                        np.tile(p.reshape(h, w, 1), (1, 1, 3)),
                        np.stack([np.zeros_like(p), p, t], 2),
                    ])
                    image_show_norm('overlay', overlay, min=0, max=1)
                    # image_show_norm('m',m,min=0,max=1)
                    # image_show_norm('t',t,min=0,max=1)
                    # image_show_norm('p',p,min=0,max=1)
                    cv2.waitKey(1)
                    cv2.imwrite(out_dir + '/train/%05d.png' % (b), (overlay * 255).astype(np.uint8))

    log.write('\n')


########################################################################
# main #################################################################
########################################################################

if __name__ == '__main__':

    # Setting seed
    seed = 36
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    ########################
    # define run arguments
    ########################
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fold", help="fold")
    parser.add_argument("-i", "--iterations", help="max iterations", default=8000)
    #----------------
    # Fold
    # ----------------
    args = parser.parse_args()
    if not args.fold:
        print("fold missing")
        sys.exit()
    elif isinstance(args.fold, int):
        fold = [int(args.fold)]
    elif isinstance(args.fold, str):
        fold = [int(c) for c in args.fold.split()]
    else:
        print("unsupported format for fold")
        sys.exit()

    ##########################################
    repo = git.Repo(search_parent_directories=True)
    model_sha = repo.head.object.hexsha[:9]
    print(f"current commit: {model_sha}")

    changedFiles = [item.a_path for item in repo.index.diff(None) if item.a_path.endswith(".py")]
    if len(changedFiles) > 0:
        print("ABORT submission -- There are unstaged files:")
        for _file in changedFiles:
            print(f" * {_file}")
    else:
        run_train(
            show_valid_images = False,
            sha               = model_sha,
            fold              = fold,
            start_lr          = 0.0005,
            batch_size        = 16,
            num_iteration     = int(args.iterations),
            iter_log          = 250,
            iter_save         = 250,
            first_iter_save   = 0,
            loss_type         = "weighted_bce",
            tile_scale        = 0.25,
            tile_size         = 320,
            image_size        = 256,
            backbone          = 'efficientnet-b0',
        )

