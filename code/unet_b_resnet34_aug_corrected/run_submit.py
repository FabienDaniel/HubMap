import argparse
import os
import sys

import torch
from torch.nn.parallel.data_parallel import data_parallel

from code.data_preprocessing.dataset import read_tiff, to_tile
from code.data_preprocessing.dataset_v2020_11_12 import make_image_id, draw_strcuture, read_json_as_df, read_mask, \
    to_mask, draw_contour_overlay, image_show_norm
from code.hubmap_v2 import data_dir, rle_encode, project_repo, raw_data_dir
from code.lib.include import IDENTIFIER
from code.lib.utility.file import Logger, time_to_str
from code.unet_b_resnet34_aug_corrected.model import Net, np_binary_cross_entropy_loss, np_dice_score, np_accuracy
from timeit import default_timer as timer

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from PIL import Image
import git
import PIL
import numpy as np
import cv2
import pandas as pd

Image.MAX_IMAGE_PIXELS = None

is_mixed_precision = False


def mask_to_csv(image_id, submit_dir):

    predicted = []
    for id in image_id:
        image_file = raw_data_dir + '/test/%s.tiff' % id
        image = read_tiff(image_file)

        height, width = image.shape[:2]
        predict_file = submit_dir + '/%s.predict.png' % id
        predict = np.array(PIL.Image.open(predict_file))
        predict = cv2.resize(predict, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
        predict = (predict > 128).astype(np.uint8) * 255

        p = rle_encode(predict)
        predicted.append(p)

    df = pd.DataFrame()
    df['id'] = image_id
    df['predicted'] = predicted
    return df


def submit(sha, server, iterations, fold):

    out_dir = project_repo + f"/result/Baseline/fold{'_'.join(map(str, fold))}"
    _checkpoint_dir = out_dir + f"/checkpoint_{sha}/"

    iter_tag = f"{int(iterations):08}"

    [model_checkpoint] = [_file for _file in os.listdir(_checkpoint_dir)
                          if iter_tag in _file.split('_')[0]]

    initial_checkpoint = out_dir + f'/checkpoint_{sha}/{model_checkpoint}'

    print(initial_checkpoint)

    print(f"submit with server={server}")

    #---
    submit_dir = out_dir + f'/predictions_{sha}/%s-%s-mean' % (server, iter_tag)
    os.makedirs(submit_dir, exist_ok=True)

    log = Logger()
    log.open(out_dir + f'/log.submit_{sha}.txt', mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))

    net = Net().cuda()
    state_dict = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)['state_dict']
    net.load_state_dict(state_dict, strict=True)
    net = net.eval()

    #---
    if server == 'local':
        valid_image_id = make_image_id('train-all')
    if server == 'kaggle':
        valid_image_id = make_image_id('test-all')


    tile_size = 640 #320
    tile_average_step = 320 #192
    tile_scale = 0.25
    tile_min_score = 0.25

    log.write('tile_size = %d \n' % tile_size)
    log.write('tile_average_step = %d \n' % tile_average_step)
    log.write('tile_scale = %f \n' % tile_scale)
    log.write('tile_min_score = %f \n' % tile_min_score)
    log.write('\n')


    start_timer = timer()
    for id in valid_image_id:
        print(50*"=")
        if server == 'local':
            image_file = raw_data_dir + '/train/%s.tiff' % id
            image = read_tiff(image_file)
            height, width = image.shape[:2]

            json_file  = raw_data_dir + '/train/%s-anatomical-structure.json' % id
            structure = draw_strcuture(read_json_as_df(json_file), height, width, structure=['Cortex'])

            try:
                mask_file = raw_data_dir + '/train/%s.corrected_shift_mask.png' % id
                mask  = read_mask(mask_file)
            except:
                mask_file  = raw_data_dir + '/train/%s.mask.png' % id
                mask  = read_mask(mask_file)

        if server == 'kaggle':
            image_file = raw_data_dir + '/test/%s.tiff' % id
            json_file  = raw_data_dir + '/test/%s-anatomical-structure.json' % id

            image = read_tiff(image_file)
            height, width = image.shape[:2]
            structure = draw_strcuture(read_json_as_df(json_file), height, width, structure=['Cortex'])

            mask = None


        #--- predict here!  ---
        # tile = to_tile(image, mask, structure, tile_scale, tile_size, tile_average_step, tile_min_score)
        tile = to_tile(image, mask, tile_scale, tile_size, tile_average_step, tile_min_score)


        tile_image = tile['tile_image']
        tile_image = np.stack(tile_image)[..., ::-1]
        tile_image = np.ascontiguousarray(tile_image.transpose(0, 3, 1, 2))
        tile_image = tile_image.astype(np.float32)/255

        tile_probability = []
        batch = np.array_split(tile_image, len(tile_image)//4)
        for t, m in enumerate(batch):
            print('\r %s  %d / %d   %s' %
                  (id, t, len(batch), time_to_str(timer() - start_timer, 'sec')),
                  end='', flush=True)
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

        print('\r' , end='', flush=True)
        log.write('%s  %d / %d   %s\n' %
                  (id, t, len(batch), time_to_str(timer() - start_timer, 'sec')))

        tile_probability = np.concatenate(tile_probability).squeeze(1)
        height, width = tile['image_small'].shape[:2]
        probability = to_mask(tile_probability, tile['coord'], height, width,
                              tile_scale, tile_size, tile_average_step, tile_min_score,
                              aggregate='mean')

        #--- show results ---
        if server == 'local':
            truth = tile['mask_small'].astype(np.float32)/255
        elif server == 'kaggle':
            truth = np.zeros((height, width), np.float32)

        overlay = np.dstack([
            np.zeros_like(truth),
            probability, #green
            truth, #red
        ])
        image_small = tile['image_small'].astype(np.float32)/255
        predict = (probability > 0.5).astype(np.float32)
        overlay1 = 1-(1-image_small)*(1-overlay)
        overlay2 = image_small.copy()
        # overlay2 = draw_contour_overlay(overlay2, tile['structure_small'], color=(1, 1, 1), thickness=3)
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
            cv2.imwrite(submit_dir + '/%s.image_small.png' % id, (image_small*255).astype(np.uint8))
            cv2.imwrite(submit_dir + '/%s.probability.png' % id, (probability*255).astype(np.uint8))
            cv2.imwrite(submit_dir + '/%s.predict.png' % id, (predict*255).astype(np.uint8))
            cv2.imwrite(submit_dir + '/%s.overlay.png' % id, (overlay*255).astype(np.uint8))
            cv2.imwrite(submit_dir + '/%s.overlay1.png' % id, (overlay1*255).astype(np.uint8))
            cv2.imwrite(submit_dir + '/%s.overlay2.png' % id, (overlay2*255).astype(np.uint8))

        #---

        if server == 'local':
            loss = np_binary_cross_entropy_loss(probability, truth)
            dice = np_dice_score(probability, truth)
            tp, tn = np_accuracy(probability, truth)
            log.write('submit_dir = %s \n' % submit_dir)
            log.write('initial_checkpoint = %s \n' % initial_checkpoint)
            log.write('loss   = %0.8f \n' % loss)
            log.write('dice   = %0.8f \n' % dice)
            log.write('tp, tn = %0.8f, %0.8f \n' % (tp, tn))
            log.write('\n')
            #cv2.waitKey(0)

    #-----
    if server == 'kaggle':
        csv_file = submit_dir + f'/submission_{sha}-%s-%s.csv' % (out_dir.split('/')[-1], iter_tag)
        df = mask_to_csv(valid_image_id, submit_dir)
        df.to_csv(csv_file, index=False)
        print(df)

    zz = 0





def run_make_csv():

    submit_dir = project_repo + '/result/Baseline/fold1'
    csv_file = submit_dir + '/kaggle-00004000_model-top1.csv'

    #-----
    image_id = make_image_id('test-all')
    predicted = []

    for id in image_id:
        print(id)
        image_file = raw_data_dir + '/test/%s.tiff' % id
        image = read_tiff(image_file)
        height, width = image.shape[:2]
        try:
            predict_file = submit_dir + '/%s.top.png' % id
            predict = np.array(PIL.Image.open(predict_file))
        except:
            predict_file = submit_dir + '/%s.predict.png' % id
            predict = np.array(PIL.Image.open(predict_file))


        predict = cv2.resize(predict, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
        predict = (predict > 128).astype(np.uint8)*255

        p = rle_encode(predict)
        predicted.append(p)

    df = pd.DataFrame()
    df['id'] = image_id
    df['predicted'] = predicted

    df.to_csv(csv_file, index=False)
    print(df)


########################################################################
# main #################################################################
########################################################################
if __name__ == '__main__':

    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument("-i", "--Iterations", help="number of iterations")
    parser.add_argument("-s", "--Server", help="run mode: server or kaggle")
    parser.add_argument("-f", "--fold", help="fold")

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

    if not args.Iterations:
        print("iterations missing")
        sys.exit()

    # if args.Iterations:
    #     print("Model taken at iterations: % s" % args.Iterations)
    #     model_checkpoint = f'{int(args.Iterations):08}_model.pth'
    #     print(f' using model: {model_checkpoint}')
    # else:
    #     print("iterations missing")
    #     sys.exit()

    if args.Server in ['kaggle', 'local']:
        print("Server: % s" % args.Server)
    else:
        print("Server missing")
        sys.exit()

    repo = git.Repo(search_parent_directories=True)
    model_sha = repo.head.object.hexsha[:9]
    print(f"current commit: {model_sha}")

    changedFiles = [item.a_path for item in repo.index.diff(None) if item.a_path.endswith(".py")]
    # if len(changedFiles) > 0:
    #     print("ABORT submission -- There are unstaged files:")
    #     for _file in changedFiles:
    #         print(f" * {_file}")
    #
    # else:
    submit(model_sha,
           server=args.Server,
           iterations=args.Iterations,
           fold=fold
           )

