import argparse
import os
import sys

import imutils
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


def get_images(server, id):
    if server == 'local':
        _tag = 'train'
    elif server == 'kaggle':
        _tag = 'test'
    image_file = raw_data_dir + f"/{_tag}/{id}.tiff"
    json_file = raw_data_dir + f"/{_tag}/{id}-anatomical-structure.json"
    image = read_tiff(image_file)

    if server == 'local':
        mask_file = raw_data_dir + f"/{_tag}/{id}.mask.png"
        mask = read_mask(mask_file)
    elif server == 'kaggle':
        mask = None

    return image, mask, json_file


def get_probas(id, net, tile_image, tile_centroids, tile, flip_predict, start_timer, log,
               tile_size, tile_average_step, tile_scale, tile_min_score, submit_dir):

    tile_probability = []
    batch = np.array_split(tile_image, len(tile_image) // 4)
    centroids = np.array_split(tile_centroids, len(tile_centroids) // 4)

    os.makedirs(submit_dir + f'/{id}', exist_ok=True)

    # num = 0
    for t, m in enumerate(batch):
        # centers = centroids[t]
        print('\r %s  %d / %d   %s' %
              (id, t, len(batch), time_to_str(timer() - start_timer, 'sec')),
              end='', flush=True)
        m = torch.from_numpy(m).cuda()

        p = []
        with torch.no_grad():
            # inference sur l'image de base
            logit = data_parallel(net, m)
            p.append(torch.sigmoid(logit))

            if flip_predict:  # inference sur les images inversées / axes x et y
                for _dim in [(2,), (3,), (2, 3)]:
                    _logit = data_parallel(net, m.flip(dims=_dim))
                    p.append(_logit.flip(dims=_dim))

        p = torch.stack(p).mean(0)
        # a = m.data.cpu().numpy()
        # b = p.data.cpu().numpy()

        # for i in range(a.shape[0]):
        #     print(f'image n°{i}', end='\r')
        #     base  = a[i, :, :, :]
        #     layer = b[i, :, :, :]
        #
        #     base = np.ascontiguousarray(base.transpose(1, 2, 0)).astype(np.float32)
        #     layer = layer.squeeze(0).astype(np.float32)
        #     overlay2 = draw_contour_overlay(base,
        #                                     layer,
        #                                     color=(0, 1, 0),
        #                                     thickness=3)
        #     num += 1
        #     print(num, centers[i])
        #     x0, y0 = centers[i][:2]
        #
        #     image_show_norm('overlay2',
        #                     overlay2,
        #                     min=0, max=1, resize=1)
        #
        #     cv2.imwrite(submit_dir + f'/{id}/y{y0}_x{x0}.png', (overlay2 * 255).astype(np.uint8))
        #
        #     cv2.waitKey(1)


        tile_probability.append(p.data.cpu().numpy())

    print('\r', end='', flush=True)
    log.write('%s  %d / %d   %s\n' %
              (id, t, len(batch), time_to_str(timer() - start_timer, 'sec')))

    # before squeeze, dimension = N_tiles x 1 x tile_x x tile_y
    # tile_probability = np.concatenate(tile_probability).squeeze(1)  # N_tiles x tile_x x tile_y
    # probability = to_mask(tile_probability,  # height x width
    #                       tile['coord'],
    #                       tile['height'],
    #                       tile['width'],
    #                       tile_scale,
    #                       tile_size,
    #                       tile_average_step,
    #                       tile_min_score,
    #                       aggregate='mean')
    # return probability

    return tile_probability


def layer2_tiling(image,
                  mask=None,
                  scale=0.25,
                  size=320,
                  step=192,
                  min_score=0.25,
                  layer1_path=None,
                  server=None
                  ):

    image_small, tile_images, tile_mask, centroids = extract_tiles_from_predictions(image, size, layer1_path, server)

    return {
        'image_small': image,
        'mask_small': mask,
        'tile_image': tile_images,
        'tile_mask': tile_mask,
        'coord': centroids,
        'reject': None,
        'height': image_small.shape[0],
        'width':  image_small.shape[1],
    }


def extract_tiles_from_predictions(id, tile_size, layer1_path, server):
    image_size = tile_size

    print(50*'-')
    print(f"processing image: {id}")

    if server == 'local':
        image_file = raw_data_dir + '/train/%s.tiff' % id
        mask_file = raw_data_dir + '/train/%s.mask.png' % id
        original_mask = read_mask(mask_file)

    elif server == 'kaggle':
        image_file = raw_data_dir + '/test/%s.tiff' % id

    image = read_tiff(image_file)

    height, width = image.shape[:2]
    print(f"image size: {height} x {width}")

    predict = np.array(PIL.Image.open(layer1_path + f'/{id}.predict.png'))
    mask = cv2.resize(predict, dsize=(width, height), interpolation=cv2.INTER_LINEAR)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # -----------------------------------------
    # loop over the contours to get centroids
    # -----------------------------------------
    centroid_list = []
    for c in cnts:
        # compute the center of the contour
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # draw the contour and center of the shape on the image
        # cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        # cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
        # cv2.putText(image, f"x={cX}, y={cY} ", (cX - 20, cY - 20),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # ---------------------------------------
        # check how close from image boundaries
        # ---------------------------------------
        if cX - image_size // 2 < 0:
            x0 = image_size // 2
        elif cX + image_size // 2 > image.shape[1]:
            x0 = image.shape[1] - image_size // 2
        else:
            x0 = cX

        if cY - image_size // 2 < 0:
            y0 = image_size // 2
        elif cY + image_size // 2 > image.shape[0]:
            y0 = image.shape[0] - image_size // 2
        else:
            y0 = cY

        centroid_list.append([x0, y0])

    # -------------------------
    # visualisation loop
    # -------------------------
    for cX, cY in centroid_list:
        resize = 1
        sub_image = image[
                    cY - image_size//2: cY + image_size//2,
                    cX - image_size//2: cX + image_size//2,
                    :]

        if server == 'local':

            # print(original_mask.shape)

            sub_mask = original_mask[
                        cY - image_size // 2: cY + image_size // 2,
                        cX - image_size // 2: cX + image_size // 2]

        if sub_image.shape[0] != sub_image.shape[1]:
            print(cX, cY, sub_image.shape, image.shape)
            sys.exit(1)

        # cv2.namedWindow("sub_Image", cv2.WINDOW_GUI_NORMAL)
        # cv2.imshow("sub_Image", sub_image)
        # cv2.resizeWindow('sub_Image', round(resize * image_size), round(resize * image_size))
        # cv2.waitKey(1)

    # -------------------------
    # loop over the centroids
    # -------------------------
    tile_image = []
    tile_mask = []
    coord = []
    for cX, cY in centroid_list:
        sub_image = image[
                    cY - image_size // 2: cY + image_size // 2,
                    cX - image_size // 2: cX + image_size // 2,
                    :]

        if server == 'local':
            sub_mask = original_mask[
                        cY - image_size // 2: cY + image_size // 2,
                        cX - image_size // 2: cX + image_size // 2]
            tile_mask.append(np.copy(sub_mask))

        coord.append([cX, cY, 1])
        tile_image.append(np.copy(sub_image))

    return image, tile_image, tile_mask, coord





def submit(sha, server, iterations, fold, flip_predict, checkpoint_sha, layer1):

    out_dir = project_repo + f"/result/Layer_2/fold{'_'.join(map(str, fold))}"
    if checkpoint_sha is not None:
        print("Checkpoint for current inference:", checkpoint_sha)
        _sha = checkpoint_sha
        _checkpoint_dir = out_dir + f"/checkpoint_{checkpoint_sha}/"
    else:
        _sha = sha
        _checkpoint_dir = out_dir + f"/checkpoint_{sha}/"

    if iterations == 'all':
        iter_tag = 'all'
        model_checkpoints = [_file for _file in os.listdir(_checkpoint_dir)]
        initial_checkpoint = [out_dir + f'/checkpoint_{_sha}/{model_checkpoint}'
                              for model_checkpoint in model_checkpoints]
    else:
        iter_tag = f"{int(iterations):08}"
        [model_checkpoint] = [_file for _file in os.listdir(_checkpoint_dir)
                              if iter_tag in _file.split('_')[0]]
        initial_checkpoint = [out_dir + f'/checkpoint_{_sha}/{model_checkpoint}']

    print("checkpoint:", initial_checkpoint)

    print(f"submit with server={server}")

    #---
    # print(checkpoint_sha, checkpoint_sha is not None)
    if checkpoint_sha is None:
        tag = ''
    else:
        tag = checkpoint_sha + '-'

    if iterations == 'all':
        submit_dir = out_dir + f'/predictions_{sha}/%s-%s-%smean' % (server, 'all', tag)
    elif flip_predict:
        submit_dir = out_dir + f'/predictions_{sha}/%s-%s-%smean' % (server, iter_tag, tag)
    else:
        submit_dir = out_dir + f'/predictions_{sha}/%s-%s-%snoflip' % (server, iter_tag, tag)

    os.makedirs(submit_dir, exist_ok=True)

    log = Logger()
    log.open(out_dir + f'/log.submit_{sha}.txt', mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))

    #---
    if server == 'local':
        valid_image_id = make_image_id('train-all')
    if server == 'kaggle':
        valid_image_id = make_image_id('test-all')

    tile_size = 640
    tile_average_step = 320
    tile_scale = 0.25
    tile_min_score = 0.25

    log.write('tile_size = %d \n' % tile_size)
    log.write('tile_average_step = %d \n' % tile_average_step)
    log.write('tile_scale = %f \n' % tile_scale)
    log.write('tile_min_score = %f \n' % tile_min_score)
    log.write('\n')

    predicted = []
    df = pd.DataFrame()

    start_timer = timer()
    for ind, id in enumerate(valid_image_id):

        # if ind != 2: continue

        print(50*"=")

        image, mask, json_file = get_images(server, id)

        print(f"Inference for image: {id}")

        tile = layer2_tiling(id, mask, tile_scale, tile_size, tile_average_step, tile_min_score, layer1, server)

        tile_image = tile['tile_image']
        tile_centroids = tile['coord']
        tile_image = np.stack(tile_image)[..., ::-1]
        tile_image = np.ascontiguousarray(tile_image.transpose(0, 3, 1, 2))
        tile_image = tile_image.astype(np.float32) / 255   # N_tiles x Colors x tile_x x tile_y

        if server == 'local':
            tile_mask = tile['tile_mask']
            tile_mask = np.ascontiguousarray(tile_mask)
            tile_mask = tile_mask.astype(np.float32) / 255  # N_tiles x Colors x tile_x x tile_y

        print(30 * '-')
        print("tile matrix shape:", tile_image.shape)


        # height, width = tile['image_small'].shape[:2]
        height = tile['height']
        width = tile['width']

        # image_small = tile['image_small'].astype(np.float32) / 255
        # print('---', image_small.shape)

        individual_probabilities = []
        overall_probabilities = []
        for _num, _checkpoint in enumerate(initial_checkpoint):

            print("processing checkpoint:", _checkpoint)

            net = Net().cuda()
            state_dict = torch.load(_checkpoint, map_location=lambda storage, loc: storage)['state_dict']
            net.load_state_dict(state_dict, strict=True)
            net = net.eval()

            tile_probability = get_probas(
                id, net, tile_image, tile_centroids, tile, flip_predict, start_timer, log,
                tile_size, tile_average_step, tile_scale, tile_min_score,
                submit_dir
            )

            # before squeeze, dimension = N_tiles x tile_x x tile_y
            tile_probability = np.concatenate(tile_probability).squeeze(1)  # N_tiles x tile_x x tile_y
            overall_probabilities.append(tile_probability)
            probas = np.mean(overall_probabilities, axis=0)

            num = 0
            for i, image in enumerate(tile_image):
                print(f'image n°{i}', end='\r')

                x0, y0 = tile_centroids[i][:2]

                if server == 'local':
                    overlay = draw_contour_overlay(
                        np.ascontiguousarray(image.transpose(1, 2, 0)).astype(np.float32),
                        tile_mask[i, :, :].astype(np.float32),
                        color=(1, 0, 0),
                        thickness=6
                    )
                else:
                    overlay = np.ascontiguousarray(image.transpose(1, 2, 0)).astype(np.float32)

                if len(overall_probabilities) == 1:
                    overlay2 = draw_contour_overlay(
                        overlay,
                        tile_probability[i, :, :].astype(np.float32),
                        color=(0, 1, 0),
                        thickness=6
                    )
                    cv2.imwrite(submit_dir + f'/{id}/y{y0}_x{x0}.png', (overlay2 * 255).astype(np.uint8))
                else:
                    overlay2 = draw_contour_overlay(
                        overlay,
                        probas[i, :, :].astype(np.float32),
                        color=(0, 1, 0),
                        thickness=6
                    )
                    cv2.imwrite(submit_dir + f'/{id}/y{y0}_x{x0}.png', (overlay2 * 255).astype(np.uint8))
                    overlay2 = draw_contour_overlay(
                        overlay2,
                        tile_probability[i, :, :].astype(np.float32),
                        color=(0, 0, 1),
                        thickness=3
                    )

                num += 1

                image_show_norm('overlay2',
                                overlay2,
                                min=0, max=1, resize=1)
                cv2.waitKey(1)
                # cv2.waitKey((0, 1)[_num == 0])

            probability = to_mask(tile_probability,  # height x width
                                  tile['coord'],
                                  tile['height'],
                                  tile['width'],
                                  tile_scale,
                                  tile_size,
                                  tile_average_step,
                                  tile_min_score,
                                  aggregate='mean')

            individual_probabilities.append(probability)

            if _num == 0: break

        probability = np.mean(individual_probabilities, axis=0)

        # sys.exit()

        # -------------------------------------------------
        # Saves the numpy array that contains probabilities
        # np.savez_compressed(submit_dir + f'/proba_{id}.npy', probability=probability)

        #--- show results ---
        if server == 'local':
            truth = tile['mask_small'].astype(np.float32)/255
        elif server == 'kaggle':
            truth = np.zeros((height, width), np.float32)


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

        elif server == 'kaggle':
            predict = (probability > 0.5).astype(np.float32)
            p = rle_encode(predict)
            predicted.append(p)
            # _df = pd.DataFrame()
            # df['id'] = id
            # df['predicted'] = predicted


    #-----
    if server == 'kaggle':
        df['id'] = valid_image_id
        df['predicted'] = predicted
        csv_file = submit_dir + f'/submission_{sha}-%s-%s%s.csv' % (out_dir.split('/')[-1], tag, iter_tag)
        df.to_csv(csv_file, index=False)
        print(df)





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
    parser.add_argument("-r", "--flip", help="flip image and merge", default=True)
    parser.add_argument("-c", "--CheckpointSha", help="checkpoint with weights", default=None)
    parser.add_argument("-l", "--layer1", help="predictions from first layer", default=None)

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

    if not args.layer1:
        print("first layer predictions missing")
        sys.exit()

    if not args.Iterations:
        print("iterations missing")
        sys.exit()

    if args.Server in ['kaggle', 'local']:
        print("Server: % s" % args.Server)
    else:
        print("Server missing")
        sys.exit()

    repo = git.Repo(search_parent_directories=True)
    model_sha = repo.head.object.hexsha[:9]
    print(f"current commit: {model_sha}")

    # changedFiles = [item.a_path for item in repo.index.diff(None) if item.a_path.endswith(".py")]
    # if len(changedFiles) > 0:
    #     print("ABORT submission -- There are unstaged files:")
    #     for _file in changedFiles:
    #         print(f" * {_file}")
    #
    # else:
    submit(model_sha,
           server=args.Server,
           iterations=args.Iterations,
           fold=fold,
           flip_predict=args.flip,
           checkpoint_sha=args.CheckpointSha,
           layer1 = args.layer1
           )

