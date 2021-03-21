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


def mix(models, server, sha):

    submit_dir = project_repo + f"/result/mix_{sha}"
    os.mkdir(submit_dir)
    print(f"submit with server={server}")

    if server == 'local':
        valid_image_id = make_image_id('train-all')
    if server == 'kaggle':
        valid_image_id = make_image_id('test-all')

    for id in valid_image_id:
        print(f"processing: {id}")
        image = []
        for _model in models:
            _tmp = np.load(_model + f'/proba_{id}.npy.npz')
            image.append(_tmp['probability'])

        probability = np.mean(image, axis=0)
        predict = (probability > 0.5).astype(np.float32)
        cv2.imwrite(submit_dir + '/%s.predict.png' % id, (predict*255).astype(np.uint8))


    if server == 'kaggle':
        csv_file = submit_dir + f'/submission_mix_{sha}.csv'
        df = mask_to_csv(valid_image_id, submit_dir)
        df.to_csv(csv_file, index=False)
        print(df)


########################################################################
# main #################################################################
########################################################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--server", help="server")
    args = parser.parse_args()

    directory = 'result/Baseline/fold2_4_5/predictions_1a9083ac2/'
    models = [
        directory + 'kaggle-00005250-mean',
        directory + 'kaggle-00005750-mean',
        directory + 'kaggle-00006250-mean',
        directory + 'kaggle-00007500-mean',
        directory + 'kaggle-00007750-mean'
    ]

    if not args.server:
        print("server missing")
        sys.exit()

    repo = git.Repo(search_parent_directories=True)
    model_sha = repo.head.object.hexsha[:9]
    print(f"current commit: {model_sha}")

    changedFiles = [item.a_path for item in repo.index.diff(None) if item.a_path.endswith(".py")]
    if len(changedFiles) > 0:
        print("ABORT submission -- There are unstaged files:")
        for _file in changedFiles:
            print(f" * {_file}")

    else:
        mix(models, args.server, model_sha)
