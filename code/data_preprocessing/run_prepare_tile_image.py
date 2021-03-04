import cv2
import pandas as pd
import numpy as np
import os

from code.data_preprocessing.dataset import to_tile, read_tiff
from code.hubmap_v2 import data_dir, image_show, raw_data_dir, project_repo

tile_scale = 0.25
tile_size  = 320
tile_average_step = 192
tile_min_score = 0.25


# --- rle ---------------------------------
def rle_decode(rle, height, width , fill=255):
    s = rle.split()
    start, length = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    start -= 1
    mask = np.zeros(height * width, dtype=np.uint8)
    for i, l in zip(start, length):
        mask[i:i+l] = fill
    mask = mask.reshape(width, height).T
    mask = np.ascontiguousarray(mask)
    return mask


def rle_encode(mask):
    m = mask.T.flatten()
    m = np.concatenate([[0], m, [0]])
    run = np.where(m[1:] != m[:-1])[0] + 1
    run[1::2] -= run[::2]
    rle = ' '.join(str(r) for r in run)
    return rle


def run_make_train_tile(train_tile_dir):

    df_train = pd.read_csv(raw_data_dir + '/train.csv')
    print(df_train)
    print(df_train.shape)

    os.makedirs(train_tile_dir, exist_ok=True)
    for i in range(0, len(df_train)):
        id, encoding = df_train.iloc[i]

        # if id != 'e79de561c': continue

        print(50*'-')
        print(f"processing image: {id}")

        image_file = raw_data_dir + '/train/%s.tiff' % id
        image = read_tiff(image_file)

        height, width = image.shape[:2]
        print(f"image size: {height} x {width}")

        mask = rle_decode(encoding, height, width, 255)
        cv2.imwrite(raw_data_dir + '/train/%s.mask.png' % id , mask)

        #make tile
        tile = to_tile(image, mask)

        coord = np.array(tile['coord'])
        df_image = pd.DataFrame()
        df_image['cx'] = coord[:, 0].astype(np.int32)
        df_image['cy'] = coord[:, 1].astype(np.int32)
        df_image['cv'] = coord[:, 2]

        # --- save ---
        os.makedirs(train_tile_dir+'/%s' % id, exist_ok=True)

        tile_id = []
        num = len(tile['tile_image'])
        for t in range(num):
            cx, cy, cv   = tile['coord'][t]
            s = 'y%08d_x%08d' % (cy, cx)
            tile_id.append(s)

            tile_image = tile['tile_image'][t]
            tile_mask  = tile['tile_mask'][t]
            cv2.imwrite(train_tile_dir + '/%s/%s.png' % (id, s), tile_image)
            cv2.imwrite(train_tile_dir + '/%s/%s.mask.png' % (id, s), tile_mask)

            image_show('tile_image', tile_image)
            image_show('tile_mask', tile_mask)
            cv2.waitKey(1)


        df_image['tile_id'] = tile_id
        df_image[['tile_id', 'cx', 'cy', 'cv']].to_csv(train_tile_dir+'/%s.csv' % id, index=False)
        #------


#make tile train image
def run_make_train_mask():

    df_train = pd.read_csv(data_dir + '/train.csv')
    print(df_train)
    print(df_train.shape)

    for i in range(0,len(df_train)):
        id, encoding = df_train.iloc[i]

        image_file = data_dir + '/train/%s.tiff' % id
        image = read_tiff(image_file)

        height, width = image.shape[:2]
        mask = rle_decode(encoding, height, width, 255)

        cv2.imwrite(data_dir + '/train/%s.mask.png' % id, mask)


# main #################################################################
if __name__ == '__main__':

    run_make_train_tile(
        train_tile_dir = project_repo + '/data/tile/0.25_320_train'
    )
