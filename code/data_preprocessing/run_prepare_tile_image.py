import cv2
import pandas as pd
import numpy as np
import os
import imutils

# from code.data_preprocessing.dataset import to_tile, read_tiff, image_show_norm
# from code.hubmap_v2 import data_dir, image_show, raw_data_dir, project_repo, draw_contour_overlay
from code.data_preprocessing.dataset import read_tiff, to_tile

import PIL
import sys

# tile_scale = 0.25
# tile_size  = 320
# tile_average_step = 192
# tile_min_score = 0.25
from code.hubmap_v2 import get_data_path

project_repo, raw_data_dir, data_dir = get_data_path('local')

# --- rle ---------------------------------
from code.lib.utility.draw import image_show


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


def extract_centroids_from_predictions(
        tile_scale,
        tile_size,
        train_tile_dir,
        sha,
        pred_tag,
        base_path
):
    """
    On récupère un .png avec les prédictions.
    On extraie les contours de ce .png.
    On compare les centroides des contours extraits aux positions des mask réels. Si la distance
     à supérieure à D (=500), l'image n'est pas conservée (car associée à un TP).
    ----------------------------------------------------------------------------------------------------
    Les sous images sauvegardées correspondent à des FP. Les sous-images sont centrées sur les positions
    des centroides des mauvaises prédictions.
    """
    df_train = pd.read_csv(raw_data_dir + '/train.csv')
    image_size = tile_size

    os.makedirs(train_tile_dir, exist_ok=True)
    for i in range(0, len(df_train)):

        id, encoding = df_train.iloc[i]

        # if id != 'c68fe75ea': continue

        real_mask_centroids = pd.read_csv(
            project_repo +
            f'/data/tile/mask_{tile_size}_{tile_scale}_centroids' + f"/centroids_{id}.csv",
            index_col=0
        )

        os.makedirs(train_tile_dir + '/%s' % id, exist_ok=True)

        print(50*'-')
        print(f"processing image: {id}")

        image_file = raw_data_dir + '/train/%s.tiff' % id
        image = read_tiff(image_file)
        height, width = image.shape[:2]
        print(f"image size: {height} x {width}")

        original_mask = rle_decode(encoding, height, width, 255)
        predict = np.array(PIL.Image.open(
            os.path.join(base_path, f"{id[:9]}.predict.png")
        ))
        mask = cv2.resize(predict, dsize=(width, height), interpolation=cv2.INTER_LINEAR)

        if tile_scale < 1:
            print(f"*** Scales down image by a factor: {tile_scale}")
            image = cv2.resize(image, dsize=None,
                                     fx=tile_scale, fy=tile_scale, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, dsize=None,
                                    fx=tile_scale, fy=tile_scale, interpolation=cv2.INTER_LINEAR)
            height, width, _ = image.shape
            print(f"\tscaled image size = {height} x {width}")

        image_copy = np.copy(image)

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

            ########################################################
            ### On vérifie la distance / à la liste des vrais masques
            ########################################################
            distance = real_mask_centroids.apply(lambda x: ((x[0] - cX)**2 + (x[1] - cY)**2)**0.5, axis=1)
            if distance.min() < 500:
                # print(f"Skips correct prediction @ {cX} {cY}")
                continue
            else:
                print(f"Wrong prediction @ {cX} {cY} d={distance.min()}")

            # draw the contour and center of the shape on the image
            cv2.drawContours(image_copy, [c], -1, (0, 255, 0), 2)
            cv2.circle(image_copy, (cX, cY), 7, (255, 255, 255), -1)
            cv2.putText(image_copy, f"x={cX}, y={cY} ", (cX - 20, cY - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            ########################################################
            ### check how close from image boundaries
            ########################################################
            if cX - image_size // 2 < 0:
                x0 = image_size // 2
            elif cX + image_size // 2 > image_copy.shape[1]:
                x0 = image_copy.shape[1] - image_size // 2
            else:
                x0 = cX

            if cY - image_size // 2 < 0:
                y0 = image_size // 2
            elif cY + image_size // 2 > image_copy.shape[0]:
                y0 = image_copy.shape[0] - image_size // 2
            else:
                y0 = cY

            centroid_list.append([x0, y0])

        # -------------------------
        # visualisation loop
        # -------------------------
        for cX, cY in centroid_list:
            resize = 1
            sub_image = image_copy[
                        cY - image_size//2: cY + image_size//2,
                        cX - image_size//2: cX + image_size//2,
                        :]

            if sub_image.shape[0] != sub_image.shape[1]:
                print(cX, cY, sub_image.shape, image_copy.shape)
                sys.exit(1)

            cv2.namedWindow("sub_Image", cv2.WINDOW_GUI_NORMAL)
            cv2.imshow("sub_Image", sub_image)
            cv2.resizeWindow('sub_Image', round(resize * image_size), round(resize * image_size))
            cv2.waitKey(1)

        ########################################################
        ### loop over the centroids.
        ### Les images et la masques sont sauvegardés sur disque.
        ########################################################
        image_size = tile_size
        for cX, cY in centroid_list:
            sub_image = image[
                        cY - image_size // 2: cY + image_size // 2,
                        cX - image_size // 2: cX + image_size // 2,
                        :]
            sub_mask = original_mask[
                       cY - image_size // 2: cY + image_size // 2,
                       cX - image_size // 2: cX + image_size // 2]

            # print(cX, cY, sub_image.shape)
            s = 'y%08d_x%08d' % (cY, cX)
            cv2.imwrite(train_tile_dir + '/%s/%s.png' % (id, s), sub_image)
            cv2.imwrite(train_tile_dir + '/%s/%s.mask.png' % (id, s), sub_mask)

        pd.DataFrame(centroid_list, columns={'x', 'y'}).to_csv(train_tile_dir + '/centroids_%s.csv' % id)


##########################################################################################################
def extract_centroids_from_L2_predictions(
        tile_scale,
        image_size,
        train_tile_dir,
        sha,
        pred_tag,
        base_path
):
    """
    On récupère un .png avec les prédictions.
    On extraie les contours de ce .png.
    On compare les centroides des contours extraits aux positions des mask réels. Si la distance
     à supérieure à D (=500), l'image n'est pas conservée (car associée à un TP).
    ----------------------------------------------------------------------------------------------------
    Les sous images sauvegardées correspondent à des FP. Les sous-images sont centrées sur les positions
    des centroides des mauvaises prédictions.
    """
    df_train = pd.read_csv(raw_data_dir + '/train.csv')
    # image_size = tile_size
    os.makedirs(train_tile_dir, exist_ok=True)

    ###########################################################
    ### On boucle sur les images de training
    ###########################################################
    for i in range(0, len(df_train)):
        id, encoding = df_train.iloc[i]

        # if id != 'c68fe75ea': continue

        os.makedirs(train_tile_dir + '/%s' % id, exist_ok=True)

        real_mask_centroids = pd.read_csv(
            project_repo +
            f'/data/tile/mask_{tile_size}_{tile_scale}_centroids' + f"/centroids_{id}.csv",
            index_col=0
        )

        print(50 * '-')
        print(f"processing image: {id}")

        #-----------------------------
        # Lecture de l'image de base
        # -----------------------------
        image_file = raw_data_dir + '/train/%s.tiff' % id
        image = read_tiff(image_file)
        height, width = image.shape[:2]
        print(f"image size: {height} x {width}")

        # ------------------------------------------
        # Extraction du masque de l'image initiale
        # ------------------------------------------
        original_mask = rle_decode(encoding, height, width, 255)

        if tile_scale < 1:
            print(f"*** Scales down image by a factor: {tile_scale}")
            image = cv2.resize(image, dsize=None,
                               fx=tile_scale, fy=tile_scale, interpolation=cv2.INTER_LINEAR)
            original_mask = cv2.resize(original_mask, dsize=None,
                              fx=tile_scale, fy=tile_scale, interpolation=cv2.INTER_LINEAR)
            height, width, _ = image.shape
            print(f"\tscaled image size = {height} x {width}")

        image_copy = np.copy(image)

        # ---------------------------------------------
        # Extraction des masques issus des prédictions
        # ---------------------------------------------

        file = os.path.join(base_path, f"{id}.csv")
        df = pd.read_csv(file, index_col=0)
        if tile_scale < 1:
            df['x'] = tile_scale * df['x']
            df['y'] = tile_scale * df['y']

        for col in ['x', 'y']:
            df[col] = df[col].astype(int)

        # print(df.head())
        # sys.exit()

        _tmp = df[df['dice'] < 0.8]
        print(_tmp.shape)
        print(_tmp.head())

        cnts = _tmp[['x', 'y', 'dice']].values

        # -----------------------------------------
        # loop over the contours to get centroids
        # -----------------------------------------
        centroid_list = []
        for cX, cY, score in cnts:
            ########################################################
            ### On vérifie la distance / à la liste des vrais masques
            ########################################################
            distance = real_mask_centroids.apply(lambda x: ((x[0] - cX) ** 2 + (x[1] - cY) ** 2) ** 0.5, axis=1)
            if distance.min() < 100 * tile_scale:
                # print(f"Skips correct prediction @ {cX} {cY}")
                continue
            else:
                print(f"Wrong prediction @ {cX} {cY} d={distance.min()}, score={score}")

            ########################################################
            ### check how close from image boundaries
            ########################################################
            if cX - image_size // 2 < 0:
                x0 = image_size // 2
            elif cX + image_size // 2 > image_copy.shape[1]:
                x0 = image_copy.shape[1] - image_size // 2
            else:
                x0 = cX

            if cY - image_size // 2 < 0:
                y0 = image_size // 2
            elif cY + image_size // 2 > image_copy.shape[0]:
                y0 = image_copy.shape[0] - image_size // 2
            else:
                y0 = cY

            centroid_list.append([int(x0), int(y0)])

        # -------------------------
        # visualisation loop
        # -------------------------
        for cX, cY in centroid_list:
            resize = 1
            sub_image = image_copy[
                        cY - image_size // 2: cY + image_size // 2,
                        cX - image_size // 2: cX + image_size // 2,
                        :]

            if sub_image.shape[0] != sub_image.shape[1]:
                print(cX, cY, sub_image.shape, image_copy.shape)
                sys.exit(1)

            cv2.namedWindow("sub_Image", cv2.WINDOW_GUI_NORMAL)
            cv2.imshow("sub_Image", sub_image)
            cv2.resizeWindow('sub_Image', round(resize * image_size), round(resize * image_size))
            cv2.waitKey(1)

        # sys.exit()

        ########################################################
        ### loop over the centroids.
        ### Les images et la masques sont sauvegardés sur disque.
        ########################################################
        # image_size = tile_size
        for cX, cY in centroid_list:
            sub_image = image[
                        cY - image_size // 2: cY + image_size // 2,
                        cX - image_size // 2: cX + image_size // 2,
                        :]
            sub_mask = original_mask[
                       cY - image_size // 2: cY + image_size // 2,
                       cX - image_size // 2: cX + image_size // 2]

            # print(cX, cY, sub_image.shape)
            s = 'y%08d_x%08d' % (cY, cX)

            # print(train_tile_dir + '/%s/%s.png' % (id, s))

            cv2.imwrite(train_tile_dir + '/%s/%s.png' % (id, s), sub_image)
            cv2.imwrite(train_tile_dir + '/%s/%s.mask.png' % (id, s), sub_mask)

        pd.DataFrame(centroid_list, columns={'x', 'y'}).to_csv(train_tile_dir + '/centroids_%s.csv' % id)


def extract_mask_centroids(
        tile_scale,
        tile_size,
        train_tile_dir
):
    image_size = tile_size
    df_train = pd.read_csv(raw_data_dir + '/train.csv')

    os.makedirs(train_tile_dir, exist_ok=True)
    for i in range(0, len(df_train)):
        id, encoding = df_train.iloc[i]

        # if id !='e79de561c': continue

        os.makedirs(train_tile_dir + '/%s' % id, exist_ok=True)

        print(50*'-')
        print(f"processing image: {id}")

        image_file = raw_data_dir + '/train/%s.tiff' % id
        image = read_tiff(image_file)

        height, width = image.shape[:2]
        print(f"image size: {height} x {width}")

        mask = rle_decode(encoding, height, width, 255)

        if tile_scale < 1:
            print(f"*** Scales down image by a factor: {tile_scale}")
            image = cv2.resize(image, dsize=None,
                                     fx=tile_scale, fy=tile_scale, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, dsize=None,
                                    fx=tile_scale, fy=tile_scale, interpolation=cv2.INTER_LINEAR)
            height, width, _ = image.shape
            print(f"\tscaled image size = {height} x {width}")

        image_copy = np.copy(image)

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

            # ---------------------------------------
            # check how close from image boundaries
            # ---------------------------------------
            if cX - image_size // 2 < 0:
                x0 = image_size // 2
            elif cX + image_size // 2 > width:
                x0 = width - image_size // 2
            else:
                x0 = cX

            if cY - image_size // 2 < 0:
                y0 = image_size // 2
            elif cY + image_size // 2 > height:
                y0 = height - image_size // 2
            else:
                y0 = cY

            # draw the contour and center of the shape on the image
            # cv2.drawContours(image_copy, [c], -1, (0, 255, 0), 2)
            # cv2.circle(image_copy, (cX, cY), 7, (255, 255, 255), -1)
            # cv2.putText(image_copy, f"x={cX}, y={cY} ", (cX - 20, cY - 20),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # draw the contour and center of the shape on the image
            cv2.drawContours(image_copy, [c], -1, (0, 255, 0), 2)
            cv2.circle(image_copy, (x0, y0), 7, (255, 255, 255), -1)
            cv2.putText(image_copy, f"x={x0}, y={y0} ", (x0 - 20, y0 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            centroid_list.append([x0, y0])

            # centroid_list.append([cX, cY])

        # -------------------------
        # visualisation loop
        # -------------------------
        image_size = tile_size
        for cX, cY in centroid_list:
            resize = 1
            sub_image = image_copy[
                        cY - image_size // 2: cY + image_size // 2,
                        cX - image_size // 2: cX + image_size // 2,
                        :]

            # print(cX, cY, sub_image.shape)

            cv2.namedWindow("sub_Image", cv2.WINDOW_GUI_NORMAL)
            cv2.imshow("sub_Image", sub_image)
            cv2.resizeWindow('sub_Image', round(resize * image_size), round(resize * image_size))
            cv2.waitKey(1)


        # -------------------------
        # loop over the centroids
        # -------------------------
        for cX, cY in centroid_list:
            sub_image = image[
                        cY - image_size // 2: cY + image_size // 2,
                        cX - image_size // 2: cX + image_size // 2,
                        :]
            sub_mask = mask[
                       cY - image_size // 2: cY + image_size // 2,
                       cX - image_size // 2: cX + image_size // 2]

            s = 'y%08d_x%08d' % (cY, cX)
            cv2.imwrite(train_tile_dir + '/%s/%s.png' % (id, s), sub_image)
            cv2.imwrite(train_tile_dir + '/%s/%s.mask.png' % (id, s), sub_mask)

        pd.DataFrame(centroid_list).to_csv(train_tile_dir + '/centroids_%s.csv' % id)


def run_make_train_tile(tile_scale,
                        tile_size,
                        train_tile_dir):

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
        tile = to_tile(image, mask, scale=tile_scale, size=tile_size)

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


#############################
# make tile train image
#############################
# def run_make_train_mask():
#
#     df_train = pd.read_csv(data_dir + '/train.csv')
#     print(df_train)
#     print(df_train.shape)
#
#     for i in range(0, len(df_train)):
#         _id, encoding = df_train.iloc[i]
#
#         image_file = data_dir + '/train/%s.tiff' % _id
#         image = read_tiff(image_file)
#
#         height, width = image.shape[:2]
#         mask = rle_decode(encoding, height, width, 255)
#
#         cv2.imwrite(data_dir + '/train/%s.mask.png' % _id, mask)


# main #################################################################
if __name__ == '__main__':

    tile_scale = 0.5
    tile_size = 700

    # run_make_train_tile(
    #     tile_scale=tile_scale,
    #     tile_size=tile_size,
    #     train_tile_dir = project_repo + f'/data/tile/{tile_scale}_{tile_size}_train'
    # )

    # extract_mask_centroids(
    #     tile_scale=tile_scale,
    #     tile_size=tile_size,
    #     train_tile_dir=project_repo + f'/data/tile/mask_{tile_size}_{tile_scale}_centroids'
    # )

    # sha = "18924a797"   #  "4707bcbcf"
    # pred_tag = 'local-all-mean'
    # base_path = f"result/Baseline/fold6_9_10/predictions_{sha}/{pred_tag}"
    #
    # extract_centroids_from_predictions(
    #     tile_scale=tile_scale,
    #     tile_size=tile_size,
    #     train_tile_dir=project_repo + f'/data/tile/predictions_{sha}_{tile_size}_{tile_scale}_centroids',
    #     sha=sha,
    #     pred_tag=pred_tag,
    #     base_path=base_path
    # )

    # sha = "680598dcf"
    # pred_tag = 'top3-587bbaf61-mean'
    # base_path = f"result/Layer_2/fold6_9_10/predictions_{sha}/local-{pred_tag}"

    sha = "8c8658346"
    pred_tag = 'top3-2d5650f29-mean'
    base_path = f"result/Layer_2/fold6_9_10/predictions_{sha}/local-{pred_tag}"

    extract_centroids_from_L2_predictions(
        tile_scale=tile_scale,
        image_size=tile_size,
        train_tile_dir=project_repo + f'/data/tile/predictions_{sha}_{pred_tag}_{tile_size}_{tile_scale}_centroids',
        sha=sha,
        pred_tag=pred_tag,
        base_path=base_path
    )
