import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

from PIL import Image
import PIL

from code.data_preprocessing.dataset import read_tiff
from code.hubmap_v2 import rle_decode, draw_contour_overlay, image_show_norm, get_data_path


def display_images(name, image, resize=0.05):
    predict = np.array(PIL.Image.open(image))
    H, W = predict.shape[0:2]
    cv2.namedWindow(name, cv2.WINDOW_GUI_NORMAL)  # WINDOW_NORMAL
    cv2.imshow(name, predict)
    cv2.resizeWindow(name, round(resize * W), round(resize * H))
    cv2.waitKey(0)


def display_predictions(submission, image_id):

    project_repo, raw_data_dir, data_dir = get_data_path('local')

    image_file = raw_data_dir + '/test/%s.tiff' % image_id
    image = read_tiff(image_file)
    image = image.astype(np.float32) / 255
    height, width = image.shape[:2]
    print(f"image size: {height} x {width}")

    print(45*"-")
    for index, sub in enumerate(submission):
        df_sub = pd.read_csv(sub)

        for i in range(0, len(df_sub)):
            id, encoding = df_sub.iloc[i]

            if id != image_id: continue

            mask = rle_decode(encoding, height, width, 255)
            if index == 0:
                print(f"GREEN: {sub.split('/')[-1]}")
                overlay2 = draw_contour_overlay(image, mask, color=(0, 1, 0), thickness=15)
            else:
                print(f"BLUE: {sub.split('/')[-1]}")
                overlay2 = draw_contour_overlay(overlay2, mask, color=(1, 0, 0), thickness=13)

    print(45 * "-")
    image_show_norm('model comparison', overlay2, min=0, max=1, resize=0.5)
    cv2.waitKey(0)
    return


########################################################################
# main #################################################################
########################################################################
if __name__ == '__main__':

    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument("-f", "--fold", help="fold number")
    parser.add_argument("-s", "--server", help="server", default="kaggle")
    parser.add_argument("-m", "--model", help="model", default="Baseline")

    args = parser.parse_args()

    if not args.fold:
        print("fold missing")
        sys.exit()

    if not args.model:
        print("model missing")
        sys.exit()
    else:
        if isinstance(args.model, str):
            models = args.model.split()

    ##################################################
    ### Get submission files
    ##################################################
    submission = []
    file_count = 1
    for model in models:
        directory = f"result/{model}/fold{args.fold}"
        print(60*"=")
        print(f"predictions available for fold n°{args.fold}, model '{model}':")
        print(60*'=')

        options = []
        i = 0
        for _file in [f for f in os.listdir(directory) if 'predictions' in f]:
            for _subdir in os.listdir(os.path.join(directory, _file)):
                if not 'kaggle' in _subdir: continue
                print(f"choice n°{i}:", os.path.join(_file, _subdir))
                options.append(os.path.join(_file, _subdir))
                i += 1

        print(35 * '-')
        while True:
            choice = input("Enter a choice: \n")
            # if file_count == 1:
            #     choice = 1
            # else:
            #     choice = 3

            try:
                if int(choice) < 0 or int(choice) >= len(options):
                    print("Choice not allowed. ", end='')
                else:
                    choice = int(choice)

                    pathdir = os.path.join(directory, options[choice])
                    filelist = [_file for _file in os.listdir(pathdir) if '.csv' in _file]
                    if len(filelist) == 1:
                        submission.append(os.path.join(pathdir, filelist[0]))
                        print(f"submission n°{file_count}: {filelist[0]} \n")
                        file_count += 1
                        break
                    else:
                        print('no valid/multiple submission files')

            except:
                print("Choice not allowed. ", end='')

    image_ids = {
        0: "2ec3f1bb9",
        1: "3589adb90",
        2: "57512b7f1",
        3: "aa05346ff",
        4: "d488c759a"
    }
    for k, v in image_ids.items():
        print(k, v)
    iid = input("Enter image id: \n")

    display_predictions(submission, image_id=image_ids[int(iid)])

