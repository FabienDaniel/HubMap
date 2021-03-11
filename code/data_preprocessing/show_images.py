import argparse
import os
import sys

import cv2
import numpy as np

from PIL import Image
import PIL


def display_images(name, image, resize=0.05):
    predict = np.array(PIL.Image.open(image))
    H, W = predict.shape[0:2]
    cv2.namedWindow(name, cv2.WINDOW_GUI_NORMAL)  # WINDOW_NORMAL
    cv2.imshow(name, predict)
    cv2.resizeWindow(name, round(resize * W), round(resize * H))
    cv2.waitKey(0)


########################################################################
# main #################################################################
########################################################################
if __name__ == '__main__':

    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument("-f", "--fold", help="fold number")
    parser.add_argument("-s", "--server", help="server", default="kaggle")
    parser.add_argument("-o", "--overlay", help="overlay", default="overlay2")

    args = parser.parse_args()

    if not args.fold:
        print("fold missing")
        sys.exit()

    directory = f"result/Baseline/fold{args.fold}"
    print(35*"=")
    print(f"predictions available for fold n°{args.fold}:")
    print(35*'=')

    options = []
    i = 0
    for _file in [f for f in os.listdir(directory) if 'predictions' in f]:
        for _subdir in os.listdir(os.path.join(directory, _file)):
            print(f"choice n°{i}:", os.path.join(_file, _subdir))
            options.append(os.path.join(_file, _subdir))
            i += 1

    print(35 * '-')
    while True:
        choice = input("Enter a choice: \n")
        try:
            if int(choice) < 0 or int(choice) >= len(options):
                print("Choice not allowed. ", end='')
            else:
                choice = int(choice)
                break
        except:
            print("Choice not allowed. ", end='')

    pathdir = os.path.join(directory, options[choice])

    print(35 * '-')
    image_options = []
    i = 0
    for _file in os.listdir(pathdir):
        # if 'overlay2' not in _file: continue
        if args.overlay not in _file: continue
        print(f"image n°{i}:", _file)
        image_options.append(_file)
        i += 1

    print(35 * '-')
    while True:
        image_choice = input("Enter a choice: \n")
        try:
            if int(image_choice) < 0 or int(image_choice) >= len(image_options):
                print("Choice not allowed. ", end='')
            else:
                image_choice = int(image_choice)
                break
        except:
            print("Choice not allowed. ", end='')

    image_name = image_options[image_choice].split('.')[0]


    resizing = {
        'b2dc8411c': 0.15,
        '26dc41664': 0.07,
        'c68fe75ea': 0.07,
        'afa5e8098': 0.07,
        'aaa6a05cc': 0.2,
    }

    display_images(
        image_name,
        os.path.join(os.path.join(pathdir, image_options[image_choice])),
        resize=resizing.get(image_name, 0.1)
    )

