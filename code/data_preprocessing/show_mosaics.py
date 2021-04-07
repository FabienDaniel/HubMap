import argparse
import os
import sys

import cv2
import numpy as np

from PIL import Image, ImageOps, ImageFont, ImageDraw
import PIL

image_size = 900


def add_border(input_image,  border=10):
    if isinstance(border, int) or isinstance(border, tuple):
        output_image = ImageOps.expand(input_image, border=border)
    else:
        raise RuntimeError('Border is not an integer or tuple!')
    return output_image


def display_mosaic(name, image_path, alpha=0.7, resize=0.1):
    images = []
    coords = []
    i = 0
    for _sub_image in os.listdir(image_path):
        if not _sub_image.endswith('.png'): continue
        images.append(os.path.join(image_path, _sub_image))

        raw_coords = _sub_image.strip('.png').split('_')
        x = int(raw_coords[1].strip('x'))
        y = int(raw_coords[0].strip('y'))
        coords.append([x, y])

        print(f"image n°{i}:", _sub_image, f'@ ({x}, {y})', end='\r')

        xmin = np.array(coords)[:, 0].min()
        xmax = np.array(coords)[:, 0].max()
        ymin = np.array(coords)[:, 1].min()
        ymax = np.array(coords)[:, 1].max()

        i += 1

    print()
    print("min/max x/y center positions:", xmin, xmax, ymin, ymax)

    image = Image.new('RGB', ((xmax-xmin) + image_size,
                              (ymax-ymin) + image_size))

    for i, _image in enumerate(images):
        predict = PIL.Image.open(_image)
        predict.putalpha(int(alpha * 256))
        predict = add_border(predict)
        x = coords[i][0] - xmin
        y = coords[i][1] - ymin
        print(f"image n°{i}", _image.split('/')[-1], (x, y), end='\r')
        image.paste(predict, (x, y), mask=predict)
    print('\n')

    to_display = np.array(image)
    H, W = to_display.shape[0:2]
    cv2.namedWindow(name, cv2.WINDOW_GUI_NORMAL)  # WINDOW_NORMAL
    cv2.imshow(name, to_display)
    cv2.resizeWindow(name, round(resize * W), round(resize * H))
    cv2.waitKey(0)

    return image


########################################################################
# main #################################################################
########################################################################
if __name__ == '__main__':

    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument("-f", "--fold", help="fold number")
    parser.add_argument("-s", "--server", help="server", default="kaggle")
    parser.add_argument("-m", "--model", help="model", default="Layer_2")

    args = parser.parse_args()

    if not args.fold:
        print("fold missing")
        sys.exit()

    directory = f"result/{args.model}/fold{args.fold}"
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

    # choice = 0

    pathdir = os.path.join(directory, options[choice])

    print(os.listdir(pathdir))

    print(35 * '-')
    image_options = []
    i = 0
    for _directory in os.listdir(pathdir):
        if not os.path.isdir(os.path.join(pathdir, _directory)): continue
        print(f"image n°{i}:", _directory)
        image_options.append(_directory)
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

    # image_choice = 0


    image_name = image_options[image_choice].split('.')[0]
    image = display_mosaic(image_name, os.path.join(pathdir, image_name))

    # width, height = image.size
    # scale = 0.1
    # image.resize((int(scale*width), int(scale*height)))
    # print("creating:", pathdir + '/%s.reconstructed.png' % image_name)
    #
    # image.save(pathdir + '/%s.reconstructed.png' % image_name, "PNG")
    # image.save(pathdir + '/%s.reconstructed.png' % image_name, quality=95)


