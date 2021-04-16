import argparse
import os
import sys

import albumentations as albu
import torch
from torch.utils.data import Dataset

from code.hubmap_v2 import *
from code.lib.include import seed_py
from code.lib.include_torch import seed_torch
from code.unet_b_resnet34_aug_corrected.image_preprocessing import do_random_crop, do_random_rotate_crop, \
    do_random_scale_crop, do_random_hsv, do_random_contast, do_random_gain, do_random_noise, \
    do_random_flip_transpose


def make_image_id(mode='train', valid_ids=None):
    train_image_id = {
        0 : '0486052bb',
        1 : '095bf7a1f',
        2 : '1e2425f28',
        3 : '26dc41664',
        4 : '2f6ecfcdf',
        5 : '4ef6695ce',
        6 : '54f2eec69',
        7 : '8242609fa',
        8 : 'aaa6a05cc',
        9 : 'afa5e8098',
        10: 'b2dc8411c',
        11: 'b9a3865fc',
        12: 'c68fe75ea',
        13: 'cb2d976f4',
        14: 'e79de561c',
    }
    test_image_id = {
        0 : '2ec3f1bb9',
        1 : '3589adb90',
        2 : 'd488c759a',
        3 : 'aa05346ff',
        4 : '57512b7f1',
    }

    if mode == 'test-all':
        return list(test_image_id.values())

    elif mode == 'train-all':
        return list(train_image_id.values())

    elif mode == 'valid':
        return [train_image_id[i] for i in valid_ids]

    elif mode == 'train':
        train_ids = [i for i in train_image_id.keys() if i not in valid_ids]
        return [train_image_id[i] for i in train_ids]


class CenteredHuDataset(Dataset):
    """ Considère des images centrées sur les glomureli.
        Les images sont extraites à partir des masques et en utilisant les FP des modèles.
    """
    def __init__(self, images, image_size, augment=None, logger=None):
        self.augment    = augment
        self.crop       = crop
        self.images     = images
        self.image_size = image_size

        self.project_repo, self.raw_data_dir, self.data_dir = get_data_path('local')

        tile_id = []
        for image_id, image_path in self.images.items():
            tile_id += image_path

        self.tile_id = tile_id
        self.len = len(self.tile_id)


    def __len__(self):
        return self.len

    def __str__(self):
        string  = ''
        string += '\t len       = %d \n' % len(self)
        # string += '\t TP image_dir = %s \n' % self.from_mask_image_dir
        # string += '\t FP image_dir = %s \n' % self.false_positive_image_dir
        # string += '\t image_id  = %s \n' % str(self.image_id)
        # string += '\t           = %d \n' % sum(len(i) for i in self.image_id)
        return string


    def __getitem__(self, index):
        id = self.tile_id[index]

        # print('/tile/%s.png'%(id))

        image = cv2.imread(self.data_dir + '%s.png' % id, cv2.IMREAD_COLOR)
        mask  = cv2.imread(self.data_dir + '%s.mask.png' % id, cv2.IMREAD_GRAYSCALE)
        #print(data_dir + '/tile/%s/%s.png'%(self.image_dir,id))

        image = image.astype(np.float32) / 255
        mask  = mask.astype(np.float32) / 255
        r = {
            'image_size': self.image_size,
            'index' : index,
            'tile_id' : id,
            'mask' : mask,
            'image' : image,
        }
        if self.augment is not None:
            r = self.augment(r)
        else:
            r = self.crop(r)


        return r


class HuDataset(Dataset):
    def __init__(self, image_id, image_dir, augment=None):
        self.augment = augment
        self.image_id = image_id
        self.image_dir = image_dir

        self.project_repo, self.raw_data_dir, self.data_dir = get_data_path('local')

        # print(self.data_dir)
        # print(image_dir)

        tile_id = []
        for i, dirpath in enumerate(image_dir):
            for id in image_id[i]:
                for _image in os.listdir(f'{self.data_dir}/tile/{dirpath}/{id}'):
                    if 'mask' in _image: continue
                    tile_id.append(f"{self.image_dir[i]}/{id}/{_image.strip('.png')}")

        # print(tile_id)
        self.tile_id = tile_id
        self.len =len(self.tile_id)


    def __len__(self):
        return self.len

    def __str__(self):
        string  = ''
        string += '\tlen  = %d\n' % len(self)
        string += '\timage_dir = %s\n' % self.image_dir
        string += '\timage_id  = %s\n' % str(self.image_id)
        string += '\t          = %d\n' % sum(len(i) for i in self.image_id)
        return string


    def __getitem__(self, index):
        id = self.tile_id[index]
        print(f'/tile/{id}.png')
        image = cv2.imread(self.data_dir + f'/tile/{id}.png', cv2.IMREAD_COLOR)
        mask  = cv2.imread(self.data_dir + f'/tile/{id}.mask.png', cv2.IMREAD_GRAYSCALE)
        #print(data_dir + '/tile/%s/%s.png'%(self.image_dir,id))

        image = image.astype(np.float32) / 255
        mask  = mask.astype(np.float32) / 255
        r = {
            'index' : index,
            'tile_id' : id,
            'mask' : mask,
            'image' : image,
        }
        if self.augment is not None:
            r = self.augment(r)

        return r


def null_collate(batch):
    batch_size = len(batch)
    index = []
    mask = []
    image = []

    for _ind, r in enumerate(batch):
        index.append(r['index'])
        mask.append(r['mask'])
        image.append(r['image'])


    image = np.stack(image)
    image = image[...,::-1]
    image = image.transpose(0,3,1,2)
    image = np.ascontiguousarray(image)

    mask  = np.stack(mask)
    mask  = np.ascontiguousarray(mask)

    #---
    image = torch.from_numpy(image).contiguous().float()
    mask  = torch.from_numpy(mask).contiguous().unsqueeze(1)
    mask  = (mask > 0.5).float()

    return {
        'index' : index,
        'mask' : mask,
        'image' : image,
    }


def train_albu_augment(record):

    verbose = record.get('verbose', False)
    image_size = record['image_size']

    image = record['image']
    mask = record['mask']

    if verbose:
        pipeline = albu.ReplayCompose
    else:
        pipeline = albu.Compose

    aug = pipeline([
        albu.OneOf([
            albu.RandomBrightnessContrast(brightness_limit = 0.2,
                                          contrast_limit = 0.2,
                                          brightness_by_max = True,
                                          always_apply = False,
                                          p = 0.7),
            albu.RandomBrightnessContrast(brightness_limit=(-0.2, 0.6),
                                          contrast_limit=.2,
                                          brightness_by_max=True,
                                          always_apply=False,
                                          p= 0.7),
            albu.RandomGamma(p=1)
        ], p=0.5),
        albu.OneOf([
            albu.Blur(blur_limit=3, p=1),
            albu.MedianBlur(blur_limit=3, p=1),
            albu.Blur(blur_limit=5, p=0.7),
            albu.MedianBlur(blur_limit=5, p=0.7)
        ], p=.25),
        albu.OneOf([
            albu.GaussNoise(0.02, p=.5),
            albu.IAAAffine(p=.5),
        ], p=.25),
        albu.RandomRotate90(p=.5),
        albu.HorizontalFlip(p=.5),
        albu.VerticalFlip(p=.5),
        albu.RandomCrop(width=image_size, height=image_size),
        albu.ShiftScaleRotate(p=.25)
    ])

    data = aug(image=image, mask=mask)
    record['image'] = data['image']
    record['mask'] = data['mask']

    if verbose:
        for transformation in data['replay']['transforms']:
            if not isinstance(transformation, dict):
                print('not a dict')
                pass
            elif transformation.get('applied', False):
                print(30*'-')
                if 'OneOf' in transformation['__class_fullname__']:
                    print(30 * '=')
                    for _trans in transformation['transforms']:
                        if not _trans.get('applied', False): continue
                        _name = _trans['__class_fullname__']
                        if 'Flip' in _name: continue

                        print(_trans['__class_fullname__'])
                        for k, v in _trans.items():
                            if k in ['__class_fullname__', 'applied', 'always_apply']: continue
                            print(f"{k}: {v}")

                else:
                    _name = transformation['__class_fullname__']
                    if 'Flip' in _name: continue
                    print(_name)
                    for k, v in transformation.items():
                        if k in ['__class_fullname__', 'applied', 'always_apply']: continue
                        print(f"{k}: {v}")

    return record


def crop(record):
    image_size = record['image_size']
    image = record['image']
    mask = record['mask']
    aug = albu.Compose([
        albu.RandomCrop(width=image_size, height=image_size),
    ])
    data = aug(image=image, mask=mask)
    record['image'] = data['image']
    record['mask'] = data['mask']
    return record


def train_augment(record):

    verbose = record.get('verbose', False)

    image_size = record['image_size']
    image = record['image']
    mask = record['mask']

    for fn in np.random.choice([
        lambda image, mask: do_random_rotate_crop(image, mask, size=image_size, mag=45, verbose=verbose),
        lambda image, mask: do_random_scale_crop(image, mask, size=image_size, mag=0.075, verbose=verbose),
        lambda image, mask: do_random_crop(image, mask, size=image_size, verbose=verbose),
    ], 1): image, mask = fn(image, mask)

    image, mask = do_random_hsv(image, mask, mag=[0.1, 0.2, 0])
    for fn in np.random.choice([
        lambda image, mask: (image, mask),
        lambda image, mask: do_random_contast(image, mask, mag=1.0, verbose=verbose),
        lambda image, mask: do_random_gain(image, mask, mag=0.9, verbose=verbose),
        # lambda image, mask : do_random_hsv(image, mask, mag=[0.1, 0.2, 0], verbose=True),
        lambda image, mask: do_random_noise(image, mask, mag=0.1, verbose=verbose),
    ], 1): image, mask = fn(image, mask)

    image, mask = do_random_flip_transpose(image, mask, verbose=verbose)
    record['mask'] = mask
    record['image'] = image
    return record


def augment(image, mask):
    #image, mask = do_random_crop(image, mask, size=320)
    #image, mask = do_random_scale_crop(image, mask, size=320, mag=0.1)
    #image, mask = do_random_rotate_crop(image, mask, size=320, mag=30 )
    #image, mask = do_random_contast(image, mask, mag=0.8 )
    return do_random_hsv(image, mask, mag=[0.1, 0.2, 0])
    image, mask = do_random_gain(image, mask, mag=0.8)
    #image, mask = do_random_noise(image, mask, mag=0.1)


def run_check_augment():

    image_size = 380
    print("initialize dataset")
    dataset = HuDataset(
        image_id  = [make_image_id('train', [0])],
        image_dir = ['mask_700_0.5_centroids'],
    )

    for i in range(1000):
        print(f"sample n°{i}") #, end='\r')
        r = dataset[i]
        image = r['image']
        mask  = r['mask']

        image_show_norm('overlay', image)
        cv2.waitKey(1)

        for i in range(100):
            print(70 * '*')
            result = train_albu_augment({
                'image_size': image_size,
                'image': image.copy(),
                'mask' : mask.copy(),
                'verbose': True
            })
            image1 = result['image']
            # overlay1 = np.hstack([image1, np.tile(mask1.reshape(*image1.shape[:2], 1), (1, 1, 3)),])
            image_show_norm('overlay1', image1)
            cv2.waitKey(0)


# main #################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", help="fold")

    args = parser.parse_args()
    if not args.mode:
        print("mode missing")
        sys.exit()
    elif args.mode == 'augmentation':
        seed = 37
        seed_py(seed)
        seed_torch(seed)
        run_check_augment()




