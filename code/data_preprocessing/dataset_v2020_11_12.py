import argparse

from code.common import *
from code.hubmap_v2 import *

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
    def __init__(self, image_id, from_mask_image_dir, false_positive_image_dir, augment=None):
        self.augment = augment
        self.image_id = image_id
        self.from_mask_image_dir = from_mask_image_dir
        self.false_positive_image_dir = false_positive_image_dir

        tile_id = []

        # ----------------------------------------
        # lecture des images issues des masques
        # ----------------------------------------
        print("True positive examples:")
        for id in self.image_id:
            # df = pd.read_csv(data_dir + '/tile/%s/%s.csv' % (self.from_mask_image_dir, id))
            # tile_id += ('%s/%s/' % (self.from_mask_image_dir, id) + df.tile_id).tolist()

            image_dir = f"/tile/{self.from_mask_image_dir}/{id}/"

            print(id,  len([f.strip('.mask.png') for f in
                        os.listdir(data_dir + image_dir)
                        if 'mask' in f]))

            tile_id += [image_dir + f.strip('.mask.png') for f in
                        os.listdir(data_dir + image_dir)
                        if 'mask' in f]

        # -----------------------------------------------
        # lecture des images issues des faux positifs
        # -----------------------------------------------
        print("False positive examples:")
        for i in range(len(self.false_positive_image_dir)):
            for id in self.image_id:

                image_dir = f"/tile/{self.false_positive_image_dir[i]}/{id}/"
                print(id, len([f.strip('.mask.png') for f in
                               os.listdir(data_dir + image_dir)
                               if 'mask' in f]))

                tile_id += [image_dir + f.strip('.mask.png') for f in
                            os.listdir(data_dir + image_dir)
                            if 'mask' in f]

                # print(tile_id)

                # df = pd.read_csv(data_dir + '/tile/%s/%s.csv' % (self.false_positive_image_dir[i], id))
                # tile_id += ('%s/%s/' % (self.false_positive_image_dir[i], id) + df.tile_id).tolist()

        self.tile_id = tile_id
        self.len = len(self.tile_id)


    def __len__(self):
        return self.len

    def __str__(self):
        string  = ''
        string += '\t len       = %d \n' % len(self)
        string += '\t TP image_dir = %s \n' % self.from_mask_image_dir
        string += '\t FP image_dir = %s \n' % self.false_positive_image_dir
        string += '\t image_id  = %s \n' % str(self.image_id)
        string += '\t           = %d \n' % sum(len(i) for i in self.image_id)
        return string


    def __getitem__(self, index):
        id = self.tile_id[index]

        # print('/tile/%s.png'%(id))

        image = cv2.imread(data_dir + '%s.png'%(id), cv2.IMREAD_COLOR)
        mask  = cv2.imread(data_dir + '%s.mask.png'%(id), cv2.IMREAD_GRAYSCALE)
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

class HuDataset(Dataset):
    def __init__(self, image_id, image_dir, augment=None):
        self.augment = augment
        self.image_id = image_id
        self.image_dir = image_dir

        tile_id = []
        for i in range(len(image_dir)):
            for id in image_id[i]:
                df = pd.read_csv(data_dir + '/tile/%s/%s.csv' % (self.image_dir[i], id))
                tile_id += ('%s/%s/' % (self.image_dir[i], id) + df.tile_id).tolist()

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
        image = cv2.imread(data_dir + '/tile/%s.png'%(id), cv2.IMREAD_COLOR)
        mask  = cv2.imread(data_dir + '/tile/%s.mask.png'%(id), cv2.IMREAD_GRAYSCALE)
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


image_size = 256  # 380  # 256

def train_augment(record):

    verbose = record.get('verbose', False)

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

    image, mask = do_random_flip_transpose(image, mask)
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

    dataset = HuDataset(
        image_id  = [make_image_id('train', [0])],
        image_dir = ['0.25_320_train'],
    )

    for i in range(1000):
    #for i in np.random.choice(len(dataset),100):
        r = dataset[i]
        image = r['image']
        mask  = r['mask']

        print('%2d --------------------------- ' % i)
        # overlay = np.hstack([image, np.tile(mask.reshape(*image.shape[:2], 1), (1, 1, 3)),])
        image_show_norm('overlay', image)
        cv2.waitKey(1)

        for i in range(100):
            print(70 * '-')
            image1, mask1 = train_augment({
                'image': image.copy(),
                'mask' : mask.copy(),
                'verbose': True
            })
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
        run_check_augment()




