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
        string += '\tlen  = %d\n'%len(self)
        string += '\timage_dir = %s\n'%self.image_dir
        string += '\timage_id  = %s\n'%str(self.image_id)
        string += '\t          = %d\n'%sum(len(i) for i in self.image_id)
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
    for r in batch:
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
    mask  = (mask>0.5).float()

    return {
        'index' : index,
        'mask' : mask,
        'image' : image,
    }


## augmentation ######################################################################
#flip
# def do_random_flip_transpose(image, mask):
#     if np.random.rand()>0.5:
#         image = cv2.flip(image,0)
#         mask = cv2.flip(mask,0)
#     if np.random.rand()>0.5:
#         image = cv2.flip(image,1)
#         mask = cv2.flip(mask,1)
#     if np.random.rand()>0.5:
#         image = image.transpose(1,0,2)
#         mask = mask.transpose(1,0)
#
#     image = np.ascontiguousarray(image)
#     mask = np.ascontiguousarray(mask)
#     return image, mask
#
# #geometric
# def do_random_crop(image, mask, size):
#     height, width = image.shape[:2]
#     x = np.random.choice(width -size)
#     y = np.random.choice(height-size)
#     image = image[y:y+size,x:x+size]
#     mask  = mask[y:y+size,x:x+size]
#     return image, mask
#
# def do_random_scale_crop(image, mask, size, mag):
#     height, width = image.shape[:2]
#
#     s = 1 + np.random.uniform(-1, 1)*mag
#     s =  int(s*size)
#
#     x = np.random.choice(width -s)
#     y = np.random.choice(height-s)
#     image = image[y:y+s,x:x+s]
#     mask  = mask[y:y+s,x:x+s]
#     if s!=size:
#         image = cv2.resize(image, dsize=(size,size), interpolation=cv2.INTER_LINEAR)
#         mask  = cv2.resize(mask, dsize=(size,size), interpolation=cv2.INTER_LINEAR)
#     return image, mask
#
# def do_random_rotate_crop(image, mask, size, mag=30 ):
#     angle = 1+np.random.uniform(-1, 1)*mag
#
#     height, width = image.shape[:2]
#     dst = np.array([
#         [0,0],[size,size], [size,0], [0,size],
#     ])
#
#     c = np.cos(angle/180*2*PI)
#     s = np.sin(angle/180*2*PI)
#     src = (dst-size//2)@np.array([[c, -s],[s, c]]).T
#     src[:,0] -= src[:,0].min()
#     src[:,1] -= src[:,1].min()
#
#     src[:,0] = src[:,0] + np.random.uniform(0,width -src[:,0].max())
#     src[:,1] = src[:,1] + np.random.uniform(0,height-src[:,1].max())
#
#     if 0: #debug
#         def to_int(f):
#             return (int(f[0]),int(f[1]))
#
#         cv2.line(image, to_int(src[0]), to_int(src[1]), (0,0,1), 16)
#         cv2.line(image, to_int(src[1]), to_int(src[2]), (0,0,1), 16)
#         cv2.line(image, to_int(src[2]), to_int(src[3]), (0,0,1), 16)
#         cv2.line(image, to_int(src[3]), to_int(src[0]), (0,0,1), 16)
#         image_show_norm('image', image, min=0, max=1)
#         cv2.waitKey(1)
#
#
#     transform = cv2.getAffineTransform(src[:3].astype(np.float32), dst[:3].astype(np.float32))
#     image = cv2.warpAffine( image, transform, (size, size), flags=cv2.INTER_LINEAR,
#                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
#     mask  = cv2.warpAffine( mask, transform, (size, size), flags=cv2.INTER_LINEAR,
#                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
#     return image, mask
#
# #warp/elastic deform ...
# #<todo>
#
# #noise
# def do_random_noise(image, mask, mag=0.1):
#     height, width = image.shape[:2]
#     noise = np.random.uniform(-1,1, (height, width,1))*mag
#     image = image + noise
#     image = np.clip(image,0,1)
#     return image, mask
#
#
# #intensity
# def do_random_contast(image, mask, mag=0.3):
#     alpha = 1 + random.uniform(-1,1)*mag
#     image = image * alpha
#     image = np.clip(image,0,1)
#     return image, mask
#
# def do_random_gain(image, mask, mag=0.3):
#     alpha = 1 + random.uniform(-1,1)*mag
#     image = image ** alpha
#     image = np.clip(image,0,1)
#     return image, mask
#
# def do_random_hsv(image, mask, mag=[0.15,0.25,0.25]):
#     image = (image*255).astype(np.uint8)
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#
#     h = hsv[:, :, 0].astype(np.float32)  # hue
#     s = hsv[:, :, 1].astype(np.float32)  # saturation
#     v = hsv[:, :, 2].astype(np.float32)  # value
#     h = (h*(1 + random.uniform(-1,1)*mag[0]))%180
#     s =  s*(1 + random.uniform(-1,1)*mag[1])
#     v =  v*(1 + random.uniform(-1,1)*mag[2])
#
#     hsv[:, :, 0] = np.clip(h,0,180).astype(np.uint8)
#     hsv[:, :, 1] = np.clip(s,0,255).astype(np.uint8)
#     hsv[:, :, 2] = np.clip(v,0,255).astype(np.uint8)
#     image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#     image = image.astype(np.float32)/255
#     return image, mask
#
#
#
# #shuffle block, etc
# #<todo>
#
#
# # post process ---
# # https://stackoverflow.com/questions/42798659/how-to-remove-small-connected-objects-using-opencv/42812226
#
# # min_radius = 50
# # min_area = 7853
# #
# #
# def filter_small(mask, min_size):
#
#     m = (mask*255).astype(np.uint8)
#
#     num_comp, comp, stat, centroid = cv2.connectedComponentsWithStats(m, connectivity=8)
#     if num_comp==1: return mask
#
#     filtered = np.zeros(comp.shape,dtype=np.uint8)
#     area = stat[:, -1]
#     for i in range(1, num_comp):
#         if area[i] >= min_size:
#             filtered[comp == i] = 255
#     return filtered

######################################################################################

# def run_check_dataset():
#
#     dataset = HuDataset(
#         image_id = [
#             make_image_id ('valid-0'),
#         ],
#         image_dir =[
#             '0.25_480_240_train',
#         ]
#     )
#     print(dataset)
#
#     for i in range(1000):
#         i = np.random.choice(len(dataset)) #98 #
#         r = dataset[i]
#
#         print(r['index'])
#         print(r['tile_id'])
#         print(r['image'].shape)
#         print(r['mask'].shape)
#         print('')
#
#         filtered = filter_small(r['mask'],min_size=800*0.25)
#
#
#         image_show_norm('image', r['image'], min=0, max=1)
#         image_show_norm('mask',  r['mask'],  min=0, max=1)
#         image_show ('filtered',  filtered)
#         cv2.waitKey(0)
#         #exit(0)


image_size = 256

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
        lambda image, mask: do_random_contast(image, mask, mag=0.5, verbose=verbose),
        lambda image, mask: do_random_gain(image, mask, mag=0.9, verbose=verbose),
        # lambda image, mask : do_random_hsv(image, mask, mag=[0.1, 0.2, 0], verbose=True),
        lambda image, mask: do_random_noise(image, mask, mag=0.03, verbose=verbose),
    ], 1): image, mask = fn(image, mask)

    image, mask = do_random_flip_transpose(image, mask)
    record['mask'] = mask
    record['image'] = image
    return record['image'], record['mask']


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
    print(dataset)

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




