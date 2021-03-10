from code.common import *
import tifffile as tiff


def read_tiff(image_file):
    image = tiff.imread(image_file)
    print(f"reads {image_file}; shape={image.shape}")
    if image.shape[0] == 3 and image.ndim == 3:
        image = image.transpose(1, 2, 0)
    elif image.shape[2] == 3 and image.ndim == 5 :
        image = np.transpose(image.squeeze(), (1, 2, 0))
    image = np.ascontiguousarray(image)
    return image


#----------------------------------
tile_scale = 0.25
tile_size  = 320
tile_average_step = 192
tile_min_score = 0.25


def to_tile(image,
            mask=None,
            scale=tile_scale,
            size=tile_size,
            step=tile_average_step,
            min_score=tile_min_score):

    half = size//2
    print(image.shape)
    print(f"1) Scales down image by a factor {scale}")
    image_small = cv2.resize(image, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    #make score
    height, width, _ = image_small.shape

    print(f"scaled image size={height} x {width}")

    print(f"2) Scales down image by a factor 1/32")
    vv = cv2.resize(image_small, dsize=None, fx=1 / 32, fy=1 / 32, interpolation=cv2.INTER_LINEAR)

    print(f"scaled image size={vv.shape[0]} x {vv.shape[0]}")

    vv = cv2.cvtColor(vv, cv2.COLOR_RGB2HSV)
    # image_show('v[0]', vv[:,:,0])
    # image_show('v[1]', vv[:,:,1])
    # image_show('v[2]', vv[:,:,2])
    # cv2.waitKey(0)
    vv = (vv[:, :, 1] > 32).astype(np.float32)
    vv = cv2.resize(vv, dsize=(width, height), interpolation=cv2.INTER_LINEAR)

    #####################
    # make coord
    #####################
    xx = np.array_split(np.arange(half, width  - half),
                        np.floor((width  - size) / step))
    yy = np.array_split(np.arange(half, height - half),
                        np.floor((height - size) / step))
    # xx = [int(x.mean()) for x in xx]
    # yy = [int(y.mean()) for y in yy]
    xx = [int(x[0]) for x in xx] + [width-half]
    yy = [int(y[0]) for y in yy] + [height-half]

    print(f"min score to reject sub-images: {min_score}")

    coord  = []
    reject = []
    for cy in yy:
        for cx in xx:
            cv = vv[cy - half:cy + half, cx - half:cx + half].mean()
            if cv > min_score:
                coord.append([cx, cy, cv])
            else:
                reject.append([cx, cy, cv])
    #-----
    if 1:
        tile_image = []
        for cx, cy, cv in coord:
            t = image_small[cy - half:cy + half, cx - half:cx + half]
            assert (t.shape == (size, size, 3))
            tile_image.append(t)

    if mask is not None:
        mask_small = cv2.resize(mask, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        tile_mask = []
        for cx,cy,cv in coord:
            t = mask_small[cy - half:cy + half, cx - half:cx + half]
            assert (t.shape == (size, size))
            tile_mask.append(t)
    else:
        mask_small = None
        tile_mask  = None

    return {
        'image_small': image_small,
        'mask_small' : mask_small,
        'tile_image' : tile_image,
        'tile_mask'  : tile_mask,
        'coord'  : coord,
        'reject' : reject,
    }


def to_mask(tile, coord, height, width,
    scale=tile_scale, size=tile_size, step=tile_average_step, min_score=tile_min_score):

    half = size//2
    mask  = np.zeros((height, width), np.float32)

    if 0:
        count = np.zeros((height, width), np.float32)
        for t, (cx, cy, cv) in enumerate(coord):
            mask [cy - half:cy + half, cx - half:cx + half] += tile[t]
            count[cy - half:cy + half, cx - half:cx + half] += 1
               # simple averge, <todo> guassian weighing?
               # see unet paper for "Overlap-tile strategy for seamless segmentation of arbitrary large images"
        m = (count != 0)
        mask[m] /= count[m]

    if 1:
        for t, (cx, cy, cv) in enumerate(coord):
            mask[cy - half:cy + half, cx - half:cx + half] = np.maximum(
                mask[cy - half:cy + half, cx - half:cx + half], tile[t] )

    return mask

def run_check_tile():

    #load a train image
    id = 'e79de561c'
    image_file = data_dir + '/train/%s.tiff' % id
    image = read_tiff(image_file)
    height, width = image.shape[:2]

    #load a mask
    df = pd.read_csv(data_dir + '/train.csv', index_col='id')
    encoding = df.loc[id,'encoding']
    mask = rle_decode(encoding, height, width, 255)

    #make tile
    tile = to_tile(image, mask)


    if 1: #debug
        overlay = tile['image_small'].copy()
        for cx,cy,cv in tile['coord']:
            cv = int(255 * cv)
            cv2.circle(overlay, (cx, cy), 64, [cv,cv,cv], -1)
            cv2.circle(overlay, (cx, cy), 64, [0, 0, 255], 16)
        for cx,cy,cv in tile['reject']:
            cv = int(255 * cv)
            cv2.circle(overlay, (cx, cy), 64, [cv,cv,cv], -1)
            cv2.circle(overlay, (cx, cy), 64, [255, 0, 0], 16)

        #---
        num = len(tile['coord'])
        cx, cy, cv = tile['coord'][num//2]
        cv2.rectangle(overlay,(cx-tile_size//2,cy-tile_size//2),(cx+tile_size//2,cy+tile_size//2), (0,0,255), 16)

        image_show('overlay', overlay, resize=0.1)
        cv2.waitKey(1)

    # make prediction for tile
    # e.g. predict = model(tile['tile_image'])
    tile_predict = tile['tile_mask'] # dummy: set predict as ground truth

    # make mask from tile
    height, width = tile['image_small'].shape[:2]
    predict = to_mask(tile_predict, tile['coord'],  height, width)

    truth = tile['mask_small']#.astype(np.float32)/255
    diff = np.abs(truth-predict)
    print('diff', diff.max(), diff.mean())

    if 1:
        image_show_norm('diff', diff, min=0, max=1, resize=0.2)
        image_show_norm('predict', predict, min=0, max=1, resize=0.2)
        cv2.waitKey(0)


#############################################################################

'''
train-0
valid-0
test
'''
def make_image_id (mode):
    train_image_id = {
        0 : '2f6ecfcdf',
        1 : 'aaa6a05cc',
        2 : 'cb2d976f4',
        3 : '0486052bb',
        4 : 'e79de561c',
        5 : '095bf7a1f',
        6 : '54f2eec69',
        7 : '1e2425f28',
    }
    test_image_id = {
        0 : 'b9a3865fc',
        1 : 'b2dc8411c',
        2 : '26dc41664',
        3 : 'c68fe75ea',
        4 : 'afa5e8098',
    }
    if 'valid' in mode or 'train' in mode:
        fold = int(mode[-1])

        valid = [fold,]
        train = list({0,1,2,3,4,5,6,7}-{fold,})
        valid_id = [ train_image_id[i] for i in valid ]
        train_id = [ train_image_id[i] for i in train ]

        if 'valid' in mode: return valid_id
        if 'train' in mode: return train_id

    if 'test'==mode:
        test_id = [ test_image_id[i] for i in [0,1,2,3,4] ]

        return test_id

class HuDataset(Dataset):
    def __init__(self, image_id, augment=None):
        self.augment = augment
        self.image_id = image_id

        tile_id = []
        for id in image_id:
            df = pd.read_csv(data_dir + '/tile/0.25_320_train/%s.csv'% id )
            tile_id += ('%s/'%id + df.tile_id).tolist()

        self.tile_id = tile_id
        self.len =len(self.tile_id)


    def __len__(self):
        return self.len

    def __str__(self):
        string  = ''
        string += '\tlen  = %d\n'%len(self)
        # string += '\timage_size = %d\n'%self.image_size
        return string


    def __getitem__(self, index):
        #index = 24969
        tile_id = self.tile_id[index]
        image = cv2.imread(data_dir + '/tile/0.25_320_train/%s.png'%tile_id, cv2.IMREAD_COLOR)
        mask  = cv2.imread(data_dir + '/tile/0.25_320_train/%s.mask.png'%tile_id, cv2.IMREAD_GRAYSCALE)
        r = {
            'index' : index,
            'tile_id' : tile_id,
            'mask' : mask,
            'image' : image,
        }
        if self.augment is not None: r = self.augment(r)
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
    image = image.transpose(0,3,1,2).astype(np.float32)/255
    mask = np.stack(mask).astype(np.float32)/255

    #---
    image = torch.from_numpy(image).contiguous().float()
    mask  = torch.from_numpy(mask).contiguous().float().unsqueeze(1)
    return {
        'index' : index,
        'mask' : mask,
        'image' : image,
    }


## augmentation ######################################################################

def do_random_flip_transpose(image, mask):
    if np.random.rand()>0.5:
        image = cv2.flip(image,0)
        mask = cv2.flip(mask,0)
    if np.random.rand()>0.5:
        image = cv2.flip(image,1)
        mask = cv2.flip(mask,1)
    if np.random.rand()>0.5:
        image = image.transpose(1,0,2)
        mask = mask.transpose(1,0)

    image = np.ascontiguousarray(image)
    mask = np.ascontiguousarray(mask)
    return image, mask


######################################################################################

def run_check_dataset():

    image_id = make_image_id ('train-0')
    dataset = HuDataset(image_id)
    print(dataset)

    for i in range(100):
        i = np.random.choice(len(dataset))
        r = dataset[i]

        print(r['index'])
        print(r['tile_id'])
        print(r['image'].shape)
        print(r['mask'].shape)
        print('')


        image_show('image', r['image'])
        image_show('mask', r['mask'])
        cv2.waitKey(0)
        #exit(0)


# main #################################################################
if __name__ == '__main__':
    #run_check_tile()
    run_check_dataset()
    #run_check_augment()




