import argparse

import rasterio

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

    print(image.shape)
    sys.exit()

    return image


# def read_tiff(image_file):
#     with rasterio.open(image_file) as file:
#         if file.count == 3:
#             image = file.read([1, 2, 3]).transpose(1, 2, 0).copy()
#         else:
#             h, w = (file.height, file.width)
#             subdatasets = file.subdatasets
#             if len(subdatasets) > 0:
#                 image = np.zeros((h, w, len(subdatasets)), dtype=np.uint8)
#                 for i, subdataset in enumerate(subdatasets, 0):
#                     with rasterio.open(subdataset) as layer:
#                         image[:, :, i] = layer.read(1)
#
#     print(image.shape)
#     sys.exit()
#
#     return image



def to_tile(image,
            mask=None,
            scale=0.25,
            size=320,
            step=192,
            min_score=0.25):
    """

    Parameters:
    -----------
    image: <type>
        large size image

    scale: float in [0:1]
        factor used to down scale the input image

    """

    half = size//2
    print("original image size:", image.shape)

    print(f"1) Scales down image by a factor: {scale}")
    image_small = cv2.resize(image, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    height, width, _ = image_small.shape
    print(f"\tscaled image size = {height} x {width}")

    print(f"2) Scales down image by a factor: 1/32")
    vv = cv2.resize(image_small, dsize=None, fx=1 / 32, fy=1 / 32, interpolation=cv2.INTER_LINEAR)
    print(f"\tscaled image size = {vv.shape[0]} x {vv.shape[0]}")

    vv = cv2.cvtColor(vv, cv2.COLOR_RGB2HSV)
    # image_show('v[0]', vv[:, :, 0])
    # image_show('v[1]', vv[:, :, 1])
    # image_show('v[2]', vv[:, :, 2])
    # cv2.waitKey(0)

    ## Image de contenant des 0 et 1 / valeur seuil de la saturation ???
    vv = (vv[:, :, 1] > 32).astype(np.float32)
    vv = cv2.resize(vv, dsize=(width, height), interpolation=cv2.INTER_LINEAR)

    #####################
    # make coord
    #####################
    xx = np.array_split(np.arange(half, width  - half), np.floor((width  - size) / step))
    yy = np.array_split(np.arange(half, height - half), np.floor((height - size) / step))
    xx = [int(x[0]) for x in xx] + [width-half]
    yy = [int(y[0]) for y in yy] + [height-half]

    print(f"min saturation score to reject sub-images: {min_score}")

    ################################################
    ## On selectionne les sous-images en fonction
    ## de la valeur moyenne de l'image de boolean
    ## créée à partir du filtre de saturation
    ################################################
    coord  = []
    reject = []
    for cy in yy:
        for cx in xx:
            cv = vv[cy - half:cy + half, cx - half:cx + half].mean()
            if cv > min_score:
                coord.append([cx, cy, cv])
            else:
                reject.append([cx, cy, cv])

    ###################################################
    ## On recupère les tuiles (image + mask) à partir des
    ## coordonnées choisies à l'étape précédente
    ## et en faisant référence à l'image initiale
    ## down-scalée du factor scale
    ###################################################
    tile_image = []
    for cx, cy, cv in coord:
        t = image_small[cy - half:cy + half, cx - half:cx + half]
        assert (t.shape == (size, size, 3))
        tile_image.append(t)

    if mask is not None:
        mask_small = cv2.resize(mask, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        tile_mask = []
        for cx, cy, cv in coord:
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


# def to_mask(tile, coord, height, width,
#             scale=tile_scale,
#             size=tile_size,
#             step=tile_average_step,
#             min_score=tile_min_score):
#
#     half = size//2
#     mask  = np.zeros((height, width), np.float32)
#
#     if 0:
#         count = np.zeros((height, width), np.float32)
#         for t, (cx, cy, cv) in enumerate(coord):
#             mask [cy - half:cy + half, cx - half:cx + half] += tile[t]
#             count[cy - half:cy + half, cx - half:cx + half] += 1
#                # simple averge, <todo> guassian weighing?
#                # see unet paper for "Overlap-tile strategy for seamless segmentation of arbitrary large images"
#         m = (count != 0)
#         mask[m] /= count[m]
#
#     if 1:
#         for t, (cx, cy, cv) in enumerate(coord):
#             mask[cy - half:cy + half, cx - half:cx + half] = np.maximum(
#                 mask[cy - half:cy + half, cx - half:cx + half], tile[t] )
#
#     return mask

# def run_check_tile():
#
#     #load a train image
#     id = 'e79de561c'
#     image_file = data_dir + '/train/%s.tiff' % id
#     image = read_tiff(image_file)
#     height, width = image.shape[:2]
#
#     #load a mask
#     df = pd.read_csv(data_dir + '/train.csv', index_col='id')
#     encoding = df.loc[id,'encoding']
#     mask = rle_decode(encoding, height, width, 255)
#
#     #make tile
#     tile = to_tile(image, mask)
#
#
#     if 1: #debug
#         overlay = tile['image_small'].copy()
#         for cx,cy,cv in tile['coord']:
#             cv = int(255 * cv)
#             cv2.circle(overlay, (cx, cy), 64, [cv,cv,cv], -1)
#             cv2.circle(overlay, (cx, cy), 64, [0, 0, 255], 16)
#         for cx,cy,cv in tile['reject']:
#             cv = int(255 * cv)
#             cv2.circle(overlay, (cx, cy), 64, [cv,cv,cv], -1)
#             cv2.circle(overlay, (cx, cy), 64, [255, 0, 0], 16)
#
#         #---
#         num = len(tile['coord'])
#         cx, cy, cv = tile['coord'][num//2]
#         cv2.rectangle(overlay,(cx-tile_size//2,cy-tile_size//2),(cx+tile_size//2,cy+tile_size//2), (0,0,255), 16)
#
#         image_show('overlay', overlay, resize=0.1)
#         cv2.waitKey(1)
#
#     # make prediction for tile
#     # e.g. predict = model(tile['tile_image'])
#     tile_predict = tile['tile_mask'] # dummy: set predict as ground truth
#
#     # make mask from tile
#     height, width = tile['image_small'].shape[:2]
#     predict = to_mask(tile_predict,
#                       tile['coord'],
#                       height,
#                       width,
#                       scale=scale,
#                       size=tile_size,
#                       step=step,
#                       min_score=min_score)
#
#
#     truth = tile['mask_small']#.astype(np.float32)/255
#     diff = np.abs(truth-predict)
#     print('diff', diff.max(), diff.mean())
#
#     if 1:
#         image_show_norm('diff', diff, min=0, max=1, resize=0.2)
#         image_show_norm('predict', predict, min=0, max=1, resize=0.2)
#         cv2.waitKey(0)

## augmentation ######################################################################
#
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


######################################################################################

# def run_check_dataset():
#
#     image_id = make_image_id ('train-0')
#     dataset = HuDataset(image_id)
#     print(dataset)
#
#     for i in range(100):
#         i = np.random.choice(len(dataset))
#         r = dataset[i]
#
#         print(r['index'])
#         print(r['tile_id'])
#         print(r['image'].shape)
#         print(r['mask'].shape)
#         print('')
#
#
#         image_show('image', r['image'])
#         image_show('mask', r['mask'])
#         cv2.waitKey(0)
#         #exit(0)



