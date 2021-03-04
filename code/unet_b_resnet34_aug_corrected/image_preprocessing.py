import random

import numpy as np
import cv2

from code.lib.include import PI


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


#geometric
def do_random_crop(image, mask, size):
    height, width = image.shape[:2]
    x = np.random.choice(width -size)
    y = np.random.choice(height-size)
    image = image[y:y+size,x:x+size]
    mask  = mask[y:y+size,x:x+size]
    return image, mask


def do_random_scale_crop(image, mask, size, mag):
    height, width = image.shape[:2]

    s = 1 + np.random.uniform(-1, 1)*mag
    s =  int(s*size)

    x = np.random.choice(width -s)
    y = np.random.choice(height-s)
    image = image[y:y+s,x:x+s]
    mask  = mask[y:y+s,x:x+s]
    if s!=size:
        image = cv2.resize(image, dsize=(size,size), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(mask, dsize=(size,size), interpolation=cv2.INTER_LINEAR)
    return image, mask


def do_random_rotate_crop(image, mask, size, mag=30 ):
    angle = 1+np.random.uniform(-1, 1)*mag

    height, width = image.shape[:2]
    dst = np.array([
        [0,0],[size,size], [size,0], [0,size],
    ])

    c = np.cos(angle/180*2*PI)
    s = np.sin(angle/180*2*PI)
    src = (dst-size//2)@np.array([[c, -s],[s, c]]).T
    src[:,0] -= src[:,0].min()
    src[:,1] -= src[:,1].min()

    src[:,0] = src[:,0] + np.random.uniform(0,width -src[:,0].max())
    src[:,1] = src[:,1] + np.random.uniform(0,height-src[:,1].max())

    if 0: #debug
        def to_int(f):
            return (int(f[0]),int(f[1]))

        cv2.line(image, to_int(src[0]), to_int(src[1]), (0,0,1), 16)
        cv2.line(image, to_int(src[1]), to_int(src[2]), (0,0,1), 16)
        cv2.line(image, to_int(src[2]), to_int(src[3]), (0,0,1), 16)
        cv2.line(image, to_int(src[3]), to_int(src[0]), (0,0,1), 16)
        image_show_norm('image', image, min=0, max=1)
        cv2.waitKey(1)


    transform = cv2.getAffineTransform(src[:3].astype(np.float32), dst[:3].astype(np.float32))
    image = cv2.warpAffine( image, transform, (size, size), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    mask  = cv2.warpAffine( mask, transform, (size, size), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image, mask

#warp/elastic deform ...
#<todo>

#noise
def do_random_noise(image, mask, mag=0.1):
    height, width = image.shape[:2]
    noise = np.random.uniform(-1,1, (height, width,1))*mag
    image = image + noise
    image = np.clip(image,0,1)
    return image, mask


#intensity
def do_random_contast(image, mask, mag=0.3):
    alpha = 1 + random.uniform(-1,1)*mag
    image = image * alpha
    image = np.clip(image,0,1)
    return image, mask

def do_random_gain(image, mask, mag=0.3):
    alpha = 1 + random.uniform(-1,1)*mag
    image = image ** alpha
    image = np.clip(image,0,1)
    return image, mask

def do_random_hsv(image, mask, mag=[0.15,0.25,0.25]):
    image = (image*255).astype(np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0].astype(np.float32)  # hue
    s = hsv[:, :, 1].astype(np.float32)  # saturation
    v = hsv[:, :, 2].astype(np.float32)  # value
    h = (h*(1 + random.uniform(-1,1)*mag[0]))%180
    s =  s*(1 + random.uniform(-1,1)*mag[1])
    v =  v*(1 + random.uniform(-1,1)*mag[2])

    hsv[:, :, 0] = np.clip(h,0,180).astype(np.uint8)
    hsv[:, :, 1] = np.clip(s,0,255).astype(np.uint8)
    hsv[:, :, 2] = np.clip(v,0,255).astype(np.uint8)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    image = image.astype(np.float32)/255
    return image, mask



#shuffle block, etc
#<todo>


# post process ---
# https://stackoverflow.com/questions/42798659/how-to-remove-small-connected-objects-using-opencv/42812226

# min_radius = 50
# min_area = 7853
#
#
def filter_small(mask, min_size):

    m = (mask*255).astype(np.uint8)

    num_comp, comp, stat, centroid = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num_comp==1: return mask

    filtered = np.zeros(comp.shape,dtype=np.uint8)
    area = stat[:, -1]
    for i in range(1, num_comp):
        if area[i] >= min_size:
            filtered[comp == i] = 255
    return filtered

