import cv2
import rasterio
import numpy as np
import os
from rasterio.windows import Window

from code.lib.utility.draw import image_show_norm

mean = np.array([0.65459856, 0.48386562, 0.69428385])
std = np.array([0.15167958, 0.23584107, 0.13146145])

sz = 1024

s_th = 40                                    # saturation blancking threshold
p_th = 1000*(sz//256)**2                     # threshold for the minimum number of pixels
identity = rasterio.Affine(1, 0, 0, 0, 1, 0)

DATA = '/home/fabien/Kaggle/HuBMAP/input/train'


class HuBMAPDataset():
    def __init__(self, idx, sz=sz):
        self.data = rasterio.open(os.path.join(DATA, idx + '.tiff'), transform=identity,
                                  num_threads='all_cpus')
        # some images have issues with their format
        # and must be saved correctly before reading with rasterio
        if self.data.count != 3:
            subdatasets = self.data.subdatasets
            self.layers = []
            if len(subdatasets) > 0:
                for i, subdataset in enumerate(subdatasets, 0):
                    self.layers.append(rasterio.open(subdataset))
        self.shape = self.data.shape
        self.reduce = 1
        self.sz = sz
        self.pad0 = (self.sz - self.shape[0] % self.sz) % self.sz
        self.pad1 = (self.sz - self.shape[1] % self.sz) % self.sz
        self.n0max = (self.shape[0] + self.pad0) // self.sz
        self.n1max = (self.shape[1] + self.pad1) // self.sz

    def __len__(self):
        return self.n0max * self.n1max

    def __getitem__(self, idx):
        # the code below may be a little bit difficult to understand,
        # but the thing it does is mapping the original image to
        # tiles created with adding padding, as done in
        # https://www.kaggle.com/iafoss/256x256-images ,
        # and then the tiles are loaded with rasterio
        # n0,n1 - are the x and y index of the tile (idx = n0*self.n1max + n1)
        n0, n1 = idx // self.n1max, idx % self.n1max
        # x0,y0 - are the coordinates of the lower left corner of the tile in the image
        # negative numbers correspond to padding (which must not be loaded)
        x0, y0 = -self.pad0 // 2 + n0 * self.sz, -self.pad1 // 2 + n1 * self.sz
        # make sure that the region to read is within the image
        p00, p01 = max(0, x0), min(x0 + self.sz, self.shape[0])
        p10, p11 = max(0, y0), min(y0 + self.sz, self.shape[1])
        img = np.zeros((self.sz, self.sz, 3), np.uint8)
        # mapping the load region to the tile
        if self.data.count == 3:
            img[(p00 - x0):(p01 - x0), (p10 - y0):(p11 - y0)] = np.moveaxis(
                self.data.read([1, 2, 3], window = Window.from_slices((p00, p01), (p10, p11))), 0, -1)
        else:
            for i, layer in enumerate(self.layers):
                img[(p00 - x0):(p01 - x0), (p10 - y0):(p11 - y0), i] = \
                    layer.read(1, window=Window.from_slices((p00, p01), (p10, p11)))

        if self.reduce != 1:
            img = cv2.resize(img, (self.sz // self.reduce, self.sz // self.reduce),
                             interpolation=cv2.INTER_AREA)
        # check for empty imges
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        if (s > s_th).sum() <= p_th or img.sum() <= p_th:
            return img / 255.0, -1
        else:
            return img / 255.0, idx


########################################################################
# main #################################################################
########################################################################
if __name__ == '__main__':

    dataset = HuBMAPDataset('cb2d976f4')


    # _id = input(f"\nGive a tile id between 0 and {len(dataset)}:\n")
    # print(dataset[int(_id)].shape)
    # print(dataset[int(_id)])
    # image_show_norm('probability', dataset[int(_id)], min=0, max=1, resize=0.5)
    # cv2.waitKey(0)

for _id in range(0, 2010):
    image, code = dataset[int(_id)]
    print(f"showing image n°{code}"+50*" ", end='\r')
    if code >= 0:
        image_show_norm('image', image, min=0, max=1, resize=0.5)
        cv2.waitKey(0)
