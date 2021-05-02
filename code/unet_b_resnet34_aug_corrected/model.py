import numpy as np
from segmentation_models_pytorch import Unet

from code.unet_b_resnet34_aug_corrected.b_resnet34 import *


def seed_torch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def np_binary_cross_entropy_loss(probability, mask):
    p = probability.reshape(-1)
    t = mask.reshape(-1)
    loss = -t * np.log(np.clip(p, 1e-6, 1)) - (1-t) * np.log(np.clip(1-p, 1e-6, 1))
    loss = loss.mean()
    return loss


def np_binary_cross_entropy_loss_optimized(probability, mask):
    p = probability.reshape(-1)
    t = mask.reshape(-1)
    size = len(p)
    batch_size = 2000000
    sum_loss, count = 0, 0
    print(f"bce - optimized; {size // batch_size + 1} iterations")
    for i in range(size // batch_size + 1):
        i1 = i * batch_size
        i2 = min((i + 1) * batch_size, size)
        loss = -t[i1:i2] * np.log(np.clip(p[i1:i2], 1e-6, 1)) - (1 - t[i1:i2]) * np.log(np.clip(1 - p[i1:i2], 1e-6, 1))
        count += len(loss)
        sum_loss += loss.sum()
        print(f"iteration n°{i}, loss={round(sum_loss / count, 5)}", end='\r')
    loss = sum_loss / count
    return loss


def np_dice_score_optimized(probability, mask):
    p = probability.reshape(-1)
    t = mask.reshape(-1)
    p = p > 0.5
    t = t > 0.5
    size = len(p)
    batch_size = 2000000
    union, overlap = 0, 0
    # print(f"dice - optimized; {size // batch_size} iterations")
    for i in range(size // batch_size + 1):
        i1 = i * batch_size
        i2 = min((i + 1) * batch_size, size)
        union += p[i1:i2].sum() + t[i1:i2].sum()
        overlap += (p[i1:i2] * t[i1:i2]).sum()

    dice = 2 * overlap / (union + 0.001)
    return dice


def np_dice_score(probability, mask):
    p = probability.reshape(-1)
    t = mask.reshape(-1)
    p = p > 0.5
    t = t > 0.5
    uion = p.sum() + t.sum()
    overlap = (p*t).sum()
    dice = 2*overlap/(uion+0.001)
    return dice


def np_accuracy(probability, mask):
    p = probability.reshape(-1)
    t = mask.reshape(-1)
    p = p > 0.5
    t = t > 0.5
    tp = (p*t).sum() / t.sum()
    tn = ((1-p)*(1-t)).sum()/(1-t).sum()

    fn = (t * (1-p)).sum() / t.sum()
    fp = ((1-t) * p).sum() / (1-t).sum()

    return tp, tn, fp, fn


def np_accuracy_optimized(probability, mask):
    p = probability.reshape(-1)
    t = mask.reshape(-1)
    p = p > 0.5
    t = t > 0.5

    size = len(p)
    batch_size = 2000000
    a, b, c, d = 0, 0, 0, 0
    print(f"accuracy - optimized; {size // batch_size + 1} iterations")
    for i in range(size // batch_size + 1):
        i1 = i * batch_size
        i2 = min((i + 1) * batch_size, size)

        a += (p[i1:i2] * t[i1:i2]).sum()
        b += t[i1:i2].sum()
        c += ((1-p[i1:i2]) * (1-t[i1:i2])).sum()
        d += (1-t[i1:i2]).sum()

    tp = a / b
    tn = c / d
    return tp, tn


def criterion_binary_cross_entropy(logit, mask):
    logit = logit.reshape(-1)
    mask = mask.reshape(-1)
    loss = F.binary_cross_entropy_with_logits(logit, mask)
    return loss
#
#
# def criterion_lovasz(logit, mask, mode='soft_hinge'):
#     logit = logit.reshape(-1)
#     mask = mask.reshape(-1)
#
#     loss = lovasz_loss(logit, mask, mode)
#     # loss = F.binary_cross_entropy_with_logits(logit, mask)
#     return loss


#unet ################################################################

def resize_like(x, reference, mode='nearest'):
    if x.shape[2:] !=  reference.shape[2:]:
        if mode == 'bilinear':
            x = F.interpolate(x, size=reference.shape[2:], mode='bilinear', align_corners=False)
        if mode == 'nearest':
            x = F.interpolate(x, size=reference.shape[2:], mode='nearest')
    return x


class SqueezeExcite(nn.Module):
    def __init__(self, in_channel, reduction=16):
        super().__init__()

        #channel squeeze excite
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, in_channel // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // reduction, in_channel, 1),
            nn.Sigmoid(),
        )

        #spatial squeeze excite
        self.spatial = nn.Sequential(
            nn.Conv2d(in_channel, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x * self.channel(x) + x * self.spatial(x)
        return x



class ResDecode(nn.Module):
    def __init__( self, in_channel, out_channel ):
        super().__init__()
        self.attent1 = SqueezeExcite(in_channel)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False),
            EnBatchNorm2d(out_channel), #nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attent2 = SqueezeExcite(out_channel)

    def forward(self, x):

        x = torch.cat(x, 1)
        x = self.attent1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attent2(x)
        return x


class CustomResnet34(nn.Module):
    # def load_pretrain(self, skip=['logit.',], is_print=True):
    #     load_pretrain(self, skip, pretrain_file=PRETRAIN_FILE, conversion=CONVERSION, is_print=is_print)

    def __init__(self):
        super(CustomResnet34, self).__init__()
        e = EnResNet34()
        self.rgb = RGB()

        self.block0 = e.block0
        self.block1 = e.block1
        self.block2 = e.block2
        self.block3 = e.block3
        self.block4 = e.block4
        e = None  #dropped

        #---
        # self.center = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Conv2d(256, 256, kernel_size=3, padding=2, dilation=2, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Conv2d(256, 256, kernel_size=3, padding=4, dilation=4, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        # )

        self.center = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=11, padding=5, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.decode1 = ResDecode(256+512, 256)
        self.decode2 = ResDecode(128+256, 128)
        self.decode3 = ResDecode( 64+128,  64)
        self.decode4 = ResDecode( 64+ 64,  32)
        self.decode5 = ResDecode(     32,  16)
        #---

        self.logit = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, image):
        batch_size,C,H,W = image.shape
        # print(batch_size, C, H, W)

        #x = self.rgb(image)
        x = image

        x0 = self.block0(x)         #;print('block0',x0.shape)
        x1 = self.block1(x0)        #;print('block1',x1.shape)
        x2 = self.block2(x1)        #;print('block2',x2.shape)
        x3 = self.block3(x2)        #;print('block3',x3.shape)
        x4 = self.block4(x3)        #;print('block4',x4.shape)

        skip = [x0,x1,x2,x3]

        #----
        z = self.center(x4)
        z = self.decode1([skip[-1], resize_like(z, skip[-1])])  #; print('d1',x.size())
        z = self.decode2([skip[-2], resize_like(z, skip[-2])])  #; print('d2',x.size())
        z = self.decode3([skip[-3], resize_like(z, skip[-3])])  #; print('d3',x.size())
        z = self.decode4([skip[-4], resize_like(z, skip[-4])])  #; print('d4',x.size())
        z = self.decode5([resize_like(z, x)])

        logit = self.logit(z)
        return logit


class Net(nn.Module):
    def __init__(self, backbone, pretrained=False):
        super(Net, self).__init__()

        if backbone == 'custom-resnet34':
            self.cnn_model = CustomResnet34()

        elif backbone in ['resnet34', 'efficientnet-b0', 'efficientnet-b1']:
            if pretrained:
                self.cnn_model = Unet(
                    encoder_name    = backbone,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                    encoder_weights = "imagenet",  # use `imagenet` pre-trained weights for encoder initialization
                    in_channels     = 3,           # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                    classes         = 1,           # model output channels (number of classes in your dataset)
                )
            else:
                self.cnn_model = Unet(
                    encoder_name    = backbone,
                    in_channels     = 3,
                    classes         = 1,
                )

    def forward(self, imgs):
        img_segs = self.cnn_model(imgs)
        return img_segs


#--------------------------------------------------------------------

def run_check_net():
    batch_size = 4
    image_size = 320

    #---
    mask  = np.random.choice(2,(batch_size,image_size,image_size))
    image = np.random.uniform(-1,1,(batch_size,3,image_size,image_size))

    mask = torch.from_numpy(mask).float()#.cuda()
    image = torch.from_numpy(image).float()#.cuda()

    net = Net()#.cuda()
    net.eval()
    logit = net(image)

    print('')
    print('mask: ',mask.shape)
    print('image: ',image.shape)
    print('logit: ',logit.shape)


# main #################################################################
if __name__ == '__main__':
    run_check_net()
