import numpy as np
import os
import random

from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter
from torchvision import transforms
from scipy.ndimage import gaussian_filter
import torch
import torchvision.transforms as T
from scipy.ndimage import zoom

def get_patch(img, gt, patch_size=16):
    th, tw = img.shape[:2]  ## HR image

    tp = round(patch_size)

    tx = random.randrange(0, (tw-tp))
    ty = random.randrange(0, (th-tp))

    return img[ty:ty + tp, tx:tx + tp], gt[ty:ty + tp, tx:tx + tp]

class BaseDataset(Dataset):
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    class ToNumpy:
        def __call__(self, sample):
            return np.array(sample)


class RGBDD_Dataset(BaseDataset):

    def __init__(self, root_dir="/opt/data/private/dataset/RGB-D-D/", scale=4, downsample='real', train=True):

        self.root_dir = root_dir

        self.scale = scale
        self.downsample = downsample
        self.train = train

        self.num_bit = 4
        self.downscale_factor = scale

        if train:
            self.GTs = []
            self.RGBs = []
            self.Seg = []
            self.NormalS = []
            list_dir = os.listdir('%s/%s/%s/' % (root_dir, "Train", "RGBDD_RGB"))#Train
            for n in list_dir:
                self.RGBs.append('%s/%s/%s/%s' % (root_dir, "Train", "RGBDD_RGB", n))
                self.GTs.append('%s/%s/%s/%s_HR_gt.png' % (root_dir, "Train", "RGBDD_GT", n[:-8]))

        else:
            self.GTs = []
            self.RGBs = []
            list_dir = os.listdir('%s/%s/%s/' % (root_dir, "Test", "RGBDD_RGB"))
            for n in list_dir:
                self.RGBs.append('%s/%s/%s/%s' % (root_dir, "Test", "RGBDD_RGB", n))
                self.GTs.append('%s/%s/%s/%s_HR_gt.png' % (root_dir, "Test", "RGBDD_GT", n[:-8]))

    def __len__(self):
        return len(self.GTs)

    def __getitem__(self, idx):

        image = Image.open(self.RGBs[idx]).convert("RGB")
        name = self.RGBs[idx][-22:-8]
        image = np.array(image).astype(np.float32)
        gt = Image.open(self.GTs[idx])
        s = self.scale
        gt = np.array(gt).astype(np.float32)

        if self.train:
            image, gt = get_patch(img=image, gt=np.expand_dims(gt, 2), patch_size=256)
            

        max_out = np.max(gt)
        min_out = np.min(gt)

        compressed_lr = gt.astype(np.uint16)

        compressed_lr = compressed_lr >> self.num_bit


        # down sampliong
        compressed_lr = zoom(compressed_lr.squeeze(), 1 / self.downscale_factor, order=2)

        # normalization
        if self.train:
            compressed_lr = (compressed_lr - min_out) / (max_out - min_out)
            gt = (gt-min_out)/(max_out-min_out)
        else:
            compressed_lr = (compressed_lr - min_out) /(max_out - min_out)

        # add noise
        noise_r = np.random.normal(0, 0.02, compressed_lr.shape)
        noise_a = np.random.normal(0, 0.05, compressed_lr.shape)
        noise = noise_r * compressed_lr + noise_a
        compressed_lr = compressed_lr + noise

        h, w = gt.shape[:2]
        lr = np.array(
            Image.fromarray(compressed_lr).resize((w, h), Image.BICUBIC))

        maxx = np.max(image)
        minn = np.min(image)
        image = (image - minn) / (maxx - minn)

        t_dep = T.Compose([
            T.ToTensor()
        ])

        image = t_dep(image).float()
        gt = t_dep(gt).float()
        lr = t_dep(lr).float()
        sample = {'rgb': image, 'dep': lr, 'gt': gt, 'max': max_out, 'min': min_out, 'name': name}

        return sample
