import numpy as np
import os
import random

from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter
from torchvision import transforms
from scipy.ndimage import gaussian_filter
import torch
import torchvision.transforms as T


#def get_patch(img, lr, gt, patch_size=16):
#    th, tw = img.shape[:2]  ## HR image
#
#    tp = round(patch_size)
#
#    tx = random.randrange(0, (tw - tp))
#    ty = random.randrange(0, (th - tp))
#
#    return img[ty:ty + tp, tx:tx + tp], lr[ty:ty + tp, tx:tx + tp], gt[ty:ty + tp, tx:tx + tp]

def get_patch(img, lr, gt, scale, patch_size=16):
    th, tw = img.shape[:2]  ## HR image

    tp = round(patch_size)

    tx = random.randrange(0, (tw-tp))
    ty = random.randrange(0, (th-tp))
    lr_tx = tx // scale
    lr_ty = ty // scale
    lr_tp = tp // scale

    return img[ty:ty + tp, tx:tx + tp], lr[lr_ty:lr_ty + lr_tp, lr_tx:lr_tx + lr_tp], gt[ty:ty + tp, tx:tx + tp]

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
    """RGB-D-D Dataset."""

    def __init__(self, root_dir="/opt/data/private/dataset/RGB-D-D/", scale=4, downsample='real', train=True,
                 isBlur=False, isNoisy=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            scale (float): dataset scale
            downsample (str): kernel type of downsample, real mean use real LR and HR data
            train (bool): train or test
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # types = ['models', 'plants', 'portraits']

        self.root_dir = root_dir

        self.scale = scale
        self.downsample = downsample
        self.train = train
        self.isBlur = isBlur
        self.isNoisy = isNoisy
        self.blur_sigma = 3.6

        if train:
            if self.downsample == 'real':
                self.GTs = []
                self.LRs = []
                self.RGBs = []
                list_dir = os.listdir('%s/%s/%s/' % (root_dir, "Train", "RGBDD_RGB"))  # Train
                for n in list_dir:
                    self.RGBs.append('%s/%s/%s/%s' % (root_dir, "Train", "RGBDD_RGB", n))
                    self.GTs.append('%s/%s/%s/%s_HR_gt.png' % (root_dir, "Train", "RGBDD_GT", n[:-8]))
                    self.LRs.append('%s/%s/%s/%s_LR_fill_depth.png' % (root_dir, "Train", "RGBDD_LR", n[:-8]))
            else:
                self.GTs = []
                self.RGBs = []
                self.Seg = []
                self.NormalS = []
                list_dir = os.listdir('%s/%s/%s/' % (root_dir, "Train", "RGBDD_RGB"))
                for n in list_dir:
                    self.RGBs.append('%s/%s/%s/%s' % (root_dir, "Train", "RGBDD_RGB", n))
                    self.GTs.append('%s/%s/%s/%s_HR_gt.png' % (root_dir, "Train", "RGBDD_GT", n[:-8]))

        else:
            if self.downsample == 'real':
                self.GTs = []
                self.LRs = []
                self.RGBs = []
                list_dir = os.listdir('%s/%s/%s/' % (root_dir, "Test", "RGBDD_RGB"))
                for n in list_dir:
                    self.RGBs.append('%s/%s/%s/%s' % (root_dir, "Test", "RGBDD_RGB", n))
                    self.GTs.append('%s/%s/%s/%s_HR_gt.png' % (root_dir, "Test", "RGBDD_GT", n[:-8]))
                    self.LRs.append('%s/%s/%s/%s_LR_fill_depth.png' % (root_dir, "Test", "RGBDD_LR", n[:-8]))
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
        if self.downsample == 'real':
            image = np.array(Image.open(self.RGBs[idx]).convert("RGB")).astype(np.float32)
            name = self.RGBs[idx][-22:-8]
            gt = np.array(Image.open(self.GTs[idx])).astype(np.float32)
            h, w = gt.shape
            s = self.scale
            lr = np.array(Image.open(self.LRs[idx]).resize((w // s, h // s), Image.BICUBIC)).astype(np.float32)
        else:
            image = Image.open(self.RGBs[idx]).convert("RGB")
            name = self.RGBs[idx][-22:-8]
            image = np.array(image).astype(np.float32)
            gt = Image.open(self.GTs[idx])
            w, h = gt.size
            s = self.scale
            lr = np.array(gt.resize((w // s, h // s), Image.BICUBIC).resize((w, h), Image.BICUBIC)).astype(np.float32)
            gt = np.array(gt).astype(np.float32)

        max_out = np.max(lr)
        min_out = np.min(lr)

        # normalization
        if self.train:
            lr = (lr - min_out) / (max_out - min_out)
            gt = (gt - min_out) / (max_out - min_out)
            image, lr, gt = get_patch(img=image, lr=lr, gt=gt, scale=self.scale, patch_size=288)
        else:
            lr = (lr - min_out) / (max_out - min_out)

        maxx = np.max(image)
        minn = np.min(image)
        image = (image - minn) / (maxx - minn)

        lr_minn = np.min(lr)
        lr_maxx = np.max(lr)

        if not self.train:
            np.random.seed(42)

        # add Blur
        if self.isBlur:
            lr = gaussian_filter(lr, sigma=self.blur_sigma)

        if self.isNoisy:
            gaussian_noise = np.random.normal(0, 0.07, lr.shape)
            lr = lr + gaussian_noise
            lr = np.clip(lr, lr_minn, lr_maxx)
        
        h1, w1 = gt.shape

        lr = np.array(Image.fromarray(lr).resize((w1, h1), Image.BICUBIC))

        t_dep = T.Compose([
            T.ToTensor()
        ])

        image = t_dep(image).float()
        gt = t_dep(np.expand_dims(gt, 2)).float()
        lr = t_dep(np.expand_dims(lr, 2)).float()

        sample = {'rgb': image, 'dep': lr, 'gt': gt, 'max': max_out, 'min': min_out, 'name': name}

        return sample
