import numpy as np
import os
import random
import h5py
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter
from torchvision import transforms
from scipy.ndimage import gaussian_filter
import torch
import torchvision.transforms as T


def get_patch(img, gt, patch_size=16):
    th, tw = img.shape[:2]  ## HR image

    tp = round(patch_size)

    tx = random.randrange(0, (tw-tp))
    ty = random.randrange(0, (th-tp))

    return img[ty:ty + tp, tx:tx + tp, :], gt[ty:ty + tp, tx:tx + tp, :]

def center_crop(image,gt, crop_width, crop_height):
    height, width = image.shape[:2]
    start_x = width // 2 - (crop_width // 2)
    start_y = height // 2 - (crop_height // 2)
    return image[start_y:start_y + crop_height, start_x:start_x + crop_width, :], gt[start_y:start_y + crop_height, start_x:start_x + crop_width]

def arugment(img,gt, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    if hflip:
        img = img[:, ::-1, :].copy()
        gt = gt[:, ::-1, :].copy()
    if vflip:
        img = img[::-1, :, :].copy()
        gt = gt[::-1, :, :].copy()
    if rot90:
        img = img.transpose(1, 0, 2).copy()
        gt = gt.transpose(1, 0, 2).copy()

    return img, gt

class SyncRGBD_Dataset(Dataset):

    def __init__(self, root_dir="/opt/data/private/dataset/", txt_file='./SyncRGBD_Test.txt',
                 scale=4, train=True):

        super(SyncRGBD_Dataset, self).__init__()
        self.root_dir = root_dir

        self.scale = scale
        self.train = train

        self.image_list = txt_file
        with open(self.image_list, 'r') as f:
            self.filename = f.readlines()

    def __len__(self):
        return len(self.filename)

    def __getitem__(self, idx):

        sample_path = self.filename[idx].strip('\n')
        sample_path_ = sample_path.split(',')
        rgb_path = sample_path_[0]
        gt_path = sample_path_[1]

        rgb_path = os.path.join(self.root_dir, rgb_path)
        gt_path = os.path.join(self.root_dir, gt_path)

        image = np.array(Image.open(rgb_path).convert("RGB")).astype(np.float32)
        with h5py.File(gt_path, 'r') as hdf5_file:
            gt_depth = hdf5_file['dataset'][:]
        gt_depth = gt_depth.astype(np.float32)

        image_max = np.max(image)
        image_min = np.min(image)
        image = (image - image_min) / (image_max - image_min)

        gt_depth_max = np.max(gt_depth)
        gt_depth_min = np.min(gt_depth)
        gt_depth = (gt_depth - gt_depth_min) / (gt_depth_max - gt_depth_min)

        if self.train:
            image, gt_depth = get_patch(img=image, gt=np.expand_dims(gt_depth,2), patch_size=256)
            image, gt_depth = arugment(img=image, gt=gt_depth) #---------------------------------------
        else:
            image, gt_depth = center_crop(image=image,gt=gt_depth, crop_width=512, crop_height=384)
        h, w = gt_depth.shape[:2]
        s = self.scale

        lr = np.array(Image.fromarray(gt_depth.squeeze()).resize((w // s, h // s), Image.BICUBIC).resize((w, h), Image.BICUBIC))

        t_dep = T.Compose([
            T.ToTensor()
        ])

        image = t_dep(image).float()
        gt_depth = t_dep(gt_depth).float()
        lr = t_dep(lr).float()

        sample = {'rgb': image, 'dep': lr, 'gt': gt_depth, 'max': gt_depth_max, 'min': gt_depth_min}

        return sample
