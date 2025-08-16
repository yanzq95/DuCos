import numpy as np
import os
import random

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torchvision.transforms as T
from scipy.ndimage import gaussian_filter

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


class TOFDSR_Dataset(BaseDataset):

    def __init__(self, root_dir="/opt/data/private/dataset/", scale=4, downsample='real', train=True, txt_file='./TOFDC_Filled_Train.txt', isBlur=False, isNoisy=False):


        self.root_dir = root_dir
        self.scale = scale
        self.downsample = downsample
        self.train = train
        self.isBlur = isBlur
        self.isNoisy = isNoisy
        self.blur_sigma = 3.6
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
        lr_path = sample_path_[2]

        name = gt_path[20:-4]
        rgb_path = os.path.join(self.root_dir, rgb_path)
        gt_path = os.path.join(self.root_dir, gt_path)
        lr_path = os.path.join(self.root_dir, lr_path)

        if self.downsample == 'real':
            image = np.array(Image.open(rgb_path).convert("RGB")).astype(np.float32)
            gt = np.array(Image.open(gt_path)).astype(np.float32)

            h, w = gt.shape
            s = self.scale
            lr = np.array(Image.open(lr_path).resize((w // s, h // s), Image.BICUBIC)).astype(np.float32)

        else:
            image = np.array(Image.open(rgb_path).convert("RGB")).astype(np.float32)
            gt = Image.open(gt_path)
            w, h = gt.size
            s = self.scale
            lr = np.array(gt.resize((w, h), Image.BICUBIC)).astype(np.float32)
            gt = np.array(gt).astype(np.float32)

        image_max = np.max(image)
        image_min = np.min(image)
        image = (image - image_min) / (image_max - image_min)

        # normalization
        if self.train:
            max_out = 5000.0
            min_out = 0.0
            lr = (lr - min_out) / (max_out - min_out)
            gt = (gt-min_out)/(max_out-min_out)
            image, lr, gt = get_patch(img=image, lr=lr, gt=gt, scale=self.scale, patch_size=256)
        else:
            max_out = 5000.0
            min_out = 0.0
            lr = (lr - min_out) / (max_out - min_out)
            
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
