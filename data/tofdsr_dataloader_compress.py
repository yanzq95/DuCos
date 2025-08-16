import numpy as np
import os
import random

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torchvision.transforms as T
from scipy.ndimage import zoom

def get_patch(img,gt, patch_size=16):
    th, tw = img.shape[:2]  ## HR image

    tp = round(patch_size)

    tx = random.randrange(0, (tw - tp))
    ty = random.randrange(0, (th - tp))

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


class TOFDSR_Dataset(BaseDataset):

    def __init__(self, root_dir="/opt/data/private/dataset/", scale=4, downsample='real', train=True, txt_file='./TOFDC_Filled_Train.txt'):


        self.root_dir = root_dir
        self.scale = scale
        self.downsample = downsample
        self.train = train
        self.image_list = txt_file
        self.num_bit = 4
        self.downscale_factor = scale
        with open(self.image_list, 'r') as f:
            self.filename = f.readlines()

    def __len__(self):
        return len(self.filename)

    def __getitem__(self, idx):

        sample_path = self.filename[idx].strip('\n')
        sample_path_ = sample_path.split(',')
        rgb_path = sample_path_[0]
        gt_path = sample_path_[1]

        name = gt_path[20:-4]
        rgb_path = os.path.join(self.root_dir, rgb_path)
        gt_path = os.path.join(self.root_dir, gt_path)

        image = np.array(Image.open(rgb_path).convert("RGB")).astype(np.float32)
        gt = Image.open(gt_path)
        
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
            mask = (gt >= 100) & (gt <= 5000)
            compressed_lr = (compressed_lr - min_out) / (max_out - min_out)
            gt = (gt-min_out)/(max_out-min_out)
        else:
            compressed_lr = (compressed_lr - min_out) / (max_out - min_out)
            mask = (gt >= 100) & (gt <= 5000)

        # add noise
        noise_r = np.random.normal(0, 0.02, compressed_lr.shape)
        noise_a = np.random.normal(0, 0.05, compressed_lr.shape)
        noise = noise_r * compressed_lr + noise_a
        compressed_lr = compressed_lr + noise

        h, w = gt.shape[:2]
        lr = np.array(
            Image.fromarray(compressed_lr).resize((w, h), Image.BICUBIC))

        image_max = np.max(image)
        image_min = np.min(image)
        image = (image - image_min) / (image_max - image_min)


        t_dep = T.Compose([
            T.ToTensor()
        ])

        image = t_dep(image).float()
        gt = t_dep(gt).float()
        lr = t_dep(lr).float()
        mask = t_dep(mask)
        sample = {'rgb': image, 'dep': lr, 'gt': gt, 'max': max_out, 'min': min_out, 'mask': mask}

        return sample
