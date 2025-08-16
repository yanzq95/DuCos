import numpy as np
import os
import random

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torchvision.transforms as T

def get_patch(img, lr, gt, patch_size=16):
    th, tw = img.shape[:2]  ## HR image

    tp = round(patch_size)

    tx = random.randrange(0, (tw - tp))
    ty = random.randrange(0, (th - tp))

    return img[ty:ty + tp, tx:tx + tp], lr[ty:ty + tp, tx:tx + tp], gt[ty:ty + tp, tx:tx + tp]

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
    """RGB-D-D Dataset."""

    def __init__(self, root_dir="/opt/data/private/dataset/", scale=4, downsample='real', train=True, txt_file='./TOFDC_Filled_Train.txt'):
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
            max_out = 5000.0
            min_out = 0.0
            h, w = gt.shape
            s = self.scale
            lr = np.array(Image.open(lr_path).resize((w, h), Image.BICUBIC)).astype(np.float32)

        else:
            image = np.array(Image.open(rgb_path).convert("RGB")).astype(np.float32)
            gt = Image.open(gt_path)
            w, h = gt.size
            s = self.scale
            lr = np.array(gt.resize((int(w // 11.6),int(h // 11.6)), Image.BICUBIC).resize((w, h), Image.BICUBIC)).astype(np.float32)
            gt = np.array(gt).astype(np.float32)
            max_out = np.max(lr)
            min_out = np.min(lr)

        image_max = np.max(image)
        image_min = np.min(image)
        image = (image - image_min) / (image_max - image_min)

        

        # normalization
        if self.train:
            image, lr, gt = get_patch(img=image, lr=np.expand_dims(lr, 2), gt=np.expand_dims(gt, 2), patch_size=256)
            mask = (gt >= 100) & (gt <= 5000)
            lr = (lr - min_out) / (max_out - min_out)
            gt = (gt-min_out)/(max_out-min_out)
        else:
            lr = (lr - min_out) / (max_out - min_out)
            mask = (gt >= 100) & (gt <= 5000)

        t_dep = T.Compose([
            T.ToTensor()
        ])

        image = t_dep(image).float()
        gt = t_dep(gt).float()
        lr = t_dep(lr).float()
        mask = t_dep(mask)
        sample = {'rgb': image, 'dep': lr, 'gt': gt, 'max': max_out, 'min': min_out, 'mask': mask, 'name': name}

        return sample
