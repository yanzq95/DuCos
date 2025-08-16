from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as T
import random
from scipy.ndimage import zoom

def get_patch(img, gt, patch_size=16):
    th, tw = img.shape[:2]

    tp = round(patch_size)

    tx = random.randrange(0, (tw-tp))
    ty = random.randrange(0, (th-tp))

    return img[ty:ty + tp, tx:tx + tp, :], gt[ty:ty + tp, tx:tx + tp, :]

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

class CompreNYU_v2_datset(Dataset):

    def __init__(self, root_dir, scale=4, train=True, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.scale = scale
        self.train = train

        self.num_bit = 4
        self.downscale_factor = scale
        
        if train:
            self.depths = np.load('%s/train_depth_split.npy'%root_dir)
            self.images = np.load('%s/train_images_split.npy'%root_dir)
        else:
            self.depths = np.load('%s/test_depth.npy'%root_dir)
            self.images = np.load('%s/test_images_v2.npy'%root_dir)

    def __len__(self):
        return self.depths.shape[0]

    def __getitem__(self, idx):
        depth = self.depths[idx]
        image = self.images[idx]

        if self.train:
            image, depth = get_patch(img=image, gt=np.expand_dims(depth,2), patch_size=256)

        compressed_lr = depth.astype(np.uint16)

        compressed_lr = compressed_lr >> self.num_bit

        compressed_lr = zoom(compressed_lr.squeeze(), 1 / self.downscale_factor, order=2)

        # add noise
        noise_r = np.random.normal(0, 0.02, compressed_lr.shape)
        noise_a = np.random.normal(0, 0.05, compressed_lr.shape)
        noise = noise_r * compressed_lr + noise_a
        compressed_lr = compressed_lr + noise

        h, w = depth.shape[:2]
        s = self.scale
        lr = np.array(
            Image.fromarray(compressed_lr).resize((w, h), Image.BICUBIC))

        t_dep = T.Compose([
            T.ToTensor()
        ])

        image = t_dep(image).float()
        depth = t_dep(depth).float()
        lr = t_dep(lr).float()

        sample = {'rgb': image, 'dep': lr, 'gt': depth}
        
        return sample