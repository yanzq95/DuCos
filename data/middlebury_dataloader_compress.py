from torchvision import transforms
import numpy as np
import os
import random
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from scipy.ndimage import zoom

def modcrop(image, modulo):
    h, w = image.shape[0], image.shape[1]
    h = h - h % modulo
    w = w - w % modulo

    return image[:h,:w]

class Middlebury_dataset(Dataset):

    def __init__(self, root_dir, scale=8, transform=None):

        self.transform = transform
        self.scale = scale
        self.GTs = []
        self.RGBs = []

        self.num_bit = 4
        self.downscale_factor = scale

        list_dir = os.listdir(root_dir)
        for name in list_dir:
            if name.find('output_color') > -1:
                self.RGBs.append('%s/%s' % (root_dir, name))
            elif name.find('output_depth') > -1:
                self.GTs.append('%s/%s' % (root_dir, name))
        self.RGBs.sort()
        self.GTs.sort()

    def __len__(self):
        return len(self.GTs)

    def __getitem__(self, idx):
        
        image = np.array(Image.open(self.RGBs[idx]).convert("RGB")).astype(np.float32)
        
        gt = np.array(Image.open(self.GTs[idx])).astype(np.float32)
        assert gt.shape[0] == image.shape[0] and gt.shape[1] == image.shape[1]
        s = self.scale  
        image = modcrop(image, 16)
        gt = modcrop(gt, 16)

        h, w = gt.shape[0], gt.shape[1]
        s = self.scale

        compressed_lr = gt.astype(np.uint16)
        compressed_lr = compressed_lr >> self.num_bit

        compressed_lr = zoom(compressed_lr.squeeze(), 1 / self.downscale_factor, order=2)

        gt = gt / 255.0
        image = image / 255.0
        compressed_lr = compressed_lr / 255.0

        # add noise
        noise_r = np.random.normal(0, 0.02, compressed_lr.shape)
        noise_a = np.random.normal(0, 0.05, compressed_lr.shape)
        noise = noise_r * compressed_lr + noise_a
        compressed_lr = compressed_lr + noise

        lr = np.array(
            Image.fromarray(compressed_lr).resize((w, h), Image.BICUBIC))

        t_dep = T.Compose([
            T.ToTensor()
        ])
        
        image = t_dep(image).float()
        gt = t_dep(np.expand_dims(gt,2)).float()
        lr = t_dep(lr).float()
            
        sample = {'rgb': image, 'dep': lr, 'gt': gt}
        return sample