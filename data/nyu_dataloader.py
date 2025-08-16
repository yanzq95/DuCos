from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as T


class NYU_v2_datset(Dataset):

    def __init__(self, root_dir, scale=4, train=True, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.scale = scale
        self.train = train
        
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

        h, w = depth.shape[:2]
        s = self.scale
        lr = np.array(Image.fromarray(depth.squeeze()).resize((w//s,h//s), Image.BICUBIC).resize((w,h), Image.BICUBIC))

        t_dep = T.Compose([
            T.ToTensor()
        ])

        image = t_dep(image).float()
        depth = t_dep(np.expand_dims(depth,2)).float()
        lr = t_dep(lr).float()

        sample = {'rgb': image, 'dep': lr, 'gt': depth}
        
        return sample