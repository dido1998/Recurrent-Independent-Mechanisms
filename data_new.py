import torch
import struct
import numpy as np
import gzip
import torchvision.transforms as transforms
from torch.utils.data import Dataset

def read_gz(filename):
	with gzip.open(filename, 'rb') as f:
		zero, data_type, dims = struct.unpack('>HBB', f.read(4))
		shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
		imgs = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
		return torch.from_numpy(imgs)

class MnistSet(Dataset):
    def __init__(self, img_dir, anno_dir, transform=None):
        self.imgs = read_gz(img_dir)
        self.annos = read_gz(anno_dir)
        self.transform = transform

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.annos[idx]
        if self.transform:
            img = self.transform(img.unsqueeze(0))
        return img, label
