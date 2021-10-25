import torch
import struct
import numpy as np
import gzip
import torchvision.transforms.functional as tf
from torch.utils.data import Dataset, DataLoader

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

class Resize(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
    
    def forward(self, img):
        img = tf.resize(img, self.size)
        return img

class ToVector(torch.nn.Module): # i think it already exists?
    def __init__(self):
        super().__init__()

    def forward(self, img):
        # vec = img.reshape((img.shape[0],-1)) # equivalent to the next line
        vec = img.flatten(start_dim=1) 
        return vec

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
        img = self.imgs[idx].unsqueeze(0)
        label = self.annos[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

train_trans = Compose([
	Resize((14,14)),
	ToVector()
])

def main():
    train_set = MnistSet(img_dir='mnist/train-images-idx3-ubyte.gz',
		anno_dir='mnist/train-labels-idx1-ubyte.gz',
		transform=train_trans)
    img, label = next(iter(train_set))
    train_loader = DataLoader(train_set, batch_size=64, 
		shuffle=True, drop_last=False, num_workers=2)
    imgs, labels = next((iter(train_loader)))   
    print(f"imgs batch {imgs.shape[0]}, labels batch {labels.shape[0]}")

if __name__ == "__main__":
    main()