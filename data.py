import torch
import struct
import numpy as np
import gzip
import cv2

def read_idx(filename):
    with gzip.open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


#if __name__ == '__main__':
#	a = read_idx('mnist/train-labels-idx1-ubyte.gz')
#	print(a.shape)



class MnistData:
	def __init__(self, batch_size, size):
		self.train_data = read_idx('mnist/train-images-idx3-ubyte.gz')
		self.train_labels = read_idx('mnist/train-labels-idx1-ubyte.gz')
		self.val_data = read_idx('mnist/t10k-images-idx3-ubyte.gz')
		self.val_labels = read_idx('mnist/t10k-labels-idx1-ubyte.gz')


		train_data_ = np.zeros((self.train_data.shape[0], size[0] * size[1]))
		val_data_ = np.zeros((self.val_data.shape[0], size[0] * size[1]))
		for i in range(self.train_data.shape[0]):
			img = self.train_data[i, :]
			img = cv2.resize(img, size, interpolation = cv2.INTER_NEAREST)
			img = (img > 128) * 1.0
			img = np.reshape(img, (-1))
			train_data_[i, :] = img

		for i in range(self.val_data.shape[0]):
			img = self.val_data[i, :]
			img = cv2.resize(img, size, interpolation = cv2.INTER_NEAREST)
			img = (img > 128) * 1.0
			img = np.reshape(img, (-1))
			val_data_[i, :] = img
		self.train_data = train_data_
		self.val_data = val_data_

		del train_data_, val_data_

		self.train_data = np.reshape(self.train_data, (self.train_data.shape[0], self.train_data.shape[1], 1))
		self.val_data = np.reshape(self.val_data, (self.val_data.shape[0], self.val_data.shape[1], 1))

		self.train_data = [self.train_data[i:i + batch_size] for i in range(0, self.train_data.shape[0], batch_size)]
		self.val_data = [self.val_data[i:i + batch_size] for i in range(0, self.val_data.shape[0], batch_size)]

		self.train_labels = [self.train_labels[i:i + batch_size] for i in range(0, self.train_labels.shape[0], batch_size)]
		self.val_labels = [self.val_labels[i:i + batch_size] for i in range(0, self.val_labels.shape[0], batch_size)]

	def train_len(self):
		return len(self.train_labels)

	def val_len(self):
		return len(self.val_labels)

	def train_get(self, i):
		return self.train_data[i], self.train_labels[i]

	def val_get(self, i):
		self.val_data[i], self.val_labels[i]




