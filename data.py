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
	def __init__(self, batch_size, size, k):
		#self.train_data = read_idx('/content/drive/My Drive/Recurrent Independent Mechanisms/mnist/train-images-idx3-ubyte.gz')
		#self.train_labels = read_idx('/content/drive/My Drive/Recurrent Independent Mechanisms/mnist/train-labels-idx1-ubyte.gz')
		#self.val_data = read_idx('/content/drive/My Drive/Recurrent Independent Mechanisms/mnist/t10k-images-idx3-ubyte.gz')
		#self.val_labels = read_idx('/content/drive/My Drive/Recurrent Independent Mechanisms/mnist/t10k-labels-idx1-ubyte.gz')


		self.train_data = read_idx('mnist/train-images-idx3-ubyte.gz')
		self.train_labels = read_idx('mnist/train-labels-idx1-ubyte.gz')
		self.val_data = read_idx('mnist/t10k-images-idx3-ubyte.gz')
		self.val_labels = read_idx('mnist/t10k-labels-idx1-ubyte.gz')

		
		train_data_ = np.zeros((self.train_data.shape[0], size[0] * size[1]))
		val_data_1 = np.zeros((self.val_data.shape[0], (size[0]  + 10)* (size[1] + 10)))
		val_data_2 = np.zeros((self.val_data.shape[0], (size[0] + 5) * (size[1] + 5)))
		val_data_3 = np.zeros((self.val_data.shape[0], (size[0] + 2) * (size[1] + 2)))
		for i in range(self.train_data.shape[0]):
			img = self.train_data[i, :]
			img = cv2.resize(img, size, interpolation = cv2.INTER_NEAREST)

			_, img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY) 

			
			
			img = np.reshape(img, (-1))
			train_data_[i, :] = img

		for i in range(self.val_data.shape[0]):
			img = self.val_data[i, :]
			img1 = cv2.resize(img, (size[0] + 10, size[1] + 10), interpolation = cv2.INTER_NEAREST)
			_, img1 = cv2.threshold(img1, 120, 255, cv2.THRESH_BINARY) 
			img1 = np.reshape(img1, (-1))
			val_data_1[i, :] = img1

			img2 = cv2.resize(img, (size[0] + 5, size[1] + 5), interpolation = cv2.INTER_NEAREST)
			_, img2 = cv2.threshold(img2, 120, 255, cv2.THRESH_BINARY) 
			img2 = np.reshape(img2, (-1))
			val_data_2[i, :] = img2

			img3 = cv2.resize(img, (size[0] + 2, size[1] + 2), interpolation = cv2.INTER_NEAREST)
			_, img3 = cv2.threshold(img3, 120, 255, cv2.THRESH_BINARY) 
			img3 = np.reshape(img3, (-1))
			val_data_3[i, :] = img3

			
		self.train_data = train_data_
		self.val_data1 = val_data_1
		self.val_data2 = val_data_2
		self.val_data3 = val_data_3

		del train_data_

		self.train_data = np.reshape(self.train_data, (self.train_data.shape[0], self.train_data.shape[1], 1))
		self.val_data1 = np.reshape(self.val_data1, (self.val_data1.shape[0], self.val_data1.shape[1], 1))
		self.val_data2 = np.reshape(self.val_data2, (self.val_data2.shape[0], self.val_data2.shape[1], 1))
		self.val_data3 = np.reshape(self.val_data3, (self.val_data3.shape[0], self.val_data3.shape[1], 1))

		self.train_data = [self.train_data[i:i + batch_size] for i in range(0, self.train_data.shape[0], batch_size)]
		self.val_data1 = [self.val_data1[i:i + 512] for i in range(0, self.val_data1.shape[0], 512)]
		self.val_data2 = [self.val_data2[i:i + 512] for i in range(0, self.val_data2.shape[0], 512)]
		self.val_data3 = [self.val_data3[i:i + 512] for i in range(0, self.val_data3.shape[0], 512)]
		self.train_labels = [self.train_labels[i:i + batch_size] for i in range(0, self.train_labels.shape[0], batch_size)]
		self.val_labels = [self.val_labels[i:i + 512] for i in range(0, self.val_labels.shape[0], 512)]

	def train_len(self):
		return len(self.train_labels)

	def val_len(self):
		return len(self.val_labels)

	def train_get(self, i):
		return self.train_data[i], self.train_labels[i]

	def val_get1(self, i):
		return self.val_data1[i], self.val_labels[i]

	def val_get2(self, i):
		return self.val_data2[i], self.val_labels[i] 

	def val_get3(self, i):
		return self.val_data3[i], self.val_labels[i]	





