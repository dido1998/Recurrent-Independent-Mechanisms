import torch
import struct
import numpy as np
import gzip
import torchvision.transforms as transforms

def read_idx(filename):
	with gzip.open(filename, 'rb') as f:
		zero, data_type, dims = struct.unpack('>HBB', f.read(4))
		shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
		imgs = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
		return torch.from_numpy(imgs)

	# with gzip.open(filename, 'rb') as f:
    #     zero, data_type, dims = struct.unpack('>HBB', f.read(4))
    #     shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
    #     return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)





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

		
		_train_data = torch.zeros(self.train_data.shape[0], size[0] * size[1])
		val_data_1 = torch.zeros((self.val_data.shape[0], (size[0]  + 10)* (size[1] + 10)))
		val_data_2 = torch.zeros((self.val_data.shape[0], (size[0] + 5) * (size[1] + 5)))
		val_data_3 = torch.zeros((self.val_data.shape[0], (size[0] + 2) * (size[1] + 2)))

		self.transforms = transforms.Compose([
			transforms.Resize(size,interpolation=transforms.InterpolationMode.NEAREST)]
			# skipped thresholding. think it's not nodig. 
		)
		
		# for i in range(self.train_data.shape[0]):
		# 	img = self.train_data[i]
		# 	# img = cv2.resize(img, size, interpolation = cv2.INTER_NEAREST)
		# 	img = self.transforms(img)
		# 	#_, img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY) 

			
			
		# 	img = torch.reshape(img, (-1))
		# 	_train_data[i] = img

		_train_data = self.transforms(self.train_data)
		val_data_1 = transforms.Resize((size[0]+10,size[1]+10),
			interpolation=transforms.InterpolationMode.NEAREST)(self.val_data)
		val_data_2 = transforms.Resize((size[0]+5,size[1]+5),
			interpolation=transforms.InterpolationMode.NEAREST)(self.val_data)
		val_data_3 = transforms.Resize((size[0]+2,size[1]+2),
			interpolation=transforms.InterpolationMode.NEAREST)(self.val_data)
		
		self.train_data = _train_data
		self.val_data1 = val_data_1
		self.val_data2 = val_data_2
		self.val_data3 = val_data_3

		del _train_data

		self.train_labels = self.train_labels.unsqueeze(1)
		self.val_labels = self.val_labels.unsqueeze(1)

		# -- Questionable -- 
		# self.train_data = self.train_data.unsqueeze(3) #unsqueeze?
		# self.val_data1 = self.val_data1.unsqueeze(3)
		# self.val_data2 = self.val_data2.unsqueeze(3)
		# self.val_data3 = self.val_data3.unsqueeze(3)

		# self.train_data = [self.train_data[i:i + batch_size] for i in range(0, self.train_data.shape[0], batch_size)]
		# self.val_data1 = [self.val_data1[i:i + 512] for i in range(0, self.val_data1.shape[0], 512)]
		# self.val_data2 = [self.val_data2[i:i + 512] for i in range(0, self.val_data2.shape[0], 512)]
		# self.val_data3 = [self.val_data3[i:i + 512] for i in range(0, self.val_data3.shape[0], 512)]
		# self.train_labels = [self.train_labels[i:i + batch_size] for i in range(0, self.train_labels.shape[0], batch_size)]
		# self.val_labels = [self.val_labels[i:i + 512] for i in range(0, self.val_labels.shape[0], 512)]

	def train_len(self):
		return len(self.train_labels)

	def val_len(self):
		return len(self.val_labels)

	def train_get(self, i):
		return self.train_data[i], self.train_labels[i] ## batch????

	def val_get1(self, i):
		return self.val_data1[i], self.val_labels[i]

	def val_get2(self, i):
		return self.val_data2[i], self.val_labels[i] 

	def val_get3(self, i):
		return self.val_data3[i], self.val_labels[i]	


if __name__ == '__main__':
	mnistdata = MnistData(128, (128,128), 10)
	print(mnistdata.train_labels.shape)


