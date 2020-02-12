import numpy as np
import torch
import sys

tensor = torch.FloatTensor

def generate_copying_sequence(T):
	
	items = [1, 2, 3, 4, 5, 6, 7, 8, 0, 9]
	x = []
	y = []

	ind = np.random.randint(8, size=10)
	for i in range(10):
		x.append([items[ind[i]]])
	for i in range(T - 1):
		x.append([items[8]])
	x.append([items[9]])
	for i in range(10):
		x.append([items[8]])

	for i in range(T + 10):
		y.append([items[8]])
	for i in range(10):
		y.append([items[ind[i]]])

	x = np.array(x)
	y = np.array(y)

	return tensor([x]), torch.LongTensor([y])	


if __name__=='__main__':
	generate_copying_sequence(300)