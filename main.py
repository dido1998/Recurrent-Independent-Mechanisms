import torch
# from data_nocv2 import MnistData
from data_new import MnistSet
from torch.utils.data import DataLoader
from networks import MnistModel, LSTM
from tqdm import tqdm
import pickle
import argparse
import numpy as np
import torchvision as tv

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

parser.add_argument('--cuda', type = str2bool, default = True, help = 'use gpu or not')
parser.add_argument('--epochs', type = int, default = 200)

parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--hidden_size', type = int, default = 100)
parser.add_argument('--input_size', type = int, default = 1)
parser.add_argument('--model', type = str, default = 'LSTM')
parser.add_argument('--train', type = str2bool, default = True)
parser.add_argument('--num_units', type = int, default = 6)
parser.add_argument('--rnn_cell', type = str, default = 'LSTM')
parser.add_argument('--key_size_input', type = int, default = 64)
parser.add_argument('--value_size_input', type = int, default =  400)
parser.add_argument('--query_size_input', type = int, default = 64)
parser.add_argument('--num_input_heads', type = int, default = 1)
parser.add_argument('--num_comm_heads', type = int, default = 4)
parser.add_argument('--input_dropout', type = float, default = 0.1)
parser.add_argument('--comm_dropout', type = float, default = 0.1)

parser.add_argument('--key_size_comm', type = int, default = 32)
parser.add_argument('--value_size_comm', type = int, default = 100)
parser.add_argument('--query_size_comm', type = int, default = 32)
parser.add_argument('--k', type = int, default = 4)

parser.add_argument('--size', type = int, default = 14)
parser.add_argument('--loadsaved', type = int, default = 0)
parser.add_argument('--log_dir', type = str, default = 'smnist_lstm_600')

args = vars(parser.parse_args())

log_dir = args['log_dir']
torch.manual_seed(10)
np.random.seed(10)
torch.cuda.manual_seed(10)

if args['model'] == 'LSTM':
	mode = LSTM
else:
	mode = MnistModel

def resize_func(size):
	trans = tv.transforms.Resize(size,
		interpolation=tv.transforms.InterpolationMode.NEAREST)
	return trans

def test_model(model, val_data):
	accuracy = 0
	loss = 0
	model.eval()
	with torch.no_grad():
		for imgs, labels in tqdm(val_data):
			test_x = model.to_device(imgs)
			test_y = model.to_device(labels).long()
			
			probs  = model(test_x)

			preds = torch.argmax(probs, dim=1)
			correct = preds == test_y
			accuracy += correct.sum().item()

	accuracy /= 100.0
	return accuracy

def train_model(model, epochs, train_data, val_data):
	acc=[]
	lossstats=[]
	best_acc = 0.0
	ctr = 0	

	test_acc = 0
	start_epoch=0
	ctr=0
	if args['loadsaved']==1:
		with open(log_dir+'/accstats.pickle','rb') as f:
			acc=pickle.load(f)
		with open(log_dir+'/lossstats.pickle','rb') as f:
			losslist=pickle.load(f)
		start_epoch=len(acc)-1
		best_acc=0
		for i in acc:
			if i[0]>best_acc:
				best_acc=i[0]
		ctr=len(losslist)-1
		saved = torch.load(log_dir + '/best_model.pt')
		model.load_state_dict(saved['net'])
	optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
		
	for epoch in range(start_epoch,epochs):
		print('epoch ' + str(epoch + 1))
		epoch_loss = 0.
		iter_ctr = 0.
		t_accuracy = 0
		norm = 0
		model.train()
		for imgs, labels in tqdm(train_data):
			iter_ctr+=1.
			
			output, l = model(imgs, labels)
			
			optimizer.zero_grad()
			l.backward()
			optimizer.step()
			norm += model.grad_norm()
			epoch_loss += l.item() # l is type?
			preds = torch.argmax(output, dim=1) # could be wrong
			
			correct = preds == labels.long()
			t_accuracy += correct.sum().item()

			ctr += 1

		v_accuracy = test_model(model, val_data) # not yet revised
		# v_accuracy2 = test_model(model, val_data, data.val_get2)
		# v_accuracy3 = test_model(model, val_data, data.val_get3)
		
		print('previous best validation accuracy ' + str(best_acc))
		print('Saving best model..')
		state = {
	       'net': model.state_dict(),	
	       'epoch':epoch,
	    'ctr':ctr,
	    'best_acc':best_acc
	    }
		with open(log_dir + '/best_model.pt', 'wb') as f:
			torch.save(state, f)
		print('epoch_loss: {}, val accuracy: {}, train_acc: {}, grad_norm: {} '.format(epoch_loss/(iter_ctr), v_accuracy, t_accuracy / 600, norm/iter_ctr))
		lossstats.append((ctr,epoch_loss/iter_ctr))
		acc.append((epoch,(v_accuracy)))
		with open(log_dir+'/lossstats.pickle','wb') as f:
			pickle.dump(lossstats,f)
		with open(log_dir+'/accstats.pickle','wb') as f:
			pickle.dump(acc,f)

def main():
	train_trans = resize_func((args['size'], args['size']))
	val1_trans = resize_func((args['size']+10, args['size']+10))
	val2_trans = resize_func((args['size']+5, args['size']+5))
	val3_trans = resize_func((args['size']+2, args['size']+2))
	train_set = MnistSet(img_dir='mnist/train-images-idx3-ubyte.gz',
		anno_dir='mnist/train-labels-idx1-ubyte.gz',
		transform=train_trans)
	val1_set = MnistSet('mnist/t10k-images-idx3-ubyte.gz',
		'mnist/t10k-labels-idx1-ubyte.gz',
		val1_trans)
	val2_set = MnistSet('mnist/t10k-images-idx3-ubyte.gz',
		'mnist/t10k-labels-idx1-ubyte.gz',
		val2_trans)
	val3_set = MnistSet('mnist/t10k-images-idx3-ubyte.gz',
		'mnist/t10k-labels-idx1-ubyte.gz',
		val3_trans)
	train_loader = DataLoader(train_set, batch_size=args['batch_size'], 
		shuffle=True, drop_last=False, num_workers=2)
	val1_loader = DataLoader(val1_set, batch_size=args['batch_size'], 
		shuffle=True, drop_last=False, num_workers=2)
	
	# data = MnistSet(args['batch_size'], (args['size'], args['size']), args['k'])
	model = mode(args).cuda() if torch.cuda.is_available() else mode(args)

	if args['train']:
		train_model(model, args['epochs'], train_loader, val1_loader)
	else:
		saved = torch.load(log_dir + '/best_model.pt')
		model.load_state_dict(saved['net'])
		v_acc = test_model(model, val1_loader)
		print('val_acc:'+str(v_acc))

if __name__ == "__main__":
	main()




