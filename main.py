import torch
from data import MnistData
from networks import MnistModel
from tqdm import tqdm
import pickle
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

parser.add_argument('--cuda', type = str2bool, default = True, help = 'use gpu or not')
parser.add_argument('--epochs', type = int, default = 100)

parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--hidden_size', type = int, default = 100)
parser.add_argument('--input_size', type = int, default = 1)

parser.add_argument('--num_units', type = int, default = 6)
parser.add_argument('--rnn_cell', type = str, default = 'LSTM')
parser.add_argument('--key_size_input', type = int, default = 64)
parser.add_argument('--value_size_input', type = int, default = 400 )
parser.add_argument('--query_size_input', type = int, default = 64)
parser.add_argument('--num_input_heads', type = int, default = 1)
parser.add_argument('--num_comm_heads', type = int, default = 4)

parser.add_argument('--key_size_comm', type = int, default = 32)
parser.add_argument('--value_size_comm', type = int, default = 100 )
parser.add_argument('--query_size_comm', type = int, default = 32)
parser.add_argument('--k', type = int, default = 4)

parser.add_argument('--size', type = int, default = 14)
parser.add_argument('--loadsaved', type = int, default = 0)


args = vars(parser.parse_args())

log_dir = 'models'


def test_model(model, loader):
	
	accuracy = 0
	loss = 0
	with torch.no_grad():
		for i in tqdm(range(loader.val_len())):
			test_x, test_y = loader.val_get(i)
			test_x = model.to_device(test_x)
			test_y = model.to_device(test_y).long()
			
			
			probs  = model(test_x)

			#loss += l.item()
			preds = torch.argmax(probs, dim=1)
			correct = preds == test_y
			accuracy += correct.sum().item()

	accuracy /= 100.0
	return accuracy

def train_model(model, epochs, data):
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
		
	for epoch in range(start_epoch,epochs):
		
		print('epoch ' + str(epoch + 1))
		epoch_loss = 0.
		iter_ctr = 0.
		for i in tqdm(range(data.train_len())):
			iter_ctr+=1.
			inp_x, inp_y = data.train_get(i)
			inp_x = model.to_device(inp_x)
			inp_y = model.to_device(inp_y)
			
			
			
			
			output, l = model(inp_x, inp_y)
			#print('-------------------------')
			model.update(l)
			epoch_loss += l.item()
			# print(z, loss_val)
			# writer.add_scalar('/hdetach:loss', loss_val, ctr)
			ctr += 1

		v_accuracy = test_model(model, data)
		if best_acc < v_accuracy:
			best_acc = v_accuracy
			print('best validation accuracy ' + str(best_acc))
			print('Saving best model..')
			state = {
	        'net': model,	
	        'epoch':epoch,
	    	'ctr':ctr,
	    	'best_acc':best_acc
	    	}
			with open(log_dir + '/best_model.pt', 'wb') as f:
				torch.save(state, f)
		print('epoch_loss: {}, val accuracy: {} '.format(epoch_loss/(iter_ctr), v_accuracy))
		lossstats.append((ctr,epoch_loss/iter_ctr))
		acc.append((epoch,v_accuracy))
		with open(log_dir+'/lossstats.pickle','wb') as f:
			pickle.dump(lossstats,f)
		with open(log_dir+'/accstats.pickle','wb') as f:
			pickle.dump(acc,f)

data = MnistData(args['batch_size'], (args['size'], args['size']))
model = MnistModel(args).cuda()

train_model(model, args['epochs'], data)




