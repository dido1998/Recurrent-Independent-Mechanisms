import torch
from networks import CopyingModel
from tqdm import tqdm
import pickle
import argparse
from generator import generate_copying_sequence
import numpy as np


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

parser.add_argument('--batch_size', type = int, default = 500)
parser.add_argument('--hidden_size', type = int, default = 100)
parser.add_argument('--input_size', type = int, default = 1)
parser.add_argument('--T', type=int, default=50, help='T')

parser.add_argument('--num_units', type = int, default = 6)
parser.add_argument('--rnn_cell', type = str, default = 'LSTM')
parser.add_argument('--key_size_input', type = int, default = 128)
parser.add_argument('--value_size_input', type = int, default = 400 )
parser.add_argument('--query_size_input', type = int, default = 128)
parser.add_argument('--num_input_heads', type = int, default = 1)
parser.add_argument('--num_comm_heads', type = int, default = 4)
parser.add_argument('--input_dropout', type = float, default = 0.1)
parser.add_argument('--comm_dropout', type = float, default = 0.1)


parser.add_argument('--key_size_comm', type = int, default = 128)
parser.add_argument('--value_size_comm', type = int, default = 100 )
parser.add_argument('--query_size_comm', type = int, default = 128)
parser.add_argument('--k', type = int, default = 4)

parser.add_argument('--size', type = int, default = 14)
parser.add_argument('--loadsaved', type = int, default = 1)
parser.add_argument('--log_dir', type = str, default = 'copying_6_4')

args = vars(parser.parse_args())

log_dir = args['log_dir']


inp_size = 1
out_size = 9
train_size = 50000
test_size = 2000

def create_dataset(size, T):
	d_x = []
	d_y = []
	for i in range(size):
		sq_x, sq_y = generate_copying_sequence(T)
		sq_x, sq_y = sq_x[0], sq_y[0]
		d_x.append(sq_x)
		d_y.append(sq_y)

	d_x = torch.stack(d_x)
	d_y = torch.stack(d_y)
	return d_x, d_y





def test_model(model, test_x, test_y):
	model.eval()
	loss = 0
	accuracy = 0
	iters = 0
	#test_x = model.to_device(test_x)
	#test_y = model.to_device(test_y)
	loss = 0
	for z in tqdm(range(test_size // args['batch_size']), total=test_size // args['batch_size']):
			iters += 1
			ind = np.random.choice(test_size, args['batch_size'])
			inp_x, inp_y = test_x[ind], test_y[ind]
			
			with torch.no_grad():
				output, _, l = model(inp_x, inp_y)
				loss+=l.item()

				output = torch.argmax(output, dim = 2)
				output = output[:, 210:]


				inp_y = torch.squeeze(inp_y)[:, 210:]
				#print(output)
				correct = output == inp_y
				accuracy += correct.sum().item()

	accuracy /= (200.0)
	loss/=4

	print('validation accuracy {} loss: {}'.format(accuracy, loss))
	return loss, accuracy

def train_model(model, epochs, state = None):
	if args['loadsaved'] == 1:
		train_x = state['train_x']
		train_y = state['train_y']
		test_x = state['test_x']
		test_y = state['test_y']
	else:	
		train_x, train_y = create_dataset(train_size, args['T'])
		test_x, test_y = create_dataset(test_size, 200)
	train_x, train_y = train_x.cuda(), train_y.cuda()
	test_x, test_y = test_x.cuda(), test_y.cuda()
	global best_acc, ctr, start_epoch
	losslist=[]
	acc=[]
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
	
	iters = -1
	p_detach=0.
	optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005)
	for epoch in range(start_epoch, epochs):
		
		print('epoch ' + str(epoch + 1))
		epoch_loss = 0
		model.train()
		epoch_iter = 0
		loss_last_10 = 0
		for z in tqdm(range(train_size // args['batch_size']), total=train_size // args['batch_size']):
			iters += 1
			epoch_iter+=1
			ind = np.random.choice(train_size, args['batch_size'])
			inp_x, inp_y = train_x[ind], train_y[ind]
			
			#inp_x = model.to_device(inp_x)
			#inp_y = model.to_device(inp_y)
			#print(inp_x.size())
			#print(inp_y.size())
			
			output, l, l1= model(inp_x, inp_y)
			optimizer.zero_grad()
			l.backward()
			#torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			optimizer.step()
			loss_val = l.item()
			epoch_loss+=loss_val
			loss_last_10+=l1.item()
			#writer.add_scalar('/hdetach:train_loss', loss_val, ctr)
			losslist.append((loss_val,ctr))
			ctr += 1


		print('loss:'+str(epoch_loss / epoch_iter))
		print('loss_last_10:'+str(loss_last_10/100))
		t_loss, accuracy = test_model(model, test_x, test_y)
	
		print('best validation accuracy ' + str(best_acc))
		print('Saving best model..')
		state = {
        'net': model,
        'epoch':epoch,
    	'ctr':ctr,
    	'train_x':train_x,
    	'train_y':train_y,
    	'test_x':test_x,
    	'test_y':test_y

    	}
		with open(log_dir + '/best_model.pt', 'wb') as f:
			torch.save(state, f)
		#writer.add_scalar('/hdetach:val_acc', accuracy, epoch)
		acc.append((accuracy,epoch))
		with open(log_dir+'/lossstats.pickle','wb') as f:
			pickle.dump(losslist,f)
		with open(log_dir+'/accstats.pickle','wb') as f:
			pickle.dump(acc,f)



print('==> Building model..')
net = CopyingModel(args).cuda()
start_epoch=0
best_acc=0
ctr=0

if args['loadsaved']==1:
	modelstate=torch.load(log_dir+'/best_model.pt')
	net.load_state_dict(modelstate['net'].state_dict())
	#optimizer.load_state_dict(modelstate['opt'.state_dict()])
	train_model(net, args['epochs'], modelstate)
else:
	train_model(net, args['epochs'])
#writer.close()


#train_x, train_y = create_dataset(10, 200)
#print(train_x.shape)
#print(train_y.shape)