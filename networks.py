import torch
import torch.nn as nn
import math
from lstm_cell import LSTM
import numpy as np

class blocked_grad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(x, mask)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, mask = ctx.saved_tensors
        return grad_output * mask, mask * 0.0


class GroupLinearLayer(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, din, dout, num_blocks):
        super(GroupLinearLayer, self).__init__()

        self.w = nn.Parameter(0.01 * torch.randn(num_blocks,din,dout))

    def forward(self,x):
        x = x.permute(1,0,2)
        x = torch.bmm(x,self.w)
        return x.permute(1,0,2)


class RIM(nn.Module):
	def __init__(self, 
		device, input_size, hidden_size, num_units, rnn_cell, input_key_size, input_value_size, input_query_size,
		num_input_heads, input_dropout, comm_key_size, comm_value_size, comm_query_size, num_comm_heads, comm_dropout,
		k 
	):#num_units, hidden_size, rnn_cell, num_input_heads, num_comm_heads, query_size, key_size, value_size, k):
		super().__init__()
		self.device = device
		self.hidden_size = hidden_size
		self.num_units =num_units
		self.rnn_cell = rnn_cell
		self.key_size = input_key_size
		self.k = k
		self.num_input_heads = num_input_heads
		self.num_comm_heads = num_comm_heads
		self.input_key_size = input_key_size
		self.input_query_size = input_query_size
		self.input_value_size = input_value_size

		self.comm_key_size = comm_key_size
		self.comm_query_size = comm_query_size
		self.comm_value_size = comm_value_size

		self.key = nn.Linear(input_size, num_input_heads * input_query_size).to(self.device)
		self.value = nn.Linear(input_size, num_input_heads * input_value_size).to(self.device)

		if self.rnn_cell == 'GRU':
			self.rnn = nn.ModuleList([nn.GRUCell(input_value_size, hidden_size) for _ in range(num_units)])
			self.query = GroupLinearLayer(hidden_size,  input_key_size * num_input_heads, self.num_units)
		else:
			self.rnn = nn.ModuleList([nn.LSTMCell(input_value_size, hidden_size) for _ in range(num_units)])
			self.query = GroupLinearLayer(hidden_size,  input_key_size * num_input_heads, self.num_units)
		self.query_ =GroupLinearLayer(hidden_size, comm_query_size * num_comm_heads, self.num_units) 
		self.key_ = GroupLinearLayer(hidden_size, comm_key_size * num_comm_heads, self.num_units)
		self.value_ = GroupLinearLayer(hidden_size, comm_value_size * num_comm_heads, self.num_units)
		self.comm_attention_output = GroupLinearLayer(num_comm_heads * comm_value_size, comm_value_size, self.num_units)
		self.comm_dropout = nn.Dropout(p =input_dropout)
		self.input_dropout = nn.Dropout(p =comm_dropout)


	def transpose_for_scores(self, x, num_attention_heads, attention_head_size):
	    new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
	    x = x.view(*new_x_shape)
	    return x.permute(0, 2, 1, 3)

	def input_attention_mask(self, x, h):
	    #print(x.type())
	    key_layer = self.key(x)
	    value_layer = self.value(x)
	    query_layer = self.query(h)

	    key_layer = self.transpose_for_scores(key_layer,  self.num_input_heads, self.input_key_size)
	    value_layer = torch.mean(self.transpose_for_scores(value_layer,  self.num_input_heads, self.input_value_size), dim = 1)
	    query_layer = self.transpose_for_scores(query_layer, self.num_input_heads, self.input_query_size)

	    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.input_key_size) 
	    attention_scores = torch.mean(attention_scores, dim = 1)
	    mask_ = torch.zeros(x.size(0), self.num_units).to(self.device)
	    
	    not_null_scores = attention_scores[:,:, 0]
	    topk1 = torch.topk(not_null_scores,self.k,  dim = 1)
	    row_index = np.arange(x.size(0))
	    row_index = np.repeat(x.size(0))

	    mask_[row_index, topk1.indices.view(-1)] = 1
	    
	    attention_probs = self.input_dropout(nn.Softmax(dim = -1)(attention_scores))
	    inputs = torch.matmul(attention_probs, value_layer) * mask_.unsqueeze(2)
	    inputs = torch.split(inputs, 1, 1)
	    return inputs, mask_

	def communication_attention(self, h, mask):
	    query_layer = []
	    key_layer = []
	    value_layer = []
	    
	    query_layer = self.query_(h)
	    key_layer = self.key_(h)
	    value_layer = self.value_(h)
	    
	    
	    query_layer = self.transpose_for_scores(query_layer, self.num_comm_heads, self.comm_query_size)
	    key_layer = self.transpose_for_scores(key_layer, self.num_comm_heads, self.comm_key_size)
	    value_layer = self.transpose_for_scores(value_layer, self.num_comm_heads, self.comm_value_size)
	    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
	    attention_scores = attention_scores / math.sqrt(self.comm_key_size)
	    
	    attention_probs = nn.Softmax(dim=-1)(attention_scores)
	    
	    mask = [mask for _ in range(attention_probs.size(1))]
	    mask = torch.stack(mask, dim = 1)
	    attention_probs = attention_probs * mask.unsqueeze(3)
	    attention_probs = self.comm_dropout(attention_probs)
	    context_layer = torch.matmul(attention_probs, value_layer)
	    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
	    new_context_layer_shape = context_layer.size()[:-2] + (self.num_comm_heads * self.comm_value_size,)
	    context_layer = context_layer.view(*new_context_layer_shape)
	    context_layer = self.comm_attention_output(context_layer)
	    context_layer = context_layer + h
	    
	    return context_layer

	def forward(self, x, hs, cs = None):
		size = x.size() # (batch_size, num_elements, feature_size)
		null_input = torch.zeros(size[0], 1, size[2]).float().to(self.device)
		x = torch.cat((x, null_input), dim = 1)
		
		inputs, mask = self.input_attention_mask(x, hs)
		h_old = hs * 1.0
		if cs is not None:
			c_old = cs * 1.0
		hs = list(torch.split(hs, 1,1))
		if cs is not None:
			cs = list(torch.split(cs, 1,1))
		for i in range(self.num_units):
			if cs is None:
				hs[:, i, :] = self.rnn[i](inputs[i], hs[:, i, :])
			else:
				
				hs[i], cs[i] = self.rnn[i](inputs[i].squeeze(), (hs[i].squeeze(), cs[i].squeeze()))
		hs = torch.stack(hs, dim = 1)
		if cs is not None:
			cs = torch.stack(cs, dim = 1)
		
		mask = mask.unsqueeze(2)
		h_new = blocked_grad.apply(hs, mask)


		h_new = self.communication_attention(h_new, mask.squeeze())

		hs = mask * h_new + (1 - mask) * h_old 
		cs = mask * cs + (1 - mask) * c_old

		return hs, cs




class MnistModel(nn.Module):
	def __init__(self, args):
		super().__init__()
		self.args = args
		if args['cuda']:
			self.device = torch.device('cuda')
		else:
			self.device = torch.device('cpu')
		self.rim_model = RIM(self.device, args['input_size'], args['hidden_size'], args['num_units'], args['rnn_cell'], args['key_size_input'], args['value_size_input'] , args['query_size_input'],
			args['num_input_heads'], args['input_dropout'], args['key_size_comm'], args['value_size_comm'], args['query_size_comm'], args['num_input_heads'], args['comm_dropout'], args['k']).to(self.device)

		self.Linear = nn.Linear(args['hidden_size'] * args['num_units'], 10)
		self.Loss = nn.CrossEntropyLoss()

	def to_device(self, x):
		return torch.from_numpy(x).to(self.device)

	def forward(self, x, y = None):
		x = x.float()
		hs = torch.randn(x.size(0), self.args['num_units'], self.args['hidden_size']).to(self.device)
		cs = None
		if self.args['rnn_cell'] == 'LSTM':
			cs = torch.randn(x.size(0), self.args['num_units'], self.args['hidden_size']).to(self.device)
		xs = torch.split(x, 1, 1)
		for x in xs:
			hs, cs = self.rim_model(x, hs, cs)
		preds = self.Linear(hs.contiguous().view(x.size(0), -1))
		if y is not None:
			y = y.long()
			probs = nn.Softmax(dim = -1)(preds)
			entropy = torch.mean(torch.sum(probs*torch.log(probs), dim = 1))
			loss = self.Loss(preds, y) - entropy
			return probs, loss
		return preds


	def grad_norm(self):
	    total_norm = 0
	    for p in self.parameters():
	    	if p.grad is not None:
		        param_norm = p.grad.data.norm(2)
		        total_norm += param_norm.item() ** 2
	    total_norm = total_norm ** (1. / 2)
	    return total_norm
	



class LSTM(nn.Module):
	def __init__(self, args):
		super().__init__()
		self.args = args
		if args['cuda']:
			self.device = torch.device('cuda')
		else:
			self.device = torch.device('cpu')
		self.hidden_size = args['hidden_size']
		self.lstm = nn.LSTMCell(args['input_size'], self.hidden_size)
		self.Linear = nn.Linear(self.hidden_size, 10)
		self.Loss = nn.CrossEntropyLoss()
		#self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.0001)

	def to_device(self, x):
		return torch.from_numpy(x).to(self.device)

	def forward(self, x, y = None):
		x = x.float()
		hs = torch.randn(x.size(0), self.hidden_size).to(self.device)
		cs = torch.randn(x.size(0), self.hidden_size).to(self.device) 
		xs = torch.split(x, 1, 1)
		for x in xs:
			x_ = torch.squeeze(x, dim = 1)
			hs, cs = self.lstm(x_, (hs, cs))
		preds = self.Linear(hs)
		if y is not None:
			y = y.long()
			probs = nn.Softmax(dim = -1)(preds)
			entropy = torch.mean(torch.sum(probs*torch.log(probs), dim = 1))
			loss = self.Loss(preds, y) - entropy
			return probs, loss
		return preds

	
	def grad_norm(self):
	    total_norm = 0
	    for p in self.parameters():
	        param_norm = p.grad.data.norm(2)
	        total_norm += param_norm.item() ** 2
	    total_norm = total_norm ** (1. / 2)
	    return total_norm




class CopyingModel(nn.Module):
	def __init__(self, args):
		super().__init__()
		self.args = args
		if args['cuda']:
			self.device = torch.device('cuda')
		else:
			self.device = torch.device('cpu')
		self.rim_model = RIM(self.device, args['input_size'], args['hidden_size'], args['num_units'], args['rnn_cell'], args['key_size_input'], args['value_size_input'] , args['query_size_input'],
			args['num_input_heads'], args['input_dropout'], args['key_size_comm'], args['value_size_comm'], args['query_size_comm'], args['num_input_heads'], args['comm_dropout'], args['k']).to(self.device)

		self.Linear = nn.Linear(args['hidden_size'] * args['num_units'], 9)
		self.Loss = nn.CrossEntropyLoss()
		
	def to_device(self, x):
		return torch.from_numpy(x).to(self.device)

	def forward(self, x, y = None):
		x = x.float()
		hs = torch.randn(x.size(0), self.args['num_units'], self.args['hidden_size']).to(self.device)
		cs = None
		if self.args['rnn_cell'] == 'LSTM':
			cs = torch.randn(x.size(0), self.args['num_units'], self.args['hidden_size']).to(self.device)

		xs = torch.split(x, 1, 1)
		preds_ = []
		loss = 0
		loss_last_10 = 0
		for i,k in enumerate(xs):
			hs, cs = self.rim_model(k, hs, cs)
			
			preds = self.Linear(hs.contiguous().view(x.size(0), -1))
			preds_.append(preds)
			if y is not None:
				loss+=self.Loss(preds, y[:,i].squeeze().long())
				if i >= len(xs) - 10:
					loss_last_10+=self.Loss(preds, y[:,i].squeeze().long())
		preds_ = torch.stack(preds_, dim = 1)
		if y is not None:
			loss/=len(xs)
			loss_last_10/=10
			return preds_, loss, loss_last_10
		return preds_

