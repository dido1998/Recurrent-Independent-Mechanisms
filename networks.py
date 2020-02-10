import torch
import torch.nn as nn
import math
from lstm_cell import LSTM
class RIM(nn.Module):
	def __init__(self, device, args):#num_units, hidden_size, rnn_cell, num_input_heads, num_comm_heads, query_size, key_size, value_size, k):
		super().__init__()
		self.device = device
		self.hidden_size = args['hidden_size']
		self.num_units = args['num_units']
		self.rnn_cell = args['rnn_cell']
		self.key_size = args['key_size_input']
		self.args = args
		self.k = args['k']
		self.key = nn.Linear(args['input_size'], args['num_input_heads'] * args['query_size_input']).to(self.device)
		self.value = nn.Linear(args['input_size'], args['num_input_heads'] * args['value_size_input']).to(self.device)

		if self.rnn_cell == 'GRU':
			self.rnn = nn.ModuleList([nn.GRUCell(args['value_size_input'], args['hidden_size']) for _ in range(args['num_units'])])
			self.query = nn.ModuleList([nn.Linear(args['hidden_size'], args['key_size_input'] * args['num_input_heads']) for _ in range(args['num_units'])])
		else:
			self.rnn = nn.ModuleList([nn.LSTMCell(args['value_size_input'], args['hidden_size']) for _ in range(args['num_units'])])
			self.query = nn.ModuleList([nn.Linear(args['hidden_size'], args['key_size_input'] * args['num_input_heads']) for _ in range(args['num_units'])])
		self.query_ = nn.ModuleList([nn.Linear(args['hidden_size'], args['query_size_comm'] * args['num_comm_heads']) for _ in range(args['num_units'])])
		self.key_ = nn.ModuleList([nn.Linear(args['hidden_size'], args['key_size_comm'] * args['num_comm_heads']) for _ in range(args['num_units'])])
		self.value_ = nn.ModuleList([nn.Linear(args['hidden_size'], args['value_size_comm'] * args['num_comm_heads']) for _ in range(args['num_units'])])
		self.comm_attention_output = nn.ModuleList([nn.Linear(args['num_comm_heads'] * args['value_size_comm'], args['value_size_comm']) for _ in range(args['num_units'])])
		self.comm_dropout = nn.Dropout(p =0.1)
		self.input_dropout = nn.Dropout(p =0.1)


	def transpose_for_scores(self, x, num_attention_heads, attention_head_size):
	    new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
	    x = x.view(*new_x_shape)
	    return x.permute(0, 2, 1, 3)

	def input_attention_mask(self, x, hs, row_index, ind):
	    #print(x.type())
	    key_layer = self.key(x)
	    value_layer = self.value(x)

	    query_layer = [self.query[i](h) for i, h in enumerate(hs)]
	    query_layer = torch.stack(query_layer, dim = 1)

	    key_layer = self.transpose_for_scores(key_layer,  self.args['num_input_heads'], self.args['key_size_input'])
	    value_layer = torch.mean(self.transpose_for_scores(value_layer,  self.args['num_input_heads'], self.args['value_size_input']), dim = 1)
	    query_layer = self.transpose_for_scores(query_layer, self.args['num_input_heads'], self.args['query_size_input'])

	    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / self.args['key_size_input']
	    attention_scores = torch.mean(attention_scores, dim = 1)
	    #mask = torch.zeros(attention_scores.size()).to(self.device) 
	    mask_ = torch.zeros(x.size(0), self.args['num_units']).to(self.device)
	    
	    not_null_scores = attention_scores[:,:, 0]
	    #null_scores = -attention_scores[:,:,0]
	    topk1 = torch.topk(not_null_scores,self.k,  dim = 1)
	    #topk2 = torch.topk(null_scores, 6 - self.k, dim = 1)
	    mask_[row_index, topk1.indices.view(-1)] = 1
	    #mask[row_index, topk1.indices.view(-1), 0] = 1
	    #mask[ind, topk2.indices.view(-1), 1] = 1

	    #attention_scores = attention_scores 
	    attention_probs = self.input_dropout(nn.Softmax(dim = -1)(attention_scores))
	    #print(value_layer)
	    inputs = torch.matmul(attention_probs, value_layer) * mask_.unsqueeze(2)
	    inputs = torch.split(inputs, 1, 1)
	    return inputs, mask_

	def communication_attention(self, hs, mask):
	    query_layer = []
	    key_layer = []
	    value_layer = []
	    for i, h in enumerate(hs):
	    	query_layer.append(self.query_[i](h) * mask[:, i].view(-1, 1))
	    	key_layer.append(self.key_[i](h))
	    	value_layer.append(self.value_[i](h))
	    query_layer = torch.stack(query_layer, dim = 1)
	    key_layer = torch.stack(key_layer, dim = 1)
	    value_layer = torch.stack(value_layer, dim = 1)
	    
	    query_layer = self.transpose_for_scores(query_layer, self.args['num_comm_heads'], self.args['query_size_comm'])
	    key_layer = self.transpose_for_scores(key_layer, self.args['num_comm_heads'], self.args['key_size_comm'])
	    value_layer = self.transpose_for_scores(value_layer, self.args['num_comm_heads'], self.args['value_size_comm'])
	    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
	    attention_scores = attention_scores / math.sqrt(self.args['key_size_comm'])
	    #attention_scores = torch.mean(attention_scores, dim = 1)
	    
	    attention_probs = nn.Softmax(dim=-1)(attention_scores)
	    #print(attention_probs.size())
	    #print(mask.unsqueeze(2).size())
	    #attention_probs = attention_probs * mask_

	    #attention_probs = attention_probs * mask.unsqueeze(2)
	    mask = [mask for _ in range(attention_probs.size(1))]
	    mask = torch.stack(mask, dim = 1)
	    
	    attention_probs = attention_probs * mask.unsqueeze(3)
	    attention_probs = self.comm_dropout(attention_probs)
	    context_layer = torch.matmul(attention_probs, value_layer)
	    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
	    new_context_layer_shape = context_layer.size()[:-2] + (self.args['num_comm_heads'] * self.args['value_size_comm'],)
	    context_layer = context_layer.view(*new_context_layer_shape)
	    
	    context_layer = torch.split(context_layer, 1, 1)
	    context_layer = [self.comm_attention_output[i](c) for i, c in enumerate(context_layer)]

	    context_layer = [torch.squeeze(c) + hs[i] for i, c in enumerate(context_layer)]

	    return context_layer

	def forward(self, row_index, ind, x, hs, cs = None):
		size = x.size() # (batch_size, num_elements, feature_size)
		null_input = torch.zeros(size[0], 1, size[2]).float().to(self.device)
		x = torch.cat((x, null_input), dim = 1)
		#x = torch.squeeze(x, dim = 1)
		#mask = torch.ones(x.size(0), self.num_units).to(self.device)
		inputs, mask = self.input_attention_mask(x, hs, row_index, ind)
		for i in range(self.num_units):
			if cs is None:
				hs[i] = self.rnn[i](inputs[i], hs[i])
			else:

				hs[i], cs[i] = self.rnn[i](inputs[i].squeeze(), (hs[i], cs[i]))
			mask_bool = (1 -mask[:, i]).view(-1).bool()
			hs[i][mask_bool, :] = hs[i][mask_bool, :].detach()

		hs = self.communication_attention(hs, mask)
		return hs, cs




class MnistModel(nn.Module):
	def __init__(self, args):
		super().__init__()
		self.args = args
		if args['cuda']:
			self.device = torch.device('cuda')
		else:
			self.device = torch.device('cpu')
		self.rim_model = RIM(self.device, args).to(self.device)

		self.Linear = nn.Linear(args['hidden_size'] * args['num_units'], 10)
		self.Loss = nn.CrossEntropyLoss()

	def to_device(self, x):
		return torch.from_numpy(x).to(self.device)

	def forward(self, row_index, ind, x, y = None):
		x = x.float()
		hs = [torch.randn(x.size(0), self.args['hidden_size']).to(self.device) for _ in range(self.args['num_units'])]
		cs = None
		if self.args['rnn_cell'] == 'LSTM':
			cs = [torch.randn(x.size(0), self.args['hidden_size']).to(self.device) for _ in range(self.args['num_units'])]
		xs = torch.split(x, 1, 1)
		#print(xs[0].size())
		#xs = [torch.squeeze(k) for k in xs]
		for x in xs:
			hs, cs = self.rim_model(row_index, ind, x, hs, cs)
		h = torch.cat(hs, dim = 1)
		preds = self.Linear(h)
		if y is not None:
			y = y.long()
			probs = nn.Softmax(dim = -1)(preds)
			entropy = torch.mean(torch.sum(probs*torch.log(probs), dim = 1))
			loss = self.Loss(preds, y) - entropy
			return probs, loss
		return preds


	def update(self, loss):
		self.zero_grad()
		loss.backward()
		self.optimizer.step()
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

	def update(self, loss):
		self.zero_grad()
		loss.backward()
		self.optimizer.step()
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
		self.rim_model = RIM(self.device, args).to(self.device)

		self.Linear = nn.Linear(args['hidden_size'] * args['num_units'], 9)
		self.Loss = nn.CrossEntropyLoss()
		#self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.005)

	def to_device(self, x):
		return torch.from_numpy(x).to(self.device)

	def forward(self, row_index, ind, x, y = None):
		x = x.float()
		hs = [torch.randn(x.size(0), self.args['hidden_size']).to(self.device) for _ in range(self.args['num_units'])]
		cs = None
		if self.args['rnn_cell'] == 'LSTM':
			cs = [torch.randn(x.size(0), self.args['hidden_size']).to(self.device) for _ in range(self.args['num_units'])]

		xs = torch.split(x, 1, 1)
		#print(xs[0].size())
		#xs = [torch.squeeze(k) for k in xs]
		preds = []
		for k in xs:
			hs, cs = self.rim_model(row_index, ind, k, hs, cs)
			h = torch.cat(hs, dim = 1)
			preds.append(self.Linear(h))
		preds = torch.stack(preds, dim = 1)
		if y is not None:
			preds_ = torch.transpose(preds, 1, 2)
			#print(preds.size())

			y = y.long()
			loss = self.Loss(preds_, torch.squeeze(y))
			return preds, loss
		return preds

