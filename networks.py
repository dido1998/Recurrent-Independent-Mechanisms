import torch
import torch.nn as nn
import math
from RIM import RIMCell
import numpy as np

class MnistModel(nn.Module):
	def __init__(self, args):
		super().__init__()
		self.args = args
		if args['cuda']:
			self.device = torch.device('cuda')
		else:
			self.device = torch.device('cpu')
		self.rim_model = RIMCell(self.device, args['input_size'], args['hidden_size'], args['num_units'], args['k'], args['rnn_cell'], args['key_size_input'], args['value_size_input'] , args['query_size_input'],
			args['num_input_heads'], args['input_dropout'], args['key_size_comm'], args['value_size_comm'], args['query_size_comm'], args['num_input_heads'], args['comm_dropout']).to(self.device)

		self.Linear = nn.Linear(args['hidden_size'] * args['num_units'], 10)
		self.Loss = nn.CrossEntropyLoss()

	def to_device(self, x):
		return torch.from_numpy(x).to(self.device)

	def forward(self, x, y = None):
		x = x.float()
		
		# initialize hidden states
		hs = torch.randn(x.size(0), self.args['num_units'], self.args['hidden_size']).to(self.device)
		cs = None
		if self.args['rnn_cell'] == 'LSTM':
			cs = torch.randn(x.size(0), self.args['num_units'], self.args['hidden_size']).to(self.device)

		xs = torch.split(x, 1, 1)

		# pass through RIMCell for all timesteps
		for x in xs:
			hs, cs = self.rim_model(x, hs, cs)
		preds = self.Linear(hs.contiguous().view(x.size(0), -1))

		if y is not None:
			# Compute Loss
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
		self.rim_model = RIM(self.device, args['input_size'], args['hidden_size'], args['num_units'],args['k'], args['rnn_cell'], args['key_size_input'], args['value_size_input'] , args['query_size_input'],
			args['num_input_heads'], args['input_dropout'], args['key_size_comm'], args['value_size_comm'], args['query_size_comm'], args['num_input_heads'], args['comm_dropout']).to(self.device)

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

