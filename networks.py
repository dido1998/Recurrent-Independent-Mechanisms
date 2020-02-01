import torch
import torch.nn as nn
import math

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
		self.key = nn.Linear(args['input_size'], args['num_input_heads'] * args['query_size_input'])
		self.value = nn.Linear(args['input_size'], args['num_input_heads'] * args['value_size_input'])

		if self.rnn_cell == 'GRU':
			self.rnn = [nn.GRUCell(args['value_size_input'], args['hidden_size']).to(self.device) for _ in range(args['num_units'])]
			self.query = [nn.Linear(args['hidden_size'], args['key_size_input'] * args['num_input_heads']).to(device) for _ in range(args['num_units'])]
		else:
			self.rnn = [nn.LSTMCell(args['value_size_input'], args['hidden_size']).to(self.device) for _ in range(args['num_units'])]
			self.query = [nn.Linear(args['hidden_size'], args['key_size_input'] * args['num_input_heads']).to(device) for _ in range(args['num_units'])]
		self.query_ = [nn.Linear(args['hidden_size'], args['query_size_comm'] * args['num_comm_heads']).to(self.device) for _ in range(args['num_units'])]
		self.key_ = [nn.Linear(args['hidden_size'], args['key_size_comm'] * args['num_comm_heads']).to(self.device) for _ in range(args['num_units'])]
		self.value_ = [nn.Linear(args['hidden_size'], args['value_size_comm'] * args['num_comm_heads']).to(self.device) for _ in range(args['num_units'])]


	def transpose_for_scores(self, x, num_attention_heads, attention_head_size):
	    new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
	    x = x.view(*new_x_shape)
	    return x.permute(0, 2, 1, 3)

	def input_attention_mask(self, x, hs):
	    query_layer = [self.query[i](h.unsqueeze(1)) for i,h in enumerate(hs)]
	    #print(x.type())
	    key_layer = self.key(x)
	    value_layer = self.value(x)
	    query_layer = [self.transpose_for_scores(q, self.args['num_input_heads'], self.args['query_size_input']) for q in query_layer]
	    key_layer = self.transpose_for_scores(key_layer,  self.args['num_input_heads'], self.args['key_size_input'])
	    value_layer = torch.mean(self.transpose_for_scores(value_layer,  self.args['num_input_heads'], self.args['value_size_input']), dim = 1)
	    attention_scores = [torch.matmul(q, key_layer.transpose(-1, -2)) for q in query_layer]
	    attention_scores = [a / math.sqrt(self.key_size) for a in attention_scores]
	    attention_scores = [-torch.squeeze(torch.mean(a, dim = 1))[:, -1] for a in attention_scores]
	    null_scores = torch.zeros(self.num_units, x.size(0)).to(self.device)
	    for i, a in enumerate(attention_scores):
	    	null_scores[i, :] = a
	    null_scores = torch.transpose(null_scores, 0, 1)
	    topk = torch.topk(null_scores, self.k, dim = 1)
	    mask = torch.zeros(x.size(0), self.num_units).to(self.device)
	    topk = topk.indices.view(-1)
	    row_index = []
	    for i in range(x.size(0)):
	    	row_index.extend([i] * self.k)
	    row_index = torch.tensor(row_index).to(self.device)
	    mask[row_index, topk] = 1.0
	    value_layer = torch.split(value_layer, 1, 1)
	    value_layer = [torch.squeeze(v) for v in value_layer]
	    inputs = [mask[:, i].view(-1, 1) * value_layer[0] for i in range(self.num_units)]

	    return inputs, mask

	def communication_attention(self, hs, mask):
	    query_layer = [self.query_[i](h) * mask[:, i].view(-1, 1) for i,h in enumerate(hs)]
	    key_layer = [self.key_[i](h) for i, h in enumerate(hs)]
	    value_layer = [self.value_[i](h) for i,h in enumerate(hs)]
	    query_layer = torch.stack(query_layer, dim = 1)
	    key_layer = torch.stack(key_layer, dim = 1)
	    value_layer = torch.stack(value_layer, dim = 1)
	    query_layer = self.transpose_for_scores(query_layer, self.args['num_comm_heads'], self.args['query_size_comm'])
	    key_layer = self.transpose_for_scores(key_layer, self.args['num_comm_heads'], self.args['key_size_comm'])
	    value_layer = torch.mean(self.transpose_for_scores(value_layer, self.args['num_comm_heads'], self.args['value_size_comm']), dim = 1)
	    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
	    attention_scores = attention_scores / math.sqrt(self.args['key_size_comm'])
	    attention_scores = torch.squeeze(torch.mean(attention_scores, dim = 1))

	    attention_probs = nn.Softmax(dim=-1)(attention_scores)
	    #print(attention_probs.size())
	    #print(mask.unsqueeze(2).size())
	    attention_probs = attention_probs * mask.unsqueeze(2)


	    context_layer = torch.matmul(attention_probs, value_layer)
	    context_layer = torch.split(context_layer, 1, 1)
	    context_layer = [torch.squeeze(c) + hs[i] for i, c in enumerate(context_layer)]
	    return context_layer

	def forward(self, x, hs, cs = None):
		size = x.size() # (batch_size, num_elements, feature_size)
		null_input = torch.zeros(size[0], 1, size[2]).float().to(self.device)
		x = torch.cat((x, null_input), dim = 1)
		inputs, mask = self.input_attention_mask(x, hs)
		for i in range(self.num_units):
			if cs is None:
				hs[i] = self.rnn[i](inputs[i], hs[i])
			else:
				hs[i], cs[i] = self.rnn[i](inputs[i], (hs[i], cs[i]))
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
		self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.0001)

	def to_device(self, x):
		return torch.from_numpy(x).to(self.device)

	def forward(self, x, y = None):
		x = x.float()
		hs = [torch.randn(x.size(0), self.args['hidden_size']).to(self.device) for _ in range(self.args['num_units'])]
		cs = None
		if self.args['rnn_cell'] == 'LSTM':
			cs = [torch.randn(x.size(0), self.args['hidden_size']).to(self.device) for _ in range(self.args['num_units'])]
		xs = torch.split(x, 1, 1)
		#print(xs[0].size())
		#xs = [torch.squeeze(k) for k in xs]
		for k in xs:
			hs, cs = self.rim_model(k, hs, cs)
		h = torch.cat(hs, dim = 1)
		probs = nn.Softmax(dim = -1)(self.Linear(h))
		if y is not None:
			y = y.long()
			loss = self.Loss(probs, y)
			return probs, loss
		return probs


	def update(self, loss):
		self.zero_grad()
		loss.backward()
		self.optimizer.step()
	







