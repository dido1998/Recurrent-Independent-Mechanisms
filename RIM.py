import torch
import torch.nn as nn
import math

import numpy as np
import torch.multiprocessing as mp


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

    def __init__(self, din, dout, num_blocks):
        super(GroupLinearLayer, self).__init__()

        self.w	 = nn.Parameter(0.01 * torch.randn(num_blocks,din,dout))

    def forward(self,x):
        x = x.permute(1,0,2)
        
        x = torch.bmm(x,self.w)
        return x.permute(1,0,2)


class GroupLSTMCell(nn.Module):
	"""
	GroupLSTMCell can compute the operation of N LSTM Cells at once.
	"""
	def __init__(self, inp_size, hidden_size, num_lstms):
		super().__init__()
		self.inp_size = inp_size
		self.hidden_size = hidden_size
		
		self.i2h = GroupLinearLayer(inp_size, 4 * hidden_size, num_lstms)
		self.h2h = GroupLinearLayer(hidden_size, 4 * hidden_size, num_lstms)
		self.reset_parameters()


	def reset_parameters(self):
		stdv = 1.0 / math.sqrt(self.hidden_size)
		for weight in self.parameters():
			weight.data.uniform_(-stdv, stdv)

	def forward(self, x, hid_state):
		"""
		input: x (batch_size, num_lstms, input_size)
			   hid_state (tuple of length 2 with each element of size (batch_size, num_lstms, hidden_state))
		output: h (batch_size, num_lstms, hidden_state)
				c ((batch_size, num_lstms, hidden_state))
		"""
		h, c = hid_state
		preact = self.i2h(x) + self.h2h(h)

		gates = preact[:, :,  :3 * self.hidden_size].sigmoid()
		g_t = preact[:, :,  3 * self.hidden_size:].tanh()
		i_t = gates[:, :,  :self.hidden_size]
		f_t = gates[:, :, self.hidden_size:2 * self.hidden_size]
		o_t = gates[:, :, -self.hidden_size:]

		c_t = torch.mul(c, f_t) + torch.mul(i_t, g_t) 
		h_t = torch.mul(o_t, c_t.tanh())

		return h_t, c_t


class GroupGRUCell(nn.Module):
    """
    GroupGRUCell can compute the operation of N GRU Cells at once.
    """
    def __init__(self, input_size, hidden_size, num_grus):
        super(GroupGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.x2h = GroupLinearLayer(input_size, 3 * hidden_size, num_grus)
        self.h2h = GroupLinearLayer(hidden_size, 3 * hidden_size, num_grus)
        self.reset_parameters()



    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data = torch.ones(w.data.size())#.uniform_(-std, std)
    
    def forward(self, x, hidden):
        """
		input: x (batch_size, num_grus, input_size)
			   hidden (batch_size, num_grus, hidden_size)
		output: hidden (batch_size, num_grus, hidden_size)
        """
        gate_x = self.x2h(x) 
        gate_h = self.h2h(hidden)
        
        i_r, i_i, i_n = gate_x.chunk(3, 2)
        h_r, h_i, h_n = gate_h.chunk(3, 2)
        
        
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))
        
        hy = newgate + inputgate * (hidden - newgate)
        
        return hy



class RIMCell(nn.Module):
	def __init__(self, 
		device, input_size, hidden_size, num_units, k, rnn_cell, input_key_size = 64, input_value_size = 400, input_query_size = 64,
		num_input_heads = 1, input_dropout = 0.1, comm_key_size = 32, comm_value_size = 100, comm_query_size = 32, num_comm_heads = 4, comm_dropout = 0.1
	):
		super().__init__()
		if comm_value_size != hidden_size:
			#print('INFO: Changing communication value size to match hidden_size')
			comm_value_size = hidden_size
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
			self.rnn = GroupGRUCell(input_value_size, hidden_size, num_units)
			self.query = GroupLinearLayer(hidden_size,  input_key_size * num_input_heads, self.num_units)
		else:
			self.rnn = GroupLSTMCell(input_value_size, hidden_size, num_units)
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
	    """
	    Input : x (batch_size, 2, input_size) [The null input is appended along the first dimension]
	    		h (batch_size, num_units, hidden_size)
	    Output: inputs (list of size num_units with each element of shape (batch_size, input_value_size))
	    		mask_ binary array of shape (batch_size, num_units) where 1 indicates active and 0 indicates inactive
		"""
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
	    row_index = np.repeat(row_index, self.k)

	    mask_[row_index, topk1.indices.view(-1)] = 1
	    
	    attention_probs = self.input_dropout(nn.Softmax(dim = -1)(attention_scores))
	    inputs = torch.matmul(attention_probs, value_layer) * mask_.unsqueeze(2)

	    return inputs, mask_

	def communication_attention(self, h, mask):
	    """
	    Input : h (batch_size, num_units, hidden_size)
	    	    mask obtained from the input_attention_mask() function
	    Output: context_layer (batch_size, num_units, hidden_size). New hidden states after communication
	    """
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
		"""
		Input : x (batch_size, 1 , input_size)
				hs (batch_size, num_units, hidden_size)
				cs (batch_size, num_units, hidden_size)
		Output: new hs, cs for LSTM
				new hs for GRU
		"""
		size = x.size()
		null_input = torch.zeros(size[0], 1, size[2]).float().to(self.device)
		x = torch.cat((x, null_input), dim = 1)

		# Compute input attention
		inputs, mask = self.input_attention_mask(x, hs)
		h_old = hs * 1.0
		if cs is not None:
			c_old = cs * 1.0
		

		# Compute RNN(LSTM or GRU) output
		
		if cs is not None:
			hs, cs = self.rnn(inputs, (hs, cs))
		else:
			hs = self.rnn(inputs, hs)

		# Block gradient through inactive units
		mask = mask.unsqueeze(2)
		h_new = blocked_grad.apply(hs, mask)

		# Compute communication attention
		h_new = self.communication_attention(h_new, mask.squeeze(2))

		hs = mask * h_new + (1 - mask) * h_old
		if cs is not None:
			cs = mask * cs + (1 - mask) * c_old
			return hs, cs

		return hs, None


class RIM(nn.Module):
	def __init__(self, device, input_size, hidden_size, num_units, k, rnn_cell, n_layers, bidirectional, **kwargs):
		super().__init__()
		if device == 'cuda':
			self.device = torch.device('cuda')
		else:
			self.device = torch.device('cpu')
		self.n_layers = n_layers
		self.num_directions = 2 if bidirectional else 1
		self.rnn_cell = rnn_cell
		self.num_units = num_units
		self.hidden_size = hidden_size
		if self.num_directions == 2:
			self.rimcell = nn.ModuleList([RIMCell(self.device, input_size, hidden_size, num_units, k, rnn_cell, **kwargs).to(self.device) if i < 2 else 
				RIMCell(self.device, 2 * hidden_size * self.num_units, hidden_size, num_units, k, rnn_cell, **kwargs).to(self.device) for i in range(self.n_layers * self.num_directions)])
		else:
			self.rimcell = nn.ModuleList([RIMCell(self.device, input_size, hidden_size, num_units, k, rnn_cell, **kwargs).to(self.device) if i == 0 else
			RIMCell(self.device, hidden_size * self.num_units, hidden_size, num_units, k, rnn_cell, **kwargs).to(self.device) for i in range(self.n_layers)])

	def layer(self, rim_layer, x, h, c = None, direction = 0):
		batch_size = x.size(1)
		xs = list(torch.split(x, 1, dim = 0))
		if direction == 1: xs.reverse()
		hs = h.squeeze(0).view(batch_size, self.num_units, -1)
		cs = None
		if c is not None:
			cs = c.squeeze(0).view(batch_size, self.num_units, -1)
		outputs = []
		for x in xs:
			x = x.squeeze(0)
			hs, cs = rim_layer(x.unsqueeze(1), hs, cs)
			outputs.append(hs.view(1, batch_size, -1))
		if direction == 1: outputs.reverse()
		outputs = torch.cat(outputs, dim = 0)
		if c is not None:
			return outputs, hs.view(batch_size, -1), cs.view(batch_size, -1)
		else:
			return outputs, hs.view(batch_size, -1)

	def forward(self, x, h = None, c = None):
		"""
		Input: x (seq_len, batch_size, feature_size
			   h (num_layers * num_directions, batch_size, hidden_size * num_units)
			   c (num_layers * num_directions, batch_size, hidden_size * num_units)
		Output: outputs (batch_size, seqlen, hidden_size * num_units * num-directions)
				h(and c) (num_layer * num_directions, batch_size, hidden_size* num_units)
		"""

		hs = torch.split(h, 1, 0) if h is not None else torch.split(torch.randn(self.n_layers * self.num_directions, x.size(1), self.hidden_size * self.num_units).to(self.device), 1, 0)
		hs = list(hs)
		cs = None
		if self.rnn_cell == 'LSTM':
			cs = torch.split(c, 1, 0) if c is not None else torch.split(torch.randn(self.n_layers * self.num_directions, x.size(1), self.hidden_size * self.num_units).to(self.device), 1, 0)
			cs = list(cs)
		for n in range(self.n_layers):
			idx = n * self.num_directions
			if cs is not None:
				x_fw, hs[idx], cs[idx] = self.layer(self.rimcell[idx], x, hs[idx], cs[idx])
			else:
				x_fw, hs[idx] = self.layer(self.rimcell[idx], x, hs[idx], c = None)
			if self.num_directions == 2:
				idx = n * self.num_directions + 1
				if cs is not None:
					x_bw, hs[idx], cs[idx] = self.layer(self.rimcell[idx], x, hs[idx], cs[idx], direction = 1)
				else:
					x_bw, hs[idx] = self.layer(self.rimcell[idx], x, hs[idx], c = None, direction = 1)

				x = torch.cat((x_fw, x_bw), dim = 2)
			else:
				x = x_fw
		hs = torch.stack(hs, dim = 0)
		if cs is not None:
			cs = torch.stack(cs, dim = 0)
			return x, hs, cs
		return x, hs