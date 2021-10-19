from RIM import RIMCell
import torch
timesteps = 50
batch_size = 32
num_units = 6
k = 4
num_input_heads = 2
input_size = 32
hidden_size = 64
# Model definition. The definition of each argument is same as above.
rim_model = RIMCell(torch.device('cpu'), input_size, hidden_size, num_units, k, 'LSTM', 
    num_input_heads=num_input_heads )

# creating hidden states and cell states
hs = torch.randn(batch_size, num_units, hidden_size)
cs = torch.randn(batch_size, num_units, hidden_size)
 
# Creating Input
xs = torch.randn(batch_size, timesteps, input_size)
xs = torch.split(xs, 1, 1)

for x in xs:
    hs, cs = rim_model(x, hs, cs)