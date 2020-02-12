# Recurrent-Independent-Mechanisms

An implementation of [Recurrent Independent Mechanisms (Goyal et al. 2019)](https://arxiv.org/pdf/1909.10893.pdf) in PyTorch.

## Requirements
```
Pytorch 1.2.0
numpy 1.18.1
```
This code was tested with python3.6

## Sequential MNIST Task
Results for MNIST task:

|      | Kt | Ka | h   | 16*16 | 19*19 | 24*24 |
|------|----|----|-----|-------|-------|-------|
|      | 6  | 6  | 600 | 80.31 | 56.19 | 37.45 |
| RIM  | 6  | 5  | 600 | 88.67 | 59.32 | 28.85 |
|      | 6  | 4  | 600 | 87.89 | 69.75 | 46.23 |
|      |    |    |     |       |       |       |
| LSTM | -  | -  | 600 | 80.43 | 39.74 | 20.48 |

This task can be run using - 
```
python3.6 main.py --args
```
`args` has the following options-
| Arguments | Description |
| ----------|-------------|
| cuda | To use GPU or not|
| epochs | number of epochs to train |
| batch_size | Batch size for training |
| hidden_Size| Per RIM hidden size |
| input_size | Input feature size|
| model | `LSTM` or `RIM` |
| train | set to `True` for training and `False` for testing. |
| rnn_cell | `LSTM` or `GRU` |
| key_size_input | Input key size |
| value_size_input | Input value size |
| query size input | Input query size |
| num_input_heads | Number of heads in input attention |
| input_dropout | Input dropout value. |
| key_size_comm | Communication key size. |
| value_size_comm | Communication value size. |
| query_size_comm | Communication query size. |
| num_comm_heads | Number of heads in communication attention  |
| comm_dropout | Communication dropout value|
| num_units| Number of RIMs (Kt) |
| k | Number of active RIMs (ka) |
| size | Image size for training. |
| loadsaved | load saved model for training from log_dir. |
| log_dir | Directory path to save meta data. |

## Using RIM as a drop-in replacement for LSTM Or GRU

```
from networks import RIM
timesteps = 50
# The definition of each argument is same as above
rim_model = RIM(torch.device(<device>), input_size, hidden_size, rnn_cell, key_size_input,
value_size_input, query_size_input, num_input_heads, input_dropout, key_size_comm, value_size_comm,
query_size_comm, num_comm_heads, comm_dropout, k)

hs = torch.randn(batch_size, num_units, hidden_size)
cs = None
if rnn_cell == 'LSTM':
  cs = torch.randn(batch_size, num_units, hidden_size)
xs = torch.randn(batch_size, timesteps, input_size)
xs = torch.split(xs, 1, 1)
for x in xs:
    hs, cs = rim_model(x, hs, cs)
```



