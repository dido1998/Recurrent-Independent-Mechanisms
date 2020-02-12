# Recurrent-Independent-Mechanisms

An implementation of Recurrent Independent Mechanisms (Goyal et al. 2019) in PyTorch.

## Requirements
```
Pytorch 1.2.0
numpy 1.18.1
```
This code was tested with python3.6

## Sequential MNIST Task
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

Results for MNIST task:

|      | Kt | Ka | h   | 16*16 | 19*19 | 24*24 |
|------|----|----|-----|-------|-------|-------|
| RIM  | 6  | 5  | 600 | 88.67 | 59.32 | 28.85 |
|      | 6  | 4  | 600 | 87.89 | 69.75 | 46.23 |
|      |    |    |     |       |       |       |
| LSTM | -  | -  | 600 | 80.43 | 39.74 | 20.48 |

