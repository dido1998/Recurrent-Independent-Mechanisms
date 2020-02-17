# Recurrent-Independent-Mechanisms

An implementation of [Recurrent Independent Mechanisms (Goyal et al. 2019)](https://arxiv.org/pdf/1909.10893.pdf) in PyTorch.

## Setup
* For using RIM as a standalone replpacement for LSTMs or GRUs
   * Install PyTorch 1.2.0 from the [official website](https://pytorch.org/).
   * Install [numpy](https://numpy.org/) 1.18.0 using `pip install numpy==1.18.0`
   
Running the **Installation** instructions below will automatically install the above libraries.

* For running the experiments below
   * Install [tqdm](https://github.com/tqdm/tqdm) using `pip install tqdm`
   * For running the RL experiments
      * Install [gym-minigrid](https://github.com/maximecb/gym-minigrid) using `pip install gym-minigrid`
      * Install [torch_ac](https://github.com/lcswillems/torch-ac) using `pip install torch_ac>=1.1.0`
      * Install [tensorboardX](https://github.com/lanpa/tensorboardX) using `pip install tensorboardX>=1.6`
## Installation
```
git clone https://github.com/dido1998/Recurrent-Independent-Mechanisms.git
cd Recurrent\ Independent\ Mechanisms
pip install -e .
```
This will allow you to use RIMs from anywhere in your system.
This code was tested with python3.6

## Documentation
```
Class RIM.RIMCell(device,
input_size,
hidden_size,
num_units,
rnn_cell,
input_key_size = 64,
input_value_size = 400,
input_query_size = 64,
num_input_heads = 1,
input_dropout = 0.1,
comm_key_size = 32,
comm_value_size = 100,
comm_query_size = 32,
num_comm_heads = 4,
comm_dropout = 0.1,
k = 4)
```
For description of the RIMCell please check the paper.

### Parameters
| Parameter | Description |
| --------- | ----------- |
| **device** | `torch.device('cuda')` or `torch.device('cpu')`. |
| **input_size** | The number of expected input features. |
| **hidden_size** | The number of hidden features. |
| **num_units** | Number of total RIM units. |
| **rnn_cell** | `'LSTM'` or `'GRU'` |
| **input_key_size** | Number of features in the input key. |
| **input_value_size** | Number of features in the input value. |
| **input_query_size** | Number of features in the input query. |
| **num_input_heads** | Number of input attention heads. |
| **input_dropout** | Dropout applied to the input attention probabilities. |
| **comm_key_size** | Number of features in the communication key. |
| **comm_value_size** | Number of features in the communication value. |
| **comm_query_size** | Number of features in the communication query. |
| **num_comm_heads** | Number of communication attention heads. |
| **comm_dropout** | Dropout applied to the communication attention probabilities. |
| **k** | Number of active RIMs at every time-step. |

### Inputs
| Input | Description |
| ----- | ----------- |
| **x** | Input of shape (*batch_size*, 1, *input_size*). |
| **hs** | Hidden state for the current time-step of shape (*batch_size, num_units, hidden_size*). |
| **cs** | This is given if *rnn_cell = 'LSTM'* else it is `None`. Cell state for the current time-step of shape (*batch_size, num_units, hidden_size*). |

### Outputs 
| Output | Description |
| ------ | ----------- |
| **hs** | The new hidden state of shape (*batch_size, num_units, hidden_size*). |
| **cs** | This is only returned if *rnn_cell = 'LSTM'*. The new cell state of shape (*batch_size, num_units, hidden_size*). |

### Example
```
from RIM import RIMCell
timesteps = 50
batch_size = 32
num_units = 6
input_size = 32
hidden_size = 64
# Model definition. The definition of each argument is same as above.
rim_model = RIM(torch.device('cuda'), input_size, hidden_size, 'LSTM')

# creating hidden states and cell states
hs = torch.randn(batch_size, num_units, hidden_size)
cs = torch.randn(batch_size, num_units, hidden_size)
 
# Creating Input
xs = torch.randn(batch_size, timesteps, input_size)
xs = torch.split(xs, 1, 1)

for x in xs:
    hs, cs = rim_model(x, hs, cs)
```



## Gym Minigrid
The minigrid environment is available [here](https://github.com/maximecb/gym-minigrid). Results for the gym minigrd environment solved using **PPO**. 

#### For all the tables, the model is trained on the star-marked column and only evaluated on the other columns.

**I report the mean return per episode in each case**

| Model | MiniGrid-Empty-5X5-V0 **\*** | MiniGrid-Empty-16X16-V0 |
| ----- | --------------------- | ----------------------- |
| RIM (Kt = 4, Ka = 3) | **0.91** | **0.92** |
| LSTM | 0.80 | 0.84 |

| Model | MiniGrid-MultiRoom-N2-S4-V0 (2 rooms) **\*** | MiniGrid-MultiRoom-N2-S5-V0 (4 rooms) | MiniGrid-MultiRoom-N6-V0 (6 rooms) |
| ----- | --------------------------- | --------------------------- | ------------------------ |
| RIM (Kt = 4, Ka = 3) | **0.81** | **0.66** | **0.05** |
| LSTM | 0.82 | 0.04 | 0.00 |

## Sequential MNIST Task
Results for MNIST task: 

#### The model has been trained on MNIST datset with indicidual image size 14*14


|      | Kt | Ka | h   | 16*16  | 19*19 | 24*24 |
|------|----|----|-----|-------|-------|-------|
|      | 6  | 6  | 600 | 80.31 | 56.19 | 37.45 |
| RIM  | 6  | 5  | 600 | **88.67** | 59.32 | 28.85 |
|      | 6  | 4  | 600 | 87.89 | **69.75** | **46.23** |
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




