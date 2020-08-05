# Recurrent-Independent-Mechanisms

An implementation of [Recurrent Independent Mechanisms (Goyal et al. 2019)](https://arxiv.org/pdf/1909.10893.pdf) in PyTorch.

## Paper Summary
This paper aims to build models that can generalize to different environments with specific factors of variation from the environment that it was trained on. To achieve this the authors build recurrent networks that are modular in nature and each module is independent of the other modules and only interact sparsely through attention. In this way each module can learn different aspects of the environment and is only responsible for ensuring similar performance on the same aspect of a different environment.

These different modules are modeled using LSTMs or GRUs. The total number of modules are fixed to Kt. At each time-step a fixed number (Ka) modules are selected to be active. These Ka active modules are selected using an input attention mechanism. The top-Ka modules that produce the highest scores for the input are selected to be active. The other modules are fed a null-input (all zeros).

Once the new states for each module (normal LSTM or GRU computation) are computed given their inputs that come from the input attention, each module can interact with each other using another attention mechanism which is called the communication attention mechanism. Only the states of the active modules are updated using this attention mechanism. The active modules can refer to the active modules as well as the inactive modules for updating their states. 

The image below has been taken from the original [paper](https://arxiv.org/pdf/1909.10893.pdf).

<p align="center">
  <img width="560" height="300" src="https://github.com/dido1998/Recurrent-Independent-Mechanisms/blob/master/README-RES/rim_image.png">
</p>

## Updates
**8/3/2020 : Implemented `GroupLSTMCell` and `GroupGRUCell` which eliminate the need for using Kt `LSTM` or `GRU` Cells. Previously, the computation of the `LSTM` or `GRU` operation required looping over Kt cells. Now, the `GroupLSTMCell` and `GroupGRUCell` can compute the `LSTM` or `GRU` operation at once (parallely) without using a loop. This results in a speed-up of the RIM computation as shown below**
<p align="center">
  <img width="500" height="450" src="https://github.com/dido1998/Recurrent-Independent-Mechanisms/blob/master/README-RES/time_comparison.png">
</p>

**7/3/2020 : Added support for n-layered and bidirectional RIM similar to `nn.LSTM` and `nn.GRU`.**

## Setup
* For using RIM as a standalone replacement for LSTMs or GRUs
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
### RIMCell
A single RIM cell similar to `nn.LSTMCell` or `nn.GRUCell`.

```
Class RIM.RIMCell(device,
input_size,
hidden_size,
num_units,
k,
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
comm_dropout = 0.1
)
```
For description of the RIMCell please check the paper.

#### Parameters
| Parameter | Description |
| --------- | ----------- |
| **device** | `torch.device('cuda')` or `torch.device('cpu')`. |
| **input_size** | The number of expected input features. |
| **hidden_size** | The number of hidden features in each unit. |
| **num_units** | Number of total RIM units. |
| **k** | Number of active RIMs at every time-step. |
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


#### Inputs
| Input | Description |
| ----- | ----------- |
| **x** | Input of shape (*batch_size*, 1, *input_size*). |
| **hs** | Hidden state for the current time-step of shape (*batch_size, num_units, hidden_size*). |
| **cs** | This is given if `rnn_cell == 'LSTM'` else it is `None`. Cell state for the current time-step of shape (*batch_size, num_units, hidden_size*). |

#### Outputs 
| Output | Description |
| ------ | ----------- |
| **hs** | The new hidden state of shape (*batch_size, num_units, hidden_size*). |
| **cs** | This is only returned if `rnn_cell == 'LSTM'`. The new cell state of shape (*batch_size, num_units, hidden_size*).|

### Example
```
from RIM import RIMCell
timesteps = 50
batch_size = 32
num_units = 6
k = 4
input_size = 32
hidden_size = 64
# Model definition. The definition of each argument is same as above.
rim_model = RIMCell(torch.device('cuda'), input_size, hidden_size, num_units, k, 'LSTM')

# creating hidden states and cell states
hs = torch.randn(batch_size, num_units, hidden_size)
cs = torch.randn(batch_size, num_units, hidden_size)
 
# Creating Input
xs = torch.randn(batch_size, timesteps, input_size)
xs = torch.split(xs, 1, 1)

for x in xs:
    hs, cs = rim_model(x, hs, cs)
```
----

### RIM
A recurrent network made up of RIM cells similar to `nn.LSTM` or `nn.GRU`.

```
class RIM.RIM(device,
input_size,
hidden_size,
num_units,
k,
rnn_cell,
n_layers,
bidirectional,
**kwargs
)
```

#### Parameters
| Parameter | Description |
| --------- | ----------- |
| **device** | `'cpu'` or `'cuda'`. |
| **input_size** | Input feature size. |
| **hidden_size** | Hidden feature size of each RIM unit. |
| **num_units** | Number of RIM units. |
| **k** | Number of active RIMs at each time-step | 
| **rnn_cell** | `'LSTM'` or `'GRU'` |
| **n_layers** | Number of RIM layers |
| **bidirectional** | `True` or `False` |

The keyword arguments are same as `RIM.RIMCell`.

#### Inputs
| Input | Description |
| ----- | ----------- |
| **x** | Input of shape (*seq_len, batch_size, input_size*) |
| **hs** | Hidden state of shape (*num_layers * num_directions, batch_size, hidden_size * num_units*). If not provided, it is randomly initialized |
| **cs** |  Provided only id `rnn_cell == LSTM`. Shape is same as **hs**. If not provided, it is randomly initialized. |

#### Outputs
| Output | Description |
| ------ | ----------- |
| **output** | Output of shape (*seq_len, batch_size, num_directions * hidden_size * num_units*) |
| **hs** | Hidden state of shape (*num_directions * num_layers, batch_size, hidden_size * num_units*) |
| **cs** | Returned if `rnn_cell == LSTM`. Cell state of shape same as **hs**. | 

#### Example
```
from RIM import RIM
rim_model = RIM('cuda', 16, 24, 6, 4, 'LSTM', 4, True)
x = torch.randn(7, 4, 16).cuda()
out, h, c = rim_model(x)
```

## Gym MiniGrid
The minigrid environment is available [here](https://github.com/maximecb/gym-minigrid). Results for the gym minigrd environment solved using **PPO**. 

You need to `cd` into the `minigrid_experiments` directory to run these experiments.

### Training
```
python3.6 train.py --algo ppo --env <Any of the available envs in the minigrid repo>
                   --model <name of the directory to store the trained model and related files>
                   --use_rim
                   --frames <num_frames>
```
You can also use `a2c` for training by changing the `--algo` option accordingly. If the `--use_rim` is not specified, the model will use a single`LSTM` for training. I recommend using a *80000* frames for task-1, *1000000* for task-2 and 300000 for task-3. I recommend keeping the other parameters same for convergence. If you tweak the other parameters and get better results let me know :)

### Evaluation
```
python3.6 evaluate.py --env <Any of the available envs in the minigrid repo>
                      --model <directory where model is stored> 
                      --use_rim
```
The `--use_rim` flag is used when your model was trained using an RIM. For simple LSTM you can leave the `--use_rim` flag.

### Visualization
```
python3.6 visualize.py --env <Any of the available envs in the minigrid repo>
                        --model <directory where model is stored> 
                        --gif <name of the gif file> 
                        --use_rim
```
The `--use_rim` flag has similar use as in evaluation.

#### For all the tables, the model is trained on the star-marked column and only evaluated on the other columns.

**I report the mean return per episode in each case**

**The environment names used below are same as the ones in the [minigrid repo](https://github.com/maximecb/gym-minigrid)**


### Task 1

The models shown in the gif have been trained on the MiniGrid-Empty-5x5-V0 environment.
| LSTM|  RIM |
:-------------------------:|:-------------------------:
![](https://github.com/dido1998/Recurrent-Independent-Mechanisms/blob/master/README-RES/5X5_LSTM.gif)  |  ![](https://github.com/dido1998/Recurrent-Independent-Mechanisms/blob/master/README-RES/16_16_RIM.gif)



| Model | MiniGrid-Empty-5x5-V0 **\*** | MiniGrid-Empty-16x16-V0 |
| ----- | --------------------- | ----------------------- |
| RIM (Kt = 4, Ka = 3) | 0.91 | 0.92 |
| RIM (Kt = 4, Ka = 2) | **0.92** | **0.95** |
| LSTM | 0.80 | 0.84 |


### Task 2

The modelS shown in the gif have been trained on the MiniGrid-MultiRoom-N2-S4-V0 (2 rooms) environment.
| LSTM|  RIM |
:-------------------------:|:-------------------------:
![](https://github.com/dido1998/Recurrent-Independent-Mechanisms/blob/master/README-RES/N4_LSTM.gif)  |  ![](https://github.com/dido1998/Recurrent-Independent-Mechanisms/blob/master/README-RES/N4_RIM.gif)

| Model | MiniGrid-MultiRoom-N2-S4-V0 (2 rooms) **\*** | MiniGrid-MultiRoom-N2-S5-V0 (4 rooms) | MiniGrid-MultiRoom-N6-V0 (6 rooms) |
| ----- | --------------------------- | --------------------------- | ------------------------ |
| RIM (Kt = 4, Ka = 3) | 0.81 | **0.66** | **0.05** |
| RIM (Kt = 4, Ka = 2) | 0.81 | 0.10 | 0.00 | 
| LSTM | **0.82** | 0.04 | 0.00 |

### Task 3

The models shown in the gif have been trained on the MiniGrid-DoorKey-5x5-V0 environment.

| LSTM|  RIM |
:-------------------------:|:-------------------------:
![](https://github.com/dido1998/Recurrent-Independent-Mechanisms/blob/master/README-RES/5x5_DoorkKey_LSTM.gif)  |  ![](https://github.com/dido1998/Recurrent-Independent-Mechanisms/blob/master/README-RES/6_6_RIM_DoorKey.gif)


| Model | MiniGrid-DoorKey-5x5-v0 **\*** | MiniGrid-DoorKey-6x6-v0 | MiniGrid-DoorKey-8x8-v0 | MiniGrid-DoorKey-16x16-v0 |
| ----- | ------------------------------ | ----------------------- | ----------------------- | ------------------------- |
| RIM (Kt=4, Ka = 3) | **0.90** | **0.68** | **0.38** | **0.18** |
| RIM (Kt = 4, Ka = 2) | 0.85 | 0.62 | 0.29 | 0.13 |
| LSTM | 0.90 | 0.63 | 0.35 | 0.12 |

**Insight**: Task 2 and Task 3 demonstrate the importance of the hyper-parameter Ka (number of active modules per timestep). We can see that reducing Ka from 3 to 2 drastically reduces performance especially in task 2. We also see that the RIM with Ka = 2 is the best performing model for task 1 but task 1 is a comparitively simple task. It would be interesting to see what causes each RIM to activate in each environment. 



## Sequential MNIST Task
Results for MNIST task: 

#### The model has been trained on MNIST datset with individual image size 14*14


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

## Contact

For any issues/questions, you can open a GitHub issue or contact [me](http://aniketdidolkar.in/) directly. 


