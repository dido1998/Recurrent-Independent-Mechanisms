#!/bin/zsh
echo Running on $HOSTNAME
source ~/.zshrc
conda activate rim

cuda=False
epochs=$1
lr=0.001
input_size=1
batch_size=32
hidden_size=100
num_units=6
k=2
size=14
loadsaved=0
model=RIM
sparse=False


name="smnist_lstm_"$hidden_size
name="${name//./}"
echo Running version $name
python3 main.py --cuda $cuda --epochs $epochs --batch_size $batch_size --input_size $input_size --hidden_size $hidden_size --size $size --loadsaved $loadsaved --model $model --sparse $sparse --num_units $num_units --k $k
