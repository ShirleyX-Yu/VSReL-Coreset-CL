#!/bin/bash
#SBATCH --job-name=cifar_qvendi_q_sweep
#SBATCH --output=../logs/cifar_qvendi_q_sweep_%A_%a.out
#SBATCH --error=../logs/cifar_qvendi_q_sweep_%A_%a.err
#SBATCH --array=0-5
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=shirley.yu@princeton.edu

# q values
q_values=(0.1 0.5 1.0 2.0 10.0 inf)
q=${q_values[$SLURM_ARRAY_TASK_ID]}

module load cuda
module load python

source activate vsrel

export CUBLAS_WORKSPACE_CONFIG=:4096:8
dataset='splitcifar'
setting='greedy'
data_path=''
buffer_size=200
alpha=20.0
beta=0.0
lr=1e-3
epochs=400
batch_size=256
mem_batch_size=32
use_cuda=1
opt_type='adam'
seed=0
slt_wo_aug=0
holdout_set='sub'
replay_mode='full'
use_bn=0
limit_per_task=1000
runner_type='coreset'
update_mode='coreset'
extra_data=''
ref_train_epoch=150
selection_steps=200
cur_train_steps=100
buffer_type='coreset'
aug_type='greedy'
ref_train_lr=3e-3
cur_train_lr=1e-2
ref_sample_per_task=0

local_path="./results/split_cifar_qvendi_q${q}/seed${seed}"

echo "Running CIFAR with q=$q on GPU $CUDA_VISIBLE_DEVICES"

# change to repository root directory
cd ..

python3 -u offline_continual_learning.py --local_path=$local_path \
    --dataset=$dataset \
    --setting=$setting \
    --data_path=$data_path \
    --buffer_size=$buffer_size \
    --alpha=$alpha \
    --beta=$beta \
    --lr=$lr \
    --epochs=$epochs \
    --batch_size=$batch_size \
    --mem_batch_size=$mem_batch_size \
    --use_cuda=$use_cuda \
    --opt_type=$opt_type \
    --seed=$seed \
    --slt_wo_aug=$slt_wo_aug \
    --holdout_set=$holdout_set \
    --replay_mode=$replay_mode \
    --use_bn=$use_bn \
    --limit_per_task=$limit_per_task \
    --runner_type=$runner_type \
    --update_mode=$update_mode \
    --extra_data=$extra_data \
    --ref_train_epoch=$ref_train_epoch \
    --selection_steps=$selection_steps \
    --cur_train_steps=$cur_train_steps \
    --ref_train_lr=$ref_train_lr \
    --cur_train_lr=$cur_train_lr \
    --buffer_type=$buffer_type \
    --ref_sample_per_task=$ref_sample_per_task \
    --aug_type=$aug_type \
    --use_qvendi=1 \
    --qvendi_q=$q
