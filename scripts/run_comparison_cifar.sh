#!/bin/bash
# Comparison script: runs exactly what run_cifar.sh does, comparing baseline vs q_vendi
# This matches the parameters from scripts/run_cifar.sh exactly

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Common parameters (matching run_cifar.sh exactly)
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

# Run 1: Baseline (loss_diff only) - exactly matching run_cifar.sh
echo "=========================================="
echo "Running BASELINE (loss_diff)"
echo "=========================================="
local_path='./results/split_cifar_loss_diff/test1'

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
	--use_qvendi=0

# Run 2: Q-Vendi method - same parameters but with q_vendi enabled
echo ""
echo "=========================================="
echo "Running Q-VENDI (loss_diff as quality)"
echo "=========================================="
local_path='./results/split_cifar_qvendi/test1'

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
	--use_qvendi=1

echo ""
echo "=========================================="
echo "Both runs complete!"
echo "=========================================="
echo "Results stored in:"
echo "  Baseline: ./results/split_cifar_loss_diff/test1/buffer/*.pkl"
echo "  Q-Vendi:  ./results/split_cifar_qvendi/test1/buffer/*.pkl"
echo ""
echo "To compare results, run:"
echo "  python analysis/summarize_table1.py ./results/split_cifar_loss_diff ./results/split_cifar_qvendi"
