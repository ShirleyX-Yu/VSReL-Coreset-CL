#!/bin/bash
# Example script to run both loss_diff (baseline) and q_vendi methods for comparison
# This runs permuted MNIST with both methods

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Common parameters
dataset='mnist'
setting='permuted'
buffer_size=200
epochs=1
batch_size=10
mem_batch_size=10
use_cuda=1
opt_type='sgd'
lr=2e-2
replay_mode='full'
use_bn=0
limit_per_task=1000
runner_type='coreset'
update_mode='coreset'
extra_data=''
ref_train_epoch=20
selection_steps=100
cur_train_steps=30
buffer_type='coreset'
aug_type='greedy'
ref_train_lr=3e-3
cur_train_lr=2e-2
seed=42

# Run 1: Baseline (loss_diff only)
echo "=========================================="
echo "Running BASELINE (loss_diff)"
echo "=========================================="
local_path='./results/permuted_mnist_loss_diff'

python permuted_mnist_cl.py \
	--local_path=$local_path \
	--dataset=$dataset \
	--setting=$setting \
	--buffer_size=$buffer_size \
	--epochs=$epochs \
	--batch_size=$batch_size \
	--mem_batch_size=$mem_batch_size \
	--use_cuda=$use_cuda \
	--opt_type=$opt_type \
	--lr=$lr \
	--replay_mode=$replay_mode \
	--use_bn=$use_bn \
	--limit_per_task=$limit_per_task \
	--runner_type=$runner_type \
	--update_mode=$update_mode \
	--extra_data=$extra_data \
	--ref_train_epoch=$ref_train_epoch \
	--selection_steps=$selection_steps \
	--cur_train_steps=$cur_train_steps \
	--buffer_type=$buffer_type \
	--aug_type=$aug_type \
	--ref_train_lr=$ref_train_lr \
	--cur_train_lr=$cur_train_lr \
	--use_qvendi=0 \
	--seed=$seed

# Run 2: Q-Vendi method
echo ""
echo "=========================================="
echo "Running Q-VENDI (loss_diff as quality)"
echo "=========================================="
local_path='./results/permuted_mnist_qvendi'

python permuted_mnist_cl.py \
	--local_path=$local_path \
	--dataset=$dataset \
	--setting=$setting \
	--buffer_size=$buffer_size \
	--epochs=$epochs \
	--batch_size=$batch_size \
	--mem_batch_size=$mem_batch_size \
	--use_cuda=$use_cuda \
	--opt_type=$opt_type \
	--lr=$lr \
	--replay_mode=$replay_mode \
	--use_bn=$use_bn \
	--limit_per_task=$limit_per_task \
	--runner_type=$runner_type \
	--update_mode=$update_mode \
	--extra_data=$extra_data \
	--ref_train_epoch=$ref_train_epoch \
	--selection_steps=$selection_steps \
	--cur_train_steps=$cur_train_steps \
	--buffer_type=$buffer_type \
	--aug_type=$aug_type \
	--ref_train_lr=$ref_train_lr \
	--cur_train_lr=$cur_train_lr \
	--use_qvendi=1 \
	--seed=$seed

echo ""
echo "=========================================="
echo "Both runs complete!"
echo "=========================================="
echo "To compare results, run:"
echo "  python analysis/compare_methods.py --baseline ./results/permuted_mnist_loss_diff --qvendi ./results/permuted_mnist_qvendi"
