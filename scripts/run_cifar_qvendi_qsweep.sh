#!/bin/bash
#SBATCH --job-name=cifar_qsweep
#SBATCH --output=../logs/cifar_qvendi_q_sweep_%j_%a.out
#SBATCH --error=../logs/cifar_qvendi_q_sweep_%j_%a.err
#SBATCH --time=2:00:00
#SBATCH --partition=cs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-6
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=shirley.yu@princeton.edu

# q-sweep for cifar-10: test q values [0.1, 0.5, 1.0, 2.0, 10.0, inf] + loss_diff baseline

# get project root from submit directory
if [[ "${SLURM_SUBMIT_DIR}" == */scripts ]]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR%/scripts}"
else
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
fi
cd "$PROJECT_ROOT"

# initialize conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate vsrel

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# map array index to q value or baseline
Q_VALUES=(0.1 0.5 1.0 2.0 10.0 inf baseline)
Q_VALUE=${Q_VALUES[$SLURM_ARRAY_TASK_ID]}

if [[ "$Q_VALUE" == "baseline" ]]; then
    USE_QVENDI=0
    METHOD_NAME="loss_diff"
    echo "=========================================="
    echo "running cifar-10 loss_diff baseline"
    echo "array task id: $SLURM_ARRAY_TASK_ID"
    echo "job id: $SLURM_JOB_ID"
    echo "=========================================="
else
    USE_QVENDI=1
    METHOD_NAME="qvendi_q${Q_VALUE}"
    echo "=========================================="
    echo "running cifar-10 q-vendi with q=${Q_VALUE}"
    echo "array task id: $SLURM_ARRAY_TASK_ID"
    echo "job id: $SLURM_JOB_ID"
    echo "=========================================="
fi

# parameters
dataset='splitcifar'
setting='greedy'
data_path=''
buffer_size=100
alpha=50.0
beta=0.0
lr=5e-4
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
ref_train_epoch=20
selection_steps=100
cur_train_steps=30
buffer_type='coreset'
aug_type='greedy'
ref_train_lr=3e-3
cur_train_lr=2e-2
ref_sample_per_task=0

local_path="./results/split_cifar_${METHOD_NAME}/test1"

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
	--use_qvendi=$USE_QVENDI \
	--qvendi_q=$Q_VALUE

if [[ "$Q_VALUE" == "baseline" ]]; then
    echo "completed cifar-10 loss_diff baseline"
else
    echo "completed cifar-10 q-vendi with q=${Q_VALUE}"
fi
