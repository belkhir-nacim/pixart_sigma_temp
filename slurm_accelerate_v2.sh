#!/bin/bash

#SBATCH -A tra24_openhack
#SBATCH -p boost_usr_prod
#SBATCH --job-name=multinode
#SBATCH -D .
#SBATCH --output=accelv2-%x.out
#SBATCH --error=accelv2-%x.err
#SBATCH --nodes=2                   # number of nodes
##SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gres=gpu:4               # number of GPUs per node
#SBATCH --cpus-per-task=16         # number of cores per tasks
##SBATCH --time=01:59:00             # maximum execution time (HH:MM:SS)
#SBATCH --qos=boost_qos_dbg
#SBATCH --export=ALL

######################
### Set enviroment ###
######################
export GPUS_PER_NODE=4
module purge
# module load python/3.11.6--gcc--8.5.0
module load cuda/12.1
module load nccl/2.19.3-1--gcc--12.2.0-cuda-12.1

source $HOME/.bashrc
conda activate pixart

export GPUS_PER_NODE=4
######################

######################
#### Set network #####
######################
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export NPROCESS=$((SLURM_NNODES * GPUS_PER_NODE))
echo $NPROCESS
######################

which python
cd /leonardo/home/usertrain/a08trb15/works/PixArt-sigma

# export LAUNCHER="accelerate launch --num_processes $NPROCESS --num_machines $SLURM_NNODES --machine_rank 0 --rdzv_backend c10d  --main_process_ip $head_node_ip --main_process_port $MASTER_PORT --multi_gpu --same_network"

export LAUNCHER="torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=2  --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:$MASTER_PORT"

export PYTHON_FILE="/leonardo/home/usertrain/a08trb15/works/PixArt-sigma/train_scripts/train.py"
export SCRIPT_ARGS=" \
        /leonardo/home/usertrain/a08trb15/works/PixArt-sigma/configs/pixart_sigma_config/PixArt_sigma_xl2_img512_internalms_custom.py \
          --load-from /leonardo/home/usertrain/a08trb15/works/PixArt-sigma/output/pretrained_models/PixArt-Sigma-XL-2-512-MS.pth \
          --work-dir /leonardo/home/usertrain/a08trb15/works/PixArt-sigma/output/your_first_pixart-exp
        "
NCCL_DEBUG=INFO 
# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $PYTHON_FILE $SCRIPT_ARGS" 
# echo $CMD
srun $CMD
