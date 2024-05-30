#!/usr/bin/env bash

#SBATCH -A tra24_openhack
#SBATCH -p boost_usr_prod
##SBATCH --time 0:30:00     # format: HH:MM:SS
#SBATCH --nodes=2            # 4 node
#SBATCH -c 16           # 1 cpu per task
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gres=gpu:4                # number of GPUs per node
#SBATCH --cpus-per-task=32         # number of cores per tasks
#SBATCH --job-name=train_pixel_cc12
#SBATCH --output=log_train_accelerate.out  # Nom du fichier de sortie
#SBATCH --error=log_train_accelerate.err  # Nom du fichier d'erreur
#SBATCH --qos=boost_qos_dbg
#SBATCH --export=ALL

export GPUS_PER_NODE=4

module purge
# module load python/3.11.6--gcc--8.5.0
module load cuda/12.1
module load nccl/2.19.3-1--gcc--12.2.0-cuda-12.1

source $HOME/.bashrc

conda activate pixart
which python

cd /leonardo/home/usertrain/a08trb15/works/PixArt-sigma

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "MASTER_ADDR:MASTER_PORT="${MASTER_ADDR}:${MASTER_PORT}

export LAUNCHER="accelerate launch \
        --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \ 
        --mixed_precision fp16 \
        --num_machines 1 \
        --rdzv_backend c10d \
        --main_process_ip $MASTER_ADDR \
        --main_process_port $MASTER_PORT \
        --machine_rank $SLURM_NODEID"

export SCRIPT="/leonardo/home/usertrain/a08trb15/works/PixArt-sigma/train_scripts/train.py"
export SCRIPT_ARGS=" \
        /leonardo/home/usertrain/a08trb15/works/PixArt-sigma/configs/pixart_sigma_config/PixArt_sigma_xl2_img512_internalms_custom.py \
          --load-from /leonardo/home/usertrain/a08trb15/works/PixArt-sigma/output/pretrained_models/PixArt-Sigma-XL-2-512-MS.pth \
          --work-dir /leonardo/home/usertrain/a08trb15/works/PixArt-sigma/output/your_first_pixart-exp
        "
export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS"
srun $CMD