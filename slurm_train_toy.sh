#!/usr/bin/env bash

#SBATCH -A tra24_openhack
#SBATCH -p boost_usr_prod
##SBATCH --time 0:30:00     # format: HH:MM:SS
##SBATCH -N 2             # 4 node
#SBATCH -c 8           # 1 cpu per task
##SBATCH --gres=gpu:1                # 1GPU
#SBATCH --job-name=train_pixel_cc12
#SBATCH --output=log_train_toy_4gpu_2node.out  # Nom du fichier de sortie
#SBATCH --error=log_train_toy_4gpu_2node.err  # Nom du fichier d'erreur
#SBATCH --qos=boost_qos_dbg
#SBATCH --export=ALL

module purge
# module load python/3.11.6--gcc--8.5.0
module load cuda/12.1
module load nccl/2.19.3-1--gcc--12.2.0-cuda-12.1

source $HOME/.bashrc

conda activate pixart
which python

cd /leonardo/home/usertrain/a08trb15/works/PixArt-sigma
# echo $HOSTNAME

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "MASTER_ADDR:MASTER_PORT="${MASTER_ADDR}:${MASTER_PORT}

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12340
export WORLD_SIZE=4

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR


# # python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --master_port=$MASTER_PORT \
# #           train_scripts/train.py \
# #           configs/pixart_sigma_config/PixArt_sigma_xl2_img512_internalms_custom.py \
# #           --load-from output/pretrained_models/PixArt-Sigma-XL-2-512-MS.pth \
# #           --work-dir output/your_first_pixart-exp

python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --world_size=2 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
          train_scripts/train.py \
          configs/pixart_sigma_config/PixArt_sigma_xl2_img512_internalms_custom.py \
          --load-from output/pretrained_models/PixArt-Sigma-XL-2-512-MS.pth \
          --work-dir output/your_first_pixart-exp