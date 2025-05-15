#!/bin/bash
#SBATCH --job-name="vel1"
####SBATCH -p gpu
#SBATCH -p geop # CWP partition
####SBATCH --exclude=g[001,002,004,005] # exclude nodes
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # number of tasks per node
##SBATCH --ntasks=1 # redundant; total number of tasks: ntasks = nodes * ntasks-per-node
#SBATCH --cpus-per-task=6
#SBATCH --mem=180GB    #ask for memory
#SBATCH --time=06:05:00 # time in HH:MM:SS
#SBATCH -o output.%j # standard print output labeled with SLURM job id %j
#SBATCH -e error.%j  # standard print error  labeled with SLURM job id %j

# export NCCL_DEBUG=INFO

### change 5-digit MASTER_PORT as you wish 0 slurm will raise Error if duplicated with others

### change WORLD_SIZE as gpus/node * num_nodes

export MASTER_PORT=12340

export WORLD_SIZE=$(( SLURM_NTASKS_PER_NODE * SLURM_NNODES ))
### export WORLD_SIZE=${SLURM_NTASKS}

### get the first node name as master address - customized for vgg slurm

### e.g. master(gnodee[2-5] 0gnoded1) == gnodee2

echo "NODELIST="${SLURM_NODELIST}

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

export MASTER_ADDR=$master_addr

echo "MASTER_ADDR="$MASTER_ADDR

export TORCH_DISTRIBUTED_DEBUG=INFO

# load python module 
module load apps/python3/2022.10

# activate conda environment required to run code
conda activate my_pytorch_12

# Start
echo "Job has started!"

##### Models predictions -  2Encoder and 4 decoders (8x8, 16x16, 32x32, 64x64)
# srun python train_mod1.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vp -r checkpoint.pth \
#             -m DualEncMultiDcoInvNetWithTailoredDecoders --tensorboard -t train_1gp.txt -v val_noiseFree_set01gp36.txt \
#             -b 100 -nb 40 --lr 0.001 \
#             -j ${SLURM_CPUS_PER_TASK} --world-size ${WORLD_SIZE} \
#             -g1_lb1 0.05 -g1_lb2 0.1 -g1_lb3 0.8 -g1_lb4 0.95 \
#             -g2_lb1 0.05 -g2_lb2 0.05 -g2_lb3 0.1 -g2_lb4 0.1

# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vp -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree.txt -r checkpoint.pth --vis -vb 2 -vsa 3
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vp -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree_set01gp33.txt -r checkpoint.pth --vis -vb 2 -vsa 3
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vp -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree_set01gp34.txt -r checkpoint.pth --vis -vb 2 -vsa 3
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vp -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree_set01gp36.txt -r checkpoint.pth --vis -vb 2 -vsa 3
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vp -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree_set01gp40.txt -r checkpoint.pth --vis -vb 2 -vsa 3

########### t03_train
# srun python train_mod1.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vp/t03_tr_mdl -r checkpoint.pth \
#             -m DualEncMultiDcoInvNetWithTailoredDecoders --tensorboard -t train_1gp.txt -v val_noiseFree_set01gp40.txt \
#             -b 100 -nb 25 \
#             -j ${SLURM_CPUS_PER_TASK} --world-size ${WORLD_SIZE} \
#             -g1_lb1 0.05 -g1_lb2 0.1 -g1_lb3 0.8 -g1_lb4 0.95 \
#             -g2_lb1 0.05 -g2_lb2 0.1 -g2_lb3 0.1 -g2_lb4 0.1

# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vp/t03_tr_mdl -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree.txt -r checkpoint.pth --vis -vb 2 -vsa 3
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vp/t03_tr_mdl -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree_set01gp34.txt -r checkpoint.pth --vis -vb 2 -vsa 3
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vp/t03_tr_mdl -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree_set01gp36.txt -r checkpoint.pth --vis -vb 2 -vsa 3
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vp/t03_tr_mdl -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree_set01gp40.txt -r checkpoint.pth --vis -vb 2 -vsa 3

######### finetune_2Dec
# srun python train_mod1.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vp/finetune_2Dec -r checkpoint.pth \
#             -m DualEncMultiDcoInvNetWithTailoredDecoders --tensorboard -t train_1gp.txt -v val_noiseFree_set01gp40.txt \
#              -b 50 -nb 15 \
#             -g1_lb1 0.9 -g1_lb2 0.9 \
#             -g2_lb1 0.01 -g2_lb2 0.01 \
#             -j ${SLURM_CPUS_PER_TASK} --world-size ${WORLD_SIZE}
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vp/finetune_2Dec -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree.txt -r checkpoint.pth --vis -vb 2 -vsa 3
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vp/finetune_2Dec -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree_set01gp34.txt -r checkpoint.pth --vis -vb 2 -vsa 3
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vp/finetune_2Dec -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree_set01gp36.txt -r checkpoint.pth --vis -vb 2 -vsa 3
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vp/finetune_2Dec -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree_set01gp40.txt -r checkpoint.pth --vis -vb 2 -vsa 3

######### finetune_2Dec_v2
# srun python train_mod1.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vp/finetune_2Dec_v2 -r checkpoint.pth\
#             -m DualEncMultiDcoInvNetWithTailoredDecoders --tensorboard -t train_1gp.txt -v val_noiseFree_set01gp40.txt \
#              -b 50 -nb 15 \
#             -g1_lb1 0.9 -g1_lb2 0.9 \
#             -g2_lb1 0.01 -g2_lb2 0.01 \
#             -j ${SLURM_CPUS_PER_TASK} --world-size ${WORLD_SIZE}
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vp/finetune_2Dec_v2 -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree.txt -r checkpoint.pth --vis -vb 2 -vsa 3
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vp/finetune_2Dec_v2 -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree_set01gp33.txt -r checkpoint.pth --vis -vb 2 -vsa 3
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vp/finetune_2Dec_v2 -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree_set01gp34.txt -r checkpoint.pth --vis -vb 2 -vsa 3
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vp/finetune_2Dec_v2 -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree_set01gp36.txt -r checkpoint.pth --vis -vb 2 -vsa 3
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vp/finetune_2Dec_v2 -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree_set01gp40.txt -r checkpoint.pth --vis -vb 2 -vsa 3

######### finetune_2Dec_v3
# srun python train_mod1.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vp/finetune_2Dec_v3 -r checkpoint.pth\
#             -m DualEncMultiDcoInvNetWithTailoredDecoders --tensorboard -t train_1gp_2400smpl.txt -v val_noiseFree.txt \
#             -b 50 -nb 25 \
#             -g1_lb1 0.0 -g1_lb2 0.0 \
#             -g2_lb1 0.5 -g2_lb2 0.5 \
#             -j ${SLURM_CPUS_PER_TASK} --world-size ${WORLD_SIZE}
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vp/finetune_2Dec_v3 -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree.txt -r checkpoint.pth --vis -vb 2 -vsa 3
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vp/finetune_2Dec_v3 -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree_set01gp33.txt -r checkpoint.pth --vis -vb 2 -vsa 3
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vp/finetune_2Dec_v3 -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree_set01gp34.txt -r checkpoint.pth --vis -vb 2 -vsa 3
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vp/finetune_2Dec_v3 -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree_set01gp36.txt -r checkpoint.pth --vis -vb 2 -vsa 3
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vp/finetune_2Dec_v3 -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree_set01gp40.txt -r checkpoint.pth --vis -vb 2 -vsa 3

######### finetune_2Dec_v4
# srun python train_mod1_reprod.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vp/finetune_2Dec_v4 -r checkpoint.pth \
#             -m DualEncMultiDcoInvNetWithTailoredDecoders --tensorboard -t train_1gp_2400smpl.txt -v val_noiseFree.txt \
#             -b 100 -nb 3\
#             -g1_lb1 1.0 -g1_lb2 1.0 \
#             -g2_lb1 0.0 -g2_lb2 0.0 \
#             -j ${SLURM_CPUS_PER_TASK} --world-size ${WORLD_SIZE}
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vp/finetune_2Dec_v4 -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree.txt -r checkpoint.pth --vis -vb 2 -vsa 3

# srun python train_mod1_reprod.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vp/finetune_2Dec_v4/t02 -r checkpoint.pth \
#             -m DualEncMultiDcoInvNetWithTailoredDecoders --tensorboard -t train_1gp_2400smpl.txt -v val_noiseFree.txt \
#             -b 100 -nb 3 \
#             -g1_lb1 0.0 -g1_lb2 0.0 \
#             -g2_lb1 1.0 -g2_lb2 1.0 \
#             -j ${SLURM_CPUS_PER_TASK} --world-size ${WORLD_SIZE}
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vp/finetune_2Dec_v4/t02 -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree.txt -r checkpoint.pth --vis -vb 2 -vsa 3

# srun python train_mod1_reprod.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vp/finetune_2Dec_v4/t03 -r checkpoint.pth \
#             -m DualEncMultiDcoInvNetWithTailoredDecoders --tensorboard -t train_1gp_2400smpl.txt -v val_noiseFree.txt \
#             -b 100 -nb 3 \
#             -g1_lb1 1.0 -g1_lb2 1.0 \
#             -g2_lb1 1.0 -g2_lb2 1.0 \
#             -j ${SLURM_CPUS_PER_TASK} --world-size ${WORLD_SIZE}
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vp/finetune_2Dec_v4/t03 -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree.txt -r checkpoint.pth --vis -vb 2 -vsa 3

# srun python train_mod1_reprod.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vp/finetune_2Dec_v4/t04 -r checkpoint.pth \
#             -m DualEncMultiDcoInvNetWithTailoredDecoders --tensorboard -t train_1gp_2400smpl.txt -v val_noiseFree.txt \
#             -b 100 -nb 3 \
#             -g1_lb1 0.5 -g1_lb2 0.5 \
#             -g2_lb1 0.5 -g2_lb2 0.5 \
#             -j ${SLURM_CPUS_PER_TASK} --world-size ${WORLD_SIZE}
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vp/finetune_2Dec_v4/t04 -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree.txt -r checkpoint.pth --vis -vb 2 -vsa 3

################# Vp/Vs ratio
# srun python train_mod1.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vpvsratio \
#             -r checkpoint.pth -m DualEncMultiDcoInvNetWithTailoredDecoders --tensorboard -t train_1gp.txt -v val_noiseFree.txt \
#             -b 50 -nb 15 --lr 0.0001 \
#             -j ${SLURM_CPUS_PER_TASK} --world-size ${WORLD_SIZE} \
#             -g1_lb1 0.1 -g1_lb2 0.3 -g1_lb3 0.7 -g1_lb4 0.9 \
#             -g2_lb1 0.0 -g2_lb2 0.0 -g2_lb3 0.0 -g2_lb4 0.0
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vpvsratio -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree.txt -r checkpoint.pth --vis -vb 2 -vsa 3
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vpvsratio -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree_set01gp33.txt -r checkpoint.pth --vis -vb 2 -vsa 3
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vpvsratio -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree_set01gp34.txt -r checkpoint.pth --vis -vb 2 -vsa 3
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vpvsratio -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree_set01gp36.txt -r checkpoint.pth --vis -vb 2 -vsa 3
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vpvsratio -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree_set01gp40.txt -r checkpoint.pth --vis -vb 2 -vsa 3


########### t04_train
# srun python train_mod1.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vpvsratio/t04_train \
#             -r checkpoint.pth -m DualEncMultiDcoInvNetWithTailoredDecoders --tensorboard -t train_1gp.txt -v val_noiseFree_set01gp36.txt \
#             -b 50 -nb 25 --lr 0.001 \
#             -j ${SLURM_CPUS_PER_TASK} --world-size ${WORLD_SIZE} \
#             -g1_lb1 0.1 -g1_lb2 0.3 -g1_lb3 0.7 -g1_lb4 0.9 \
#             -g2_lb1 0.05 -g2_lb2 0.1 -g2_lb3 0.3 -g2_lb4 0.4
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vpvsratio/t04_train -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree.txt -r checkpoint.pth --vis -vb 2 -vsa 3
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vpvsratio/t04_train -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree_set01gp34.txt -r checkpoint.pth --vis -vb 2 -vsa 3
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vpvsratio/t04_train -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree_set01gp36.txt -r checkpoint.pth --vis -vb 2 -vsa 3
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vpvsratio/t04_train -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree_set01gp40.txt -r checkpoint.pth --vis -vb 2 -vsa 3

######### finetune_2Dec
# srun python train_mod1.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vpvsratio/finetune_2Dec -r checkpoint.pth \
#             -m DualEncMultiDcoInvNetWithTailoredDecoders --tensorboard -t train_1gp.txt -v val_noiseFree.txt \
#              -b 50 -nb 15 \
#             -g1_lb1 0.2 -g1_lb2 0.9 \
#             -g2_lb1 0.01 -g2_lb2 0.01 \
#             -j ${SLURM_CPUS_PER_TASK} --world-size ${WORLD_SIZE}
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vpvsratio/finetune_2Dec -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree.txt -r checkpoint.pth --vis -vb 2 -vsa 3
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vpvsratio/finetune_2Dec -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree_set01gp33.txt -r checkpoint.pth --vis -vb 2 -vsa 3
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vpvsratio/finetune_2Dec -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree_set01gp34.txt -r checkpoint.pth --vis -vb 2 -vsa 3
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vpvsratio/finetune_2Dec -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree_set01gp36.txt -r checkpoint.pth --vis -vb 2 -vsa 3
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/vpvsratio/finetune_2Dec -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree_set01gp40.txt -r checkpoint.pth --vis -vb 2 -vsa 3

################ Acoustic Impedence IP
# srun python train_mod1.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/ip -r checkpoint.pth \
#             -m DualEncMultiDcoInvNetWithTailoredDecoders --tensorboard -t train_1gp.txt -v val_noiseFree.txt \
#             -b 50 -nb 35 --lr 0.001\
#             -g1_lb1 0.01 -g1_lb2 0.1 -g1_lb3 0.5 -g1_lb4 0.95 \
#             -g2_lb1 0.01 -g2_lb2 0.1 -g2_lb3 0.5 -g2_lb4 0.95 \
#             -j ${SLURM_CPUS_PER_TASK} --world-size ${WORLD_SIZE}
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/ip -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree.txt -r checkpoint.pth --vis -vb 2 -vsa 3
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/ip -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree_set01gp33.txt -r checkpoint.pth --vis -vb 2 -vsa 3
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/ip -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree_set01gp34.txt -r checkpoint.pth --vis -vb 2 -vsa 3
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/ip -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree_set01gp36.txt -r checkpoint.pth --vis -vb 2 -vsa 3
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/ip -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree_set01gp40.txt -r checkpoint.pth --vis -vb 2 -vsa 3

################ Acoustic Impedence IP (128x128), (256,256)
# srun python train_mod1.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/ip/finetune_2Dec -r checkpoint.pth \
#             -m DualEncMultiDcoInvNetWithTailoredDecoders --tensorboard -t train_1gp.txt -v val_noiseFree.txt \
#              -b 50 -nb 15 \
#             -g1_lb1 0.2 -g1_lb2 0.9 \
#             -g2_lb1 0.01 -g2_lb2 0.1 \
#             -j ${SLURM_CPUS_PER_TASK} --world-size ${WORLD_SIZE}
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/ip/finetune_2Dec -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree.txt -r checkpoint.pth --vis -vb 2 -vsa 3
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/ip/finetune_2Dec -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree_set01gp33.txt -r checkpoint.pth --vis -vb 2 -vsa 3
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/ip/finetune_2Dec -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree_set01gp34.txt -r checkpoint.pth --vis -vb 2 -vsa 3
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/ip/finetune_2Dec -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree_set01gp36.txt -r checkpoint.pth --vis -vb 2 -vsa 3
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/ip/finetune_2Dec -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree_set01gp40.txt -r checkpoint.pth --vis -vb 2 -vsa 3


# srun python train_mod1.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/ip/finetune_2Dec_t02 -r checkpoint.pth \
#             -m DualEncMultiDcoInvNetWithTailoredDecoders --tensorboard -t train_1gp_3200smpl.txt -v val_noiseFree.txt \
#             -b 50 -nb 10 \
#             -g1_lb1 0.2 -g1_lb2 0.9 \
#             -g2_lb1 0.01 -g2_lb2 0.1 \
#             -j ${SLURM_CPUS_PER_TASK} --world-size ${WORLD_SIZE}
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/ip/finetune_2Dec_t02 -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree_set01gp36.txt -r checkpoint.pth --vis -vb 2 -vsa 3
# srun python test.py -ds kimberlina -n ~/scratch/kimb_iso_mra_singleparameter_lstm/ip/finetune_2Dec_t02 -m DualEncMultiDcoInvNetWithTailoredDecoders -v val_noiseFree_set01gp40.txt -r checkpoint.pth --vis -vb 2 -vsa 3




echo "Job has finished!"

