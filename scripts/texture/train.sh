#!/bin/bash

#SBATCH --job-name graphnn
##SBATCH --nodes=1
##SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --gpus=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yawar.siddiqui@tum.de
##SBATCH --exclude=char,pegasus,tarsonis,gondor,moria,seti,sorona,umoja,lothlann
#SBATCH --exclude=char,pegasus,tarsonis,balrog,daidalos,gimli,himring,hithlum
#SBATCH --partition=debug
#SBATCH --qos=normal

## run the training
cd /rhome/ysiddiqui/MeshCNN
python train.py --dataroot datasets/cube_tex/ --name texture_test --arch meshunet --dataset_mode texture --ncf 32 64 128 256 --ninput_edges 7440 --pool_res 5580 3720 1860 --resblocks 3 --batch_size 4 --lr 0.0002 --num_aug 0 --num_threads 8 --save_epoch_freq 1  --save_latest_freq 1500 --run_test_freq 1