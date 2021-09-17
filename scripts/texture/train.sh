#!/usr/bin/env bash

## run the training
python train.py --dataroot datasets/cube_tex/ --name texture_test --arch meshunet --dataset_mode texture --ncf 32 64 128 256 --ninput_edges 7440 --pool_res 5580 3720 1860 --resblocks 3 --batch_size 2 --lr 0.001 --num_aug 0 --use_single_view --num_threads 0 --save_epoch_freq 50  --save_latest_freq 500 --run_test_freq 50 --continue_train