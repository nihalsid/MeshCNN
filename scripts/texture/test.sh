#!/usr/bin/env bash

## run the test and export collapses
python test.py
--dataroot datasets/cube_tex
--name texture_test
--arch meshunet
--dataset_mode texture
--ncf 32 64 128 256
--ninput_edges 7440
--pool_res 5580 3720 1860
--resblocks 3
--batch_size 2
--export_folder meshes
--use_single_view
