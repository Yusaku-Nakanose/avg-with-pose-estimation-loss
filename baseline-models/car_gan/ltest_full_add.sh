#!/usr/bin/env bash
#satoshi
python3 test.py --data_dir ../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/test_v_02_3d_data_v2.txt --ngf 64 --batch_size 1 --model test_full_add --name 0822_5 --checkpoints_dir ../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/res --norm batch --gpu_ids 2 --eval --verbose --results_dir ../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results