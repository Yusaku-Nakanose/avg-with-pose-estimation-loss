#!/usr/bin/env bash
python3 test.py --data_dir ../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/test_con_v_02_data.txt --ngf 64 --batch_size 1 --model test --name 0712_1 --checkpoints_dir ../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/res --norm batch --gpu_ids 0 --eval --verbose --results_dir ../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results