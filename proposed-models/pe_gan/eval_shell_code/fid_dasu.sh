#!/usr/bin/env bash
VAR=0905_1
VAR3=0905_2
VAR1=clarinet01
VAR2=cl
echo $VAR >> ../../../../../mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/fid_res/val_res.txt
python3 -m pytorch_fid ../../../../../mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"$VAR1"/ ../../../../../mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"$VAR2"_"$VAR" --device cuda:0 --batch-size 4 --num-workers 0 >> ../../../../../mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/fid_res/val_res.txt
#
echo $VAR3 >> ../../../../../mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/fid_res/val_res.txt
python3 -m pytorch_fid ../../../../../mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"$VAR1"/ ../../../../../mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"$VAR2"_"$VAR3" --device cuda:0 --batch-size 4 --num-workers 0 >> ../../../../../mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/fid_res/val_res.txt
