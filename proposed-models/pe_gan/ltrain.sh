#!/usr/bin/env bash
python3 train.py --data_dir /mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/txt/bassoon_txt/train_b_00_data.txt --ngf 64 --ndf 2 --batch_size 4 --gan_mode vanilla --model pix2pix --name 03_ours --checkpoints_dir /mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/res --dataset_mode spectram --no_flip --verbose --norm batch --save_epoch_freq 100 --niter 100

<< COMMENTOUT
01_ours     /mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/txt/sax_txt/train_s_00_data.txt
02_ours     /mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/txt/flute_txt/train_f_00_data.txt
03_ours     /mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/txt/bassoon_txt/train_b_00_data.txt
04_ours     /mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/txt/horn_txt/train_h_00_data.txt
05_ours     /mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/txt/clarinet_txt/train_cl_00_data.txt
06_ours     /mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/txt/cello_txt/train_c_00_data.txt
07_ours     /mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/txt/trumpet_txt/train_tp_00_data.txt
08_ours     /mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/txt/oboe_txt/train_ob_00_data.txt
09_ours     /mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/txt/trombone_txt/train_tb_00_data.txt
10_ours     /mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/txt/violin_txt/train_vi_00_data.txt
11_ours     /mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/txt/double_bass_txt/train_d_00_data.txt
12_ours     /mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/txt/tuba_txt/train_tu_00_data.txt
13_ours     /mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/txt/viola_txt/train_vl_00_data.txt
COMMENTOUT