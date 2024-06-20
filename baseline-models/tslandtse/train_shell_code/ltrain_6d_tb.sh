#!/usr/bin/env bash satoshi
python3 ../train.py --data_dir /mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/txt/trombone_txt/train_tb_3d_data.txt --ngf 64 --ndf 2 --batch_size 4 --gan_mode vanilla --model pix2pix_con_full_add --name 09_3 --checkpoints_dir /mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/res --gpu_id 0 --dataset_mode spectram --no_flip --verbose --norm batch  --save_epoch_freq 100 --niter 100
