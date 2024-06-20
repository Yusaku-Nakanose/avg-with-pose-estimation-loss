#!/usr/bin/env bash
python3 ../train.py --data_dir /mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/txt/violin_txt/train_vi_00_data.txt --ngf 64 --ndf 2 --batch_size 4 --gan_mode vanilla --model pix2pix --name 10_3 --checkpoints_dir /mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/res --dataset_mode spectram --no_flip --gpu_id 1 --verbose --norm batch --save_epoch_freq 1 --niter 1 --niter_decay 1
