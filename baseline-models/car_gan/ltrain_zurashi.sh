#!/usr/bin/env bash
#!/usr/bin/env bash
python3 train.py --data_dir ../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/train_zurashi_v_02_03_data.txt --ngf 64 --ndf 2 --batch_size 4 --gan_mode vanilla --model pix2pix --name 0910_4_2 --checkpoints_dir ../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/res --gpu_ids 0 --dataset_mode spectram --no_flip --verbose --norm batch  --save_epoch_freq  100 --niter 100
python3 train.py --data_dir ../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/train_zurashi_v_02_02_data.txt --ngf 64 --ndf 2 --batch_size 4 --gan_mode vanilla --model pix2pix --name 0910_5_2 --checkpoints_dir ../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/res --gpu_ids 0 --dataset_mode spectram --no_flip --verbose --norm batch  --save_epoch_freq  100 --niter 100
python3 train.py --data_dir ../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/train_zurashi_v_02_01_data.txt --ngf 64 --ndf 2 --batch_size 4 --gan_mode vanilla --model pix2pix --name 0910_6_2 --checkpoints_dir ../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/res --gpu_ids 0 --dataset_mode spectram --no_flip --verbose --norm batch  --save_epoch_freq  100 --niter 100
python3 train.py --data_dir ../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/train_zurashi_v_02_04_data.txt --ngf 64 --ndf 2 --batch_size 4 --gan_mode vanilla --model pix2pix --name 0910_3_2 --checkpoints_dir ../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/res --gpu_ids 0 --dataset_mode spectram --no_flip --verbose --norm batch  --save_epoch_freq  100 --niter 100