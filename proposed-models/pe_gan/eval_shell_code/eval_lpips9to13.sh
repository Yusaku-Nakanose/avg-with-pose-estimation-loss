#!/usr/bin/env bash
#python3 test.py --data_dir ../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/test_zurashi_v_02_04_data.txt --ngf 64 --batch_size 1 --model test --name 0909_3 --checkpoints_dir ../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/res --norm batch --gpu_ids 0 --eval --verbose --results_dir ../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results
#python3 lpips_2imgs.py -p0 imgs/ex_ref.png -p1 imgs/ex_p0.png --use_gpu
#######
##proposed##
VAR1=0911_1
VAR11=0911_2
VAR12=0911_3
VAR2=d
VAR3=double_bass01
# python3 lpips_2dirs.py -d0 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"$VAR2"_"$VAR1" -d1 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"$VAR3"_128 -o ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/lpips_res/lp_res_alex_$VAR1.txt --use_gpu

# python3 lpips_2dirs.py -d0 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"$VAR2"_"$VAR11" -d1 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"$VAR3"_128 -o ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/lpips_res/lp_res_alex_$VAR11.txt --use_gpu

# python3 lpips_2dirs.py -d0 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"$VAR2"_"$VAR12" -d1 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"$VAR3"_128 -o ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/lpips_res/lp_res_alex_$VAR12.txt --use_gpu
#
VAR1=0912_1
VAR11=0912_2
VAR12=0912_3
VAR2=tu
VAR3=tuba04
python3 lpips_2dirs.py -d0 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"$VAR2"_"$VAR1" -d1 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"$VAR3"_128 -o ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/lpips_res/lp_res_alex_$VAR1.txt --use_gpu

# python3 lpips_2dirs.py -d0 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"$VAR2"_"$VAR11" -d1 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"$VAR3"_128 -o ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/lpips_res/lp_res_alex_$VAR11.txt --use_gpu

# python3 lpips_2dirs.py -d0 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"$VAR2"_"$VAR12" -d1 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"$VAR3"_128 -o ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/lpips_res/lp_res_alex_$VAR12.txt --use_gpu
#
VAR1=0913_1
VAR11=0913_2
VAR12=0913_3
VAR2=vl
VAR3=viola01
# python3 lpips_2dirs.py -d0 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"$VAR2"_"$VAR1" -d1 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"$VAR3"_128 -o ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/lpips_res/lp_res_alex_$VAR1.txt --use_gpu

# python3 lpips_2dirs.py -d0 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"$VAR2"_"$VAR11" -d1 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"$VAR3"_128 -o ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/lpips_res/lp_res_alex_$VAR11.txt --use_gpu

# python3 lpips_2dirs.py -d0 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"$VAR2"_"$VAR12" -d1 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"$VAR3"_128 -o ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/lpips_res/lp_res_alex_$VAR12.txt --use_gpu

VAR1=1909_1
VAR11=1909_2
VAR12=1909_3
VAR2=tb
VAR3=trombone02
#python3 lpips_2dirs.py -d0 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"$VAR2"_"$VAR1" -d1 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"$VAR3"_128 -o ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/lpips_res/lp_res_alex_$VAR1.txt --use_gpu

#python3 lpips_2dirs.py -d0 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"$VAR2"_"$VAR11" -d1 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"$VAR3"_128 -o ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/lpips_res/lp_res_alex_$VAR11.txt --use_gpu

#python3 lpips_2dirs.py -d0 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"$VAR2"_"$VAR12" -d1 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"$VAR3"_128 -o ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/lpips_res/lp_res_alex_$VAR12.txt --use_gpu
#
VAR1=1910_1
VAR11=1910_2
VAR12=1910_3
VAR2=vi
VAR3=violin06
#python3 lpips_2dirs.py -d0 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"$VAR2"_"$VAR1" -d1 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"$VAR3"_128 -o ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/lpips_res/lp_res_alex_$VAR1.txt --use_gpu

#python3 lpips_2dirs.py -d0 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"$VAR2"_"$VAR11" -d1 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"$VAR3"_128 -o ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/lpips_res/lp_res_alex_$VAR11.txt --use_gpu

#python3 lpips_2dirs.py -d0 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"$VAR2"_"$VAR12" -d1 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"$VAR3"_128 -o ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/lpips_res/lp_res_alex_$VAR12.txt --use_gpu