#!/usr/bin/env bash
#python3 test.py --data_dir ../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/test_zurashi_v_02_04_data.txt --ngf 64 --batch_size 1 --model test --name 0909_3 --checkpoints_dir ../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/res --norm batch --gpu_ids 0 --eval --verbose --results_dir ../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results
#python3 lpips_2imgs.py -p0 imgs/ex_ref.png -p1 imgs/ex_p0.png --use_gpu
#######
##proposed##
# VAR1=0907_1
# VAR11=0907_2
# VAR2=tp
# VAR3=trumpet01
# python3 lpips_2dirs.py -d0 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"$VAR2"_"$VAR1" -d1 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"$VAR3"_128 -o ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/lpips_res/lp_res_alex_$VAR1.txt --use_gpu
# #
# python3 lpips_2dirs.py -d0 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"$VAR2"_"$VAR11" -d1 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"$VAR3"_128 -o ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/lpips_res/lp_res_alex_$VAR11.txt --use_gpu
# #
# VAR1=0906_1
# VAR11=0906_2
# VAR2=c
# VAR3=cello01
# python3 lpips_2dirs.py -d0 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"$VAR2"_"$VAR1" -d1 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"$VAR3"_128 -o ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/lpips_res/lp_res_alex_$VAR1.txt --use_gpu
# #
# python3 lpips_2dirs.py -d0 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"$VAR2"_"$VAR11" -d1 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"$VAR3"_128 -o ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/lpips_res/lp_res_alex_$VAR11.txt --use_gpu
#
VAR1=0905_1
VAR11=0905_2
VAR2=cl
VAR3=clarinet01
#python3 lpips_2dirs.py -d0 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"$VAR2"_"$VAR1" -d1 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"$VAR3"_128 -o ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/lpips_res/lp_res_alex_$VAR1.txt --use_gpu
#
python3 lpips_2dirs.py -d0 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"$VAR2"_"$VAR11" -d1 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"$VAR3"_128 -o ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/lpips_res/lp_res_alex_$VAR11.txt --use_gpu
#
# VAR1=0904_1
# VAR11=0904_2
# VAR2=h
# VAR3=horn04
# python3 lpips_2dirs.py -d0 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"$VAR2"_"$VAR1" -d1 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"$VAR3"_128 -o ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/lpips_res/lp_res_alex_$VAR1.txt --use_gpu
# #
# python3 lpips_2dirs.py -d0 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"$VAR2"_"$VAR11" -d1 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"$VAR3"_128 -o ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/lpips_res/lp_res_alex_$VAR11.txt --use_gpu
# #
# VAR1=0903_1
# VAR11=0903_2
# VAR2=b
# VAR3=bassoon01
# python3 lpips_2dirs.py -d0 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"$VAR2"_"$VAR1" -d1 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"$VAR3"_128 -o ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/lpips_res/lp_res_alex_$VAR1.txt --use_gpu
# #
# python3 lpips_2dirs.py -d0 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"$VAR2"_"$VAR11" -d1 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"$VAR3"_128 -o ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/lpips_res/lp_res_alex_$VAR11.txt --use_gpu
# #
# VAR1=0902_1
# VAR11=0902_2
# VAR2=f
# VAR3=flute01
# python3 lpips_2dirs.py -d0 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"$VAR2"_"$VAR1" -d1 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"$VAR3"_128 -o ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/lpips_res/lp_res_alex_$VAR1.txt --use_gpu
# #
# python3 lpips_2dirs.py -d0 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"$VAR2"_"$VAR11" -d1 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"$VAR3"_128 -o ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/lpips_res/lp_res_alex_$VAR11.txt --use_gpu
# #
# VAR1=0901_1
# VAR11=0901_2
# VAR2=s
# VAR3=sax01
# python3 lpips_2dirs.py -d0 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"$VAR2"_"$VAR1" -d1 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"$VAR3"_128 -o ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/lpips_res/lp_res_alex_$VAR1.txt --use_gpu
# #
# python3 lpips_2dirs.py -d0 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"$VAR2"_"$VAR11" -d1 ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"$VAR3"_128 -o ../../../../../mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/lpips_res/lp_res_alex_$VAR11.txt --use_gpu
# #