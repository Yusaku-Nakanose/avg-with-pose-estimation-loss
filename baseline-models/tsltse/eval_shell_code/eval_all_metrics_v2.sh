#!/bin/bash
#sumika
#eval_all
#
#########################
settings=(
    '07_param_0point0 tp trumpet01'
    '10_param_0point0 vi violin06'
    '11_param_0point0 d double_bass01'
)


############################
#
#copy_image.sh
for ((i=0; ${#settings[*]}>$i; i++))
    do
        tmp=(${settings[$i]})
        VAR1=${tmp[0]}
        VAR2=${tmp[1]}
        VAR3=${tmp[2]}
        #
        #copy_image.sh
        mkdir /mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/results/gif_"$VAR2"_$VAR1/
        cp -r /mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/results/$VAR1/test_latest/images/*2_1.png /mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/results/gif_"$VAR2"_$VAR1/ 
        #
        #fid_dasu
        echo $VAR1 >> /mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/fid_res/val_res_all.txt
        python3 -m pytorch_fid /mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"$VAR3"/ /mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/results/gif_"$VAR2"_$VAR1 --batch-size 4 --num-workers 0 >> /mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/fid_res/val_res_all.txt
        #
        #lpips
        python3 lpips_2dirs.py -d0 /mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/results/gif_"$VAR2"_$VAR1 -d1 /mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"$VAR3"_128 -o /mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/lpips_res/lp_res_alex_$VAR1.txt --use_gpu
        #
        #gif_make
        python3 gif_make_shell.py -d "$VAR1" -i "$VAR2"
    done
exit