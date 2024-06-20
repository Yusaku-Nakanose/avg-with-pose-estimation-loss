#!/bin/bash
#sumika
#eval_all
#
#########################
<< COMMENTOUT
settings=(
    '01_op_3 s sax01'
    '02_op_3 f flute01'
    '03_op_3 b bassoon01'
    '04_op_3 h horn04'
    '05_op_3 cl clarinet01'
    '06_op_3 c cello01'
    '07_op_3 tp trumpet01'
    '08_op_3 o oboe04'
    '09_op_3 tb trombone02'
    '10_op_3 vi violin06'
    '11_op_3 d double_bass01'
    '12_op_3 tu tuba04'
    '13_op_3 vl viola01'
)
COMMENTOUT
settings=(
    '04_op_3_2 h horn04'
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