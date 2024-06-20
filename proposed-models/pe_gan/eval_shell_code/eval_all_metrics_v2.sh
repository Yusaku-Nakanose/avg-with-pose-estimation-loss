#!/bin/bash
#sumika
#eval_all
#
#########################
<< COMMENTOUT
settings=(
    '04_0 h horn04'
    '04_1 h horn04'
    '04_2 h horn04'
    '05_0 cl clarinet01'
    '05_1 cl clarinet01'
    '05_2 cl clarinet01'
    '06_0 c cello01'
    '06_1 c cello01'
    '06_2 c cello01'
    '08_0 o oboe04'
    '08_1 o oboe04'
    '08_2 o oboe04'
    '09_0 tb trombone02'
    '09_1 tb trombone02'
    '09_2 tb trombone02'
    '12_0 tu tuba04'
    '12_1 tu tuba04'
    '12_2 tu tuba04'
    '13_0 vl viola01'
    '13_1 vl viola01'
    '13_2 vl viola01'
)
COMMENTOUT
settings=(
    '01_0 s sax01'

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
        cp -r /mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/results/$VAR1/test_latest/images/*2.png /mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/results/gif_"$VAR2"_$VAR1/ 
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