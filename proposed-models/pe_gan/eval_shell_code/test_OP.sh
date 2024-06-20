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
        python3 gif_make_shell_OP.py -d "$VAR1" -i "$VAR2"
    done
exit