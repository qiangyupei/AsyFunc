#!/bin/bash
# nohup bash ./profiler.sh <$modelName> > log_<modelName>.txt 2>&1 &
for ((core_num = 2; core_num <= 16; core_num+=2))
do
    for ((batch_size=1; batch_size <= 8; batch_size++))
    do  
        python ./profiler.py ./model_weight/$1_layers/ $core_num $batch_size
    done
done