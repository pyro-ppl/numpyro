#!/usr/bin/env bash
set -xe

device=cpu
N=100
benchmark_dir=$( cd $(dirname "$0") ; pwd -P )

for backend in numpyro stan; do
    for seed in 1 2 3 4 5; do
        for P in 10 20 30 40 50 60 70 80; do
            if [[ ${device} = "cpu" ]]; then
                python ${benchmark_dir}/sparse_regression.py --backend ${backend} --num-data ${N} \
                    --num-dimensions ${P} --device ${device} --seed ${seed} --x64 #--disable-progbar
            else
                python ${benchmark_dir}/sparse_regression.py --backend ${backend} --num-data ${N} \
                    --num-dimensions ${P} --device ${device} --seed ${seed} #--disable-progbar
            fi
        done
    done
done
