#!/usr/bin/env bash
set -xe

device=cpu
x64=false  #true/false
backend=numpyro  #numpyro/stan
N=200
benchmark_dir=$( cd $(dirname "$0") ; pwd -P )


for seed in 1 2 3 4 5; do
    for P in 8 16 32 64 128 256 512; do
        if [[ ${x64} = "true" ]]; then
            python ${benchmark_dir}/sparse_regression.py --backend ${backend} --num-data ${N} \
                --num-dimensions ${P} --device ${device} --seed ${seed} --x64 --disable-progbar
        else
            python ${benchmark_dir}/sparse_regression.py --backend ${backend} --num-data ${N} \
                --num-dimensions ${P} --device ${device} --seed ${seed} --disable-progbar
        fi
    done
done
