#!/usr/bin/env bash
set -xe

device=cpu
x64=false  #true
backend=edward  #numpyro/stan/pyro/edward
benchmark_dir=$( cd $(dirname "$0") ; pwd -P )


for seed in 1 2 3 4 5; do
    if [[ ${x64} = "true" ]]; then
        python ${benchmark_dir}/covtype.py --backend ${backend} \
            --device ${device} --seed ${seed} --x64 --disable-progbar
    else
        python ${benchmark_dir}/covtype.py --backend ${backend} \
            --device ${device} --seed ${seed} --disable-progbar
    fi
done
