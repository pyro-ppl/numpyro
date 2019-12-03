#!/usr/bin/env bash
set -xe

device=cpu
backend=stan  #numpyro/stan/pyro
benchmark_dir=$( cd $(dirname "$0") ; pwd -P )


for seed in 1 2 3 4 5; do
    if [[ ${device} = "cpu" ]]; then
        python ${benchmark_dir}/hmm.py --backend ${backend} \
            --device ${device} --seed ${seed} --x64 --disable-progbar
    else
        python ${benchmark_dir}/hmm.py --backend ${backend} \
            --device ${device} --seed ${seed} --disable-progbar
    fi
done
