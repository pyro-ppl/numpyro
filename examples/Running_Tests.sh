#!/bin/sh
#python logistic_hmcecs.py -num_samples 100 -num_warmup 50 -ecs_algo NUTS -algo NUTS -map_init NUTS &
#python logistic_hmcecs.py -num_samples 100 -num_warmup 50 -ecs_algo NUTS -algo NUTS -map_init HMC &
#python logistic_hmcecs.py -num_samples 100 -num_warmup 50 -ecs_algo NUTS -algo NUTS -map_init SVI & #Slow, wrong number of epochs,repeat

echo NUTS,HMC,NUTS
python logistic_hmcecs.py -num_samples 100 -num_warmup 50 -ecs_algo NUTS -algo HMC -map_init NUTS &
echo NUTS,HMC,HMC
#python logistic_hmcecs.py -num_samples 100 -num_warmup 50 -ecs_algo NUTS -algo HMC -map_init HMC &
echo NUTS,HMC,SVI
#python logistic_hmcecs.py -num_samples 100 -num_warmup 50 -ecs_algo NUTS -algo HMC -map_init SVI &

echo HMC,NUTS,NUTS
python logistic_hmcecs.py -num_samples 100 -num_warmup 50 -ecs_algo HMC -algo NUTS -map_init NUTS &
echo HMC,NUTS,HMC
python logistic_hmcecs.py -num_samples 100 -num_warmup 50 -ecs_algo HMC -algo NUTS -map_init HMC &
echo HMC,NUTS,SVI
python logistic_hmcecs.py -num_samples 100 -num_warmup 50 -ecs_algo HMC -algo NUTS -map_init SVI &

echo HMC,HMC,NUTS
python logistic_hmcecs.py -num_samples 100 -num_warmup 50 -ecs_algo HMC -algo HMC -map_init NUTS &
echo HMC,HMC,HMC
python logistic_hmcecs.py -num_samples 100 -num_warmup 50 -ecs_algo HMC -algo HMC -map_init HMC &
echo HMC,HMC,SVI
python logistic_hmcecs.py -num_samples 100 -num_warmup 50 -ecs_algo HMC -algo HMC -map_init SVI &
