export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SLURM_ARRAY_TASK_ID=1 python quick_f11fix2.py > logs/w1.log 2>&1 &
while [ ! -f out_f11free/conditions_config.csv ]; do sleep 1; done
seq 2 12 | xargs -P 2 -I{} sh -c 'SLURM_ARRAY_TASK_ID={} python quick_f11fix2.py > logs/w{}.log 2>&1'
wait
