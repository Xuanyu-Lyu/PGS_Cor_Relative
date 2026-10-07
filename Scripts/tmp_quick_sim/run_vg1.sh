export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SLURM_ARRAY_TASK_ID=1 python quick_vg1fix.py > logs/v1.log 2>&1 &
while [ ! -f out_vg1fix/conditions_config.csv ]; do sleep 1; done
for i in $(seq 2 12); do SLURM_ARRAY_TASK_ID=$i python quick_vg1fix.py > logs/v$i.log 2>&1 & done
wait
