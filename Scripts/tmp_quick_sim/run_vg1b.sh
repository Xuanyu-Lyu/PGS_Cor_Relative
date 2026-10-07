export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for i in $(seq 1 12); do case "  " in *" $i "*) ;; *) echo $i;; esac; done | xargs -P 3 -I{} sh -c 'SLURM_ARRAY_TASK_ID={} python quick_vg1fix.py > logs/v{}.log 2>&1'
