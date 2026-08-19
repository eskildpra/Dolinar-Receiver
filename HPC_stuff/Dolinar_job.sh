#!/bin/sh
#BSUB -q hpc
### -- Job Array of 48 independent tasks --
#BSUB -J dolinar_sweep[7-12]
### -- Request 4 CPU cores per sub-job (48 * 4 = 192 cores total) --
#BSUB -n 16
#BSUB -R "span[hosts=1]"
### -- Walltime: 12 Hours --
#BSUB -W 12:00
### -- 2GB RAM per core (8GB total per 4-core task) --
#BSUB -R "rusage[mem=2GB]"
### -- Notifications --
#BSUB -u s234463@student.dtu.dk
#BSUB -B
#BSUB -N
### -- Per-Job Log Routing --
#BSUB -o logs/run_%I_%J.out
#BSUB -e logs/run_%I_%J.err

mkdir -p logs

# Load environment
source .venv/bin/activate

# Run worker with 4 cores
python3 Dolinar_Reciever_Sweep.py $LSB_JOBINDEX