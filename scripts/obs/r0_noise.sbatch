#!/bin/bash
# Noise floor and drift of the R0 (MW alone) expansions vs V21's LMC signal (scripts/r0_noise.py).
# Submit after (or with --dependency on) the R0 run:
#   sbatch [--dependency=afterok:<R0 job>] ~/projects/glossary/scripts/obs/r0_noise.sbatch [EVERY]
# Output: $RUN_DIR/r0_noise.npz and ~/projects/glossary/plots/r0_noise.png
#
#SBATCH --job-name=r0-noise
#SBATCH --partition=obs
#SBATCH --nodelist=obs-2026-001
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.log

set -euo pipefail
EVERY=${1:-4}
RUN_DIR=/carnegie/nobackup/users/abonaca/v21_sim/runs/R0
REPO=$HOME/projects/glossary
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "[$(date +%H:%M:%S)] node $(hostname), job $SLURM_JOB_ID, $OMP_NUM_THREADS threads"
cd "$REPO/scripts"
MPLBACKEND=Agg "$HOME/projects/python/nbody-env/bin/python" -u -c "
import numpy as np, r0_noise as rn
res = rn.analyse_r0('$RUN_DIR/R0_evol.nemo', every=$EVERY)
np.savez('$RUN_DIR/r0_noise.npz', **res)
rn.plot_noise(res)
"
echo "[$(date +%H:%M:%S)] done"
