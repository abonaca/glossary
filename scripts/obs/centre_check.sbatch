#!/bin/bash
# t=0 centres of every run of a tuning round with several velocity apertures (scripts/v21_centre_check.py), plus a
# bootstrap of the base run -> TUNE/roundNN/centre_check.npz. One round per array task.
# Submit:  sbatch --array=0-16 ~/projects/glossary/scripts/obs/centre_check.sbatch [NBOOT]   (task id = round)
#
#SBATCH --job-name=centre-check
#SBATCH --partition=obs
#SBATCH --nodelist=obs-2026-001
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%A_%a.log

set -euo pipefail
ROUND=${SLURM_ARRAY_TASK_ID:?array task id = round}
NBOOT=${1:-20}
echo "[$(date +%H:%M:%S)] node $(hostname), job $SLURM_JOB_ID, round $ROUND, nboot $NBOOT"
set +u; source "$HOME/projects/python/nemo/nemo_start.sh"; set -u
cd "$HOME/projects/glossary/scripts"
OMP_NUM_THREADS=1 "$HOME/projects/python/nbody-env/bin/python" -u -c "
import v21_centre_check as cc
cc.measure_round($ROUND, nboot=$NBOOT)
"
echo "[$(date +%H:%M:%S)] done"
