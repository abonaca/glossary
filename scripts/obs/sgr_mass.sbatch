#!/bin/bash
# Sgr mass within 3, 5, 10 kpc at t=0 in every run of a round and along its base run (tune_analysis/sgr_mass.py).
# Submit:  sbatch --array=11-15 ~/projects/glossary/scripts/obs/sgr_mass.sbatch   (task id = round)
#
#SBATCH --job-name=sgr-mass
#SBATCH --partition=obs
#SBATCH --nodelist=obs-2026-001
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%A_%a.log

set -euo pipefail
ROUND=${SLURM_ARRAY_TASK_ID:?array task id = round}
echo "[$(date +%H:%M:%S)] node $(hostname), job $SLURM_JOB_ID, round $ROUND"
set +u; source "$HOME/projects/python/nemo/nemo_start.sh"; set -u
cd "$HOME/projects/glossary/scripts/tune_analysis"
OMP_NUM_THREADS=1 "$HOME/projects/python/nbody-env/bin/python" -u -c "
import sgr_mass
sgr_mass.measure($ROUND)
"
echo "[$(date +%H:%M:%S)] done"
