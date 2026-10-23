#!/bin/bash
# All-particle movie frames of the dense-output adopted run (r8_all_particles.adopted_frames): array task k of N
# draws a contiguous block of the 769 snapshots into ADOPTED/frames/.
# Submit:  sbatch --array=0-31%16 ~/projects/glossary/scripts/obs/adopted_frames.sbatch 32
#
#SBATCH --job-name=adopted-frames
#SBATCH --partition=obs
#SBATCH --nodelist=obs-2026-001
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=6:00:00
#SBATCH --output=%x-%A_%a.log

set -euo pipefail
NTASK=${1:?number of array tasks}
cd "$HOME/projects/glossary/scripts/tune_analysis"
OMP_NUM_THREADS=1 "$HOME/projects/python/nbody-env/bin/python" -u r8_all_particles.py adopted "$SLURM_ARRAY_TASK_ID" "$NTASK"
echo "done"
