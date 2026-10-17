#!/bin/bash
# After a tuning round: centres and endpoints of its 13 runs, the Gauss-Newton step, and round.json
# for the next round (scripts/v21_tune.py finish_round). Perturbed runs' snapshots go to scrap/tune.
# Submit:  sbatch --dependency=afterok:<array job> ~/projects/glossary/scripts/obs/tune_endpoints.sbatch ROUND [block]
#
#SBATCH --job-name=tune-endpoints
#SBATCH --partition=obs
#SBATCH --nodelist=obs-2026-001
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.log

set -euo pipefail
ROUND=${1:?round number}
BLOCK=$([ "${2:-}" = block ] && echo True || echo False)
echo "[$(date +%H:%M:%S)] node $(hostname), job $SLURM_JOB_ID, round $ROUND, block=$BLOCK"
cd "$HOME/projects/glossary/scripts"
"$HOME/projects/python/nbody-env/bin/python" -u -c "
import v21_tune as vt
vt.finish_round($ROUND, block=$BLOCK)
"
echo "[$(date +%H:%M:%S)] done"
