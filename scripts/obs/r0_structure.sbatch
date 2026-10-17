#!/bin/bash
# R0 stellar density maps (every ~0.5 Gyr) and disk/bulge/halo coefficient time series
# (every snapshot), scripts/r0_structure.py.
# Submit:  sbatch ~/projects/glossary/scripts/obs/r0_structure.sbatch
# Output:  $RUN_DIR/r0_structure.pkl and ~/projects/glossary/plots/r0_{stellar_maps,coefs_*}.png
#
#SBATCH --job-name=r0-structure
#SBATCH --partition=obs
#SBATCH --nodelist=obs-2026-001
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.log

set -euo pipefail
RUN_DIR=/carnegie/nobackup/users/abonaca/v21_sim/runs/R0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "[$(date +%H:%M:%S)] node $(hostname), job $SLURM_JOB_ID, $OMP_NUM_THREADS threads"
cd "$HOME/projects/glossary/scripts"
MPLBACKEND=Agg "$HOME/projects/python/nbody-env/bin/python" -u -c "
import pickle, r0_structure as rs
maps = rs.stellar_maps('$RUN_DIR/R0_evol.nemo')
rs.plot_maps(maps)
coefs = rs.coefficient_series('$RUN_DIR/R0_evol.nemo')
pickle.dump(dict(maps=maps, coefs=coefs), open('$RUN_DIR/r0_structure.pkl', 'wb'))
for c in ('disk', 'bulge', 'halo'):
    rs.plot_coefficients(coefs, c)
"
echo "[$(date +%H:%M:%S)] done"
