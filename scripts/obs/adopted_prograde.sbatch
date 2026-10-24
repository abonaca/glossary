#!/bin/bash
# Adopted run with the MW stars' velocities reversed (disc rotating as in the MW, mw_prograde.build), evolved with
# the same gyrfalcON settings and dense output as adopted_run.sbatch (snapshot every STEP, default 1/256).
# Submit:  sbatch ~/projects/glossary/scripts/obs/adopted_prograde.sbatch ROUND RUN [STEP]
#          e.g. sbatch ... 9 2   -> /carnegie/nobackup/users/abonaca/v21_sim/adopted/r09run02_prograde
#
#SBATCH --job-name=adopted-prograde
#SBATCH --partition=obs
#SBATCH --nodelist=obs-2026-001
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.log

set -euo pipefail
ROUND=${1:?round number}
RUN=${2:?run number}
STEP=${3:-0.00390625}
D=$(printf "/carnegie/nobackup/users/abonaca/v21_sim/adopted/r%02drun%02d_prograde" "$ROUND" "$RUN")
say() { echo "[$(date +%H:%M:%S)] $*"; }

say "node $(hostname), job ${SLURM_JOB_ID}, output $D, step $STEP"
[ -e "$D/evol.nemo" ] && { say "$D/evol.nemo exists; move it away first"; exit 1; }
cd "$HOME/projects/glossary/scripts/tune_analysis"
OMP_NUM_THREADS=1 "$HOME/projects/python/nbody-env/bin/python" -u mw_prograde.py "$ROUND" "$RUN"
md5sum "$D/ics.nemo" | tee "$D/ics.md5"
set +u; source "$HOME/projects/python/nemo/nemo_start.sh"; set -u
say "gyrfalcON round $ROUND run $RUN, MW stars reversed, step $STEP"
/usr/bin/time -v gyrfalcON "$D/ics.nemo" "$D/evol.nemo" tstop=0 step=$STEP logfile="$D/evol.log" \
    eps=0.1 kmax=8 Nlev=3 fea=0.2 2> "$D/evol.time"
grep -E "Maximum resident|Elapsed" "$D/evol.time"
say "done"
