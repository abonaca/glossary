#!/bin/bash
# Rerun an adopted tuning run with dense output: copies the run's ics.nemo (byte-identical initial conditions) to
# OUT and evolves it with the same gyrfalcON settings as tune_round.sbatch, but writing a snapshot every STEP
# (default 1/256 = the largest gyrfalcON step, 2^-8 time units = 3.82 Myr; 769 snapshots, ~138 GB).
# Submit:  sbatch ~/projects/glossary/scripts/obs/adopted_run.sbatch ROUND RUN [STEP]
#          e.g. sbatch ... 9 2   -> /carnegie/nobackup/users/abonaca/v21_sim/adopted/r09run02
#
#SBATCH --job-name=adopted-run
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
V=/carnegie/nobackup/users/abonaca/v21_sim
SRC=$(printf "%s/tune/round%02d/run%02d" "$V" "$ROUND" "$RUN")
D=$(printf "%s/adopted/r%02drun%02d" "$V" "$ROUND" "$RUN")
say() { echo "[$(date +%H:%M:%S)] $*"; }

say "node $(hostname), job ${SLURM_JOB_ID}, source $SRC, output $D, step $STEP"
mkdir -p "$D"
[ -e "$D/evol.nemo" ] && { say "$D/evol.nemo exists; move it away first"; exit 1; }
cp -p "$SRC/ics.nemo" "$D/ics.nemo"
md5sum "$SRC/ics.nemo" "$D/ics.nemo" | tee "$D/ics.md5"
set +u; source "$HOME/projects/python/nemo/nemo_start.sh"; set -u
say "gyrfalcON round $ROUND run $RUN, step $STEP"
/usr/bin/time -v gyrfalcON "$D/ics.nemo" "$D/evol.nemo" tstop=0 step=$STEP logfile="$D/evol.log" \
    eps=0.1 kmax=8 Nlev=3 fea=0.2 2> "$D/evol.time"
grep -E "Maximum resident|Elapsed" "$D/evol.time"
say "done"
