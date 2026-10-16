#!/bin/bash
# Evolve one of the V21 rerun initial snapshots (R0-R3) with gyrfalcON, V21's run_sim.sh settings,
# writing full snapshots every STEP time units into one multi-snapshot NEMO file.
#
# Submit:  sbatch ~/projects/glossary/scripts/obs/run_gyrfalcon.sbatch R0 [STEP] [TSTOP]
#   STEP  default 0.03125 (1/32 time unit = 30.6 Myr);  TSTOP default 0 (present day)
# Output:  $OUT/<run>/<run>_evol.nemo (+ gyrfalcON log); extract times with `s2s ... times=`.
#
#SBATCH --job-name=gyrfalcon-run
#SBATCH --partition=obs
#SBATCH --nodelist=obs-2026-001
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.log

set -euo pipefail
RUN=${1:?run name (R0-R3)}
STEP=${2:-0.03125}
TSTOP=${3:-0}
ICS=/carnegie/nobackup/users/abonaca/v21_sim/ics/runs_mw_triax_full
OUT=/carnegie/nobackup/users/abonaca/v21_sim/runs
NEMO_DIR=$HOME/projects/python/nemo
say() { echo "[$(date +%H:%M:%S)] $*"; }

d=$OUT/$RUN
mkdir -p "$d"
[ -e "$d/${RUN}_evol.nemo" ] && { say "$d/${RUN}_evol.nemo exists; move it away first"; exit 1; }
say "node $(hostname), job $SLURM_JOB_ID, CPUs $(grep Cpus_allowed_list /proc/self/status | awk '{print $2}')"
set +u; source "$NEMO_DIR/nemo_start.sh"; set -u
cp "$ICS/$RUN.json" "$d/"
say "$RUN: $(tsf "$ICS/$RUN.nemo" | grep -m1 'double Time' | awk '{print $NF}') -> $TSTOP, snapshots every $STEP"
/usr/bin/time -v gyrfalcON "$ICS/$RUN.nemo" "$d/${RUN}_evol.nemo" tstop=$TSTOP step=$STEP \
    logfile="$d/${RUN}_evol.log" eps=0.1 kmax=8 Nlev=3 fea=0.2 2> "$d/${RUN}_evol.time"
grep -E "Maximum resident|Elapsed" "$d/${RUN}_evol.time"
say "done"
