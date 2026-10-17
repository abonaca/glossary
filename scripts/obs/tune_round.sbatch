#!/bin/bash
# One round of the joint LMC+Sgr tuning (scripts/v21_tune.py): task k builds the initial snapshot of
# run k (0 = base, 1-12 = one unknown nudged) and evolves it with gyrfalcON from t=-3 to 0, writing
# snapshots every 1/8 time unit. Needs TUNE/roundNN/round.json (v21_tune.new_round / finish_round).
# Submit:  sbatch --array=0-12 ~/projects/glossary/scripts/obs/tune_round.sbatch ROUND
#
#SBATCH --job-name=tune-round
#SBATCH --partition=obs
#SBATCH --nodelist=obs-2026-001
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%A_%a.log

set -euo pipefail
ROUND=${1:?round number}
RUN=${SLURM_ARRAY_TASK_ID:?submit as an array job}
D=$(printf "/carnegie/nobackup/users/abonaca/v21_sim/tune/round%02d/run%02d" "$ROUND" "$RUN")
say() { echo "[$(date +%H:%M:%S)] $*"; }

say "node $(hostname), job ${SLURM_ARRAY_JOB_ID}_${RUN}, CPUs $(grep Cpus_allowed_list /proc/self/status | awk '{print $2}')"
cd "$HOME/projects/glossary/scripts"
OMP_NUM_THREADS=1 "$HOME/projects/python/nbody-env/bin/python" -u -c "
import v21_tune as vt
vt.build_ics($ROUND, $RUN)
"
set +u; source "$HOME/projects/python/nemo/nemo_start.sh"; set -u
[ -e "$D/evol.nemo" ] && { say "$D/evol.nemo exists; move it away first"; exit 1; }
say "gyrfalcON round $ROUND run $RUN"
/usr/bin/time -v gyrfalcON "$D/ics.nemo" "$D/evol.nemo" tstop=0 step=0.125 logfile="$D/evol.log" \
    eps=0.1 kmax=8 Nlev=3 fea=0.2 2> "$D/evol.time"
grep -E "Maximum resident|Elapsed" "$D/evol.time"
say "done"
