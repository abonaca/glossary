#!/bin/bash
# Time gyrfalcON on one of the V21 rerun initial snapshots (default R2 = MW+LMC+Sgr, 6.4e6
# particles) for 4 of the longest block steps, with V21's run_sim.sh settings, and record its
# peak memory. Same measurement as v21_ics.time_gyrfalcon on the laptop (R2: 493 s there).
#
# Needs: install_stack.sbatch done; the snapshot rsynced to $RUNS on OBS.
# Submit:  sbatch ~/projects/glossary/scripts/obs/time_gyrfalcon.sbatch [R0|R1|R2|R3]
#
#SBATCH --job-name=gyrfalcon-timing
#SBATCH --partition=obs
#SBATCH --nodelist=obs-2026-001
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.log

set -euo pipefail
RUN=${1:-R2}
RUNS=/carnegie/nobackup/users/abonaca/v21_sim/ics/runs_mw_triax_full
NEMO_DIR=$HOME/projects/python/nemo
say() { echo "[$(date +%H:%M:%S)] $*"; }

say "node $(hostname), job $SLURM_JOB_ID, CPUs $(grep Cpus_allowed_list /proc/self/status | awk '{print $2}')"
lscpu | grep -E "Model name|MHz"
set +u; source "$NEMO_DIR/nemo_start.sh"; set -u

snap=$RUNS/$RUN.nemo
t0=$(tsf "$snap" | grep -m1 'double Time' | awk '{print $NF}')
# kmax=8, Nlev=3: longest step 2^-6; 4 of them
tstop=$(python3 -c "print($t0 + 4 * 0.5**6)")
out=$RUNS/${RUN}_obs_timing
rm -f "$out.log" "$out.out"
say "$RUN: t=$t0 -> $tstop"
start=$(date +%s.%N)
/usr/bin/time -v gyrfalcON "$snap" "$out.out" tstop=$tstop step=$(python3 -c "print($tstop - $t0)") \
    startout=f logfile="$out.log" eps=0.1 kmax=8 Nlev=3 fea=0.2 2> "$out.time"
wall=$(python3 -c "print($(date +%s.%N) - $start)")
grep -E "Maximum resident|Elapsed" "$out.time"
python3 -c "
w = $wall; per_unit = w / (4 * 0.5**6)
print('%s: %.0f s for 4 long steps (laptop: R0 500, R1 476, R2 493, R3 362 s)' % ('$RUN', w))
print('    %.0f s per time unit, %.2f h per Gyr, %.1f h for 3 time units' % (per_unit, per_unit / 0.9778 / 3600, per_unit * 3 / 3600))
"
rm -f "$out.out"
