# wait for round-6 base run, then compute its endpoint in a small job on the same node
until [ "$(timeout 60 ssh obs 'sacct -j 286576_0 -X -n -o State' | tr -d ' ')" = "COMPLETED" ]; do
    s=$(timeout 60 ssh obs 'sacct -j 286576_0 -X -n -o State' | tr -d ' ')
    case "$s" in FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_ME*) echo "run 0 ended: $s"; exit 1;; esac
    sleep 300
done
J=$(timeout 60 ssh obs 'cd ~/projects/python/slurm_logs && sbatch --parsable -J r6-base-early -p obs -w obs-2026-001 -c 1 --mem=6G -t 0:30:00 -o %x-%j.log --wrap "cd ~/projects/glossary/scripts && ~/projects/python/nbody-env/bin/python -u -c \"
import json, os, numpy as np, v21_tune as vt
np.set_printoptions(precision=3, suppress=True, linewidth=200)
comps = json.load(open(vt.BASE_ICS + \\\".json\\\"))[\\\"components\\\"]
tr = vt.track_centres(os.path.join(vt.run_dir(6, 0), \\\"evol.nemo\\\"), comps, vt.run_theta(6, 0))
F = vt.endpoint(tr); r = F - vt.TARGET
print(\\\"endpoint\\\", F); print(\\\"residual\\\", r); print(\\\"resid/TOL\\\", r / vt.TOL)
\""')
echo "early job $J"
while timeout 60 ssh obs "squeue -h -j $J | grep -q ."; do sleep 60; done
timeout 60 ssh obs "sacct -j $J -X -n -o JobID,State,Start,Elapsed; tail -5 ~/projects/python/slurm_logs/r6-base-early-$J.log"
