#!/bin/bash
# Build the N-body stack (NEMO + gyrfalcON, Agama + plugin, Python requirements) on the OBS
# Zen 5c node. Both falcON and Agama compile with -march=native, so they must be built on the
# node type the simulations run on; this job pins itself to obs-2026-001.
#
# Before submitting, from the laptop:   rsync the repo to ~/projects/glossary on OBS
# Submit on the login node:             sbatch ~/projects/glossary/scripts/obs/install_stack.sbatch
#
#SBATCH --job-name=nbody-install
#SBATCH --partition=obs
#SBATCH --nodelist=obs-2026-001
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.log

set -euo pipefail
REPO=$HOME/projects/glossary
export PREFIX=$HOME/projects/python                 # NEMO, Agama sources and build logs
VENV=$PREFIX/nbody-env                               # Python 3.11 venv for the whole stack
export PYTHON=$VENV/bin/python
say() { echo "[$(date +%H:%M:%S)] $*"; }

say "node $(hostname), job $SLURM_JOB_ID"
lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core|Socket|MHz"
cpus=$(grep Cpus_allowed_list /proc/self/status | awk '{print $2}')
say "allocated CPUs: $cpus"
# are we given whole cores (both hyperthreads of each) or single threads?
for r in ${cpus//,/ }; do
    for c in $(seq ${r%-*} ${r#*-}); do
        echo "    cpu$c siblings: $(cat /sys/devices/system/cpu/cpu$c/topology/thread_siblings_list)"
    done
done
gcc --version | head -1

say "network from the compute node"
net=1
for url in https://github.com https://pypi.org/simple/numpy/ https://gitlab.com https://raw.githubusercontent.com; do
    if curl -sS -o /dev/null --max-time 15 "$url"; then echo "    ok   $url"; else echo "    FAIL $url"; net=0; fi
done
[ $net = 1 ] || { say "no network on the compute node: stage sources on the login node instead"; exit 2; }

# Lmod is not always initialised in batch shells; its scripts also read unset variables
set +u
type module >/dev/null 2>&1 || source /usr/share/lmod/lmod/init/bash
module load python/3.11_conda
set -u
if [ ! -x "$PYTHON" ]; then
    say "creating venv $VENV from $(which python)"
    python -m venv "$VENV"
    "$PYTHON" -m pip install -q --upgrade pip
fi

say "running install_nbody_stack.sh (all steps)"
"$REPO/scripts/install_nbody_stack.sh"
say "install job finished"
