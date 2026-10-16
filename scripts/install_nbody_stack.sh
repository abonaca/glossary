#!/usr/bin/env bash
# Pinned install of the N-body stack used to rerun the Vasiliev, Belokurov & Erkal (2021)
# MW+LMC+Sgr simulations: NEMO + gyrfalcON, Agama (with CVXOPT, GLPK, UNSIO) and the
# Agama acceleration plugin for gyrfalcON.
#
# Usage:
#   PREFIX=~/software PYTHON=~/venvs/nbody/bin/python scripts/install_nbody_stack.sh [step ...]
# Steps (run in this order; default is all of them):
#   check   - verify compilers and system libraries
#   tirpc   - build libtirpc locally, only needed when the system has no RPC headers (no sudo)
#   nemo    - clone, configure and build NEMO and falcON
#   python  - install requirements_nbody.txt (numpy, scipy, ..., cvxopt) into $PYTHON's environment
#   agama   - clone Agama + UNSIO, build, install into $PYTHON's environment, copy the NEMO plugin
#   verify  - solveOpt test and a gyrfalcON+Agama plugin vs agama.orbit comparison
#   versions- write $PREFIX/nbody_stack_versions.txt
# Optional step (not run by default):
#   pyanalysis - install requirements_analysis.txt (the jax stack used by agama_jax.py)
#
# Prerequisites:
# - A Python (tested: 3.11) with pip and its C headers (Python.h; pyenv builds include them,
#   system pythons need python3-devel), passed as PYTHON=. Best a fresh venv.
#   Python >= 3.12 is untested (Agama's setup.py relies on setuptools' distutils shim).
# - System packages (openSUSE names): gcc gcc-c++ gcc-fortran make git unzip curl tar bzip2
#   tcsh gsl-devel libtirpc-devel; optional: glpk-devel cmake hdf5-devel.
# - Network access (GitHub, PyPI, GitLab for Eigen, SourceForge for the libtirpc fallback).
#
# Notes:
# - Agama compiles with -march=native: build on the node type the jobs will run on.
# - NEMO's top-level Makefile runs `make -j` with an unlimited job count, which corrupts
#   libnemo.a (parallel `ar` on one archive). We always build it with JOBS=1.
# - NEMO's Fortran test programs fail with gfortran < 10 (-fallow-argument-mismatch); harmless.
# - agama.readSnapshot() on a multi-snapshot NEMO file returns only the first snapshot;
#   extract times with `s2s in out times=...` first.

set -euo pipefail

# ---- pinned versions ---------------------------------------------------------------------
NEMO_REPO=https://github.com/teuben/nemo.git
NEMO_COMMIT=5a9b5d82bd36dd6fa597be8c1f36a81f91eb8e91       # 2026-09-24
AGAMA_REPO=https://github.com/GalacticDynamics-Oxford/Agama.git
AGAMA_COMMIT=f392b9d6e59269d451b37b3ada2508a44bd3a80c      # 2025-05-21 (== Agama-master.zip used locally)
UNSIO_REPO=https://github.com/GalacticDynamics-Oxford/unsio.git
UNSIO_COMMIT=e3cf2e6e3c5c42bca06f19fe73de7cdfb796e155
CVXOPT_VERSION=1.3.3                # must match requirements_nbody.txt (C headers for Agama)
TIRPC_VERSION=1.3.4
# Eigen 3.3.9 is downloaded by Agama's setup.py from a fixed URL (pinned upstream).

PREFIX=${PREFIX:-$HOME/projects/python}
PYTHON=${PYTHON:-python}
NEMO_DIR=${NEMO_DIR:-$PREFIX/nemo}
AGAMA_DIR=${AGAMA_DIR:-$PREFIX/agama}
LOGDIR=${LOGDIR:-$PREFIX/nbody_stack_logs}
TIRPC_PREFIX=${TIRPC_PREFIX:-$PREFIX/tirpc}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

mkdir -p "$PREFIX" "$LOGDIR"
say()  { echo "[$(date +%H:%M:%S)] $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }
# nemo_start.sh reads unset variables (e.g. PKG_CONFIG_PATH), so source it without `set -u`
nemo_env() { set +u; source "$NEMO_DIR/nemo_start.sh"; set -u; }

# locate RPC/XDR headers (glibc >= 2.32 no longer ships them); sets TIRPC_CPATH/TIRPC_LIB
find_tirpc() {
    TIRPC_CPATH=""; TIRPC_LIB=""
    for inc in /usr/include/tirpc /usr/include "$TIRPC_PREFIX/include/tirpc"; do
        if [ -f "$inc/rpc/rpc.h" ]; then TIRPC_CPATH=$inc; break; fi
    done
    for lib in /usr/lib64 /usr/lib /usr/lib/x86_64-linux-gnu "$TIRPC_PREFIX/lib64" "$TIRPC_PREFIX/lib"; do
        if ls "$lib"/libtirpc.so* >/dev/null 2>&1; then TIRPC_LIB=$lib; break; fi
    done
}

# gcc honours CPATH/LIBRARY_PATH/LD_RUN_PATH, so a local libtirpc needs no makefile patches
tirpc_env() {
    find_tirpc
    case "$TIRPC_CPATH" in
        "$TIRPC_PREFIX"*) export CPATH="$TIRPC_CPATH${CPATH:+:$CPATH}"
                          export LIBRARY_PATH="$TIRPC_LIB${LIBRARY_PATH:+:$LIBRARY_PATH}"
                          export LD_RUN_PATH="$TIRPC_LIB${LD_RUN_PATH:+:$LD_RUN_PATH}" ;;
        /usr/include/tirpc) export CPATH="$TIRPC_CPATH${CPATH:+:$CPATH}" ;;
    esac
}

step_check() {
    say "checking prerequisites"
    local missing=""
    for t in gcc g++ gfortran make git unzip curl tar bzip2; do
        command -v $t >/dev/null || missing="$missing $t"
    done
    command -v csh >/dev/null || command -v tcsh >/dev/null || missing="$missing csh/tcsh"
    [ -z "$missing" ] || die "missing tools:$missing"
    command -v gsl-config >/dev/null || [ -f /usr/include/gsl/gsl_math.h ] || die "GSL headers not found (gsl-devel)"
    "$PYTHON" -m pip --version >/dev/null 2>&1 || die "$PYTHON has no pip"
    "$PYTHON" -c "import sysconfig, os, sys; sys.exit(not os.path.isfile(os.path.join(sysconfig.get_paths()['include'], 'Python.h')))" \
        || die "$PYTHON has no C headers (Python.h); install python3-devel or use a pyenv build"
    "$PYTHON" -c "import sys; sys.exit(sys.version_info >= (3, 12))" \
        || say "WARNING: Python >= 3.12 is untested with Agama's setup.py (tested: 3.11)"
    find_tirpc
    if [ -z "$TIRPC_CPATH" ] || [ -z "$TIRPC_LIB" ]; then
        say "RPC headers/libtirpc not found: run the 'tirpc' step (or install libtirpc-devel)"
    else
        say "RPC headers: $TIRPC_CPATH/rpc/rpc.h, libtirpc in $TIRPC_LIB"
    fi
    say "gcc $(gcc -dumpversion), gfortran $(gfortran -dumpversion), python $("$PYTHON" -c 'import sys;print(sys.version.split()[0])')"
}

step_tirpc() {
    find_tirpc
    if [ -n "$TIRPC_CPATH" ] && [ -n "$TIRPC_LIB" ]; then say "libtirpc already available, skipping"; return; fi
    say "building libtirpc $TIRPC_VERSION into $TIRPC_PREFIX"
    local src=$PREFIX/src_tirpc
    mkdir -p "$src"
    curl -sSfL -o "$src/libtirpc-$TIRPC_VERSION.tar.bz2" \
        "https://downloads.sourceforge.net/project/libtirpc/libtirpc/$TIRPC_VERSION/libtirpc-$TIRPC_VERSION.tar.bz2"
    tar xjf "$src/libtirpc-$TIRPC_VERSION.tar.bz2" -C "$src"
    (cd "$src/libtirpc-$TIRPC_VERSION" && ./configure --prefix="$TIRPC_PREFIX" --disable-gssapi \
        && make -j4 && make install) > "$LOGDIR/tirpc.log" 2>&1
    find_tirpc
    [ -n "$TIRPC_CPATH" ] || die "libtirpc build failed, see $LOGDIR/tirpc.log"
}

step_nemo() {
    [ -e "$NEMO_DIR" ] && die "$NEMO_DIR exists; move it away first"
    tirpc_env
    say "cloning NEMO $NEMO_COMMIT"
    git clone -q "$NEMO_REPO" "$NEMO_DIR"
    (cd "$NEMO_DIR" && git checkout -q "$NEMO_COMMIT")
    say "configuring NEMO"
    (cd "$NEMO_DIR" && ./configure --with-yapp=ps) > "$LOGDIR/nemo_configure.log" 2>&1
    say "building NEMO serially (JOBS=1); this takes a while"
    (cd "$NEMO_DIR" && make build JOBS=1) > "$LOGDIR/nemo_build.log" 2>&1
    # falcON only links libtirpc when headers sit in /usr/include/tirpc; link it explicitly
    say "rebuilding falcON linked against libtirpc"
    find_tirpc
    local lutil="-L../utils/lib -lWDutils"
    [ -n "$TIRPC_LIB" ] && lutil="$lutil -L$TIRPC_LIB -ltirpc"
    (cd "$NEMO_DIR" && nemo_env && cd usr/dehnen && make -i clean all LUTIL="$lutil") \
        > "$LOGDIR/falcon_build.log" 2>&1
    (nemo_env
     for exe in gyrfalcON s2s s2a snapscale snapshift mkplummer; do
         command -v $exe >/dev/null || die "NEMO build incomplete: $exe missing (see $LOGDIR)"
     done
     # no `grep -q` here: with pipefail its early exit can SIGPIPE nm and fail the check
     [ "$(nm "$NEMO_DIR/lib/libnemo.a" 2>/dev/null | grep -c " T get_acceleration")" -ge 1 ] \
         || die "libnemo.a incomplete")
    say "NEMO ok: source $NEMO_DIR/nemo_start.sh"
}

step_python() {
    say "installing requirements_nbody.txt into $("$PYTHON" -c 'import sys;print(sys.prefix)')"
    "$PYTHON" -m pip install -q -r "$SCRIPT_DIR/requirements_nbody.txt"
    "$PYTHON" -c "import cvxopt, sys; sys.exit(cvxopt.__version__ != '$CVXOPT_VERSION')" \
        || die "cvxopt version differs from CVXOPT_VERSION=$CVXOPT_VERSION"
}

step_pyanalysis() {
    say "installing requirements_analysis.txt (jax stack) into $("$PYTHON" -c 'import sys;print(sys.prefix)')"
    "$PYTHON" -m pip install -q -r "$SCRIPT_DIR/requirements_analysis.txt"
    "$PYTHON" -c "import jax, equinox, diffrax, interpax; print('jax', jax.__version__, jax.devices())"
}

step_agama() {
    [ -e "$AGAMA_DIR" ] && die "$AGAMA_DIR exists; move it away first"
    [ -f "$NEMO_DIR/nemo_start.sh" ] || die "build NEMO first (the Agama build installs the gyrfalcON plugin)"
    say "cloning Agama $AGAMA_COMMIT"
    git clone -q "$AGAMA_REPO" "$AGAMA_DIR"
    (cd "$AGAMA_DIR" && git checkout -q "$AGAMA_COMMIT")
    local extras=$AGAMA_DIR/extras
    mkdir -p "$extras/include" "$extras/lib"
    say "building UNSIO $UNSIO_COMMIT (as Agama's setup.py would, but pinned)"
    git clone -q "$UNSIO_REPO" "$extras/unsio-src"
    (cd "$extras/unsio-src" && git checkout -q "$UNSIO_COMMIT" && make) > "$LOGDIR/unsio_build.log" 2>&1
    for h in componentrange.h ctools.h snapshotinterface.h uns.h userselection.h; do
        cp "$extras/unsio-src/unsio/$h" "$extras/include/"
    done
    cp "$extras/unsio-src/libnemo.a" "$extras/unsio-src/libunsio.a" "$extras/lib/"
    say "fetching CVXOPT $CVXOPT_VERSION C headers"
    for h in cvxopt.h blas_redefines.h; do
        curl -sSfL -o "$extras/include/$h" "https://raw.githubusercontent.com/cvxopt/cvxopt/$CVXOPT_VERSION/src/C/$h"
    done
    say "building and installing Agama (setup.py install --yes)"
    (nemo_env && cd "$AGAMA_DIR" && \
        CFLAGS="-I$extras/include" LDFLAGS="-L$extras/lib" "$PYTHON" setup.py install --yes) \
        > "$LOGDIR/agama_build.log" 2>&1
    local flags
    flags=$(grep -o -E -- "-DHAVE_[A-Z]+" "$AGAMA_DIR/Makefile.local" | sort -u | tr '\n' ' ')
    say "Agama features: $flags"
    for f in CVXOPT EIGEN UNSIO; do
        [[ "$flags" == *"HAVE_$f"* ]] || die "Agama built without $f (see $LOGDIR/agama_build.log)"
    done
    (nemo_env; [ -f "$NEMOOBJ/acc/agama.so" ] || cp "$AGAMA_DIR/agama.so" "$NEMOOBJ/acc/agama.so")
}

step_verify() {
    say "verifying Agama optimisation solver"
    (cd "$LOGDIR" && "$PYTHON" -c "
import agama, numpy as np
np.random.seed(1); A = np.random.rand(3, 5); b = A @ np.ones(5)
for kw in (dict(xpenq=np.ones(5)*0.1), dict(xpenl=np.ones(5))):
    x = agama.solveOpt(matrix=A, rhs=b, **kw)
    assert np.abs(A @ x - b).max() < 1e-6, kw
print('solveOpt ok')
" 2>&1 | grep -E "ok|Error|assert")
    say "verifying gyrfalcON + Agama plugin against agama.orbit"
    local d=$LOGDIR/verify
    rm -rf "$d"; mkdir -p "$d"
    cat > "$d/pot.ini" <<EOF
[Potential]
type=NFW
mass=430000
scaleRadius=15
EOF
    (cd "$d" && "$PYTHON" -c "
import agama, numpy as np
np.random.seed(3); n = 200
x = np.random.normal(size=(n, 3))*15 + [20, 0, 10]
v = np.random.normal(size=(n, 3))*80
np.savetxt('ic.txt', np.hstack((x, v)))
agama.writeSnapshot('ic.nemo', (np.hstack((x, v)), np.full(n, 1e-10)), 'nemo')
")
    (nemo_env && cd "$d" && \
        gyrfalcON ic.nemo out.nemo eps=0.1 kmax=10 Nlev=1 tstop=0.5 step=0.5 accname=agama accfile=pot.ini \
            > gyr.log 2>&1 && s2s out.nemo final.nemo times=0.5 2>/dev/null)
    (cd "$d" && "$PYTHON" -c "
import agama, numpy as np
ic = np.loadtxt('ic.txt'); fin = agama.readSnapshot('final.nemo')[0]
o = agama.orbit(potential=agama.Potential('pot.ini'), ic=ic, time=0.5, trajsize=1001, accuracy=1e-12, verbose=False)
ref = np.vstack([t[1][-1] for t in o]); rmin = np.array([np.linalg.norm(t[1][:, :3], axis=1).min() for t in o])
dx = np.linalg.norm(fin[:, :3] - ref[:, :3], axis=1)[rmin > 1]
print('plugin orbits (rmin>1 kpc, N=%d): |dx| median %.1e, max %.1e kpc' % (len(dx), np.median(dx), dx.max()))
assert np.median(dx) < 0.01, 'gyrfalcON and agama.orbit disagree'
print('plugin ok')
")
}

step_versions() {
    local f=$PREFIX/nbody_stack_versions.txt
    {
        echo "# N-body stack installed by install_nbody_stack.sh on $(date -I) on $(hostname)"
        echo "nemo    $(cd "$NEMO_DIR" && git rev-parse HEAD)"
        echo "agama   $(cd "$AGAMA_DIR" && git rev-parse HEAD)"
        echo "unsio   $(cd "$AGAMA_DIR/extras/unsio-src" && git rev-parse HEAD)"
        echo "cvxopt  $("$PYTHON" -c 'import cvxopt;print(cvxopt.__version__)')"
        echo "eigen   $(grep -E 'define EIGEN_(WORLD|MAJOR|MINOR)_VERSION' -r "$AGAMA_DIR"/extras/include/Eigen/src/Core/util/Macros.h 2>/dev/null | awk '{printf "%s.", $3}' | sed 's/\.$//')"
        find_tirpc; echo "libtirpc $TIRPC_LIB"
        echo "gcc     $(gcc -dumpversion)  gfortran $(gfortran -dumpversion)"
        echo "python  $("$PYTHON" -c 'import sys, numpy;print(sys.version.split()[0], "numpy", numpy.__version__)')"
        echo "agama features: $(grep -o -E -- '-DHAVE_[A-Z]+' "$AGAMA_DIR/Makefile.local" | sort -u | tr '\n' ' ')"
    } > "$f"
    cat "$f"
}

steps=${*:-check tirpc nemo python agama verify versions}
for s in $steps; do
    declare -f "step_$s" >/dev/null || die "unknown step '$s'"
    "step_$s"
done
say "done: $steps"
