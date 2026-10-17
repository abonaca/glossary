"""
Noise floor of the basis-function expansions of the live Milky Way model (run R0, MW alone).

For snapshots of R0 the Milky Way is re-centred (shrinking sphere on the stars) and expanded as
V21 did for their evolving halo (Multipole, lmax=4, gridSizeR=25, no symmetry, 0.38-262.6 kpc),
plus a CylSpline for the disk+bulge (mmax=4). Three force comparisons, all relative to the total
Milky Way force at random points in radial shells:
  noise  - half-split: expansions of two random halves of the particles; the noise of the full
           expansion is |F_A - F_B| / 2 (each half has twice the variance of the full set)
  drift  - expansion at time t vs the one at the first snapshot (includes sqrt(2) x noise)
  signal - V21's evolving MW halo (mwNN.pot, MW+LMC run) vs their initial mw00.pot, i.e. the
           LMC-induced deformation the noise should be compared with

Units: kpc, km/s, mass unit 232500 Msun; times in kpc/(km/s) = 0.978 Gyr.

Usage (ipython):
    import r0_noise as rn
    res = rn.analyse_r0('/carnegie/nobackup/users/abonaca/v21_sim/runs/R0/R0_evol.nemo')
    rn.plot_noise(res)
"""
import os
import subprocess
import numpy as np
import agama
from configparser import RawConfigParser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V21_POT = os.path.join(ROOT, 'data', 'tango', 'potentials_triax')
NEMO_START = os.path.expanduser('~/projects/python/nemo/nemo_start.sh')
NSTARS = 1000000             # R0 = MW stars (disk+bulge) first, then the halo (v21_ics.assemble_runs)
HALO_EXP = dict(type='Multipole', lmax=4, symmetry='n', gridSizeR=25, rmin=0.38, rmax=262.6)
DISK_EXP = dict(type='CylSpline', mmax=4, symmetry='n', gridSizeR=20, gridSizez=20,
                Rmin=0.2, Rmax=60., zmin=0.05, zmax=20.)
SHELLS = np.array([2., 5., 10., 20., 40., 80., 160.])     # kpc, edges of the radial shells
V21_TIMES = -1.5 + np.arange(29) / 16.                     # mw00.pot ... mw28.pot


def read_snapshot(evol, t, tmpdir=None):
    """Snapshot at time t from a multi-snapshot NEMO file (agama.readSnapshot reads only the first)."""
    tmpdir = tmpdir or os.path.dirname(evol)
    tmp = os.path.join(tmpdir, '_snap_%.6f.nemo' % t)
    if os.path.exists(tmp):
        os.remove(tmp)
    subprocess.run(['bash', '-c', 'set +u; source %s >/dev/null 2>&1; s2s %s %s times=%.6f' %
                    (NEMO_START, evol, tmp, t)], check=True, capture_output=True)
    xv, m = agama.readSnapshot(tmp)
    os.remove(tmp)
    return xv, m


def snapshot_times(evol):
    """Times of all snapshots in a NEMO file."""
    out = subprocess.run(['bash', '-c', "set +u; source %s >/dev/null 2>&1; tsf %s | grep 'double Time'" %
                          (NEMO_START, evol)], check=True, capture_output=True, text=True).stdout
    return np.array([float(l.split()[-1]) for l in out.splitlines()])


def centre(xv, m, r_start=30., r_stop=1., shrink=0.8):
    """Shrinking-sphere centre position and the mean velocity of the particles inside r_stop."""
    c, r = np.median(xv[:, :3], axis=0), r_start
    while r > r_stop:
        sel = np.sum((xv[:, :3] - c)**2, axis=1) < r**2
        c = np.average(xv[sel, :3], axis=0, weights=m[sel])
        r *= shrink
    sel = np.sum((xv[:, :3] - c)**2, axis=1) < r_stop**2
    return c, np.average(xv[sel, 3:6], axis=0, weights=m[sel])


def fit(xv, m, which):
    """Expansion of a set of particles: which='halo' (Multipole) or 'disk' (CylSpline)."""
    return agama.Potential(particles=(xv[:, :3], m), **(HALO_EXP if which == 'halo' else DISK_EXP))


def fit_mw(xv, m, recentre=True):
    """Halo and disk+bulge expansions of a R0 snapshot about the Milky Way centre."""
    xv = xv.copy()
    if recentre:
        c, v = centre(xv[:NSTARS], m[:NSTARS])
        xv[:, :3] -= c
        xv[:, 3:6] -= v
    return dict(halo=fit(xv[NSTARS:], m[NSTARS:], 'halo'), disk=fit(xv[:NSTARS], m[:NSTARS], 'disk')), xv


def half_split(xv, m, which, seed=0):
    """Expansions of two random halves of the particles (each with doubled masses)."""
    sl = slice(NSTARS, None) if which == 'halo' else slice(None, NSTARS)
    xv, m = xv[sl], m[sl]
    half = np.random.default_rng(seed).permutation(len(m)) < len(m) // 2
    return fit(xv[half], 2 * m[half], which), fit(xv[~half], 2 * m[~half], which)


def eval_points(npershell=400, seed=1):
    """Isotropic random points, log-uniform in radius within each shell; returns points and shell index."""
    rng = np.random.default_rng(seed)
    pts, idx = [], []
    for i, (r1, r2) in enumerate(zip(SHELLS[:-1], SHELLS[1:])):
        r = np.exp(rng.uniform(np.log(r1), np.log(r2), npershell))
        u = rng.normal(size=(npershell, 3))
        pts.append(u / np.linalg.norm(u, axis=1)[:, None] * r[:, None])
        idx.append(np.full(npershell, i))
    return np.vstack(pts), np.concatenate(idx)


def v21_static():
    """V21's analytic bulge and disk (mw_model_triax.ini), used for the total-force normalisation."""
    ini = RawConfigParser()
    ini.optionxform = str
    ini.read(os.path.join(ROOT, 'data', 'tango', 'scripts', 'mw_model_triax.ini'))
    return [agama.Potential(**dict(ini.items(s))) for s in ('Potential bulge', 'Potential disk')]


def rel_force(pa, pb, pts, ftot):
    """|F_a - F_b| / |F_total| at the points."""
    return np.linalg.norm(pa.force(pts) - pb.force(pts), axis=1) / ftot


def shell_median(x, idx):
    return np.array([np.median(x[idx == i]) for i in range(len(SHELLS) - 1)])


def v21_signal(pts, idx, ftot):
    """Per-shell median |F(mwNN) - F(mw00)| / |F_total| for V21's evolving halo; returns times, array."""
    ref = agama.Potential(os.path.join(V21_POT, 'mw00.pot'))
    out = [shell_median(rel_force(agama.Potential(os.path.join(V21_POT, 'mw%02d.pot' % k)), ref, pts, ftot), idx)
           for k in range(len(V21_TIMES))]
    return V21_TIMES, np.array(out)


def analyse_r0(evol, times=None, every=4, seed=0, npershell=400):
    """
    Noise, drift and V21 signal for the R0 snapshots at `times` (default: every `every`-th one).
    Returns a dict of per-shell medians, shells along the last axis.
    """
    all_times = snapshot_times(evol)
    times = all_times[::every] if times is None else np.atleast_1d(times)
    pts, idx = eval_points(npershell)
    static = v21_static()
    ftot = np.linalg.norm(sum(p.force(pts) for p in static) +
                          agama.Potential(os.path.join(V21_POT, 'mw00.pot')).force(pts), axis=1)
    res = dict(times=times, shells=SHELLS, noise_halo=[], noise_disk=[], drift_halo=[], drift_disk=[],
               centre=[])
    first = None
    for t in times:
        print('t = %.4f' % t, flush=True)
        xv, m = read_snapshot(evol, t)
        c, v = centre(xv[:NSTARS], m[:NSTARS])
        res['centre'].append(np.hstack((c, v)))
        pots, xvc = fit_mw(xv, m)
        for which in ('halo', 'disk'):
            a, b = half_split(xvc, m, which, seed)
            res['noise_' + which].append(shell_median(rel_force(a, b, pts, ftot) / 2, idx))
        if first is None:
            first = pots
        for which in ('halo', 'disk'):
            res['drift_' + which].append(shell_median(rel_force(pots[which], first[which], pts, ftot), idx))
    for k in list(res):
        res[k] = np.array(res[k])
    res['v21_times'], res['v21_signal'] = v21_signal(pts, idx, ftot)
    return res


def plot_noise(res, fname=os.path.join(ROOT, 'plots', 'r0_noise.png')):
    """
    Noise and change since t0 of R0's expansions against V21's LMC-induced halo deformation, per
    radial shell. Every curve is a force difference between two expansions, |F_a - F_b| / |F_MW|.
    """
    import matplotlib.pyplot as plt
    nsh = len(res['shells']) - 1
    fig, axes = plt.subplots(2, (nsh + 1) // 2, figsize=(4 * ((nsh + 1) // 2), 6.5), sharex=True)
    for i, ax in enumerate(axes.flat[:nsh]):
        ax.semilogy(res['times'], res['noise_halo'][:, i], 'k-', label='halo: noise (half-split)')
        ax.semilogy(res['times'], res['noise_disk'][:, i], 'b-', label='disk+bulge: noise (half-split)')
        ax.semilogy(res['times'][1:], res['drift_halo'][1:, i], 'k--', label='halo: change since t0')
        ax.semilogy(res['times'][1:], res['drift_disk'][1:, i], 'b--', label='disk+bulge: change since t0')
        ax.semilogy(res['v21_times'][1:], res['v21_signal'][1:, i], 'r-', label='V21 halo (MW+LMC): change since mw00')
        ax.set_title('%g-%g kpc' % (res['shells'][i], res['shells'][i + 1]))
    for ax in axes.flat[nsh:]:
        ax.axis('off')
    for ax in axes[-1]:
        ax.set_xlabel('t [kpc/(km/s)]')
    for ax in axes[:, 0]:
        ax.set_ylabel(r'median $|\Delta F| / |F_{\rm MW}|$')
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    print('saved', fname)
    return fig
