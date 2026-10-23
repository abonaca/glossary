"""All particles of a tuning run (default round 8, run 3), relative to the MW centre, in Galactocentric Cartesian
projections: MW dark matter, LMC, Sgr dark matter, MW stars and Sgr stars in separate colours.

    plot()                     # t=0 from the locally saved positions (all_t0_pos.npz)
    frames(8, 3, outdir)       # every snapshot, read from evol.nemo (run on OBS)
"""
import sys
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
sys.path.insert(0, '/home/abonaca/projects/glossary/scripts')
import os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import v21_ics as vi
plt.style.use('default')
F = '/home/ana/projects/glossary/output/v21_sim/tune/round08/run03/all_t0_pos.npz'
COMPS = dict(mw=(0, 5000000), lmc=(5000000, 6000000), sgr=(6000000, 6400000))   # R2.json
LIM = 110.


def groups(x):
    nmw, nsgr = vi.MW_COUNTS['nbody_stars'], vi.SGR_COUNTS['nstar']
    a, b = COMPS['mw']; mw_star, mw_dm = x[a:a + nmw], x[a + nmw:b]
    a, b = COMPS['lmc']; lmc = x[a:b]
    a, b = COMPS['sgr']; sgr_star, sgr_dm = x[a:a + nsgr], x[a + nsgr:b]
    # drawing order: dark matter first, stars on top
    return [('MW dark matter', mw_dm, '#08306b', 0.15),
            ('LMC (dark matter)', lmc, '#2171b5', 0.25),
            ('Sgr dark matter', sgr_dm, '#9ecae1', 0.5),
            ('MW stars', mw_star, '#e6550d', 0.5),
            ('Sgr stars', sgr_star, '#fec44f', 0.5)]


def figure(x, label, out, lim=LIM):
    g = groups(x)
    fig, ax = plt.subplots(1, 3, figsize=(18, 6.3))
    for p, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
        a = ax[p]
        for lab, xx, col, al in g:
            a.plot(xx[:, i], xx[:, j], ',', color=col, alpha=al, rasterized=True)
        a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_aspect('equal')
        a.set_xlabel('%s [kpc]' % 'xyz'[i]); a.set_ylabel('%s [kpc]' % 'xyz'[j])
    for lab, xx, col, al in g:
        ax[0].plot([], [], 'o', color=col, ms=6, label='%s (%d)' % (lab, len(xx)))
    ax[0].legend(fontsize=8, loc='lower left')
    ax[0].text(0.02, 0.98, label, transform=ax[0].transAxes, va='top', fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(out, flush=True)


def plot(out='/home/ana/projects/glossary/plots/tune_r8_run03_all_particles_xyz.png'):
    figure(np.load(F)['x'], 'r8 run 3, t = 0 Gyr, relative to the MW centre', out)


def frames(k=8, i=3, outdir=None):
    """Every snapshot of round k run i (on OBS), positions relative to the MW centre at t=0 (fixed frame)."""
    import v21_tune as vt, r0_noise as rn
    d = vt.run_dir(k, i)
    outdir = outdir or os.path.join(d, 'frames')
    os.makedirs(outdir, exist_ok=True)
    ev = os.path.join(d, 'evol.nemo')
    c0 = np.load(os.path.join(d, 'centres.npz'))['mw'][-1, :3]
    for n, t in enumerate(rn.snapshot_times(ev)):
        xv, _ = rn.read_snapshot(ev, t)
        figure(xv[:, :3] - c0, 'r%d run %d, t = %+.3f Gyr, relative to the t = 0 MW centre' % (k, i, t),
               os.path.join(outdir, 'tune_r%d_run%02d_all_particles_xyz_%02d.png' % (k, i, n)))


ADOPTED = '/carnegie/nobackup/users/abonaca/v21_sim/adopted/r09run02'
GYR = 0.9778                    # Gyr per model time unit


def adopted_frames(task, ntask, d=ADOPTED, step=1 / 256., k=9, i=2):
    """Frames of the dense-output adopted run (on OBS), split over ntask array tasks: task handles a contiguous
    block of snapshot times, cut out of evol.nemo with snapsplit (one pass) into a task-local temporary directory,
    each split file deleted once drawn. Frame n is snapshot t = -3 + n * step, relative to the t=0 MW centre of
    round k run i (the same initial conditions)."""
    import subprocess, glob, agama, v21_tune as vt, r0_noise as rn
    times = -3 + np.arange(int(round(3 / step)) + 1) * step
    blocks = np.array_split(np.arange(len(times)), ntask)[task]
    lo, hi = times[blocks[0]], times[blocks[-1]]
    c0 = np.load(os.path.join(vt.run_dir(k, i), 'centres.npz'))['mw'][-1, :3]
    outdir = os.path.join(d, 'frames')
    tmp = os.path.join(d, 'split_%03d' % task)
    os.makedirs(outdir, exist_ok=True); os.makedirs(tmp, exist_ok=True)
    subprocess.run(['bash', '-c', 'set +u; source %s >/dev/null 2>&1; snapsplit %s %s/s_%%04d.nemo nbody=%d '
                    'times=%.8f:%.8f' % (rn.NEMO_START, os.path.join(d, 'evol.nemo'), tmp, COMPS['sgr'][1],
                                         lo - step / 4, hi + step / 4)], check=True, capture_output=True)
    files = sorted(glob.glob(os.path.join(tmp, 's_*.nemo')))
    if len(files) != len(blocks):
        raise RuntimeError('task %d: %d split files for %d times' % (task, len(files), len(blocks)))
    for n, f in zip(blocks, files):
        xv, _ = agama.readSnapshot(f)
        figure(xv[:, :3] - c0, 'adopted model (r%d run %d), t = %+.3f Gyr, relative to the t = 0 MW centre'
               % (k, i, times[n] * GYR), os.path.join(outdir, 'adopted_all_particles_xyz_%03d.png' % n))
        os.remove(f)
    os.rmdir(tmp)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'frames':
        frames(*map(int, sys.argv[2:4]))
    elif len(sys.argv) > 1 and sys.argv[1] == 'adopted':
        adopted_frames(*map(int, sys.argv[2:4]))
    else:
        plot()
