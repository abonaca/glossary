"""Adopted model (round 9 run 2) with the MW stars rotating in the observed sense (Lz < 0 in V21's astropy
Galactocentric frame, where the Sun moves with vy = +245.6 km/s at x = -8.1 kpc).

The model disc rotates with Lz > 0 (Agama QuasiIsothermal DF in V21's create_mw.py; see mw_ELz.py). The fix reverses
the velocities of the MW star particles, v -> -v, and leaves their positions, the MW dark matter, the LMC and Sgr as
in round 9 run 2's ics.nemo. Reversing v is the time reverse of the star particles' orbits, so it keeps them in
equilibrium in any static potential, including V21's tilted triaxial halo. Reversing only (vx, vy) would also need
the potential to be symmetric in z, and the tilted halo isn't. The MW starts at rest at the origin, so no bulk
velocity is involved.

    build()     # on OBS: -> v21_sim/adopted/r09run02_prograde/ics.nemo (time -3)
Run with obs/adopted_prograde.sbatch (build + gyrfalcON with the adopted_run.sbatch settings).
    centres()   # on OBS: MW, LMC, Sgr centres on the tuning-run snapshot grid (every 1/8) -> centres.npz
    compare()   # locally: t=0 LMC and Sgr misses of this run and of round 9 run 2
"""
import sys, os
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
sys.path.insert(0, '/home/abonaca/projects/glossary/scripts')
import numpy as np
NSTARS = 1000000                    # MW stars are particles 0 .. NSTARS-1 (R2.json)
ADOPTED = '/carnegie/nobackup/users/abonaca/v21_sim/adopted'


def vphi_summary(xv, label):
    R = np.hypot(xv[:, 0], xv[:, 1])
    vphi = (xv[:, 0] * xv[:, 4] - xv[:, 1] * xv[:, 3]) / R
    s = (R > 6) & (R < 10) & (np.abs(xv[:, 2]) < 0.5)
    print('%s: mean v_phi (R 6-10, |z|<0.5) = %+.1f km/s, Lz>0 fraction %.4f (N=%d)'
          % (label, vphi[s].mean(), np.mean(vphi[s] > 0), s.sum()), flush=True)


def build(k=9, i=2, out=None):
    import agama, v21_tune as vt, v21_ics as vi
    src = os.path.join(vt.run_dir(k, i), 'ics.nemo')
    d = out or os.path.join(ADOPTED, 'r%02drun%02d_prograde' % (k, i))
    os.makedirs(d, exist_ok=True)
    f, tmp = os.path.join(d, 'ics.nemo'), os.path.join(d, 'ics_t0.nemo')
    if os.path.exists(f):
        raise RuntimeError('%s exists; move it away first' % f)
    xv, m = agama.readSnapshot(src)
    vphi_summary(xv[:NSTARS], 'source stars')
    vphi_summary(xv[NSTARS:5000000], 'source MW dark matter')
    xv[:NSTARS, 3:6] *= -1
    vphi_summary(xv[:NSTARS], 'flipped stars')
    agama.writeSnapshot(tmp, (xv, m), 'nemo')
    vi._nemo('s2s %s %s time=%g' % (tmp, f, vt.T_START))
    os.remove(tmp)
    print(src, '->', f, flush=True)
    return f


def centres(k=9, i=2, d=None):
    """Centres (v21_tune.track_centres) of the reversed-disc run on the tuning runs' snapshot grid (every 1/8 time
    unit, 25 snapshots), so the procedure, and the membership tracking, are the same as for round k run i. The
    grid snapshots are cut out of the dense evol.nemo (three s2s passes) into a temporary evol_grid.nemo."""
    import json, numpy as np, v21_tune as vt, v21_ics as vi, r0_noise as rn
    d = d or os.path.join(ADOPTED, 'r%02drun%02d_prograde' % (k, i))
    grid = vt.T_START + np.arange(25) * vt.SNAP_STEP
    f = os.path.join(d, 'evol_grid.nemo')
    if os.path.exists(f):
        os.remove(f)
    # s2s mis-selects with a long list of time ranges (25 ranges gave wrong snapshots after the 13th), so cut the
    # grid in chunks of 9 ranges and concatenate the chunk files (NEMO snapshot files concatenate with cat)
    parts = []
    for n, a in enumerate(range(0, len(grid), 9)):
        parts.append(f + '.%d' % n)
        vi._nemo('s2s %s %s times=%s' % (os.path.join(d, 'evol.nemo'), parts[-1],
                                         ','.join('%.6f:%.6f' % (t - 1e-4, t + 1e-4) for t in grid[a:a + 9])))
    vi._nemo('cat %s > %s' % (' '.join(parts), f))
    for p in parts:
        os.remove(p)
    times = rn.snapshot_times(f)
    if len(times) != len(grid) or np.max(np.abs(times - grid)) > 1e-6:
        raise RuntimeError('grid snapshots: %s' % times)
    comps = json.load(open(vt.BASE_ICS + '.json'))['components']
    tr = vt.track_centres(f, comps, vt.run_theta(k, i))
    np.savez(os.path.join(d, 'centres.npz'), **tr)
    os.remove(f)
    print(os.path.join(d, 'centres.npz'), flush=True)


LOCAL = '/home/ana/projects/glossary/output/v21_sim/adopted/'


def compare(k=9, i=2):
    """t=0 misses (endpoint - TARGET, kpc and km/s) of the reversed-disc run and of round k run i, and the
    differences of their tracks over time."""
    import numpy as np, v21_tune as vt
    base = np.load('/home/ana/projects/glossary/output/v21_sim/tune/round%02d/run%02d/centres.npz' % (k, i))
    flip = np.load(LOCAL + 'r%02drun%02d_prograde/centres.npz' % (k, i))
    for lab, c in (('r%d run %d (Lz > 0 disc)' % (k, i), base), ('reversed (MW-like) disc', flip)):
        miss = vt.endpoint(c) - vt.TARGET
        print('%-26s LMC %s | %s   Sgr %s | %s' % (lab, np.array2string(miss[:3], precision=3),
              np.array2string(miss[3:6], precision=2), np.array2string(miss[6:9], precision=3),
              np.array2string(miss[9:], precision=2)))
    print('tolerance: LMC 0.2 kpc / 1 km/s, Sgr 0.1 kpc / 1 km/s')
    for comp in ('lmc', 'sgr'):
        dr = np.linalg.norm((flip[comp] - flip['mw'])[:, :3] - (base[comp] - base['mw'])[:, :3], axis=1)
        dv = np.linalg.norm((flip[comp] - flip['mw'])[:, 3:] - (base[comp] - base['mw'])[:, 3:], axis=1)
        print('%s: |dr| [kpc] / |dv| [km/s] (reversed - base) at t = %s Gyr' % (comp, ', '.join('%.2f' % t for t in base['times'][::4] * 0.9778)))
        print('    ', ' '.join('%.2f/%.1f' % (a, b) for a, b in zip(dr[::4], dv[::4])))
    return base, flip


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'centres':
        centres()
    else:
        build(*map(int, sys.argv[1:3]))
