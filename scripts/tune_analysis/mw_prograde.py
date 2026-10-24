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


if __name__ == '__main__':
    build(*map(int, sys.argv[1:3]))
