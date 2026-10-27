"""Sgr mass within 3, 5 and 10 kpc of its centre (all Sgr particles; V21's present-day constraint is (3-4)e8 Msun
within 5 kpc): at t=0 in every run of a round, and at every snapshot of the round's base run.

    measure(k)      # on OBS -> TUNE/roundKK/sgr_mass_runs.npz (t=0, every run) and sgr_mass_base.npz (run 0 vs time)
    table(rounds)   # locally, from copies in output/v21_sim/tune/roundKK/
Submit:  sbatch --array=11-15 ~/projects/glossary/scripts/obs/sgr_mass.sbatch
"""
import sys
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
sys.path.insert(0, '/home/abonaca/projects/glossary/scripts')
import json, os
import numpy as np

RADII = (3., 5., 10.)
MUNIT = 232500.                      # Msun per N-body mass unit
LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../output/v21_sim/tune/')


def _within(xv, m, a, b, c):
    r = np.linalg.norm(xv[a:b, :3] - c[:3], axis=1)
    return [m[a:b][r < R].sum() * MUNIT for R in RADII]


def measure(k):
    import v21_tune as vt, r0_noise as rn, v21_centre_check as cc
    comps = json.load(open(vt.BASE_ICS + '.json'))['components']
    a, b = comps['sgr']
    runs, mt0 = cc.run_ids(k), []
    for i in runs:
        tr = np.load(os.path.join(vt.run_dir(k, i), 'centres.npz'))
        xv, m = rn.read_snapshot(cc.final_snapshot(k, i), vt.T_END, tmpdir=vt.run_dir(k, i))
        mt0.append(_within(xv, m, a, b, tr['sgr'][-1]))
        print('round %d run %2d t=0: M(<3,5,10 kpc) = %s Msun' % (k, i, ' '.join('%.3g' % x for x in mt0[-1])), flush=True)
    np.savez(os.path.join(vt.round_dir(k), 'sgr_mass_runs.npz'), run=np.array(runs), mass=np.array(mt0),
             radii=np.array(RADII))
    tr = np.load(os.path.join(vt.run_dir(k, 0), 'centres.npz'))
    evol, mb = os.path.join(vt.run_dir(k, 0), 'evol.nemo'), []
    for t, c in zip(tr['times'], tr['sgr']):
        xv, m = rn.read_snapshot(evol, t, tmpdir=vt.run_dir(k, 0))
        mb.append(_within(xv, m, a, b, c))
    np.savez(os.path.join(vt.round_dir(k), 'sgr_mass_base.npz'), times=tr['times'], mass=np.array(mb),
             radii=np.array(RADII))
    print('base run 0: M(<5 kpc) from %.3g to %.3g Msun' % (mb[0][1], mb[-1][1]), flush=True)


def table(rounds=range(11, 17)):
    for k in rounds:
        f = os.path.join(LOCAL, 'round%02d' % k, 'sgr_mass_runs.npz')
        if not os.path.exists(f):
            continue
        d = np.load(f)
        m5 = d['mass'][:, 1]
        print('round %d: base M(<5) %.2e; runs %.2e-%.2e (mean %.2e); M(<3) base %.2e, M(<10) base %.2e Msun'
              % (k, m5[0], m5.min(), m5.max(), m5.mean(), d['mass'][0, 0], d['mass'][0, 2]))
