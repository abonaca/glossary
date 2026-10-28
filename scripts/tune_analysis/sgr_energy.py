"""Sgr orbital energy at t=0, E = v^2/2 + Phi_MW(x) + Phi_LMC(x - x_LMC), relative to the MW centre (as the
endpoints), against the target's energy in the same potential. The MW and LMC potentials are multipole fits to the
run's own t=0 particles about their centres (Sgr's particles excluded), so model and target share one field and the
difference does not depend on where along its orbit Sgr is (arrival time). Each run is also evaluated in the field
of the first run listed, to separate a change of Sgr's state from a change of the field.

    measure(runs)   # on OBS, runs = [(round, run), ...] -> TUNE/sgr_energy.npz
    table()         # locally, from output/v21_sim/tune/sgr_energy.npz
"""
import sys
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
sys.path.insert(0, '/home/abonaca/projects/glossary/scripts')
import json, os
import numpy as np

RUNS = ((16, 2), (15, 0), (17, 5), (18, 0))
LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../output/v21_sim/tune/sgr_energy.npz')


def _field(xv, m, comps, tr):
    import agama
    pots = []
    for c in ('mw', 'lmc'):
        a, b = comps[c]
        ctr = tr[c][-1, :3]
        p = agama.Potential(type='Multipole', particles=(xv[a:b, :3] - ctr, m[a:b]), symmetry='none',
                            lmax=12 if c == 'mw' else 6, rmin=0.1, rmax=500, gridSizeR=50)
        pots.append((p, ctr))
    return pots


def _energy(pots, s, mwc):
    """Energy of phase-space points s (relative to the MW centre, km/s) in the field pots."""
    s = np.atleast_2d(s)
    (pmw, cmw), (plmc, clmc) = pots
    return 0.5 * np.sum(s[:, 3:]**2, 1) + pmw.potential(s[:, :3]) + plmc.potential(s[:, :3] + cmw - clmc)


def measure(runs=RUNS):
    import agama, v21_tune as vt, r0_noise as rn, v21_centre_check as cc
    agama.setUnits(length=1, velocity=1, mass=232500)
    comps = json.load(open(vt.BASE_ICS + '.json'))['components']
    fields, sgr = [], []
    for k, i in runs:
        tr = np.load(os.path.join(vt.run_dir(k, i), 'centres.npz'))
        xv, m = rn.read_snapshot(cc.final_snapshot(k, i), vt.T_END, tmpdir=vt.run_dir(k, i))
        fields.append(_field(xv, m, comps, tr))
        sgr.append(tr['sgr'][-1] - tr['mw'][-1])
        print('round %d run %d: field fitted' % (k, i), flush=True)
    tgt = vt.TARGET[6:]
    E_own = np.array([_energy(f, s, None) for f, s in zip(fields, sgr)]).ravel()
    E_tgt_own = np.array([_energy(f, tgt, None) for f in fields]).ravel()
    E_ref = np.array([_energy(fields[0], s, None) for s in sgr]).ravel()
    E_tgt_ref = _energy(fields[0], tgt, None)[0]
    np.savez(os.path.join(vt.TUNE, 'sgr_energy.npz'), runs=np.array(runs), sgr=np.array(sgr), E_own=E_own,
             E_tgt_own=E_tgt_own, E_ref=E_ref, E_tgt_ref=E_tgt_ref)
    for (k, i), e, et, er in zip(runs, E_own, E_tgt_own, E_ref):
        print('round %d run %d: E - E_target = %+.0f (own field), %+.0f (field of r%d run %d) (km/s)^2'
              % (k, i, e - et, er - E_tgt_ref, runs[0][0], runs[0][1]), flush=True)


def table(f=LOCAL):
    import v21_tune as vt
    d = np.load(f)
    v = np.linalg.norm(vt.TARGET[9:])
    for (k, i), e, et, er in zip(d['runs'], d['E_own'], d['E_tgt_own'], d['E_ref']):
        de = e - et
        print('round %d run %d: E - E_target = %+6.0f (km/s)^2 own field, %+6.0f in r%d run %d field; '
              '= speed change %+.2f km/s at the target position'
              % (k, i, de, er - d['E_tgt_ref'], d['runs'][0][0], d['runs'][0][1], de / v))
