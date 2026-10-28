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


TIMES = (-1.375, -1.25, -1.125, -1.0, -0.875)        # snapshots around the previous pericentre (~ -1.15 Gyr)
LOCAL_ALONG = LOCAL.replace('.npz', '_along.npz')


def _peri(pots, s, t0, dt):
    """Pericentre of Sgr's orbit from state s at t0, integrated over dt [Gyr] (sign = direction) in the fixed field
    pots: (time [Gyr], radius [kpc], energy) at the pericentre. 1 kpc/(km/s) = 0.9778 Gyr."""
    from scipy.integrate import solve_ivp
    (pmw, cmw), (plmc, clmc) = pots

    def f(_, w):
        return np.r_[w[3:], pmw.force(w[:3]) + plmc.force(w[:3] + cmw - clmc)]
    o = solve_ivp(f, (0, dt / 0.9778), s, rtol=1e-10, atol=1e-10, dense_output=True, max_step=1e-3)
    tt = np.linspace(0, dt / 0.9778, 20001)
    r = np.linalg.norm(o.sol(tt)[:3], axis=0)
    j = np.argmin(r)
    return t0 + tt[j] * 0.9778, r[j], _energy(pots, o.sol(tt[j]), None)[0]


def along(runs=RUNS, times=TIMES):
    """Sgr energy at the snapshots `times` around the previous pericentre, each in the run's own field at that
    time; the pericentre times of the most recent passage (t=0 state integrated 0.1 Gyr back in the t=0 field, for
    each run and the target) and of the previous one (state at -1.25 integrated 0.25 Gyr forward in that field).
    -> TUNE/sgr_energy_along.npz"""
    import agama, v21_tune as vt, r0_noise as rn, v21_centre_check as cc
    agama.setUnits(length=1, velocity=1, mass=232500)
    comps = json.load(open(vt.BASE_ICS + '.json'))['components']
    E, peri_now, peri_prev, peri_tgt = [], [], [], []
    for k, i in runs:
        tr = np.load(os.path.join(vt.run_dir(k, i), 'centres.npz'))
        e = []
        for t in tuple(times) + (vt.T_END,):
            j = int(np.argmin(np.abs(tr['times'] - t)))
            xv, m = rn.read_snapshot(cc.final_snapshot(k, i), tr['times'][j], tmpdir=vt.run_dir(k, i))
            trj = {c: tr[c][:j + 1] for c in ('mw', 'lmc')}
            pots = _field(xv, m, comps, trj)
            s = tr['sgr'][j] - tr['mw'][j]
            e.append(_energy(pots, s, None)[0])
            if abs(t + 1.25) < 1e-6:
                peri_prev.append(_peri(pots, s, t, 0.25))
            if t == vt.T_END:
                peri_now.append(_peri(pots, s, 0., -0.1))
                peri_tgt.append(_peri(pots, vt.TARGET[6:], 0., -0.1))
            print('round %d run %d t=%.3f: E = %.0f (km/s)^2' % (k, i, tr['times'][j], e[-1]), flush=True)
        E.append(e)
        print('  pericentres: previous t=%.3f Gyr r=%.1f kpc; latest t=%.4f Gyr r=%.1f kpc (target %.4f, %.1f)'
              % (peri_prev[-1][:2] + peri_now[-1][:2] + peri_tgt[-1][:2]), flush=True)
    np.savez(os.path.join(vt.TUNE, 'sgr_energy_along.npz'), runs=np.array(runs), times=np.r_[times, vt.T_END],
             E=np.array(E), peri_now=np.array(peri_now), peri_prev=np.array(peri_prev), peri_tgt=np.array(peri_tgt))
