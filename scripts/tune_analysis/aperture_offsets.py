"""Does the persistent Sgr velocity miss at the right arrival time depend on how the centres are measured?

Reads TUNE/roundNN/centre_check.npz (v21_centre_check.measure_round: t=0 centres of every run with several velocity
apertures; MW velocity from its stars and, separately, from its dark matter around the stellar centre) for rounds
0-16. Rounds 0-10 have V21's disc (rotating with Lz > 0), rounds 11-16 the reversed, MW-like disc.

    mw_velocity()     # MW velocity of each estimator minus the v21_tune one (stars within 3 kpc), per round group
    sgr_offset()      # Sgr velocity miss at zero arrival offset (fit over runs) for each MW / Sgr estimator
"""
import sys
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
sys.path.insert(0, '/home/abonaca/projects/glossary/scripts')
import os
import numpy as np
import v21_tune as vt
from r9_analysis import timing_split

T = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../output/v21_sim/tune/')
OLD, NEW = range(0, 11), range(11, 17)


def load(rounds):
    out = []
    for k in rounds:
        f = os.path.join(T, 'round%02d' % k, 'centre_check.npz')
        if os.path.exists(f):
            out.append((k, dict(np.load(f))))
    return out


def estimators(c):
    """MW velocity estimators of one round: label -> (n_run, 3)."""
    est = {'stars %g kpc' % r: c['mw'][:, j, 3:] for j, r in enumerate(c['r_vel_mw'])}
    est.update({'dark matter %g kpc' % r: c['mwdm'][:, j, 3:] for j, r in enumerate(c['r_vel_mwdm'])})
    return est


def mw_velocity(rounds_groups=(('V21 disc, rounds 0-10', OLD), ('reversed disc, rounds 11-16', NEW))):
    """Mean (and run-to-run std) of each MW velocity estimator minus stars within 3 kpc, km/s."""
    res = {}
    for lab, rounds in rounds_groups:
        d = {}
        for k, c in load(rounds):
            ref = c['mw'][:, list(c['r_vel_mw']).index(3.), 3:]
            for name, v in estimators(c).items():
                d.setdefault(name, []).append(v - ref)
        res[lab] = {name: (np.vstack(v).mean(0), np.vstack(v).std(0)) for name, v in d.items()}
        print('\n%s: MW velocity minus stars within 3 kpc [km/s], mean (std over runs)' % lab)
        for name, (m, s) in res[lab].items():
            print('  %-20s %s  (%s)' % (name, np.array2string(m, precision=2), np.array2string(s, precision=2)))
    return res


def sgr_offset(rounds=NEW, mw='stars 3 kpc', sgr_r=5.):
    """Sgr t=0 miss (positions kpc, velocities km/s) at zero arrival offset: intercept of a fit of the miss against the
    arrival offset over all runs of the rounds, with the given MW velocity estimator and Sgr median radius."""
    miss = []
    for k, c in load(rounds):
        mwv = estimators(c)[mw]
        s = c['sgr'][:, list(c['r_sgr']).index(sgr_r)]
        e = np.hstack((s[:, :3] - c['mw'][:, 0, :3], s[:, 3:] - mwv))
        miss.append(e - vt.TARGET[6:])
    miss = np.vstack(miss)
    a = np.array([timing_split(m, vt.TARGET[9:])[0] for m in miss])
    X = np.vstack((np.ones_like(a), a)).T
    C, *_ = np.linalg.lstsq(X, miss, rcond=None)
    return C[0], C[1], np.std(miss - X @ C, axis=0), len(a)


def table():
    np.set_printoptions(precision=2, suppress=True, linewidth=160)
    mw_velocity()
    for lab, rounds in (('V21 disc, rounds 0-10', OLD), ('reversed disc, rounds 11-16', NEW)):
        print('\n%s: Sgr miss at zero arrival offset (x y z kpc | vx vy vz km/s)' % lab)
        c0 = load(rounds)[0][1]
        for mw in estimators(c0):
            for sr in (5.,) if mw != 'stars 3 kpc' else c0['r_sgr']:
                i0, sl, rms, n = sgr_offset(rounds, mw, sr)
                print('  MW %-18s Sgr %g kpc: %s | %s   (n=%d)' % (mw, sr, i0[:3], i0[3:], n))


if __name__ == '__main__':
    table()
