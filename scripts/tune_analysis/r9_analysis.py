"""Round 9: Sgr/LMC misses, J8 forecasts vs outcomes for the damped steps (runs 0-3), the forward-difference J9 around
run 2 (lambda=3e-3) from the nudge runs 4-9, and the along-orbit (timing) part of each Sgr miss."""
import json, numpy as np
import r8_jacobian as rj
T = '/home/ana/projects/glossary/output/v21_sim/tune/'
R9 = T + 'round09/round.json'


def load():
    d = json.load(open(R9))
    return d, np.array(d['endpoints'])[:, 6:], np.array(d['misses'])


def timing_split(miss, v):
    """Split a Sgr miss into a shift along the target orbit (dt in Myr: miss ~ dt * (v, a)) and the remainder."""
    vt = np.asarray(v, float) * 1.0227e-3                  # kpc/Myr
    dt = miss[:3] @ vt / (vt @ vt)
    return dt, miss[:3] - dt * vt


def report():
    np.set_printoptions(precision=3, suppress=True, linewidth=170)
    d, F, M = load()
    j8 = rj.jacobians()
    r3 = j8['f0'] - j8['target']                               # round-8 run 3 Sgr miss
    off = np.array(d['offsets'])
    print('Sgr miss (x,y,z kpc | vx,vy,vz km/s), LMC ok, along-orbit shift dt [Myr], position remainder [kpc]')
    for i, lab in enumerate(d['labels']):
        m = M[i, 6:]
        dt, rem = timing_split(m, d['target'][9:])
        print('  %d %-18s %s | %s  LMC %-5s dt %+5.2f  rem %s' % (i, lab, m[:3], m[3:], d['lmc_ok'][i], dt, rem))
    print('\nLMC miss')
    for i, lab in enumerate(d['labels']):
        print('  %d %-18s %s' % (i, lab, M[i, :6]))
    print('\nJ8 forecast vs outcome (Sgr), runs 0-3')
    for i in range(4):
        f = r3 + j8['central'] @ off[i]
        print('  %d forecast %s\n    outcome  %s' % (i, f, M[i, 6:]))
    J9 = (F[4:10] - F[2]).T / rj.DELTA
    print('\nJ9 (forward, from run 2), response per nudge; rows x,y,z,vx,vy,vz; columns', ' '.join(rj.NAMES))
    print(J9 * rj.DELTA)
    print('J8 central, per nudge')
    print(j8['central'] * rj.DELTA)
    print('J9 - J8 per nudge')
    print((J9 - j8['central']) * rj.DELTA)
    A = np.diag(1 / rj.TOL) @ J9 @ np.diag(rj.DELTA)
    print('J9 singular values (tolerance units per nudge):', np.linalg.svd(A, compute_uv=False))
    return dict(J9=J9, misses=M, offsets=off)


if __name__ == '__main__':
    report()
