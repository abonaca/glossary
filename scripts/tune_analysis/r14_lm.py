"""Levenberg-Marquardt steps from round 14's base on J14 (full 12x12 and LMC/Sgr blocks), with their linear forecasts
(round-14 miss + J14 step, full J14 including the cross-terms), next to the uniformly scaled Gauss-Newton step.

Damping as in r8_jacobian.lm_steps: in tolerance-per-nudge units A = W J D^-1, b = W r,
    (A^T A + lam diag(A^T A)) u = -A^T b,  step = D^-1 u.
For the block version the LMC and Sgr halves are solved separately with their own diagonal blocks of J14.

    table()     # prints steps (kpc, km/s) and forecast misses (kpc, km/s)
"""
import sys
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
sys.path.insert(0, '/home/abonaca/projects/glossary/scripts')
import json, os
import numpy as np
import v21_tune as vt

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../output/v21_sim/tune/round%02d/round.json')
LAMS = (0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)


def load(k=14):
    d = json.load(open(R % k))
    F = np.array(d['endpoints'])
    J = (F[1:] - F[0]).T / np.array(d['delta'])[None, :]
    return J, F[0] - vt.TARGET, np.array(d['delta'])


def lm(J, r, delta, tol, lam):
    A = J / tol[:, None] * delta[None, :]
    M = A.T @ A
    u = -np.linalg.solve(M + lam * np.diag(np.diag(M)), A.T @ (r / tol))
    return u * delta


def lm_block(J, r, delta, lam):
    s = np.zeros(12)
    for b in (slice(0, 6), slice(6, 12)):
        s[b] = lm(J[b, b], r[b], delta[b], vt.TOL[b], lam)
    return s


def table(k=14, lams=LAMS, scales=(0.25, 0.5)):
    J, r, delta = load(k)
    gn = np.linalg.solve(J, -r)
    rows = [('GN x%g' % f, f * gn) for f in scales]
    rows += [('LM full %g' % lam, lm(J, r, delta, vt.TOL, lam)) for lam in lams]
    rows += [('LM block %g' % lam, lm_block(J, r, delta, lam)) for lam in lams if lam > 0]
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    hdr = 'x y z [kpc], vx vy vz [km/s]: LMC | Sgr'
    print('round %d miss  %s' % (k, hdr))
    print('%-16s %s' % ('', r))
    print('\nstep at t=-3 (nudges %s)' % delta)
    for lab, s in rows:
        print('%-16s %s   max |step|/nudge %.2f' % (lab, s, np.max(np.abs(s) / delta)))
    print('\nforecast miss with J%d (out of tolerance: *)' % k)
    for lab, s in rows:
        m = r + J @ s
        print('%-16s %s  %s' % (lab, m, ''.join('*' if o else '.' for o in np.abs(m) > vt.TOL)))
    return rows


if __name__ == '__main__':
    table()
