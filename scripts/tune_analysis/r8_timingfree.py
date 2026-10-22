"""
Timing-free linear fit for round 8: for the 19 runs sharing the round-6 LMC (round-6 base + Sgr nudges,
all round-7 runs), the Sgr velocity and position offsets from the target at closest approach to the target
position (present-day state integrated +-20 Myr in round 7's t=0 fields) vs the Sgr start change from the
round-7 base. Fit y = a + B dtheta, validate, and solve for the start shift that zeroes y.
"""
import sys
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import json, numpy as np, agama, v21_tune as vt
from scipy.integrate import solve_ivp
agama.setUnits(length=1, velocity=1, mass=232500)
np.set_printoptions(precision=3, suppress=True, linewidth=170)
T = '/home/ana/projects/glossary/output/v21_sim/tune/'
D = T + 'round07/run00/expansions_hi'
mw = agama.Potential(agama.Potential(D + '/halo_24.coef'), agama.Potential(D + '/disk_24.coef'))
lmc = agama.Potential(D + '/lmc_24.coef')
tgt = vt.TARGET[6:]; TU = 977.8
ts = np.linspace(-20, 20, 8001) / TU
i0 = 4000; ts[i0] = 0.
SCALE = np.array([0.05, 0.05, 0.05, 0.25, 0.25, 0.25])     # round-7 nudge scale (x0, z0; y0 nudge was 0.25 kpc)


def orbit(s, l):
    xl = lambda t: l[:3] + l[3:] * t
    f = lambda t, w: np.hstack((w[3:], mw.force(w[:3]) + lmc.force(w[:3] - xl(t)) - lmc.force(-xl(t))))
    up = solve_ivp(f, (0, ts[-1]), s, t_eval=ts[i0:], rtol=1e-11, atol=1e-11, method='DOP853').y.T
    dn = solve_ivp(f, (0, ts[0]), s, t_eval=ts[i0::-1], rtol=1e-11, atol=1e-11, method='DOP853').y.T[::-1]
    W = np.vstack((dn[:-1], up))
    xls = l[:3] + l[3:] * ts[:, None]
    E = 0.5 * np.sum(W[:, 3:]**2, 1) + mw.potential(W[:, :3]) + lmc.potential(W[:, :3] - xls)
    return W, E


def collect():
    names = ['x0', 'y0', 'z0', 'vx0', 'vy0', 'vz0']
    i6 = json.load(open(T + 'round06/round.json')); th6 = np.array(i6['theta']); d6 = np.array(i6['delta'])
    i7 = json.load(open(T + 'round07/round.json')); b7 = np.array(i7['runs'][0])[6:]
    c = np.load(T + 'round06/centre_check.npz')
    rows = []
    for i in [0] + list(range(7, 13)):
        th = th6 + (d6 * np.eye(12)[i - 1] if i else 0)
        rows.append(('r6 ' + ('base' if i == 0 else '+' + names[i - 7]), th[6:] - b7, c['sgr'][i, 2] - c['mw'][i, 3], c['lmc'][i, 3] - c['mw'][i, 3]))
    lab7 = ['base'] + ['+' + n for n in names] + ['%+g Myr' % x for x in i7['scan_dt']]
    for i in range(13):
        r = np.load(T + 'round07/run%02d/centres.npz' % i)
        rows.append(('r7 ' + lab7[i], np.array(i7['runs'][i])[6:] - b7, r['sgr'][-1] - r['mw'][-1], r['lmc'][-1] - r['mw'][-1]))
    out = []
    for lab, dth, s, l in rows:
        W, E = orbit(s, l)
        o = W - tgt
        k = np.argmin(np.linalg.norm(o[:, :3], axis=1))
        Et = 0.5 * tgt[3:] @ tgt[3:] + mw.potential(tgt[:3]) + lmc.potential(tgt[:3] - (l[:3] + l[3:] * ts[k]))
        dL = np.cross(W[k, :3], W[k, 3:]) - np.cross(tgt[:3], tgt[3:])
        out.append(dict(lab=lab, dth=dth, tc=ts[k] * TU, dx=o[k, :3], dv=o[k, 3:], dE=E[k] - Et, dL=dL))
    return out


def fit(X, Y):
    A = np.column_stack((np.ones(len(X)), X))
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    return coef, A @ coef


if __name__ == '__main__':
    runs = collect()
    X = np.array([r['dth'] for r in runs]) / SCALE                     # start change in round-7 nudge units
    Y = np.array([np.r_[r['dv'], r['dx']] for r in runs])               # closest-approach dv (km/s), dx (kpc)
    Yel = np.array([np.r_[r['dE'], r['dL']] for r in runs])
    print('run             t_close  dv at closest (km/s)       dx at closest (kpc)        dE     dLz')
    for r in runs:
        print('%-15s %+6.2f   %+6.2f %+6.2f %+6.2f   %+6.3f %+6.3f %+6.3f   %+6.0f %+6.1f' % (r['lab'], r['tc'], *r['dv'], *r['dx'], r['dE'], r['dL'][2]))
    coef, Yhat = fit(X, Y)
    res = Y - Yhat
    loo = []
    for i in range(len(X)):
        m = np.arange(len(X)) != i
        c, _ = fit(X[m], Y[m])
        loo.append(Y[i] - np.r_[1, X[i]] @ c)
    loo = np.array(loo)
    print('\nfit residual rms  dv %s km/s  dx %s kpc' % (np.sqrt(np.mean(res[:, :3]**2, 0)), np.sqrt(np.mean(res[:, 3:]**2, 0))))
    print('leave-one-out rms dv %s km/s  dx %s kpc' % (np.sqrt(np.mean(loo[:, :3]**2, 0)), np.sqrt(np.mean(loo[:, 3:]**2, 0))))
    r7 = np.array([r['lab'].startswith('r7') for r in runs])
    print('r7 runs only, rms of dv about their mean (spread):', Y[r7, :3].std(0))
    print('intercept (prediction at r7 base start): dv %s, dx %s' % (coef[0, :3], coef[0, 3:]))
    B = coef[1:].T                                                       # (6 outputs, 6 inputs)
    print('B (per round-7 nudge unit), rows dvx dvy dvz dx dy dz:\n', B)
    print('singular values of B scaled by (1 km/s, 0.1 kpc):', np.linalg.svd(B / np.r_[1, 1, 1, .1, .1, .1][:, None], compute_uv=False))
    # minimum-norm shift (nudge units) zeroing dv and dx at closest approach, weighted by tolerance
    Wt = 1 / np.r_[1, 1, 1, .1, .1, .1]
    shift_u = -np.linalg.pinv(Wt[:, None] * B, rcond=1e-2) @ (Wt * coef[0])
    shift = shift_u * SCALE
    print('\nshift (min-norm, nudge units):', shift_u)
    print('shift physical: dx0 %s kpc  dv0 %s km/s' % (shift[:3], shift[3:]))
    print('predicted closest-approach dv %s, dx %s' % (coef[0, :3] + B[:3] @ shift_u, coef[0, 3:] + B[3:] @ shift_u))
    rng = np.random.default_rng(0); bs = []
    for _ in range(500):
        k = rng.integers(0, len(X), len(X))
        c, _ = fit(X[k], Y[k])
        bs.append(-np.linalg.pinv(Wt[:, None] * c[1:].T, rcond=1e-2) @ (Wt * c[0]) * SCALE)
    bs = np.array(bs)
    print('bootstrap shift median %s\n          16-84%% half-width %s' % (np.median(bs, 0), 0.5 * (np.percentile(bs, 84, 0) - np.percentile(bs, 16, 0))))
    ce, _ = fit(X, Yel)
    print('E/L fit: intercept dE %.0f dL %s; predicted at shift: dE %.0f dL %s' % (ce[0, 0], ce[0, 1:], ce[0, 0] + ce[1:, 0] @ shift_u,
          ce[0, 1:] + ce[1:, 1:].T @ shift_u))
    np.savez('r8_timingfree.npz', X=X, Y=Y, coef=coef, shift=shift, bs=bs, labels=[r['lab'] for r in runs])
