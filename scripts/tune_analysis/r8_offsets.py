"""Round-8 Sgr start offsets (relative to the round-7 base; LMC fixed at round 6/7)."""
import json, numpy as np
from r8_timingfree import fit, SCALE
np.set_printoptions(precision=4, suppress=True, linewidth=170)
T = '/home/ana/projects/glossary/output/v21_sim/tune/'
i7 = json.load(open(T + 'round07/round.json'))
J = np.array(i7['jacobian_sgr']); base = np.array(i7['runs'][0])
tol = np.array(i7['tol'])[6:]
# A: timing-free fit, 3 strongest directions (exact)
d = np.load('r8_timingfree.npz'); X, Y = d['X'], d['Y']
Wt = 1 / np.r_[1, 1, 1, .1, .1, .1]
c, _ = fit(X, Y)
U, S, Vt = np.linalg.svd(Wt[:, None] * c[1:].T)
A = -sum(Vt[i] * (U[:, i] @ (Wt * c[0])) / S[i] for i in range(3)) * SCALE
# B: round-7 Jacobian solved for vx only
Bv = np.linalg.solve(J, np.r_[0, 0, 0, 239.5 - (np.array(i7['endpoints'])[0][9]), 0, 0])
# C: speed along the start velocity
v0 = base[9:12]; C = np.r_[0, 0, 0, v0 / np.linalg.norm(v0)]
# D: backward-integration direction (higher-order fields, smooth lnL(t), M(<20 kpc)), scaled to |A| in nudge units
Dr = np.array([2.841, -3.030, 5.999, -28.632, -0.308, 10.660])
nu = lambda s: np.linalg.norm(s / SCALE)
Dn = Dr * nu(A) / nu(Dr)
runs = []
# A, C, D: amplitudes so that the J7-predicted present-day position change is 0.5, 1.0, 1.5 kpc (~1.5-4.5 Myr of timing)
pos = lambda o: np.linalg.norm(J[:3] @ o)
for name, o in (('A timing-free', A), ('C speed', C), ('D backward', Dr)):
    pass
for target in (0.5, 1.0, 1.5): runs.append(('A timing-free x%.3f' % (target / pos(A)), target / pos(A) * A))
for a in (0.5, 1., 1.5): runs.append(('B J7 vx-only x%g' % a, a * Bv))
for target in (0.5, 1.0, 1.5): runs.append(('C speed +%.3f km/s' % (target / pos(C)), target / pos(C) * C))
for target in (0.5, 1.0, 1.5): runs.append(('D backward x%.4f' % (target / pos(Dr)), target / pos(Dr) * Dr))
# random: Gaussian in the 5 directions orthogonal to J7's dominant (timing) direction, tolerance-weighted
Jn = (J / tol[:, None]) * SCALE[None, :]
u, s, vt = np.linalg.svd(Jn)
rng = np.random.default_rng(2026)
sig = np.minimum(2.0, 0.5 / s[1:])            # nudge units: predicted change <= ~0.5 tolerance per direction, capped at 2 nudges
print('J7 singular values (tolerance units per nudge unit):', s)
print('random-draw sigma per direction (nudge units):', sig)
for k in range(12):
    z = rng.normal(0, 1, 5) * sig
    runs.append(('R random %02d' % (k + 1), (z @ vt[1:]) * SCALE))
print('\nrun  label                    dx0 (kpc)                 dv0 (km/s)                    | nudge-units  J7-pred end change: pos (kpc) / vel (km/s)')
for i, (lab, o) in enumerate(runs):
    p = J @ o
    print('%2d   %-22s %+8.4f %+8.4f %+8.4f   %+8.4f %+8.4f %+8.4f | %5.2f   %+6.3f %+6.3f %+6.3f / %+6.2f %+6.2f %+6.2f' % (i + 1, lab, *o, nu(o), *p))
json.dump(dict(base=list(base), offsets=[list(o) for _, o in runs], labels=[l for l, _ in runs]), open('r8_offsets.json', 'w'), indent=1)
