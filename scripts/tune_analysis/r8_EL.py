"""Sgr energy and angular momentum along the round-8 runs (round-7 base-run expansions_hi), with the round-7 base,
the target at t=0 and the tolerance region."""
import sys
sys.path.insert(0, "/home/ana/projects/glossary/scripts")
import json, numpy as np, agama, v21_tune as vt
from itertools import product
from scipy.spatial import ConvexHull
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('default')
agama.setUnits(length=1, velocity=1, mass=232500)
T = '/home/ana/projects/glossary/output/v21_sim/tune/'
D = T + 'round07/run00/expansions_hi'
base = dict(np.load(T + 'round07/run00/centres.npz'))
n = len(base['times'])
mw = [agama.Potential(agama.Potential(D + '/halo_%02d.coef' % j), agama.Potential(D + '/disk_%02d.coef' % j)) for j in range(n)]
lmc = [agama.Potential(D + '/lmc_%02d.coef' % j) for j in range(n)]
xl = base['lmc'][:, :3] - base['mw'][:, :3]


def EL(s, j):
    s = np.atleast_2d(s)
    x, v = s[:, :3], s[:, 3:]
    return 0.5 * np.sum(v**2, 1) + mw[j].potential(x) + lmc[j].potential(x - xl[j]), np.cross(x, v)


def track(f):
    r = dict(np.load(f))
    rel = r['sgr'] - r['mw']
    E, L = zip(*[EL(rel[j], j) for j in range(n)])
    return np.ravel(E), np.vstack(L)


info = json.load(open(T + 'round08/round.json'))
runs = [('r7 base', 'k', track(T + 'round07/run00/centres.npz'))]
for i, lab in enumerate(info['labels']):
    runs.append(('r8 ' + lab.replace('J7 LM lambda', '$\\lambda$'), plt.cm.viridis(i / 6),
                 track(T + 'round08/run%02d/centres.npz' % i)))
tgt = vt.TARGET[6:]
Et, Lt = EL(tgt, n - 1)
Et, Lt = Et[0], Lt[0]
# tolerance region: states within 0.1 kpc and 1 km/s of the target in every component
box = np.array(list(product(*[(-1, 1)] * 6)), float)
box = np.vstack((box, np.random.default_rng(1).uniform(-1, 1, (20000, 6)))) * vt.TOL[6:] + tgt
Eb, Lb = EL(box, n - 1)

fig, ax = plt.subplots(2, 3, figsize=(15, 9))
for row in range(2):
    for p, k in enumerate((2, 1, 0)):
        a = ax[row, p]
        sel = slice(None) if row == 0 else slice(-2, None)
        for lab, col, (E, L) in runs:
            a.plot(L[sel, k], E[sel], '-', color=col, lw=2 if lab == 'r7 base' else 0.9, label=lab,
                   zorder=3 if lab == 'r7 base' else 2)
            a.plot(L[-1, k], E[-1], 'o', color=col, ms=6, mec='k', mew=0.5, zorder=4)
            if row == 0:
                a.plot(L[0, k], E[0], 's', color=col, ms=4, zorder=4)
        pts = np.column_stack((Lb[:, k], Eb))
        h = ConvexHull(pts).vertices
        a.fill(pts[h, 0], pts[h, 1], color='r', alpha=0.15, lw=0, zorder=0, label='tolerance region')
        a.plot(np.r_[pts[h, 0], pts[h[0], 0]], np.r_[pts[h, 1], pts[h[0], 1]], 'r-', lw=0.8, zorder=0)
        a.plot(Lt[k], Et, '*', color='r', ms=16, mec='k', zorder=5, label='target, t=0')
        a.set_xlabel('$L_%s$ [kpc km/s]' % 'xyz'[k])
        a.set_ylabel('E (MW + LMC) [(km/s)$^2$]')
        if row == 1:
            near = [L for lab, _, (E, L) in runs]
            Ee = np.r_[[E[-1] for _, _, (E, L) in runs], Eb]
            Le = np.r_[[L[-1, k] for _, _, (E, L) in runs], Lb[:, k]]
            pe, pl = 0.08 * np.ptp(Ee), 0.08 * np.ptp(Le)
            a.set_xlim(Le.min() - pl, Le.max() + pl); a.set_ylim(Ee.min() - pe, Ee.max() + pe)
ax[0, 2].legend(fontsize=7, loc='best')
ax[0, 0].text(0.02, 0.97, 'squares: t = -3, circles: t = 0', transform=ax[0, 0].transAxes, va='top', fontsize=8)
ax[1, 0].text(0.02, 0.97, 'zoom: last snapshot interval (t = -0.12 to 0)', transform=ax[1, 0].transAxes,
              va='top', fontsize=8)
fig.tight_layout()
f = '/home/ana/projects/glossary/plots/tune_r8_sgr_EL.png'
fig.savefig(f, dpi=200)
print(f)
print('tolerance region: E %.1f .. %.1f, Lx %.1f .. %.1f, Ly %.1f .. %.1f, Lz %.1f .. %.1f' % (
    Eb.min(), Eb.max(), *np.ravel([(Lb[:, k].min(), Lb[:, k].max()) for k in range(3)])))
for lab, _, (E, L) in runs:
    print('%-22s E0 %9.1f E_end %9.1f  L_end %s' % (lab, E[0], E[-1], np.round(L[-1], 1)))
print('%-22s          E     %9.1f  L     %s' % ('target', Et, np.round(Lt, 1)))
