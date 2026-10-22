"""
Sgr's present-day state of each round-6/7 run (shared LMC) integrated +-20 Myr in round 7's t=0 fields;
E (MW + LMC at its moving position) vs Lz, Ly, Lx along each, with t=0, the best-velocity point within the
position tolerance (all |dx_i| <= 0.1 kpc) and the target.
"""
import sys
sys.path.insert(0, "/home/ana/projects/glossary/scripts")
import json, numpy as np, agama, v21_tune as vt
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('default')
agama.setUnits(length=1, velocity=1, mass=232500)
T = '/home/ana/projects/glossary/output/v21_sim/tune/'
D = T + 'round07/run00/expansions_hi'
mw = agama.Potential(agama.Potential(D + '/halo_24.coef'), agama.Potential(D + '/disk_24.coef'))
lmc = agama.Potential(D + '/lmc_24.coef')
tgt = vt.TARGET[6:]; TU = 977.8
ts = np.linspace(-20, 20, 8001) / TU
i0 = 4000
ts[i0] = 0.


def orbit(s, l):
    xl = lambda t: l[:3] + l[3:] * t
    f = lambda t, w: np.hstack((w[3:], mw.force(w[:3]) + lmc.force(w[:3] - xl(t)) - lmc.force(-xl(t))))
    up = solve_ivp(f, (0, ts[-1]), s, t_eval=ts[i0:], rtol=1e-11, atol=1e-11, method='DOP853').y.T
    dn = solve_ivp(f, (0, ts[0]), s, t_eval=ts[i0::-1], rtol=1e-11, atol=1e-11, method='DOP853').y.T[::-1]
    W = np.vstack((dn[:-1], up))
    xls = l[:3] + l[3:] * ts[:, None]
    E = 0.5 * np.sum(W[:, 3:]**2, 1) + mw.potential(W[:, :3]) + lmc.potential(W[:, :3] - xls)
    return W, E, np.cross(W[:, :3], W[:, 3:])


names = ['x0', 'y0', 'z0', 'vx0', 'vy0', 'vz0']
runs = []
c = np.load(T + 'round06/centre_check.npz')
for i in [0] + list(range(7, 13)):
    runs.append(('r6 ' + ('base' if i == 0 else '+' + names[i - 7]), c['sgr'][i, 2] - c['mw'][i, 3], c['lmc'][i, 3] - c['mw'][i, 3]))
i7 = json.load(open(T + 'round07/round.json'))
lab7 = ['base'] + ['+' + n for n in names] + ['%+g Myr' % x for x in i7['scan_dt']]
for i in range(13):
    r = np.load(T + 'round07/run%02d/centres.npz' % i)
    runs.append(('r7 ' + lab7[i], r['sgr'][-1] - r['mw'][-1], r['lmc'][-1] - r['mw'][-1]))
l7 = runs[7][2]
xlt = l7[:3]
Et = 0.5 * tgt[3:] @ tgt[3:] + mw.potential(tgt[:3]) + lmc.potential(tgt[:3] - xlt)
Lt = np.cross(tgt[:3], tgt[3:])
fig, ax = plt.subplots(1, 3, figsize=(17, 6))
for n, (lab, s, l) in enumerate(runs):
    W, E, L = orbit(s, l)
    o = W - tgt
    ok = np.all(np.abs(o[:, :3]) <= 0.1, axis=1)
    col = plt.cm.tab20(n % 20)
    ls = ':' if lab.startswith('r6') else ('-' if 'Myr' not in lab else '--')
    for p, k in enumerate((2, 1, 0)):
        a = ax[p]
        a.plot(L[:, k], E, ls, color=col, lw=0.8, label=lab if p == 0 else None)
        a.plot(L[i0, k], E[i0], 'o', color=col, ms=4)
        if ok.any():
            idx = np.flatnonzero(ok)
            a.plot(L[idx, k], E[idx], '-', color=col, lw=4, alpha=0.9, solid_capstyle='round')
            kb = idx[np.argmin(np.linalg.norm(o[idx, 3:], axis=1))]
            a.plot(L[kb, k], E[kb], 'D', color=col, mec='k', ms=7, zorder=5)
# tolerance region: every state with |dx_i| <= 0.1 kpc and |dv_i| <= 1 km/s of the target (box corners + random fill)
from itertools import product
from scipy.spatial import ConvexHull
tol = vt.TOL[6:]
box = np.array(list(product(*[(-1, 1)] * 6)), float)
box = np.vstack((box, np.random.default_rng(1).uniform(-1, 1, (20000, 6)))) * tol + tgt
Eb = 0.5 * np.sum(box[:, 3:]**2, 1) + mw.potential(box[:, :3]) + lmc.potential(box[:, :3] - xlt)
Lb = np.cross(box[:, :3], box[:, 3:])
print('tolerance region: E %.1f .. %.1f, Lx %.1f .. %.1f, Ly %.1f .. %.1f, Lz %.1f .. %.1f' % (
    Eb.min(), Eb.max(), *np.ravel([(Lb[:, k].min(), Lb[:, k].max()) for k in range(3)])))
for p, k in enumerate((2, 1, 0)):
    pts = np.column_stack((Lb[:, k], Eb))
    h = ConvexHull(pts)
    ax[p].fill(pts[h.vertices, 0], pts[h.vertices, 1], color='r', alpha=0.15, lw=0, zorder=0,
               label='tolerance region' if p == 0 else None)
    ax[p].plot(np.r_[pts[h.vertices, 0], pts[h.vertices[0], 0]], np.r_[pts[h.vertices, 1], pts[h.vertices[0], 1]],
               'r-', lw=0.8, zorder=0)
for p, k in enumerate((2, 1, 0)):
    ax[p].plot(Lt[k], Et, '*', color='r', mec='k', ms=18, zorder=6, label='target' if p == 0 else None)
    ax[p].set_xlabel('$L_%s$ [kpc km/s]' % 'xyz'[k])
    ax[p].set_ylabel('E (MW + LMC) [(km/s)$^2$]')
ax[0].text(0.02, 0.98, 'lines: $\\pm$20 Myr around t = 0 (circles); thick: all $|\\Delta x_i| \\leq$ 0.1 kpc;\n'
           'diamonds: smallest $|\\Delta v|$ within that; dotted: r6, solid: r7 nudges, dashed: r7 timing scan;\n'
           'red region: states within 0.1 kpc and 1 km/s of the target in every component',
           transform=ax[0].transAxes, va='top', fontsize=7)
fig.legend(loc='center right', fontsize=7)
fig.tight_layout(rect=(0, 0, 0.9, 1))
f = '/home/ana/projects/glossary/plots/tune_r6r7_sgr_EL_window.png'
fig.savefig(f, dpi=200)
print(f, 'target E %.1f L %s' % (Et, np.round(Lt, 1)))
