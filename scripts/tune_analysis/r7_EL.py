"""Sgr energy and angular momentum along the round-7 runs (base-run expansions), with the target at t=0."""
import json, numpy as np, agama, r7_df as rd, v21_tune as vt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('default')
D = rd.D + '/run00/expansions_hi'
base = dict(np.load(rd.D + '/run00/centres.npz'))
n = len(base['times'])
mw = [agama.Potential(agama.Potential(D + '/halo_%02d.coef' % j), agama.Potential(D + '/disk_%02d.coef' % j)) for j in range(n)]
lmc = [agama.Potential(D + '/lmc_%02d.coef' % j) for j in range(n)]
xl = base['lmc'][:, :3] - base['mw'][:, :3]


def EL(s, j):
    x, v = s[:3], s[3:]
    return 0.5 * v @ v + mw[j].potential(x) + lmc[j].potential(x - xl[j]), np.cross(x, v)


labels = ['base', '+x0', '+y0', '+z0', '+vx0', '+vy0', '+vz0', '-1 Myr', '-0.5 Myr', '-0.25 Myr', '+0.25 Myr', '+0.5 Myr', '+1 Myr']
tracks = []
for i in range(13):
    r = dict(np.load(rd.D + '/run%02d/centres.npz' % i))
    rel = r['sgr'] - r['mw']
    E, L = zip(*[EL(rel[j], j) for j in range(n)])
    tracks.append((np.array(E), np.array(L)))
Et, Lt = EL(vt.TARGET[6:], n - 1)
fig, ax = plt.subplots(2, 3, figsize=(15, 9))
cols = ['k'] + ['C%d' % k for k in range(6)] + [plt.cm.viridis(k / 5) for k in range(6)]
for row in range(2):
    for p, k in enumerate((2, 1, 0)):
        a = ax[row, p]
        for i, (E, L) in enumerate(tracks):
            sel = slice(None) if row == 0 else slice(-2, None)
            a.plot(L[sel, k], E[sel], '-', color=cols[i], lw=2 if i == 0 else 0.8, ls='-' if i < 7 else '--',
                   label=labels[i], zorder=3 if i == 0 else 2)
            a.plot(L[-1, k], E[-1], 'o', color=cols[i], ms=5 if i else 7, zorder=4)
            if row == 0:
                a.plot(L[0, k], E[0], 's', color=cols[i], ms=4, zorder=4)
        a.plot(Lt[k], Et, '*', color='r', ms=16, mec='k', zorder=5, label='target, t=0')
        a.set_xlabel('$L_%s$ [kpc km/s]' % 'xyz'[k])
        a.set_ylabel('E (MW + LMC) [(km/s)$^2$]')
        if row == 1:
            Ee = np.array([E[-1] for E, _ in tracks] + [Et]); Le = np.array([L[-1, k] for _, L in tracks] + [Lt[k]])
            pe, pl = np.ptp(Ee) + 50, np.ptp(Le) + 10
            a.set_xlim(Le.min() - pl, Le.max() + pl); a.set_ylim(Ee.min() - pe, Ee.max() + pe)
ax[0, 2].legend(fontsize=7, ncol=2, loc='best')
ax[1, 0].text(0.02, 0.97, 'zoom: last snapshot interval (t = -0.12 to 0)', transform=ax[1, 0].transAxes, va='top', fontsize=8)
ax[0, 0].text(0.02, 0.97, 'squares: t = -3, circles: t = 0', transform=ax[0, 0].transAxes, va='top', fontsize=8)
fig.tight_layout()
fig.savefig('/home/ana/projects/glossary/plots/tune_r7_sgr_EL.png', dpi=200)
for i, (E, L) in enumerate(tracks):
    print('%-10s E0 %9.1f E_end %9.1f  L_end %s' % (labels[i], E[0], E[-1], np.round(L[-1], 1)))
print('target      E %9.1f  L %s' % (Et, np.round(Lt, 1)))
