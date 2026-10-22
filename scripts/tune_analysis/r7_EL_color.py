"""Sgr E-L tracks of the round-7 runs, coloured by the initial potential or kinetic energy (exact starts)."""
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
runs0 = np.array(json.load(open(rd.D + '/round.json'))['runs'])[:, 6:]


def parts(s, j):
    x, v = s[:3], s[3:]
    return 0.5 * v @ v, mw[j].potential(x) + lmc[j].potential(x - xl[j]), np.cross(x, v)


labels = ['base', '+x0', '+y0', '+z0', '+vx0', '+vy0', '+vz0', '-1 Myr', '-0.5 Myr', '-0.25 Myr', '+0.25 Myr', '+0.5 Myr', '+1 Myr']
tracks = []
for i in range(13):
    r = dict(np.load(rd.D + '/run%02d/centres.npz' % i))
    rel = r['sgr'] - r['mw']
    P = [parts(rel[j], j) for j in range(n)]
    tracks.append((np.array([a + b for a, b, _ in P]), np.array([c for _, _, c in P])))
K0, P0 = np.array([parts(s, 0)[:2] for s in runs0]).T
kt, pt, Lt = parts(vt.TARGET[6:], n - 1)
Et = kt + pt
print('run        K0 [(km/s)^2]  Phi0          E0          E_end')
for i in range(13):
    print('%-10s %10.1f %12.1f %12.1f %10.1f' % (labels[i], K0[i], P0[i], K0[i] + P0[i], tracks[i][0][-1]))
for key, q, lab in (('Epot', P0, 'initial potential energy $\\Phi_{MW}+\\Phi_{LMC}$ [(km/s)$^2$]'),
                    ('Ekin', K0, 'initial kinetic energy [(km/s)$^2$]')):
    norm = plt.Normalize(q.min(), q.max())
    cmap = plt.cm.viridis
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    for row in range(2):
        for p, k in enumerate((2, 1, 0)):
            a = ax[row, p]
            for i, (E, L) in enumerate(tracks):
                sel = slice(None) if row == 0 else slice(-2, None)
                c = cmap(norm(q[i]))
                a.plot(L[sel, k], E[sel], color=c, lw=2 if i == 0 else 0.9, ls='-' if i < 7 else '--', zorder=2)
                a.plot(L[-1, k], E[-1], 'o', color=c, mec='k' if i == 0 else c, ms=7 if i == 0 else 5, zorder=4)
                if row == 0:
                    a.plot(L[0, k], E[0], 's', color=c, ms=4, zorder=4)
                else:
                    a.annotate(labels[i], (L[-1, k], E[-1]), fontsize=6, xytext=(3, 2), textcoords='offset points')
            a.plot(Lt[k], Et, '*', color='r', ms=16, mec='k', zorder=5)
            a.set_xlabel('$L_%s$ [kpc km/s]' % 'xyz'[k])
            a.set_ylabel('E (MW + LMC) [(km/s)$^2$]')
            if row == 1:
                Ee = np.array([E[-1] for E, _ in tracks] + [Et]); Le = np.array([L[-1, k] for _, L in tracks] + [Lt[k]])
                pe, pl = np.ptp(Ee) * 0.15 + 30, np.ptp(Le) * 0.15 + 5
                a.set_xlim(Le.min() - pl, Le.max() + pl); a.set_ylim(Ee.min() - pe, Ee.max() + pe)
    ax[0, 0].text(0.02, 0.97, 'squares: t = -3, circles: t = 0, star: target; solid: nudges, dashed: timing scan',
                  transform=ax[0, 0].transAxes, va='top', fontsize=7)
    ax[1, 0].text(0.02, 0.97, 'zoom: last snapshot interval (t = -0.12 to 0)', transform=ax[1, 0].transAxes, va='top', fontsize=8)
    fig.tight_layout(rect=(0, 0, 0.92, 1))
    cax = fig.add_axes([0.93, 0.1, 0.012, 0.8])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label=lab)
    f = '/home/ana/projects/glossary/plots/tune_r7_sgr_EL_%s.png' % key
    fig.savefig(f, dpi=200)
    print(f)
