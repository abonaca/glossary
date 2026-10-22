"""Initial (t=-3) Sgr star vs dark-matter distribution in R2: masses, radii, particle masses, profiles."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('default')
MU = 232500.
s = np.load('/home/ana/projects/glossary/output/v21_sim/tune/sgr_init_R2.npz')
xv, m = s['xv'].astype(float), s['m'] * MU
st = np.arange(len(m)) < 200000
c = np.average(xv[st, :3], axis=0, weights=m[st]); cv = np.average(xv[:, 3:], axis=0, weights=m)
r = np.linalg.norm(xv[:, :3] - c, axis=1)
v = np.linalg.norm(xv[:, 3:] - cv, axis=1)
for lab, w in (('stars', st), ('DM', ~st), ('total', np.ones_like(st))):
    rr, mm = r[w], m[w]
    o = np.argsort(rr); cm = np.cumsum(mm[o]) / mm.sum()
    q = [rr[o][np.searchsorted(cm, f)] for f in (0.1, 0.5, 0.9, 0.99)]
    print('%-6s N %6d  M %.3g Msun  r10/50/90/99 %s kpc  rmax %.2f  sigma_1D %.1f km/s  particle mass %s' % (
        lab, w.sum(), mm.sum(), np.round(q, 2), rr.max(), np.sqrt(np.average((xv[w, 3:] - cv)**2, axis=0, weights=mm).mean()),
        np.unique(np.round(mm, 1))[:4]))
for R in (1, 2, 5, 10):
    print('within %2d kpc: stars %.3g, DM %.3g Msun, DM/stars %.1f' % (R, m[st & (r < R)].sum(), m[~st & (r < R)].sum(),
          m[~st & (r < R)].sum() / m[st & (r < R)].sum()))
mdm = m[~st]
for u in np.unique(np.round(mdm, 1)):
    w = np.round(mdm, 1) == u
    print('DM particle mass %.1f Msun: N %d, radii median %.2f, range %.2f-%.2f kpc' % (u, w.sum(), np.median(r[~st][w]),
          r[~st][w].min(), r[~st][w].max()))
e = np.logspace(-2, 1.3, 50); rc = np.sqrt(e[1:] * e[:-1]); vol = 4 / 3 * np.pi * np.diff(e**3)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
for lab, w, col in (('stars', st, 'tab:red'), ('dark matter', ~st, 'k'), ('total', np.ones_like(st), '0.6')):
    ax[0].loglog(rc, np.histogram(r[w], e, weights=m[w])[0] / vol, '-', color=col, label=lab)
    o = np.argsort(r[w]); ax[1].semilogx(r[w][o], np.cumsum(m[w][o]), '-', color=col, label=lab)
    sig = [np.std(xv[w][(r[w] > a) & (r[w] < b), 3:] - cv) if ((r[w] > a) & (r[w] < b)).sum() > 30 else np.nan
           for a, b in zip(e[:-1], e[1:])]
    ax[2].semilogx(rc, sig, '-', color=col, label=lab)
for k, mm in enumerate((1, 2.5, 10)):
    w = ~st & (np.abs(m / m[~st].min() - mm) < 0.01)
    ax[0].loglog(rc, np.histogram(r[w], e, weights=m[w])[0] / vol, ':', color='C%d' % k, label='DM, %g$\\times$ particle mass' % mm)
ax[0].set_ylabel('density [M$_\\odot$ kpc$^{-3}$]'); ax[1].set_ylabel('enclosed mass [M$_\\odot$]')
ax[2].set_ylabel('1D velocity dispersion [km/s]')
for a in ax: a.set_xlabel('r [kpc]'); a.set_xlim(0.03, 20)
ax[0].set_ylim(1e3, 1e10); ax[0].legend(fontsize=8)
fig.tight_layout()
f = '/home/ana/projects/glossary/plots/sgr_init_R2_profiles.png'; fig.savefig(f, dpi=200); print(f)
