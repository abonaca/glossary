import numpy as np, r7_df as rd, r7_smooth as sm, v21_surrogate as vs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('default')
m = rd.load(cubic=True)
t, live = m['times'], m['live']
d = np.load('r7_smooth.npz')
lin = sm.linearise(m, d['L'])
Ls = {'smooth, $\\alpha$ = %.3g (CV)' % d['alpha']: d['L'],
      '$\\alpha$ = 0.01 (nearly unregularised)': sm.solve(lin, 0.01),
      'constant 2.09': np.full(len(t), 2.094),
      'V21, 3': np.full(len(t), 3.)}
gyr = rd.TU / 1e3
ts = np.linspace(t[0], t[-1], 1201)
r = rd.integrate(m, d['L'], t[0], t[-1], live[0], t_eval=t)[1]          # only for the r(t) panel: use the live track
from scipy.interpolate import CubicHermiteSpline
lv = CubicHermiteSpline(t, live[:, :3], live[:, 3:])
x, v = lv(ts), lv.derivative()(ts)
ms = np.interp(ts, m['tr']['sgr_t'], m['tr']['sgr_m'])
ch1 = np.array([vs._friction_rho(m['ev'].halo_density(xi, ti), vi, mi, 1., xi) @ vi / np.linalg.norm(vi)
                for xi, vi, ti, mi in zip(x, v, ts, ms)])
fig, ax = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
ax[0].plot(ts * gyr, np.linalg.norm(x, axis=1), 'k-')
ax[0].set_ylabel('Sgr distance [kpc]')
ax[1].plot(ts * gyr, ms * 232500 / 1e8, 'k-')
ax[1].set_ylabel('Sgr mass within 5 kpc\n[$10^8$ M$_\\odot$]')
for k, (lab, L) in enumerate(Ls.items()):
    ls = '-' if k < 2 else ':'
    Lt = np.interp(ts, t, L)
    ax[2].plot(ts * gyr, Lt, ls, label=lab)
    ax[3].plot(ts * gyr, ch1 * Lt / gyr, ls, label=lab)
ax[2].plot(t * gyr, d['L'], 'C0o', ms=3)
ax[2].set_ylabel('ln$\\Lambda$')
ax[2].legend(fontsize=8)
ax[3].set_ylabel('drag along $v$\n[km/s/Gyr]')
ax[3].set_xlabel('time [Gyr]')
for a in ax:
    for tj in t * gyr:
        a.axvline(tj, color='0.9', lw=0.5, zorder=0)
fig.tight_layout()
fig.savefig('/home/ana/projects/glossary/plots/tune_r7_sgr_lnL_smooth.png', dpi=200)
print('alpha=0.01 lnL:', np.round(sm.solve(lin, 0.01), 2))
