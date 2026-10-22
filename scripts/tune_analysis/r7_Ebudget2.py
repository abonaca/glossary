"""
Sgr energy budget per snapshot interval along model orbits restarted from the live state at each snapshot
(higher-order expansions, no drag), so the path is a real orbit. Terms as in r7_Ebudget; with no drag the
terms should add up to the model's dE, and live dE - model dE is the friction + force-model error.
"""
import numpy as np, r7_df as rd, v21_surrogate as vs
from scipy.interpolate import CubicHermiteSpline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('default')
m = rd.load(cubic=True, mass=20., exp='expansions_hi')
ev, run, t, live = m['ev'], m['run'], m['times'], m['live']
lrel = run['lmc'] - run['mw']
Lsp = CubicHermiteSpline(t, lrel[:, :3], lrel[:, 3:])
ms = lambda tt: np.interp(tt, m['tr']['sgr_t'], m['tr']['sgr_m'])


def phis(x, tt, xl):
    k, c = ev._cw(tt)
    return (sum(ci * ev.mw[ki].potential(x) for ki, ci in zip(k, c)), sum(ci * ev.lmc[ki].potential(x - xl) for ki, ci in zip(k, c)))


def lmc_force(x, tt):
    k, c = ev._cw(tt)
    return sum(ci * ev.lmc[ki].force(x) for ki, ci in zip(k, c))


def E_of(s, tt):
    pm, pl = phis(s[:3], tt, Lsp(tt))
    return 0.5 * s[3:] @ s[3:] + pm + pl


h = 1e-4
rows, curves = [], []
for j in range(len(t) - 1):
    ts = np.linspace(t[j], t[j + 1], 121)
    rel = rd.integrate(m, 0., t[j], t[j + 1], live[j], t_eval=ts)[1]
    E, fr, dmw, dl = [], [], [], []
    for tt, s in zip(ts, rel):
        x, v, xl = s[:3], s[3:], Lsp(tt)
        E.append(E_of(s, tt))
        fr.append(-v @ (lmc_force(-xl, tt) + ms(tt) * x / (x @ x + vs.SGR_PLUMMER**2)**1.5))
        tp, tm = min(tt + h, t[-1]), max(tt - h, t[0])
        dmw.append((phis(x, tp, xl)[0] - phis(x, tm, xl)[0]) / (tp - tm))
        dl.append((phis(x, tp, Lsp(tp))[1] - phis(x, tm, Lsp(tm))[1]) / (tp - tm))
    E = np.array(E)
    I = lambda y: np.trapezoid(y, ts)
    dE_live = E_of(live[j + 1], t[j + 1]) - E_of(live[j], t[j])
    rows.append((t[j], E[-1] - E[0], I(fr), I(dmw), I(dl), dE_live))
    curves.append((ts, E, np.array(fr), np.array(dmw), np.array(dl)))
rows = np.array(rows)
print('interval [Myr]   model dE   frame(-v.a_MW)  dPhi_MW/dt  dPhi_LMC/dt  sum-of-terms | live dE   live-model (friction+error)')
for r in rows:
    print('%6.0f..%5.0f  %9.1f  %10.1f  %10.1f  %10.1f  %10.1f  | %9.1f  %9.1f' % (r[0] * rd.TU, (r[0] + 0.125) * rd.TU, r[1], r[2], r[3], r[4],
          r[2] + r[3] + r[4], r[5], r[5] - r[1]))
tot = rows[:, 1:].sum(0)
print('total        %9.1f  %10.1f  %10.1f  %10.1f  %10.1f  | %9.1f  %9.1f' % (tot[0], tot[1], tot[2], tot[3], tot[1] + tot[2] + tot[3], tot[4], tot[4] - tot[0]))
gyr = rd.TU / 1e3
fig, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
for k, (ts, E, fr, dmw, dl) in enumerate(curves):
    ax[0].plot(ts * gyr, E, 'k-', lw=1)
    for y, c, lab in ((fr, 'C0', 'frame: $-v\\cdot a_{MW}$'), (dmw, 'C1', '$\\partial\\Phi_{MW}/\\partial t$'),
                      (dl, 'C2', '$\\partial\\Phi_{LMC}/\\partial t$ (LMC motion + shape)')):
        ax[1].plot(ts * gyr, y / gyr, color=c, label=lab if k == 0 else None)
ax[0].plot(t * gyr, [E_of(live[j], t[j]) for j in range(len(t))], 'ro', ms=4, label='live snapshots')
ax[0].set_ylabel('E (MW + LMC) [(km/s)$^2$]'); ax[0].legend()
ax[1].axhline(0, color='0.5', lw=0.8)
ax[1].set_ylabel('dE/dt [(km/s)$^2$/Gyr]'); ax[1].set_xlabel('time [Gyr]'); ax[1].legend(fontsize=8)
for a in ax:
    for tj in t * gyr:
        a.axvline(tj, color='0.9', lw=0.5, zorder=0)
fig.tight_layout()
fig.savefig('/home/ana/projects/glossary/plots/tune_r7_sgr_energy_budget.png', dpi=200)
