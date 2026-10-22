"""
Energy budget of Sgr along round 7's base run (live track), E = v^2/2 + Phi_MW(x, t) + Phi_LMC(x - x_LMC, t)
with x, v relative to the MW centre and the potentials from the higher-order expansions (cubic in time).
dE/dt = -v.a_MW (frame) + dPhi_MW/dt|x + dPhi_LMC/dt|x (LMC motion + shape) + v.a_nc (remainder: friction,
force-model error).
"""
import numpy as np, r7_df as rd, v21_surrogate as vs
from scipy.interpolate import CubicHermiteSpline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('default')
np.set_printoptions(precision=1, suppress=True, linewidth=160)
m = rd.load(cubic=True, mass=20., exp='expansions_hi')
ev, run, t = m['ev'], m['run'], m['times']
rel = run['sgr'] - run['mw']
S = CubicHermiteSpline(t, rel[:, :3], rel[:, 3:])
lrel = run['lmc'] - run['mw']
Lsp = CubicHermiteSpline(t, lrel[:, :3], lrel[:, 3:])
ms = lambda tt: np.interp(tt, m['tr']['sgr_t'], m['tr']['sgr_m'])


def phis(x, tt, xl):
    k, c = ev._cw(tt)
    return (sum(ci * ev.mw[ki].potential(x) for ki, ci in zip(k, c)),
            sum(ci * ev.lmc[ki].potential(x - xl) for ki, ci in zip(k, c)))


def lmc_force(x, tt):
    k, c = ev._cw(tt)
    return sum(ci * ev.lmc[ki].force(x) for ki, ci in zip(k, c))


ts = np.linspace(t[0], t[-1], 24 * 40 + 1)
h = 1e-4
E, frame_tr, frame_ph, dmw, dlmc = [], [], [], [], []
aMW_track = ev.xm.derivative(2)
for tt in ts:
    x, v, xl = S(tt), S.derivative()(tt), Lsp(tt)
    pm, pl = phis(x, tt, xl)
    E.append(0.5 * v @ v + pm + pl)
    a_tr = aMW_track(tt)
    d = x                                                           # Sgr - MW
    a_ph = lmc_force(-xl, tt) + ms(tt) * d / (d @ d + vs.SGR_PLUMMER**2)**1.5
    frame_tr.append(-v @ a_tr); frame_ph.append(-v @ a_ph)
    tp, tm = min(tt + h, t[-1]), max(tt - h, t[0])
    dmw.append((phis(x, tp, xl)[0] - phis(x, tm, xl)[0]) / (tp - tm))
    dlmc.append((phis(x, tp, Lsp(tp))[1] - phis(x, tm, Lsp(tm))[1]) / (tp - tm))
E, frame_tr, frame_ph, dmw, dlmc = map(np.array, (E, frame_tr, frame_ph, dmw, dlmc))
dEdt = np.gradient(E, ts)
rem = dEdt - frame_ph - dmw - dlmc
I = lambda y, a, b: np.trapezoid(y[(ts >= a - 1e-9) & (ts <= b + 1e-9)], ts[(ts >= a - 1e-9) & (ts <= b + 1e-9)])
print('interval [Myr]       dE     frame(track) frame(LMC+Sgr pull)  dPhi_MW/dt  dPhi_LMC/dt  remainder (friction+error)')
for j in range(len(t) - 1):
    a, b = t[j], t[j + 1]
    print('%6.0f..%5.0f  %8.1f  %10.1f  %10.1f  %10.1f  %10.1f  %10.1f' % (a * rd.TU, b * rd.TU, I(dEdt, a, b), I(frame_tr, a, b),
          I(frame_ph, a, b), I(dmw, a, b), I(dlmc, a, b), I(rem, a, b)))
print('total         %8.1f  %10.1f  %10.1f  %10.1f  %10.1f  %10.1f' % (I(dEdt, t[0], t[-1]), I(frame_tr, t[0], t[-1]), I(frame_ph, t[0], t[-1]),
      I(dmw, t[0], t[-1]), I(dlmc, t[0], t[-1]), I(rem, t[0], t[-1])))
gyr = rd.TU / 1e3
fig, ax = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
ax[0].plot(ts * gyr, np.linalg.norm(S(ts), axis=1), 'k-'); ax[0].set_ylabel('Sgr distance [kpc]')
ax[1].plot(ts * gyr, E, 'k-', label='E (MW + LMC), live track')
ax[1].plot(t * gyr, [E[np.argmin(abs(ts - tj))] for tj in t], 'ko', ms=3, label='snapshots')
ax[1].set_ylabel('E [(km/s)$^2$]'); ax[1].legend()
for y, lab in ((frame_ph, 'frame: $-v\\cdot a_{MW}$'), (dmw, '$\\partial\\Phi_{MW}/\\partial t$'),
               (dlmc, '$\\partial\\Phi_{LMC}/\\partial t$ (LMC motion + shape)'), (rem, 'remainder (friction + model error)')):
    ax[2].plot(ts * gyr, y / gyr, label=lab)
ax[2].axhline(0, color='0.5', lw=0.8)
ax[2].set_ylabel('dE/dt [(km/s)$^2$/Gyr]'); ax[2].set_xlabel('time [Gyr]'); ax[2].legend(fontsize=8)
for a in ax:
    for tj in t * gyr:
        a.axvline(tj, color='0.9', lw=0.5, zorder=0)
fig.tight_layout()
fig.savefig('/home/ana/projects/glossary/plots/tune_r7_sgr_energy_budget.png', dpi=200)
