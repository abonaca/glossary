"""Single lnL for Sgr in round 7's base run; backward integration from the live t=0 state."""
import numpy as np, r7_df as rd
from scipy.optimize import minimize_scalar
np.set_printoptions(precision=3, suppress=True, linewidth=160)
m = rd.load(cubic=True)
t, live = m['times'], m['live']
W = rd.W


def interval_misses(L):
    return np.array([rd.integrate(m, L, t[j], t[j + 1], live[j])[1][-1] - live[j + 1] for j in range(len(t) - 1)])


def forward(L):
    return rd.integrate(m, L, t[0], t[-1], live[0], t_eval=t)[1]


fits = {}
r = minimize_scalar(lambda L: np.sum((interval_misses(L) * W)**2), bounds=(0, 10), method='bounded', options=dict(xatol=1e-3))
fits['per-interval'] = r.x
r = minimize_scalar(lambda L: np.sum(((forward(L) - live)[1:] * W)**2), bounds=(0, 10), method='bounded', options=dict(xatol=1e-3))
fits['whole-orbit forward'] = r.x
for name, L in fits.items():
    mi = interval_misses(L)
    fw = forward(L)
    bw = rd.integrate(m, L, t[-1], t[0], live[-1], t_eval=t[::-1])[1][::-1]
    print('=== %s fit: lnL = %.3f' % (name, L))
    print('  interval misses, rms: %.3f kpc %.2f km/s' % (np.sqrt(np.mean(np.sum(mi[:, :3]**2, 1))), np.sqrt(np.mean(np.sum(mi[:, 3:]**2, 1)))))
    print('  forward from live t=-3, at t=0 minus live:', fw[-1] - live[-1])
    print('  backward from live t=0, at t=-3 minus live t=-3:', bw[0] - live[0])
    print('  backward from live t=0, at t=-3 minus round-7 start theta:', bw[0] - m['theta'][6:])
    print('  max |backward - live| along orbit: %.2f kpc %.2f km/s' % (np.linalg.norm(bw[:, :3] - live[:, :3], axis=1).max(),
                                                                      np.linalg.norm(bw[:, 3:] - live[:, 3:], axis=1).max()))
    print('     t[Myr]   |fw-live| kpc km/s   |bw-live| kpc km/s')
    for j in range(0, len(t), 3):
        print('    %6.0f   %6.2f %6.2f   %6.2f %6.2f' % (t[j] * rd.TU, np.linalg.norm(fw[j, :3] - live[j, :3]), np.linalg.norm(fw[j, 3:] - live[j, 3:]),
                                                     np.linalg.norm(bw[j, :3] - live[j, :3]), np.linalg.norm(bw[j, 3:] - live[j, 3:])), flush=True)
print('live t=-3 measured centre minus theta start:', live[0] - m['theta'][6:])
for L in (0., 3.):
    bw0 = rd.integrate(m, L, t[-1], t[0], live[-1])[1][-1]
    print('lnL = %g: backward t=-3 minus theta start:' % L, bw0 - m['theta'][6:])
