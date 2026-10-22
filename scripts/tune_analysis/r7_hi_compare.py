"""Current (lmax 4) vs higher-order (lmax 8, mmax 6) expansions: interval misses, smooth lnL(t) with M(<20 kpc), backward starts."""
import numpy as np, r7_df as rd, r7_smooth as sm, v21_tune as vt
np.set_printoptions(precision=3, suppress=True, linewidth=170)
alphas = np.logspace(-3, 4, 15)
for exp in ('expansions', 'expansions_hi'):
    m = rd.load(cubic=True, mass=20., exp=exp)
    t, live = m['times'], m['live']
    m0 = sm.misses(m, 0.)
    lin = sm.linearise(m, np.full(len(t), 2.))
    e = sm.cv(lin, alphas)
    a = alphas[np.argmin(e)]
    L = sm.solve(sm.linearise(m, sm.solve(lin, a)), a)
    mi = sm.misses(m, L)
    rms = lambda q: (np.sqrt(np.mean(np.sum(q[:, :3]**2, 1))), np.sqrt(np.mean(np.sum(q[:, 3:]**2, 1))))
    e1 = live[-1].copy(); e1[3] = vt.TARGET[9]
    s0 = rd.integrate(m, L, t[-1], t[0], live[-1])[1][-1]
    s1 = rd.integrate(m, L, t[-1], t[0], e1)[1][-1]
    fw = rd.integrate(m, L, t[0], t[-1], live[0])[1][-1] - live[-1]
    print('=== %s' % exp)
    print('  interval misses, no drag: rms %.3f kpc %.2f km/s' % rms(m0))
    print('  per-interval |dv| no drag:', np.linalg.norm(m0[:, 3:], axis=1))
    print('  smooth lnL(t), M(<20): alpha %.3g, CV %.2f, chi2 %.2f, rms %.3f kpc %.2f km/s' % ((a, e.min(), np.sum((mi * rd.W)**2)) + rms(mi)))
    print('  per-interval |dv| fitted:', np.linalg.norm(mi[:, 3:], axis=1))
    print('  lnL:', L)
    print('  forward from live t=-3, t=0 minus live:', fw)
    print('  backward live end   -> start', s0, '  minus r7 start', s0 - m['theta'][6:])
    print('  backward vx target  -> start', s1)
    print('  difference', s1 - s0, flush=True)
