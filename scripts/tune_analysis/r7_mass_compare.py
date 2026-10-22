"""Smooth lnL(t) (curvature penalty, CV) with each Sgr mass series in the drag; backward start error."""
import numpy as np, r7_df as rd, r7_smooth as sm
np.set_printoptions(precision=3, suppress=True, linewidth=160)
alphas = np.logspace(-3, 4, 15)
res = {}
import sys
MASSES = {'5': None, '10': 10., '20': 20., 'bound': 'bound', 'bound_held': 'bound_held'}
for mass in [MASSES[a] for a in sys.argv[1:]]:
    m = rd.load(cubic=True, mass=mass)
    t, live = m['times'], m['live']
    lin = sm.linearise(m, np.full(len(t), 2.))
    e = sm.cv(lin, alphas)
    a = alphas[np.argmin(e)]
    L = sm.solve(sm.linearise(m, sm.solve(lin, a)), a)
    mi = sm.misses(m, L)
    bw = rd.integrate(m, L, t[-1], t[0], live[-1])[1][-1] - m['theta'][6:]
    fw = rd.integrate(m, L, t[0], t[-1], live[0])[1][-1] - live[-1]
    name = 'M(<5 kpc)' if mass is None else (mass if isinstance(mass, str) else 'M(<%g kpc)' % mass)
    print('=== %s: alpha %.3g, CV %.2f, chi2 %.2f' % (name, a, e.min(), np.sum((mi * rd.W)**2)))
    print('  lnL:', L)
    print('  lnL x M [1e8 Msun] at snapshots:', (L * m['tr']['sgr_m'] * 232500 / 1e8).round(1))
    print('  backward start - true start:', bw, ' |dx| %.2f kpc |dv| %.1f km/s' % (np.linalg.norm(bw[:3]), np.linalg.norm(bw[3:])))
    print('  forward end - live end:', fw, ' |dx| %.2f kpc |dv| %.1f km/s' % (np.linalg.norm(fw[:3]), np.linalg.norm(fw[3:])), flush=True)
    res[name] = dict(L=L, alpha=a, cv=e, mass=m['tr']['sgr_m'])
np.save('r7_mass_compare_%s.npy' % '_'.join(sys.argv[1:]), res, allow_pickle=True)
