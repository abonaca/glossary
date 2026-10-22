import numpy as np
exec(open('/home/ana/projects/glossary/scripts/tune_analysis/sgr_restricted.py').read().split('# 1) calibrate')[0])
th0 = np.array(info['theta']); q = 3.
d = vs.best_delay(tr); DL, DS = -d['lmc'][0], -d['sgr'][0]
th_ind = vs.delayed_theta(th0, DL, DS, dict(lnL_sgr=q), tr['sgr_m'][0])
drv = drivers(base, delay=DL); xm, xl, td = drv
ts = np.linspace(-3, 0, 25)
print('driver check: LMC-MW at t=0 (should be ~ round-0 at -0.148)', xl(0.) - xm(0.), ' target', vt.TARGET[:3])
print('MW pos at -3, -2.852:', xm(-3.), xm(-3 + DL))
a = sgr_orbit(th_ind[6:], drv, q, tr['sgr_t'], tr['sgr_m'], ts, delay=DS)
b = sgr_orbit(th0[6:], drivers(base), q, tr['sgr_t'], tr['sgr_m'], ts)
live = base['sgr'] - base['mw']
print('  t   r_delayedmodel  r_round0model  r_live')
for j in range(25): print('%6.3f %7.1f %7.1f %7.1f' % (ts[j], np.linalg.norm(a[j, :3]), np.linalg.norm(b[j, :3]), np.linalg.norm(live[j, :3])))
th = th_ind[6:].copy(); th[0] += 0.3
print('f(th+dx) - f(th):', sgr_orbit(th, drv, q, tr['sgr_t'], tr['sgr_m'], np.array([-3., 0.]), delay=DS)[-1] - a[-1])
