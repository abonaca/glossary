import numpy as np
exec(open('/home/ana/projects/glossary/scripts/tune_analysis/sgr_restricted.py').read().split('# 1) calibrate')[0])
q = 3.0
TS = vt.TARGET[6:]; W = 1 / vt.TOL[6:]
th0 = np.array(info['theta']); live = base['sgr'] - base['mw']
F0 = sgr_orbit(th0[6:], drivers(base), q, tr['sgr_t'], tr['sgr_m'], base['times'])[-1]
offset = F0 - live[-1]
d = vs.best_delay(tr); DL, DS = -d['lmc'][0], -d['sgr'][0]
th_ind = vs.delayed_theta(th0, DL, DS, dict(lnL_sgr=q), tr['sgr_m'][0])     # independent delays
drv = drivers(base, delay=DL)
tend = np.array([vt.T_START, vt.T_END])
f = lambda th: sgr_orbit(th, drv, q, tr['sgr_t'], tr['sgr_m'], tend, delay=DS)[-1]
print('Sgr endpoint with LMC delayed, Sgr independent-delay start: (model - target)/TOL', (f(th_ind[6:]) - TS) / W**-1)
print('   same, offset-corrected prediction of the live run:          ', (f(th_ind[6:]) - offset - TS) * W)
for name, off in (('offset-corrected', offset), ('uncorrected', np.zeros(6))):
    r = least_squares(lambda th: (f(th) - off - TS) * W, th_ind[6:], x_scale=np.array([1, 1, 1, 10, 10, 10.]))
    print('%s: max|resid|/TOL %.2f\n  Sgr theta %s\n  shift vs independent delay %s\n  shift vs round 0 %s' % (name,
          np.max(np.abs(r.fun)), r.x, r.x - th_ind[6:], r.x - th0[6:]))
    if name == 'offset-corrected':
        np.save('/home/ana/projects/glossary/scripts/tune_analysis/theta_round1.npy', np.concatenate((th_ind[:6], r.x)))
