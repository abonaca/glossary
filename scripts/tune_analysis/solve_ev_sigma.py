import sys, json, pickle, numpy as np
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import v21_surrogate as vs, v21_tune as vt
from scipy.optimize import least_squares
np.set_printoptions(precision=3, suppress=True, linewidth=200)
O = '/home/ana/projects/glossary/output/v21_sim/tune/'
tr = vs.load_tracks(1); info = tr['info']; base = tr['runs'][0]; th1 = np.array(info['theta'])
ev = vs.EvolvingField(O + 'round01/expansions', base)
c = pickle.load(open(O + 'round01/round2_candidates.pkl', 'rb'))
vs.set_sigma_profile(tuple(np.load('/home/ana/projects/glossary/output/v21_sim/ics/mw_triax_full/halo_sigma.npy')))
L_EV, L_R = 1.89, 2.60
live = vs._live_rel(base)[-1, 6:]; W = 1 / vt.TOL[6:]
off_ev = vs.sgr_evolving(ev, tr, base, th1[6:], [0.], lnL=L_EV)[-1] - live
drv = vs.drivers(tr, base); ends = np.array([-3., 0.])
off_r = vs.sgr_restricted(tr, th1[6:], drv, ends, lnL=L_R)[-1] - live
f_ev = lambda s: (vs.sgr_evolving(ev, tr, base, s, [0.], lnL=L_EV)[-1] - off_ev - vt.TARGET[6:]) * W
f_r = lambda s: (vs.sgr_restricted(tr, s, drv, ends, lnL=L_R)[-1] - off_r - vt.TARGET[6:]) * W
print('previous rigid solution:  evolving sigma(r) resid/TOL', f_ev(c['rigid'][6:]), '\n                          rigid sigma(r)    resid/TOL', f_r(c['rigid'][6:]), flush=True)
res = least_squares(f_ev, c['rigid'][6:], x_scale=np.array([0.1, 0.1, 0.1, 1, 1, 1.]), diff_step=1e-5)
s = res.x
print('evolving sigma(r) solution: own resid/TOL', res.fun, '\n                            rigid sigma(r) resid/TOL', f_r(s))
print('Sgr start', s, '\n  minus previous rigid solution', s - c['rigid'][6:], '\n  minus round 1', s - th1[6:])
c['ev_sigma'] = np.concatenate((c['theta_lmc'], s)); c['lnL_ev_sigma'] = L_EV
pickle.dump(c, open(O + 'round01/round2_candidates.pkl', 'wb'))
th0 = np.array(json.load(open(O + 'round00/round.json'))['theta'])
starts = {'round 0 start': th0, 'round 1 start': th1, 'round 2 alt. (delay)': np.concatenate((c['theta_lmc'], c['sgr_delay'])),
          'round 2 rigid-model solution': c['rigid'], 'round 2 evolving sigma(r) solution': c['ev_sigma']}
vs.plot_orbits(ev, tr, starts, s, '/home/ana/projects/glossary/plots/tune_r2_orbits.png', lnL=L_EV)
