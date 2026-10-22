import sys, time, pickle, numpy as np
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import v21_surrogate as vs, v21_tune as vt
from scipy.optimize import least_squares
np.set_printoptions(precision=3, suppress=True, linewidth=200)
O = '/home/ana/projects/glossary/output/v21_sim/tune/'
vs.set_sigma_profile(tuple(np.load('/home/ana/projects/glossary/output/v21_sim/ics/mw_triax_full/halo_sigma.npy')))
tr = vs.load_tracks(2); info = tr['info']; base = tr['runs'][0]
c = pickle.load(open(O + 'round02/round3_candidates.pkl', 'rb')); L = c['lnL']; off = c['offset']
ev = vs.EvolvingField(O + 'round02/expansions', base)
th2 = np.array(info['theta'])[6:]; W = 1 / vt.TOL[6:]
guess = np.array([64.848, -2.523, -16.268, -34.407, -47.107, 109.217])   # first backward pass
n = [0]
def g(s):
    n[0] += 1
    r = (vs.sgr_evolving(ev, tr, base, s, [vt.T_END], lnL=L)[-1] - off - vt.TARGET[6:]) * W
    if n[0] % 10 == 1:
        print('  eval %4d  max|resid|/TOL %8.2f  start %s' % (n[0], np.max(np.abs(r)), s), flush=True)
    return r
t = time.time()
print('guess: resid/TOL', g(guess), flush=True)
res = least_squares(g, guess, x_scale=np.array([0.1, 0.1, 0.1, 1, 1, 1.]), diff_step=1e-6, xtol=1e-12, ftol=1e-12,
                    max_nfev=400)
s = res.x
print('\n%s after %d evaluations, %.0f s' % (res.message, n[0], time.time() - t))
print('far solution: resid/TOL', res.fun, '\n  start', s, '\n  minus round 2', s - th2,
      '\n  minus first backward pass', s - guess)
c['sgr_far'] = s; c['sgr_far_fun'] = res.fun
pickle.dump(c, open(O + 'round02/round3_candidates.pkl', 'wb'))
