import sys, json, pickle, numpy as np
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import v21_surrogate as vs, v21_tune as vt
from scipy.optimize import least_squares
np.set_printoptions(precision=2, suppress=True, linewidth=200)
O = '/home/ana/projects/glossary/output/v21_sim/tune/'
R = pickle.load(open(O + 'round03/rigid_jacobian.pkl', 'rb')); F0, p = R['F0'], R['p']
sm = np.load(O + 'round02/sgr_mass.npz'); i3 = json.load(open(O + 'round03/round.json'))
th3, r3 = np.array(i3['theta']), np.array(i3['residual']); W = 1 / vt.TOL
full = lambda d: W * (vs.endpoint(th3 + np.concatenate((d, np.zeros(6))), p, sm['times'], sm['mass']) - F0)
g = lambda d: r3[:6] * W[:6] + full(d)[:6]
# integrator noise check: does the response change smoothly at small steps?
for h in (1e-6, 1e-4, 1e-2):
    print('d z0 = %g: response/h' % h, (g(np.array([0, 0, h, 0, 0, 0])) - g(np.zeros(6))) / h)
out = {}
for lab, x0 in (('from 0', np.zeros(6)), ('from lam 0.1', np.array([0.99, 0.12, -5.75, -0.89, 0.2, 0.35])),
                ('from full GN', np.array([0.81, -1.14, -9.44, -0.83, 1.2, 0.93]))):
    res = least_squares(g, x0, diff_step=1e-3, x_scale=np.array([1, 1, 1, .3, .3, .3]), xtol=1e-10, ftol=1e-10, max_nfev=300)
    e = full(res.x)[6:]
    print('%-13s %s  nfev %d\n   step %s\n   LMC miss /TOL %s  (max %.2f)\n   rigid effect on Sgr /TOL %s' % (
        lab, res.message, res.nfev, res.x, res.fun, abs(res.fun).max(), e))
    out[lab] = res.x
pickle.dump(out, open(O + 'round03/rigid_lmc_solve.pkl', 'wb'))
s = out['from full GN']; Jr = R['J']; J3 = np.array(i3['jacobian'])
print('\nper-column contributions to the LMC change (rigid J, TOL):')
print(W[:6, None] * Jr[:6, :6] * s)
print('J3 linear prediction of LMC miss for this step:', r3[:6] * W[:6] + W[:6] * (J3[:6, :6] @ s))
for f in (0.25, 0.5, 0.75):
    print('fraction %.2f: rigid LMC miss %s, Sgr effect %s' % (f, g(f * s), full(f * s)[6:]))
