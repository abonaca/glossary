import sys, json, pickle, numpy as np
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import v21_surrogate as vs, v21_tune as vt
from scipy.optimize import least_squares
np.set_printoptions(precision=2, suppress=True, linewidth=200)
O = '/home/ana/projects/glossary/output/v21_sim/tune/'
I = {k: json.load(open(O + 'round%02d/round.json' % k)) for k in (3, 4)}
J = {k: np.array(I[k]['jacobian']) for k in I}; th = {k: np.array(I[k]['theta']) for k in I}; r = {k: np.array(I[k]['residual']) for k in I}
W = 1 / vt.TOL; p = pickle.load(open(O + 'round03/rigid_jacobian.pkl', 'rb'))['p']
sm = np.load(O + 'round02/sgr_mass.npz'); dl4 = np.array(I[4]['delta'])
F4, Jr4 = vs.jacobian(th[4], p, sm['times'], sm['mass'], delta=dl4)
Jr3 = pickle.load(open(O + 'round03/rigid_jacobian.pkl', 'rb'))['J']
B = slice(0, 6)
print('LMC->LMC block, per nudge (TOL), column = LMC start component nudged:')
for lab, A in (('live J4', J[4]), ('rigid @ r4', Jr4), ('live J3', J[3]), ('rigid @ r3', Jr3)):
    print('  %-11s' % lab, 'column norms', np.linalg.norm(W[B, None] * A[B, B] * dl4[B], axis=0))
rd = lambda A, C: np.linalg.norm(W[B, None] * (A - C)[B, B], axis=0) / np.linalg.norm(W[B, None] * C[B, B], axis=0)
print('  rel diff rigid@r4 vs J4:', rd(Jr4, J[4]), '\n  rel diff J3 vs J4      :', rd(J[3], J[4]), '\n  rel diff rigid@r3 vs rigid@r4:', rd(Jr3, Jr4))
print('\nfull LMC block, J4 (TOL per unit):\n', W[B, None] * J[4][B, B], '\nrigid @ r4:\n', W[B, None] * Jr4[B, B])
# ratio per element for the big entries
M = np.abs(W[B, None] * J[4][B, B]) > 3
print('ratio J4/rigid where |J4|>3 TOL/unit:\n', np.where(M, (J[4] / Jr4)[B, B], np.nan))
# rigid offset-corrected nonlinear LMC solve from round 4
full = lambda d: W * (vs.endpoint(th[4] + np.concatenate((d, np.zeros(6))), p, sm['times'], sm['mass']) - F4)
g = lambda d: r[4][B] * W[B] + full(d)[B]
sol = {}
for lab, x0 in (('from GN(rigid)', -np.linalg.solve(W[B, None] * Jr4[B, B], r[4][B] * W[B])), ('from GN(J4)', -np.linalg.solve(W[B, None] * J[4][B, B], r[4][B] * W[B]))):
    res = least_squares(g, x0, diff_step=1e-3, x_scale=np.array([1, 1, 1, .3, .3, .3]), xtol=1e-10, ftol=1e-10, max_nfev=300)
    sol[lab] = res.x
    print('\nrigid solve %-14s start %s\n   step %s  LMC miss %s\n   J4-linear miss for this step %s\n   rigid effect on Sgr %s' % (
        lab, x0, res.x, res.fun, r[4][B] * W[B] + W[B] * (J[4][B, B] @ res.x), full(res.x)[6:]))
print('\nGN step from J4 LMC block alone:', -np.linalg.solve(W[B, None] * J[4][B, B], r[4][B] * W[B]))
pickle.dump(dict(F0=F4, J=Jr4, p=p, solve=sol), open(O + 'round04/rigid_jacobian.pkl', 'wb'))
d = -np.linalg.solve(W[B, None] * J[4][B, B], r[4][B] * W[B])
print('\nJ4 GN step in the rigid model: LMC miss', g(d), ' Sgr effect', full(d)[6:])
for f in (0.25, 0.5):
    s = f * sol['from GN(J4)']
    print('%.2f x rigid solution: rigid LMC miss %s  J4-linear miss %s' % (f, g(s), r[4][B] * W[B] + W[B] * (J[4][B, B] @ s)))
