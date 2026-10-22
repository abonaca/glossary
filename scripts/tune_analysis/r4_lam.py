import sys, json, pickle, numpy as np
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import v21_surrogate as vs, v21_tune as vt
from scipy.optimize import least_squares
np.set_printoptions(precision=2, suppress=True, linewidth=200)
O = '/home/ana/projects/glossary/output/v21_sim/tune/'
R = pickle.load(open(O + 'round03/rigid_jacobian.pkl', 'rb')); Jr, F0, p = R['J'], R['F0'], R['p']
sm = np.load(O + 'round02/sgr_mass.npz'); i3 = json.load(open(O + 'round03/round.json'))
th3, r3 = np.array(i3['theta']), np.array(i3['residual']); W = 1 / vt.TOL
J3 = np.array(i3['jacobian']); Jh = J3.copy(); Jh[:, [0, 2]] = Jr[:, [0, 2]]
rig = lambda d: W[:6] * (vs.endpoint(th3 + np.concatenate((d, np.zeros(6))), p, sm['times'], sm['mass']) - F0)[:6]
print(' lam   |step|z0  linear LMC miss (max)   rigid-forecast LMC miss (max)    step')
for lab, J in (('hybrid', Jh), ('rigid', Jr)):
    Jw = W[:6, None] * J[:6, :6]; rw = r3[:6] * W[:6]; A = Jw.T @ Jw
    for lam in (0, 0.03, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0):
        dl = -np.linalg.solve(A + lam * np.diag(np.diag(A)), Jw.T @ rw)
        lin, act = rw + Jw @ dl, rw + rig(dl)
        print('%-6s %4.2f  %6.2f   %4.1f  %s   %4.1f  %s   %s' % (lab, lam, dl[2], abs(lin).max(), lin, abs(act).max(), act, dl))
# offset-corrected nonlinear solve in the rigid model (like the Sgr solve in the evolving-field model)
res = least_squares(lambda d: r3[:6] * W[:6] + rig(d), np.zeros(6), x_scale=np.array([1, 1, 1, .1, .1, .1]))
print('rigid nonlinear solve: step', res.x, ' forecast LMC miss', res.fun)
