import sys, json, pickle, numpy as np, agama
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import v21_surrogate as vs, v21_tune as vt
from scipy.integrate import solve_ivp
np.set_printoptions(precision=2, suppress=True, linewidth=200)
O = '/home/ana/projects/glossary/output/v21_sim/tune/'
I = {k: json.load(open(O + 'round%02d/round.json' % k)) for k in (2, 3, 4)}
J = {k: np.array(I[k]['jacobian']) for k in I}; th = {k: np.array(I[k]['theta']) for k in I}; r = {k: np.array(I[k]['residual']) for k in I}
W = 1 / vt.TOL; T = vt.TARGET
R = pickle.load(open(O + 'round03/rigid_jacobian.pkl', 'rb')); F0, p, Jr = R['F0'], R['p'], R['J']
sm = np.load(O + 'round02/sgr_mass.npz')
d = th[4] - th[3]
print('r3 resid/TOL', r[3] * W, '\nr4 resid/TOL', r[4] * W)
act = (r[4] - r[3]) * W
rig = W * (vs.endpoint(th[3] + d, p, sm['times'], sm['mass']) - F0)
print('\nr3->r4 change /TOL\n  actual      ', act, '\n  rigid model ', rig)
for k in (2, 3, 4):
    print('  J%d linear   ' % k, W * (J[k] @ d))
# Sgr timing in t=0 field of round 3 (round-4 expansions not built): best time shift of the r4 base
E = O + 'round03/expansions/'; n = len(np.load(E + 'times.npy')) - 1
mw = agama.Potential(agama.Potential(E + 'halo_%02d.coef' % n), agama.Potential(E + 'disk_%02d.coef' % n)); lmc = agama.Potential(E + 'lmc_%02d.coef' % n)
rel = vs._live_rel(dict(np.load(O + 'round04/run00_centres.npz')))[-1]; L0, S0 = rel[:6], rel[6:]
rhs = lambda t, w: np.hstack((w[3:], mw.force(w[:3]) + lmc.force(w[:3] - L0[:3]) - lmc.force(-L0[:3])))
for sgn in (-1, 1):
    te = np.linspace(0, sgn * 10 / 977.8, 2001)
    w = solve_ivp(rhs, (0, te[-1]), S0, t_eval=te, rtol=1e-11, atol=1e-11, method='DOP853').y.T
    chi = np.linalg.norm((w - T[6:]) * W[6:], axis=1); dx = np.linalg.norm(w[:, :3] - T[6:9], axis=1)
    j, i = np.argmin(chi), np.argmin(dx)
    print('Sgr %s: best overall %.2f at %+.2f Myr resid/TOL %s; closest %.3f kpc at %+.2f Myr' % ('back' if sgn < 0 else 'fwd', chi[j], te[j] * 977.8, (w[j] - T[6:]) * W[6:], dx[i], te[i] * 977.8))
# Jacobian consistency
def cd(A, B, sl):
    return np.linalg.norm(W[:, None] * (A - B), axis=0)[sl] / np.linalg.norm(W[:, None] * B, axis=0)[sl]
print('\nrel diff J4-J3 per column', cd(J[4], J[3], slice(None)))
print('rel diff J4-rigid         ', cd(J[4], Jr, slice(None)))
print('LMC block col norms J4', np.linalg.norm(W[:6, None] * J[4][:6, :6], axis=0), 'J3', np.linalg.norm(W[:6, None] * J[3][:6, :6], axis=0))
print('round-5 auto GN step', np.array(I[4]['dtheta']) if 'dtheta' in I[4] else None)
