import sys, json, pickle, numpy as np, agama
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import v21_surrogate as vs, v21_tune as vt
from scipy.integrate import solve_ivp
np.set_printoptions(precision=2, suppress=True, linewidth=200)
O = '/home/ana/projects/glossary/output/v21_sim/tune/'; W = 1 / vt.TOL; T = vt.TARGET
I = {k: json.load(open(O + 'round%02d/round.json' % k)) for k in (4, 5)}
th = {k: np.array(I[k]['theta']) for k in I}; r = {k: np.array(I[k]['residual']) for k in I}; J = {k: np.array(I[k]['jacobian']) for k in I}
print('r5 residual (physical)', r[5], '\nr5 resid/TOL', r[5] * W)
d = th[5] - th[4]
print('r4->r5 change /TOL actual', (r[5] - r[4]) * W, '\n  J4 linear', W * (J[4] @ d))
R = pickle.load(open(O + 'round04/rigid_recal.pkl', 'rb'))['res']['new: r4 lnL, r4 Sgr mass, sigma(r)']
vs.set_sigma_profile(tuple(np.load('/home/ana/projects/glossary/output/v21_sim/ics/mw_triax_full/halo_sigma.npy')))
tr4 = vs.load_tracks(4)
print('  rigid    ', W * (vs.endpoint(th[5], R['p'], tr4['sgr_t'], tr4['sgr_m']) - R['F4']))
print('forecast J4 /TOL', (r[4] + J[4] @ d) * W)
E = O + 'round03/expansions/'; n = len(np.load(E + 'times.npy')) - 1
mw = agama.Potential(agama.Potential(E + 'halo_%02d.coef' % n), agama.Potential(E + 'disk_%02d.coef' % n)); lmc = agama.Potential(E + 'lmc_%02d.coef' % n)
rel = vs._live_rel(dict(np.load(O + 'round05/run00_centres.npz')))[-1]; L0, S0 = rel[:6], rel[6:]
rhs = lambda t, w: np.hstack((w[3:], mw.force(w[:3]) + lmc.force(w[:3] - L0[:3]) - lmc.force(-L0[:3])))
for sgn in (-1, 1):
    te = np.linspace(0, sgn * 5 / 977.8, 1001)
    w = solve_ivp(rhs, (0, te[-1]), S0, t_eval=te, rtol=1e-11, atol=1e-11, method='DOP853').y.T
    chi = np.linalg.norm((w - T[6:]) * W[6:], axis=1); j = np.argmin(chi)
    print('Sgr %s: best %.2f at %+.2f Myr, resid/TOL %s' % ('back' if sgn < 0 else 'fwd', chi[j], te[j] * 977.8, (w[j] - T[6:]) * W[6:]))
print('rel diff J5-J4 per column', np.linalg.norm(W[:, None] * (J[5] - J[4]), axis=0) / np.linalg.norm(W[:, None] * J[4], axis=0))
