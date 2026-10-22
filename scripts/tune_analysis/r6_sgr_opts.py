import sys, json, pickle, numpy as np, agama
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import v21_surrogate as vs, v21_tune as vt
from scipy.integrate import solve_ivp
np.set_printoptions(precision=2, suppress=True, linewidth=200)
O = '/home/ana/projects/glossary/output/v21_sim/tune/'; W = 1 / vt.TOL; S = slice(6, 12)
I = {k: json.load(open(O + 'round%02d/round.json' % k)) for k in (5, 6)}
th6, r6 = np.array(I[6]['theta']), np.array(I[6]['residual']); J = {k: np.array(I[k]['jacobian']) for k in I}
# timing fix: Sgr start advanced 1.07 Myr along its own orbit (round-3 t=-3 field, LMC at its round-6 start)
E = O + 'round03/expansions/'
mw = agama.Potential(agama.Potential(E + 'halo_00.coef'), agama.Potential(E + 'disk_00.coef')); lmc = agama.Potential(E + 'lmc_00.coef')
xl = th6[:3]
rhs = lambda t, w: np.hstack((w[3:], mw.force(w[:3]) + lmc.force(w[:3] - xl) - lmc.force(-xl)))
sT = solve_ivp(rhs, (0, 1.07 / 977.8), th6[S], rtol=1e-11, atol=1e-11, method='DOP853').y[:, -1] - th6[S]
# J6 Sgr-block GN step
sJ = -np.linalg.solve(W[S, None] * J[6][S, S], r6[S] * W[S])
R = pickle.load(open(O + 'round04/rigid_recal.pkl', 'rb'))['res']['new: r4 lnL, r4 Sgr mass, sigma(r)']
vs.set_sigma_profile(tuple(np.load('/home/ana/projects/glossary/output/v21_sim/ics/mw_triax_full/halo_sigma.npy')))
tr = vs.load_tracks(4); Fr = lambda q: vs.endpoint(q, R['p'], tr['sgr_t'], tr['sgr_m']); F6 = Fr(th6)
print('round 6 Sgr miss /TOL', r6[S] * W[S])
for name, s in (('timing 1.07 Myr', sT), ('J6 Sgr block', sJ)):
    d = np.concatenate((np.zeros(6), s)); f = (r6 + Fr(th6 + d) - F6) * W
    print('\n%s: Sgr start change %s' % (name, s))
    print('  Sgr miss /TOL  J6 %s\n                 J5 %s\n              rigid %s' % (
        (r6[S] + J[6][S, S] @ s) * W[S], (r6[S] + J[5][S, S] @ s) * W[S], f[S]))
    print('  LMC miss /TOL  J6 %s\n              rigid %s' % ((r6[:6] + J[6][:6, S] @ s) * W[:6], f[:6]))
print('\nJ6-block step minus timing step', sJ - sT)
