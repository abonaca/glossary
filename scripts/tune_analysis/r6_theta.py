import sys, json, pickle, numpy as np, agama
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import v21_surrogate as vs, v21_tune as vt
from scipy.integrate import solve_ivp
np.set_printoptions(precision=3, suppress=True, linewidth=200)
O = '/home/ana/projects/glossary/output/v21_sim/tune/'; W = 1 / vt.TOL; B = slice(0, 6)
i5 = json.load(open(O + 'round05/round.json')); th5, r5, J5 = np.array(i5['theta']), np.array(i5['residual']), np.array(i5['jacobian'])
dl = -np.linalg.solve(W[B, None] * J5[B, B], r5[B] * W[B])
E = O + 'round03/expansions/'
mw = agama.Potential(agama.Potential(E + 'halo_00.coef'), agama.Potential(E + 'disk_00.coef')); lmc = agama.Potential(E + 'lmc_00.coef')
xl = th5[:3]
rhs = lambda t, w: np.hstack((w[3:], mw.force(w[:3]) + lmc.force(w[:3] - xl) - lmc.force(-xl)))
s6 = solve_ivp(rhs, (0, 0.6 / 977.8), th5[6:], rtol=1e-11, atol=1e-11, method='DOP853').y[:, -1]
th6 = np.concatenate((th5[:6] + dl, s6)); d = th6 - th5
print('LMC step', dl, '\nSgr step', s6 - th5[6:], '\ntheta6', repr(th6.tolist()))
R = pickle.load(open(O + 'round04/rigid_recal.pkl', 'rb'))['res']['new: r4 lnL, r4 Sgr mass, sigma(r)']
vs.set_sigma_profile(tuple(np.load('/home/ana/projects/glossary/output/v21_sim/ics/mw_triax_full/halo_sigma.npy')))
tr = vs.load_tracks(4); p = R['p']
Fr = lambda q: vs.endpoint(q, p, tr['sgr_t'], tr['sgr_m'])
F5 = Fr(th5)
print('\nr5 live resid/TOL          ', r5 * W)
print('rigid forecast r6 resid/TOL', (r5 + Fr(th6) - F5) * W)
print('  LMC step only           ', (r5 + Fr(th5 + np.concatenate((dl, np.zeros(6)))) - F5) * W)
print('  Sgr step only           ', (r5 + Fr(th5 + np.concatenate((np.zeros(6), s6 - th5[6:]))) - F5) * W)
print('J5 linear forecast         ', (r5 + J5 @ d) * W)
print('J5 linear, LMC-block only  ', np.concatenate((r5[B] * W[B] + W[B] * (J5[B, B] @ dl), [np.nan] * 6)))
json.dump(th6.tolist(), open('/home/ana/projects/glossary/scripts/tune_analysis/theta6.json', 'w'))
