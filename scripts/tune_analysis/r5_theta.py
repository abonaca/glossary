import sys, json, pickle, numpy as np, agama
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import v21_surrogate as vs, v21_tune as vt
from scipy.integrate import solve_ivp
np.set_printoptions(precision=4, suppress=True, linewidth=200)
O = '/home/ana/projects/glossary/output/v21_sim/tune/'; W = 1 / vt.TOL; B = slice(0, 6)
i4 = json.load(open(O + 'round04/round.json')); th4, r4, J4 = np.array(i4['theta']), np.array(i4['residual']), np.array(i4['jacobian'])
dl = -np.linalg.solve(W[B, None] * J4[B, B], r4[B] * W[B])
# LMC step's effect on Sgr, expressed as Myr along Sgr's present-day orbit (TOL per Myr from the t=0 forward integration)
ts = np.array([14.1, -1.7, 12.2, -12.7, -1.4, 6.5]) / 5.6; u = ts / np.linalg.norm(ts)
R = pickle.load(open(O + 'round04/rigid_recal.pkl', 'rb'))['res']['new: r4 lnL, r4 Sgr mass, sigma(r)']
vs.set_sigma_profile(tuple(np.load('/home/ana/projects/glossary/output/v21_sim/ics/mw_triax_full/halo_sigma.npy')))
tr4 = vs.load_tracks(4)
eR = W[6:] * (vs.endpoint(th4 + np.concatenate((dl, np.zeros(6))), R['p'], tr4['sgr_t'], tr4['sgr_m']) - R['F4'])[6:]
eJ = W[6:] * (J4[6:, :6] @ dl)
for lab, e in (('rigid (refit)', eR), ('J4', eJ)):
    print('LMC-step effect on Sgr via %-13s %s -> %+.2f Myr along the orbit' % (lab, e, (e @ u) / np.linalg.norm(ts)))
print('LMC step', dl)
E = O + 'round03/expansions/'
mw = agama.Potential(agama.Potential(E + 'halo_00.coef'), agama.Potential(E + 'disk_00.coef')); lmc = agama.Potential(E + 'lmc_00.coef')
xl = th4[:3]
rhs = lambda t, w: np.hstack((w[3:], mw.force(w[:3]) + lmc.force(w[:3] - xl) - lmc.force(-xl)))
s5 = solve_ivp(rhs, (0, -4.3 / 977.8), th4[6:], rtol=1e-11, atol=1e-11, method='DOP853').y[:, -1]
th5 = np.concatenate((th4[:6] + dl, s5))
print('Sgr step', s5 - th4[6:], '\ntheta5', repr(th5.tolist()))
json.dump(th5.tolist(), open('/home/ana/projects/glossary/scripts/tune_analysis/theta5.json', 'w'))
