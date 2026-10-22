import sys, json, pickle, numpy as np, agama
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
from scipy.integrate import solve_ivp
np.set_printoptions(precision=4, suppress=True, linewidth=200)
O = '/home/ana/projects/glossary/output/v21_sim/tune/'
i3 = json.load(open(O + 'round03/round.json')); th3 = np.array(i3['theta'])
ex = pickle.load(open(O + 'round03/rigid_lmc_solve.pkl', 'rb'))['from full GN']
E = O + 'round03/expansions/'
mw = agama.Potential(agama.Potential(E + 'halo_00.coef'), agama.Potential(E + 'disk_00.coef')); lmc = agama.Potential(E + 'lmc_00.coef')
xl = th3[:3]
rhs = lambda t, w: np.hstack((w[3:], mw.force(w[:3]) + lmc.force(w[:3] - xl) - lmc.force(-xl)))
s4 = solve_ivp(rhs, (0, 4.8 / 977.8), th3[6:], rtol=1e-11, atol=1e-11, method='DOP853').y[:, -1]
th4 = np.concatenate((th3[:6] + ex, s4))
print('LMC step', ex, '\nSgr step', s4 - th3[6:], '\ntheta4', repr(th4.tolist()))
json.dump(th4.tolist(), open('/home/ana/projects/glossary/scripts/tune_analysis/theta4.json', 'w'))
