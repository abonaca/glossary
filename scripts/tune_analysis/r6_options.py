import sys, json, pickle, time, numpy as np
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import v21_surrogate as vs, v21_tune as vt
from scipy.optimize import least_squares
np.set_printoptions(precision=2, suppress=True, linewidth=200)
O = '/home/ana/projects/glossary/output/v21_sim/tune/'; W = 1 / vt.TOL; T = vt.TARGET
vs.set_sigma_profile(tuple(np.load('/home/ana/projects/glossary/output/v21_sim/ics/mw_triax_full/halo_sigma.npy')))
R = pickle.load(open(O + 'round04/rigid_recal.pkl', 'rb'))['res']['new: r4 lnL, r4 Sgr mass, sigma(r)']
tr4 = vs.load_tracks(4); st, sm = tr4['sgr_t'], tr4['sgr_m']
th6 = np.array(json.load(open('theta6.json')))
live6 = vs._live_rel(dict(np.load(O + 'round06/run00_centres.npz')))[-1]
F6 = vs.endpoint(th6, R['p'], st, sm)
fc = lambda th: (live6 + vs.endpoint(th, R['p'], st, sm) - F6 - T) * W
# J6 LMC block from the printed endpoints
E = np.array([[-0.65,-41.24,-27.01,-61.94,-210.29,202.93],[-0.82,-40.2,-28.07,-67.24,-216.61,202.5],
 [0.235,-38.4,-29.9,-65.5,-222,201],[-0.51,-40.56,-27.43,-64.42,-213.57,203.43],[0.68,-39.01,-30.22,-76.38,-217.23,197.39],
 [2.34,-31.25,-36.83,-62.32,-238.48,180.04],[-2.16,-42.8,-19.21,-68.86,-200.75,221.69]])
d = np.array([3,1,3,3,2,3.]); J6 = (E[1:] - E[0]).T / d; r = (live6 - T)[:6]
sA = -np.linalg.solve(J6, r)
thA = th6.copy(); thA[:6] += sA
print('A: J6 step', sA, '\n   J6 linear LMC /TOL', (r + J6 @ sA) * W[:6], '\n   rigid forecast /TOL', fc(thA), flush=True)
t = time.time()
g = lambda q: fc(np.concatenate((q, th6[6:])))[:6]
res = least_squares(g, th6[:6] + 0.5 * sA, x_scale=np.array([1, 1, 1, .3, .3, .3]), diff_step=1e-3)
sB = res.x - th6[:6]; thB = np.concatenate((res.x, th6[6:]))
print('B: rigid LMC solve (%.0f s)' % (time.time() - t), sB, '\n   rigid forecast /TOL', fc(thB),
      '\n   J6 linear LMC /TOL', (r + J6 @ sB) * W[:6], '\n   B - A', sB - sA)
pickle.dump(dict(sA=sA, sB=sB, J6=J6), open('r6_options.pkl', 'wb'))
