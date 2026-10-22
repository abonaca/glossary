import sys, json, pickle, numpy as np
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import v21_surrogate as vs, v21_tune as vt
np.set_printoptions(precision=2, suppress=True, linewidth=200)
O = '/home/ana/projects/glossary/output/v21_sim/tune/'; W = 1 / vt.TOL; L, S = slice(0, 6), slice(6, 12)
I6 = json.load(open(O + 'round06/round.json')); th6 = np.array(I6['theta']); J6 = np.array(I6['jacobian'])
th7 = np.array(json.load(open('theta7.json')))
R = pickle.load(open(O + 'round04/rigid_recal.pkl', 'rb'))['res']['new: r4 lnL, r4 Sgr mass, sigma(r)']
vs.set_sigma_profile(tuple(np.load('/home/ana/projects/glossary/output/v21_sim/ics/mw_triax_full/halo_sigma.npy')))
tr = vs.load_tracks(4); st, sm = tr['sgr_t'], tr['sgr_m']; p = R['p']
F = vs.endpoint(th6, p, st, sm); F0 = vs.endpoint(th6, p, st, 0 * sm)
print('Sgr mass at t=-3 and t=0 (model units):', sm[0], sm[-1])
dLMC = (F - F0)[L]
print('1. Sgr present vs massless, LMC endpoint change: phys', dLMC, ' /TOL', dLMC * W[L])
th = th6.copy(); th[S] = th7[S]
dS = (vs.endpoint(th, p, st, sm) - F)[L]
print('2. r7 Sgr step only, LMC endpoint change: rigid /TOL', dS * W[L], '  J6 /TOL', J6[L, S] @ (th7 - th6)[S] * W[L])
Jr = vs.jacobian(th6, p, st, sm, np.array(I6['delta']))[1] if isinstance(vs.jacobian(th6, p, st, sm, np.array(I6['delta'])), tuple) else vs.jacobian(th6, p, st, sm, np.array(I6['delta']))
print('3. Sgr->LMC block /TOL per unit (cols Sgr x0..vz0)\n rigid\n', Jr[L, S] * W[L, None], '\n J6\n', J6[L, S] * W[L, None])
