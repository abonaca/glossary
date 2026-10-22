import sys, json, pickle, numpy as np
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import v21_surrogate as vs, v21_tune as vt
np.set_printoptions(precision=3, suppress=True, linewidth=200)
O = '/home/ana/projects/glossary/output/v21_sim/tune/'; W = 1 / vt.TOL; L, S = slice(0, 6), slice(6, 12)
I = {k: json.load(open(O + 'round%02d/round.json' % k)) for k in (5, 6)}
th6, r6 = np.array(I[6]['theta']), np.array(I[6]['residual']); J6, J5 = (np.array(I[k]['jacobian']) for k in (6, 5))
d = np.zeros(12)
for B in (L, S):
    d[B] = -np.linalg.solve(W[B, None] * J6[B, B], r6[B] * W[B])
th7 = th6 + d
print('LMC step', d[L], '\nSgr step', d[S], '\ntheta6', th6, '\ntheta7', th7, '\ntheta7 full', repr(th7.tolist()))
R = pickle.load(open(O + 'round04/rigid_recal.pkl', 'rb'))['res']['new: r4 lnL, r4 Sgr mass, sigma(r)']
vs.set_sigma_profile(tuple(np.load('/home/ana/projects/glossary/output/v21_sim/ics/mw_triax_full/halo_sigma.npy')))
tr = vs.load_tracks(4); Fr = lambda q: vs.endpoint(q, R['p'], tr['sgr_t'], tr['sgr_m'])
print('\nr6 miss /TOL              ', r6 * W)
print('J6 blocks only (no cross) ', np.concatenate(((r6[L] + J6[L, L] @ d[L]) * W[L], (r6[S] + J6[S, S] @ d[S]) * W[S])))
print('J6 full (with cross terms)', (r6 + J6 @ d) * W)
print('J5 full                   ', (r6 + J5 @ d) * W)
print('rigid                     ', (r6 + Fr(th7) - Fr(th6)) * W)
json.dump(th7.tolist(), open('theta7.json', 'w'))
