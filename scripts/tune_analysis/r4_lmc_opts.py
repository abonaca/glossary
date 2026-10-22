import sys, json, pickle, numpy as np
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import v21_surrogate as vs, v21_tune as vt
np.set_printoptions(precision=2, suppress=True, linewidth=200)
O = '/home/ana/projects/glossary/output/v21_sim/tune/'
R = pickle.load(open(O + 'round03/rigid_jacobian.pkl', 'rb')); F0, p, Jr = R['F0'], R['p'], R['J']
sm = np.load(O + 'round02/sgr_mass.npz'); i3 = json.load(open(O + 'round03/round.json')); i2 = json.load(open(O + 'round02/round.json'))
th3, r3 = np.array(i3['theta']), np.array(i3['residual']); W = 1 / vt.TOL
J3, J2 = np.array(i3['jacobian']), np.array(i2['jacobian'])
ex = pickle.load(open(O + 'round03/rigid_lmc_solve.pkl', 'rb'))['from full GN']
full = lambda d: W * (vs.endpoint(th3 + np.concatenate((d, np.zeros(6))), p, sm['times'], sm['mass']) - F0)
opts = {'damped lam 0.1': np.array([0.99, 0.12, -5.75, -0.89, 0.2, 0.35])}
for f in (0.1, 0.25, 0.5): opts['%.2f x exact' % f] = f * ex
print('%-15s %-48s %-38s %-38s %-38s %s' % ('option', 'step', 'rigid LMC miss', 'J3 linear LMC miss', 'J2 linear LMC miss', 'rigid Sgr effect'))
for k, d in opts.items():
    print('%-15s %s  %s  %s  %s  %s' % (k, d, r3[:6] * W[:6] + full(d)[:6], r3[:6] * W[:6] + W[:6] * (J3[:6, :6] @ d),
                                     r3[:6] * W[:6] + W[:6] * (J2[:6, :6] @ d), full(d)[6:]))
