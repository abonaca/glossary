import sys, json, pickle, numpy as np
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import v21_surrogate as vs, v21_tune as vt
np.set_printoptions(precision=2, suppress=True, linewidth=200)
O = '/home/ana/projects/glossary/output/v21_sim/tune/'
R = pickle.load(open(O + 'round03/rigid_jacobian.pkl', 'rb')); Jr, F0, p = R['J'], R['F0'], R['p']
sm = np.load(O + 'round02/sgr_mass.npz'); th3 = np.array(json.load(open(O + 'round03/round.json'))['theta']); W = 1 / vt.TOL
steps = {'hybrid full': [0.81, -1.14, -9.44, -0.83, 1.2, 0.93], 'hybrid damped': [0.26, 0.06, -3.62, -0.53, 0.09, 0.16],
         'rigid full': [-1.53, -2.23, -12.25, -0.67, 2.02, 1.05], 'z0 -3 only': [0, 0, -3, 0, 0, 0]}
for k, s in steps.items():
    d = np.concatenate((s, np.zeros(6)))
    F = vs.endpoint(th3 + d, p, sm['times'], sm['mass'])
    lin, act = W * (Jr @ d), W * (F - F0)
    print('%-14s LMC: linear %s actual %s | nonlin %.0f%%' % (k, lin[:6], act[:6], 100 * np.linalg.norm(act[:6] - lin[:6]) / np.linalg.norm(act[:6])))
