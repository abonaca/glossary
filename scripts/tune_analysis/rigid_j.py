import sys, json, pickle, numpy as np, time
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import v21_surrogate as vs, v21_tune as vt
np.set_printoptions(precision=2, suppress=True, linewidth=200)
O = '/home/ana/projects/glossary/output/v21_sim/tune/'
p = pickle.load(open(O + 'round00/surrogate_calib.pkl', 'rb'))['p']
sm = np.load(O + 'round02/sgr_mass.npz')
i2 = json.load(open(O + 'round02/round.json')); i3 = json.load(open(O + 'round03/round.json'))
J2, J3 = np.array(i2['jacobian']), np.array(i3['jacobian'])
th3 = np.array(i3['theta']); W = 1 / vt.TOL
t = time.time()
F0, Jr = vs.jacobian(th3, p, sm['times'], sm['mass'], delta=np.array(i3['delta']))
print('rigid Jacobian in %.0f s' % (time.time() - t))
print('rigid model t=0 minus live r3 base /TOL:', (F0 - (np.array(i3['residual']) + vt.TARGET)) * W)
def rel(A, B):  # per-column relative difference in TOL-weighted norm
    return np.linalg.norm(W[:, None] * (A - B), axis=0) / np.linalg.norm(W[:, None] * B, axis=0)
for blk, sl_r, sl_c in (('LMC->LMC', slice(0, 6), slice(0, 6)), ('LMC->Sgr', slice(6, 12), slice(0, 6)), ('Sgr->LMC', slice(0, 6), slice(6, 12))):
    w = W[sl_r, None]
    n = lambda A: np.linalg.norm(w * A[sl_r, sl_c], axis=0)
    d = lambda A, B: np.linalg.norm(w * (A - B)[sl_r, sl_c], axis=0) / n(B)
    print('%s column norms  rigid %s  J2 %s  J3 %s' % (blk, n(Jr), n(J2), n(J3)))
    print('   rel diff rigid-J3 %s  rigid-J2 %s  J2-J3 %s' % (d(Jr, J3), d(Jr, J2), d(J2, J3)))
pickle.dump(dict(F0=F0, J=Jr, p=p), open(O + 'round03/rigid_jacobian.pkl', 'wb'))
