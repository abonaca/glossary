import sys, json, pickle, numpy as np
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import v21_tune as vt
np.set_printoptions(precision=2, suppress=True, linewidth=200)
O = '/home/ana/projects/glossary/output/v21_sim/tune/'
i2 = json.load(open(O + 'round02/round.json')); i3 = json.load(open(O + 'round03/round.json'))
J2, J3 = np.array(i2['jacobian']), np.array(i3['jacobian']); Jr = pickle.load(open(O + 'round03/rigid_jacobian.pkl', 'rb'))['J']
r3 = np.array(i3['residual']); W = 1 / vt.TOL
Jh = J3.copy(); Jh[:, [0, 2]] = Jr[:, [0, 2]]          # live J3, but LMC x0, z0 columns from the rigid model
ts = np.array([14.1, -1.7, 12.2, -12.7, -1.4, 6.5]) / 5.6   # Sgr change per Myr of advance (TOL/Myr), forward integration
u = ts / np.linalg.norm(ts)
for lab, J in (('J3', J3), ('rigid', Jr), ('hybrid', Jh), ('J2', J2)):
    Jw = W[:6, None] * J[:6, :6]; rw = r3[:6] * W[:6]; A = Jw.T @ Jw
    for lam in (0., 0.3):
        dl = -np.linalg.solve(A + lam * np.diag(np.diag(A)), Jw.T @ rw)
        pr = '%-6s lam %.1f  LMC step %s  pred LMC /TOL %s' % (lab, lam, dl, rw + Jw @ dl)
        effs = {k: W[6:] * (Jk[6:, :6] @ dl) for k, Jk in (('J3', J3), ('rigid', Jr), ('J2', J2))}
        print(pr)
        for k, e in effs.items():
            print('        effect on Sgr via %-5s /TOL %s  -> along timing %+.1f Myr, rest %.1f' % (k, e, (e @ u) / np.linalg.norm(ts), np.linalg.norm(e - (e @ u) * u)))
