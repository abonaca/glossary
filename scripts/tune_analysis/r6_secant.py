import sys, json, numpy as np
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import v21_tune as vt
np.set_printoptions(precision=2, suppress=True, linewidth=200)
O = '/home/ana/projects/glossary/output/v21_sim/tune/'; TL = vt.TOL[:6]
I = {k: json.load(open(O + 'round%02d/round.json' % k)) for k in (5, 6)}
th = {k: np.array(I[k]['theta'])[:6] for k in I}; r = {k: np.array(I[k]['residual'])[:6] for k in I}
J = {k: np.array(I[k]['jacobian'])[:6, :6] for k in I}
d = np.array([3, 1, 3, 3, 2, 3.])
dth, dr = th[6] - th[5], r[6] - r[5]
print('r5 -> r6 start change', dth, '\nactual end change /TOL', dr / TL)
for k in (5, 6): print('  J%d predicts      /TOL' % k, J[k] @ dth / TL)
# 1. best point on the line through the two starts: r(a) = r5 + a (r6 - r5)
a = -np.dot(r[5] / TL, dr / TL) / np.dot(dr / TL, dr / TL)
print('\n1. line through r5 and r6: best a = %.2f (0 = r5, 1 = r6), predicted /TOL %s, total %.2f'
      % (a, (r[5] + a * dr) / TL, np.linalg.norm((r[5] + a * dr) / TL)), '\n   start', th[5] + a * dth)
# 2. Broyden: correct J6 along the r5->r6 direction (in nudge units) so it reproduces the actual change
u = dth / d
Jb = J[6] + np.outer(dr - J[6] @ dth, u / d) / (u @ u)
print('   check Broyden reproduces actual change:', np.allclose(Jb @ dth, dr))
for name, Jx in (('J6', J[6]), ('Broyden J6', Jb), ('mean J5,J6', (J[5] + J[6]) / 2)):
    Jw = Jx / TL[:, None]; s = -np.linalg.solve(Jw, r[6] / TL)
    print('\n%s step from r6: %s\n   start %s' % (name, s, th[6] + s))
    for nm, Jy in (('J6', J[6]), ('Broyden', Jb), ('J5', J[5])):
        print('   predicted /TOL with %-7s %s' % (nm, (r[6] + Jy @ s) / TL))
print('\ncond (TOL-weighted, nudge-scaled): J5 %.0f  J6 %.0f  Broyden %.0f' % tuple(
    np.linalg.cond(Jx / TL[:, None] * d) for Jx in (J[5], J[6], Jb)))
