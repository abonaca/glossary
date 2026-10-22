import sys, json, numpy as np, agama
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import v21_tune as vt
from scipy.integrate import solve_ivp
np.set_printoptions(precision=2, suppress=True, linewidth=200)
O = '/home/ana/projects/glossary/output/v21_sim/tune/'; W = 1 / vt.TOL; L, S = slice(0, 6), slice(6, 12)
c = np.load(O + 'round06/centre_check.npz'); info = json.load(open(O + 'round06/round.json'))
F = np.hstack((c['lmc'][:, 3] - c['mw'][:, 3], c['sgr'][:, 2] - c['mw'][:, 3]))   # LMC 5, Sgr 5, MW 3 kpc
th6 = np.array(info['theta']); d = np.array(info['delta']); q = (F - vt.TARGET) * W
print('round-6 runs with the LMC at its round-6 start (miss /TOL):')
for i in [0] + list(range(7, 13)):
    print('  run %2d %-7s LMC %s (max %.2f) | Sgr %s (total %.2f)' % (i, '' if i == 0 else '+' + vt.NAMES[i - 1][4:],
          q[i, L], np.abs(q[i, L]).max(), q[i, S], np.linalg.norm(q[i, S])))
Js = (F[7:, S] - F[0, S]).T / d[S]; Jw = W[S, None] * Js; r = q[0, S]
print('\nJ6 Sgr block, TOL units per nudge (cols x0..vz0, nudges %s):\n' % d[S], Jw * d[S])
print('singular values (nudge-scaled):', np.linalg.svd(Jw * d[S], compute_uv=False))
# timing direction at the t=-3 start (round-3 t=-3 field, LMC at its round-6 start)
E = O + 'round03/expansions/'
mw = agama.Potential(agama.Potential(E + 'halo_00.coef'), agama.Potential(E + 'disk_00.coef')); lmc = agama.Potential(E + 'lmc_00.coef')
xl = th6[:3]
rhs = lambda t, w: np.hstack((w[3:], mw.force(w[:3]) + lmc.force(w[:3] - xl) - lmc.force(-xl)))
shift = lambda dt: solve_ivp(rhs, (0, dt / 977.8), th6[S], rtol=1e-11, atol=1e-11, method='DOP853').y[:, -1] - th6[S]
def show(nm, s):
    print('\n%s\n  Sgr start change %s  (in nudge units %s)\n  J6 forecast Sgr /TOL %s  total %.2f' % (
        nm, s, s / d[S], r + Jw @ s, np.linalg.norm(r + Jw @ s)))
for dt in (0.5, 1.07, 1.5):
    show('timing shift %.2f Myr along the orbit' % dt, shift(dt))
s_gn = -np.linalg.solve(Jw, r); show('J6 Sgr-block Gauss-Newton (undamped)', s_gn)
Js_ = Jw * d[S]
for lam in (0.01, 0.1, 1.):
    A = Js_.T @ Js_; u = -np.linalg.solve(A + lam * np.diag(np.diag(A)), Js_.T @ r)
    show('J6 Sgr-block Levenberg-Marquardt lambda=%g' % lam, u * d[S])
np.save('r9_sgr_gn.npy', s_gn)
