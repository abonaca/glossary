import sys, json, numpy as np, agama
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import v21_tune as vt
from scipy.integrate import solve_ivp
np.set_printoptions(precision=4, suppress=True, linewidth=200)
O = '/home/ana/projects/glossary/output/v21_sim/tune/'; W = 1 / vt.TOL; S = slice(6, 12)
c = np.load(O + 'round06/centre_check.npz'); info = json.load(open(O + 'round06/round.json'))
F = np.hstack((c['lmc'][:, 3] - c['mw'][:, 3], c['sgr'][:, 2] - c['mw'][:, 3]))
th6 = np.array(info['theta']); d = np.array(info['delta']); r = (F[0, S] - vt.TARGET[S]) * W[S]
Jn = W[S, None] * (F[7:, S] - F[0, S]).T / d[S] * d[S]
A = Jn.T @ Jn; lam = 3e-3
s = -np.linalg.solve(A + lam * np.diag(np.diag(A)), Jn.T @ r) * d[S]
th7 = th6.copy(); th7[S] += s
print('Sgr step', s, '\ntheta7', repr(th7.tolist()))
E = O + 'round03/expansions/'
mw = agama.Potential(agama.Potential(E + 'halo_00.coef'), agama.Potential(E + 'disk_00.coef')); lmc = agama.Potential(E + 'lmc_00.coef')
xl = th7[:3]
rhs = lambda t, w: np.hstack((w[3:], mw.force(w[:3]) + lmc.force(w[:3] - xl) - lmc.force(-xl)))
Jp = (F[7:, S] - F[0, S]).T / d[S]
v = np.linalg.norm(vt.TARGET[9:]) * 1.0227e-3
print('Sgr present-day speed %.3f kpc/Myr -> 0.1 kpc = %.2f Myr' % (v, 0.1 / v))
for dt in (-1, -0.5, -0.25, 0.25, 0.5, 1):
    w = solve_ivp(rhs, (0, dt / 977.8), th7[S], rtol=1e-11, atol=1e-11, method='DOP853').y[:, -1]
    ds = w - th7[S]; dF = Jp @ ds
    # arrival shift implied by J6: projection of the position change on the present-day velocity direction
    u = vt.TARGET[9:] / np.linalg.norm(vt.TARGET[9:])
    print('start %+5.2f Myr: start change %s | J6 endpoint change pos %s vel %s | arrival %+5.2f Myr' % (
        dt, ds, dF[:3], dF[3:], (dF[:3] @ u) / v))
