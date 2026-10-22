import sys, json, numpy as np, agama
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import v21_surrogate as vs, v21_tune as vt
from scipy.integrate import solve_ivp
np.set_printoptions(precision=2, suppress=True, linewidth=200)
O = '/home/ana/projects/glossary/output/v21_sim/tune/'
E = O + 'round02/expansions/'
n = len(np.load(E + 'times.npy')) - 1                 # last expansion = t=0 of the round-2 base run
mw = agama.Potential(agama.Potential(E + 'halo_%02d.coef' % n), agama.Potential(E + 'disk_%02d.coef' % n))
lmc = agama.Potential(E + 'lmc_%02d.coef' % n)
r3 = dict(np.load(O + 'round03/run00_centres.npz'))
rel = vs._live_rel(r3)[-1]; L0, S0 = rel[:6], rel[6:]   # round-3 base run, relative to the MW, t=0
T, W = vt.TARGET[6:], 1 / vt.TOL[6:]
MYR = 1 / 977.8                                       # time unit = 0.9778 Gyr, velocities in km/s
print('round-3 Sgr at t=0:', S0, '\ntarget            :', T, '\nresid/TOL at t=0  :', (S0 - T) * W)
def run(moving_lmc, tmax=10):
    def rhs(t, w):
        xl = L0[:3] + (L0[3:] * t if moving_lmc else 0)
        a = mw.force(w[:3]) + lmc.force(w[:3] - xl) - lmc.force(-xl)   # minus the LMC's pull on the MW centre
        return np.hstack((w[3:], a))
    te = np.linspace(0, tmax * MYR, 2001)
    return te / MYR, solve_ivp(rhs, (0, te[-1]), S0, t_eval=te, rtol=1e-11, atol=1e-11, method='DOP853').y.T
for lab, mv in (('LMC fixed at its t=0 position', False), ('LMC moving with its t=0 velocity', True)):
    t, w = run(mv)
    dx = np.linalg.norm(w[:, :3] - T[:3], axis=1); chi = np.linalg.norm((w - T) * W, axis=1)
    i, j = np.argmin(dx), np.argmin(chi)
    print('\n%s' % lab)
    print('  closest in position: %.3f kpc at +%.2f Myr; resid/TOL there %s' % (dx[i], t[i], (w[i] - T) * W))
    print('  closest overall (TOL-weighted, all 6): %.2f at +%.2f Myr; resid/TOL there %s' % (chi[j], t[j], (w[j] - T) * W))
    for tt in (2, 4, 6, 8, 10):
        k = np.argmin(abs(t - tt)); print('   +%2d Myr  |dx| %.3f kpc  resid/TOL %s' % (tt, dx[k], (w[k] - T) * W))
