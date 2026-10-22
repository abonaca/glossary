import sys, json, numpy as np, agama
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import v21_surrogate as vs, v21_tune as vt
from scipy.integrate import solve_ivp
np.set_printoptions(precision=2, suppress=True, linewidth=200)
O = '/home/ana/projects/glossary/output/v21_sim/tune/'; W = 1 / vt.TOL; T = vt.TARGET
b = dict(np.load(O + 'round07/run00_centres.npz'))
rel = vs._live_rel(b)
print('snapshot times near the end (Myr):', (b['times'][-4:] - b['times'][-1]) * 977.8)
print('live resid/TOL at t=0:', (rel[-1] - T) * W)
E = O + 'round03/expansions/'; n = len(np.load(E + 'times.npy')) - 1
mw = agama.Potential(agama.Potential(E + 'halo_%02d.coef' % n), agama.Potential(E + 'disk_%02d.coef' % n))
lmc = agama.Potential(E + 'lmc_%02d.coef' % n)
L0, S0 = rel[-1, :6], rel[-1, 6:]
# LMC relative to the MW centre: MW force on the LMC minus the LMC's pull on the MW centre
rl = lambda t, w: np.hstack((w[3:], mw.force(w[:3]) - lmc.force(-w[:3])))
for sgn in (-1, 1):
    te = np.linspace(0, sgn * 10 / 977.8, 2001)
    wl = solve_ivp(rl, (0, te[-1]), L0, t_eval=te, rtol=1e-11, atol=1e-11, method='DOP853').y.T
    # Sgr, with the LMC moving along its own orbit
    xl = lambda t: np.interp(t * sgn, te * sgn, wl[:, 0]), 
    def rs(t, w):
        x = np.array([np.interp(t * sgn, te * sgn, wl[:, k]) for k in range(3)])
        return np.hstack((w[3:], mw.force(w[:3]) + lmc.force(w[:3] - x) - lmc.force(-x)))
    ws = solve_ivp(rs, (0, te[-1]), S0, t_eval=te, rtol=1e-11, atol=1e-11, method='DOP853').y.T
    for name, w, Tt, Wt in (('LMC', wl, T[:6], W[:6]), ('Sgr', ws, T[6:], W[6:])):
        q = (w - Tt) * Wt; chi = np.linalg.norm(q, axis=1); j = np.argmin(chi)
        pos = np.linalg.norm(q[:, :3], axis=1); jp = np.argmin(pos)
        print('%s %s: |resid/TOL| now %.2f; best total %.2f at %+.2f Myr %s; best position %.2f at %+.2f Myr %s' % (
            name, 'back' if sgn < 0 else 'fwd ', chi[0], chi[j], te[j] * 977.8, q[j], pos[jp], te[jp] * 977.8, q[jp]))
