import sys, json, numpy as np, agama
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import v21_surrogate as vs, v21_tune as vt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
np.set_printoptions(precision=2, suppress=True, linewidth=200)
O = '/home/ana/projects/glossary/output/v21_sim/tune/'; W = 1 / vt.TOL; T = vt.TARGET; L, S = slice(0, 6), slice(6, 12)
E = O + 'round03/expansions/'; n = len(np.load(E + 'times.npy')) - 1
mw = agama.Potential(agama.Potential(E + 'halo_%02d.coef' % n), agama.Potential(E + 'disk_%02d.coef' % n)); lmc = agama.Potential(E + 'lmc_%02d.coef' % n)
I = {k: json.load(open(O + 'round%02d/round.json' % k)) for k in (4, 5, 6, 7)}
th = {k: np.array(I[k]['theta']) for k in I}; J = {k: np.array(I[k]['jacobian']) for k in I}; r = {k: np.array(I[k]['residual']) for k in I}

def orbit(S0, L0, tmax):
    """Sgr from its present-day state, LMC on its own orbit, both in the t=0 field; returns w(t) callable"""
    rl = lambda t, w: np.hstack((w[3:], mw.force(w[:3]) - lmc.force(-w[:3])))
    def rs(t, w):
        xl = w[6:9]
        return np.hstack((w[3:6], mw.force(w[:3]) + lmc.force(w[:3] - xl) - lmc.force(-xl), w[9:], mw.force(xl) - lmc.force(-xl)))
    return solve_ivp(rs, (0, tmax), np.hstack((S0, L0)), dense_output=True, rtol=1e-11, atol=1e-11, method='DOP853').sol

def lateness(k):
    """Myr by which Sgr is behind the target along its own orbit (best total chi over +-10 Myr)"""
    rel = th[k] * 0 + (r[k] + T)          # present-day base state = target + residual
    out = []
    for sgn in (1, -1):
        sol = orbit(rel[S], rel[L], sgn * 10 / 977.8)
        f = lambda dt: np.linalg.norm((sol(sgn * dt / 977.8)[:6] - T[S]) * W[S])
        m = minimize_scalar(f, bounds=(0, 10), method='bounded', options=dict(xatol=1e-3))
        out.append((m.fun, sgn * m.x))
    return min(out)[1]

# timing direction u = d(Sgr state)/dt at the present day, in TOL units, per Myr
def u_dir(k):
    s = r[k][S] + T[S]; l = r[k][L] + T[L]
    a = mw.force(s[:3]) + lmc.force(s[:3] - l[:3]) - lmc.force(-l[:3])
    return np.hstack((s[3:] / 977.8 * 977.8 / 977.8, a / 977.8)) * W[S]   # km/s/Myr ~ kpc/Myr scale below
# careful units: dx/dt [kpc/Myr] = v[km/s] * 1.0227e-3 ; dv/dt [km/s/Myr] = a[(km/s)^2/kpc] * 1.0227e-3
def u_dir(k):
    s = r[k][S] + T[S]; l = r[k][L] + T[L]
    a = mw.force(s[:3]) + lmc.force(s[:3] - l[:3]) - lmc.force(-l[:3])
    return np.hstack((s[3:], a)) * 1.0227e-3 * W[S]

lt = {k: lateness(k) for k in I}
print('Sgr lateness (Myr, + = late):', {k: round(-v, 2) for k, v in lt.items()})
print('\nSgr arrival shift per LMC nudge unit (Myr per kpc or km/s, + = later), and fraction of the column along timing')
tau = {}
for k in I:
    u = u_dir(k); C = W[S, None] * J[k][S, L]
    tau[k] = -(u @ C) / (u @ u)
    frac = np.abs(u @ C) / (np.linalg.norm(u) * np.linalg.norm(C, axis=0))
    print(' J%d  tau %s   |cos| %s' % (k, tau[k], frac))
print('\nLMC step effect on Sgr timing (Myr, + = later): predicted by J(k) vs J(k+1), and Sgr own-step timing')
for k in (4, 5, 6):
    dl = (th[k + 1] - th[k])[L]
    us = u_dir(k); Cs = W[S, None] * J[k][S, S]
    own = -(us @ (Cs @ (th[k + 1] - th[k])[S])) / (us @ us)
    print(' r%d->r%d: LMC effect J%d %+.2f  J%d %+.2f | Sgr own step (J%d) %+.2f | lateness %+.2f -> %+.2f (change %+.2f)' % (
        k, k + 1, k, tau[k] @ dl, k + 1, tau[k + 1] @ dl, k, own, -lt[k], -lt[k + 1], -lt[k + 1] + lt[k]))
