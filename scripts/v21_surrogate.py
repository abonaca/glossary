"""
Cheap surrogate of the live R2 run for starting guesses of the LMC+Sgr tuning (v21_tune).

Three point-centred bodies in V21's rigid initial potentials (their eqs. 2-3, extended to Sgr):
    MW   moves in the LMC potential (lmc00.pot) and the Sgr potential (Plummer, mass M_Sgr(t));
    LMC  moves in the MW potential (mw00.pot halo + bulge + disk) with Chandrasekhar friction;
    Sgr  moves in the MW and LMC potentials with Chandrasekhar friction, mass M_Sgr(t).
The friction strengths (Coulomb logarithms) are calibrated on the centre tracks of a tuning round;
M_Sgr(t) is the mass within 5 kpc of the Sgr centre in that round's base run (V21's definition).

theta = (LMC x, v, Sgr x, v) at t=-3 in the frame where the MW starts at rest at the origin, as in
v21_tune; endpoints = present-day LMC and Sgr relative to the MW.

Usage:
    import v21_surrogate as vs
    # on OBS (once per calibration round):  vs.sgr_mass_series(0)  -> TUNE/round00/sgr_mass.npz
    tr = vs.load_tracks(0)                     # round 0 centre tracks (13 runs) + Sgr mass series
    p = vs.calibrate(tr)                       # Coulomb logarithms
    vs.validate(tr, p)                         # endpoints and Jacobian vs the live runs
    theta = vs.solve(p)                        # surrogate initial conditions reaching TARGET
"""
import os
import json
import math
import numpy as np
import agama
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import v21_ics as vi
import r0_noise as rn
import v21_tune as vt

agama.setUnits(length=1, velocity=1, mass=232500)
POT = os.path.join(vi.ROOT, 'data', 'tango', 'potentials_triax')
LOCAL_TRACKS = os.path.join(vi.ROOT, 'output', 'v21_sim', 'tune')   # local copies of centres/sgr_mass
SIGMA = 120.          # MW velocity dispersion in the friction formula (V21)
SGR_PLUMMER = 2.      # scale radius of the Sgr potential acting on the MW [kpc]

_halo = agama.Potential(os.path.join(POT, 'mw00.pot'))
_mw = agama.Potential(_halo, *rn.v21_static())
_lmc = agama.Potential(os.path.join(POT, 'lmc00.pot'))
M_LMC = _lmc.totalMass()


# ------------------------------------------------------------------------------------ live inputs
def sgr_mass_series(k, radius=5.):
    """On OBS: mass of all Sgr particles within `radius` of the Sgr centre in round k's base run."""
    comps = json.load(open(vt.BASE_ICS + '.json'))['components']
    evol = os.path.join(vt.run_dir(k, 0), 'evol.nemo')
    tr = np.load(os.path.join(vt.run_dir(k, 0), 'centres.npz'))
    a, b = comps['sgr']
    mass = []
    for t, c in zip(tr['times'], tr['sgr']):
        xv, m = rn.read_snapshot(evol, t)
        sel = np.sum((xv[a:b, :3] - c[:3])**2, axis=1) < radius**2
        mass.append(m[a:b][sel].sum())
        print('t = %6.3f  M_Sgr(<%g kpc) = %.3g Msun' % (t, radius, mass[-1] * 232500), flush=True)
    out = os.path.join(vt.round_dir(k), 'sgr_mass.npz')
    np.savez(out, times=tr['times'], mass=np.array(mass))
    return out


def load_tracks(k, d=None):
    """Centre tracks of round k (runs 0-12), the round's theta/deltas and the base run's Sgr mass."""
    d = os.path.join(LOCAL_TRACKS, 'round%02d' % k) if d is None else d
    info = json.load(open(os.path.join(d, 'round.json')))
    runs = [dict(np.load(os.path.join(d, 'run%02d_centres.npz' % i))) for i in range(13)]
    sm = np.load(os.path.join(d, 'sgr_mass.npz'))
    return dict(info=info, runs=runs, sgr_t=sm['times'], sgr_m=sm['mass'])


# ------------------------------------------------------------------------------------ the model
def _friction(dx, dv, mass, lnL):
    """Chandrasekhar acceleration (V21 eq. 2) on a body of `mass` at dx, dv relative to the MW."""
    v = np.linalg.norm(dv)
    if lnL == 0 or v == 0:
        return np.zeros(3)
    X = v / (np.sqrt(2) * SIGMA)
    rho = _halo.density(dx)
    return -4 * np.pi * rho * lnL * mass * dv / v**3 * (
        math.erf(X) - 2 / np.sqrt(np.pi) * X * np.exp(-X**2))


def integrate(theta, p, sgr_t, sgr_m, t_eval=None):
    """
    Integrate MW, LMC and Sgr from T_START to T_END. p = dict(lnL_lmc, lnL_sgr). Returns
    (times, rel) with rel[:, :6] = LMC - MW and rel[:, 6:] = Sgr - MW phase-space coordinates.
    """
    msgr = lambda t: np.interp(t, sgr_t, sgr_m)

    def rhs(t, w):
        xm, vm, xl, vl, xs, vs_ = w[0:3], w[3:6], w[6:9], w[9:12], w[12:15], w[15:18]
        ms = msgr(t)
        dl, ds = xl - xm, xs - xm
        # Plummer pull of Sgr on the MW
        r2 = np.sum(ds**2) + SGR_PLUMMER**2
        am = _lmc.force(xm - xl) + ms * ds / r2**1.5
        al = _mw.force(dl) + _friction(dl, vl - vm, M_LMC, p['lnL_lmc'])
        as_ = _mw.force(ds) + _lmc.force(xs - xl) + _friction(ds, vs_ - vm, ms, p['lnL_sgr'])
        return np.concatenate((vm, am, vl, al, vs_, as_))

    w0 = np.concatenate((np.zeros(6), theta))
    sol = solve_ivp(rhs, (vt.T_START, vt.T_END), w0, t_eval=t_eval, rtol=1e-9, atol=1e-7,
                    method='DOP853')
    w = sol.y.T
    rel = np.hstack((w[:, 6:12] - w[:, 0:6], w[:, 12:18] - w[:, 0:6]))
    return sol.t, rel


def endpoint(theta, p, sgr_t, sgr_m):
    return integrate(theta, p, sgr_t, sgr_m, t_eval=[vt.T_END])[1][-1]


def jacobian(theta, p, sgr_t, sgr_m, delta=vt.DELTA):
    """Forward-difference Jacobian of the surrogate endpoints, same steps as the live rounds."""
    F0 = endpoint(theta, p, sgr_t, sgr_m)
    J = np.empty((12, 12))
    for j in range(12):
        th = np.array(theta, float)
        th[j] += delta[j]
        J[:, j] = (endpoint(th, p, sgr_t, sgr_m) - F0) / delta[j]
    return F0, J


# ------------------------------------------------------------------------------------ calibration
def run_theta(info, i):
    """theta of run i of a round (as v21_tune.run_theta, from the round.json contents)."""
    theta = np.array(info['theta'], float)
    if i > 0:
        theta[i - 1] += info['delta'][i - 1]
    return theta


def _live_rel(run):
    return np.hstack((run['lmc'] - run['mw'], run['sgr'] - run['mw']))


def calibrate(tr, runs=range(13), t_max=vt.T_END, p0=(3., 3.)):
    """
    Coulomb logarithms that best reproduce the live LMC and Sgr tracks (relative positions, scaled
    by the tuning tolerances x 10) of the given runs, over snapshots up to t_max.
    """
    def resid(q):
        p = dict(lnL_lmc=q[0], lnL_sgr=q[1])
        out = []
        for i in runs:
            run = tr['runs'][i]
            sel = run['times'] <= t_max + 1e-9
            t, rel = integrate(run_theta(tr['info'], i), p, tr['sgr_t'], tr['sgr_m'],
                               t_eval=run['times'][sel])
            dx = rel - _live_rel(run)[sel]
            out.append(np.hstack((dx[:, 0:3] / 2., dx[:, 6:9] / 1.)).ravel())
        return np.concatenate(out)
    res = least_squares(resid, p0, bounds=([0, 0], [20, 20]), x_scale=1., diff_step=1e-2)
    p = dict(lnL_lmc=res.x[0], lnL_sgr=res.x[1])
    print('calibrated: lnL_lmc = %.2f, lnL_sgr = %.2f  (rms scaled residual %.2f)' %
          (p['lnL_lmc'], p['lnL_sgr'], np.sqrt(np.mean(res.fun**2))))
    return p


def validate(tr, p):
    """Surrogate vs live: base-run track, endpoints of all 13 runs, and the Jacobian."""
    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    info = tr['info']
    run = tr['runs'][0]
    t, rel = integrate(np.array(info['theta']), p, tr['sgr_t'], tr['sgr_m'], t_eval=run['times'])
    live = _live_rel(run)
    print('   t   |dx_LMC| |dv_LMC|  |dx_Sgr| |dv_Sgr|   (surrogate - live, base run)')
    for j in range(0, len(t), 2):
        d = rel[j] - live[j]
        print('%6.3f %7.2f %7.2f %8.2f %7.2f' % (t[j], *[np.linalg.norm(d[s:s + 3]) for s in (0, 3, 6, 9)]))
    F_live = np.array([_live_rel(r)[-1] for r in tr['runs']])
    F0, J = jacobian(np.array(info['theta']), p, tr['sgr_t'], tr['sgr_m'], info['delta'])
    J_live = (F_live[1:] - F_live[0]).T / np.array(info['delta'])[None, :]
    print('base endpoint  live %s\n          surrogate %s' % (F_live[0], F0))
    print('(surrogate - live)/TOL %s' % ((F0 - F_live[0]) / vt.TOL))
    W = 1 / vt.TOL
    dlive = np.linalg.solve(W[:, None] * J_live, W * (vt.TARGET - F_live[0]))
    dsur = np.linalg.solve(W[:, None] * J, W * (vt.TARGET - F_live[0]))
    print('Gauss-Newton step from round %d with live J %s\n                       surrogate J %s' %
          (info['round'], dlive, dsur))
    rel_err = np.linalg.norm(W[:, None] * (J - J_live), axis=0) / np.linalg.norm(W[:, None] * J_live, axis=0)
    print('relative Jacobian column error %s' % rel_err)
    return dict(F0=F0, J=J, F_live=F_live, J_live=J_live)


# ------------------------------------------------------------------------------------ solve
def solve(p, tr, theta0=None, offset=None, niter=10):
    """
    Initial conditions with surrogate endpoints = TARGET - offset (offset = live - surrogate at a
    calibration point, or zero), by Gauss-Newton on the surrogate starting from theta0.
    """
    theta = np.array(tr['info']['theta'] if theta0 is None else theta0, float)
    offset = np.zeros(12) if offset is None else np.asarray(offset)
    W = 1 / vt.TOL
    for it in range(niter):
        F, J = jacobian(theta, p, tr['sgr_t'], tr['sgr_m'])
        r = vt.TARGET - offset - F
        print('iteration %d: max |residual|/TOL = %.2f' % (it, np.max(np.abs(r * W))), flush=True)
        if np.all(np.abs(r * W) < 0.1):
            break
        # damped step: shrink until the residual decreases
        step = np.linalg.solve(W[:, None] * J, W * r)
        for lam in (1, 0.5, 0.25, 0.125, 0.0625):
            Fn = endpoint(theta + lam * step, p, tr['sgr_t'], tr['sgr_m'])
            if np.sum(((vt.TARGET - offset - Fn) * W)**2) < np.sum((r * W)**2):
                break
        theta = theta + lam * step
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    print('theta = %s\nshift from start = %s' % (theta, theta - np.array(tr['info']['theta'])))
    return theta
