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
    theta = vs.restricted_start(tr)            # preferred: LMC time delay + restricted Sgr solve

The rigid three-body model misses the halo response near the LMC pericentre (LMC 8 kpc off by
t=-0.5 in round 0), so starting guesses use restricted_start instead.
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


# ------------------------------------------------------------------------------------ time delays
def best_delay(tr, i=0, taus=np.arange(-0.4, 1e-4, 0.002)):
    """
    Times tau at which the live LMC and Sgr tracks of run i come closest to TARGET (chi^2 in units
    of TOL, cubic Hermite interpolation between snapshots). Returns {'lmc': (tau, resid/TOL), 'sgr': ...}.
    """
    from scipy.interpolate import CubicHermiteSpline
    run = tr['runs'][i]
    t, rel = run['times'], _live_rel(run)
    out = {}
    for name, s in (('lmc', 0), ('sgr', 6)):
        x = CubicHermiteSpline(t, rel[:, s:s + 3], rel[:, s + 3:s + 6])
        v = CubicHermiteSpline(t, rel[:, s + 3:s + 6], np.gradient(rel[:, s + 3:s + 6], t, axis=0))
        chi = (np.hstack((x(taus), v(taus))) - vt.TARGET[s:s + 6]) / vt.TOL[s:s + 6]
        j = np.argmin(np.sum(chi**2, axis=1))
        out[name] = (taus[j], chi[j])
    return out


def delayed_theta(theta, dt_lmc, dt_sgr, p, sgr_mass):
    """
    Start each body earlier on its own orbit: the LMC is integrated backward by dt_lmc (>0) in the
    rigid MW+LMC two-body problem (which the live runs follow at these distances), Sgr backward by
    dt_sgr in the rigid MW with friction (lnL_sgr, constant mass sgr_mass). Relative coordinates.
    """
    theta = np.asarray(theta, float)

    def two_body(t, w):
        r = w[:3]
        return np.hstack((w[3:], _mw.force(r) - _lmc.force(-r)))

    def sgr(t, w):
        return np.hstack((w[3:], _mw.force(w[:3]) + _friction(w[:3], w[3:], sgr_mass, p['lnL_sgr'])))
    lmc = solve_ivp(two_body, (0, -dt_lmc), theta[:6], rtol=1e-10, atol=1e-10).y[:, -1]
    s = solve_ivp(sgr, (0, -dt_sgr), theta[6:], rtol=1e-10, atol=1e-10).y[:, -1]
    return np.concatenate((lmc, s))


# ------------------------------------------------------------------------- restricted Sgr model
# Sgr as a test body (with friction and its own MW reflex) in the rigid MW and LMC potentials moving
# along a live round's MW and LMC tracks, optionally delayed. The live MW track contains the MW's
# reflex to Sgr; it is removed (sgr_reflex) and recomputed along the model's Sgr orbit, otherwise a
# delayed MW track wobbles out of phase with Sgr.
def sgr_reflex(tr, run):
    """MW reflex (position, velocity; from rest at T_START) to the live Sgr of a run, Plummer of its bound mass."""
    from scipy.interpolate import CubicHermiteSpline
    rel = run['sgr'] - run['mw']
    x = CubicHermiteSpline(run['times'], rel[:, :3], rel[:, 3:])
    m = lambda t: np.interp(t, tr['sgr_t'], tr['sgr_m'])
    f = lambda t, w: np.hstack((w[3:], m(t) * x(t) / (np.sum(x(t)**2) + SGR_PLUMMER**2)**1.5))
    return solve_ivp(f, (vt.T_START, vt.T_END), np.zeros(6), t_eval=run['times'], rtol=1e-9, atol=1e-9).y.T


def drivers(tr, run, delay=0.):
    """
    Splines of the MW (Sgr reflex removed) and LMC absolute tracks of a live run, delayed by `delay`;
    before the run's start both follow the rigid two-body orbit backwards. With delay > 0 the frame
    is shifted so that the MW starts at rest at the origin.
    """
    from scipy.interpolate import CubicHermiteSpline
    t = run['times'] + delay
    mw, lmc = run['mw'] - sgr_reflex(tr, run), run['lmc'].copy()
    if delay > 0:
        def two(tt, w):
            xm, vm, xl, vl = w[:3], w[3:6], w[6:9], w[9:]
            return np.concatenate((vm, _lmc.force(xm - xl), vl, _mw.force(xl - xm)))
        tb = np.linspace(vt.T_START + delay, vt.T_START, 9)
        w = solve_ivp(two, (tb[0], tb[-1]), np.concatenate((mw[0], lmc[0])), t_eval=tb,
                      rtol=1e-10, atol=1e-10).y.T[::-1][:-1]
        t = np.concatenate((tb[::-1][:-1], t))
        mw, lmc = np.vstack((w[:, :6], mw)), np.vstack((w[:, 6:], lmc))
        mw0, dt = mw[0].copy(), t - t[0]
        for a in (mw, lmc):
            a[:, :3] -= mw0[:3] + mw0[3:] * dt[:, None]
            a[:, 3:] -= mw0[3:]
    return [CubicHermiteSpline(t, a[:, :3], a[:, 3:]) for a in (mw, lmc)]


def sgr_restricted(tr, th_sgr, drv, t_eval, mass_delay=0., lnL=3.):
    """Sgr phase-space coordinates relative to the MW centre (driver + Sgr reflex) at t_eval."""
    xb, xl = drv
    vb = xb.derivative()
    m = lambda t: np.interp(t - mass_delay, tr['sgr_t'], tr['sgr_m'])

    def rhs(t, w):
        x, v, R, VR = w[:3], w[3:6], w[6:9], w[9:]
        d = x - xb(t) - R
        ms = m(t)
        return np.hstack((v, _mw.force(d) + _lmc.force(x - xl(t)) + _friction(d, v - vb(t) - VR, ms, lnL),
                          VR, ms * d / (np.sum(d**2) + SGR_PLUMMER**2)**1.5))
    w0 = np.hstack((th_sgr + np.hstack((xb(vt.T_START), vb(vt.T_START))), np.zeros(6)))
    w = solve_ivp(rhs, (vt.T_START, t_eval[-1]), w0, t_eval=t_eval, rtol=1e-9, atol=1e-8,
                  method='DOP853').y.T
    return w[:, :6] - np.hstack((xb(t_eval), vb(t_eval))) - w[:, 6:]


def restricted_start(tr, lnL=3.):
    """
    Starting guess from a live round: LMC delayed onto its target (best_delay + two-body backward
    integration); Sgr solved in the restricted model driven by the delayed MW/LMC tracks, with the
    model's error on the round's base run subtracted from the target.
    """
    info, base = tr['info'], tr['runs'][0]
    th0 = np.array(info['theta'])
    d = best_delay(tr)
    DL, DS = -d['lmc'][0], -d['sgr'][0]
    th = delayed_theta(th0, DL, DS, dict(lnL_sgr=lnL), tr['sgr_m'][0])
    ends = np.array([vt.T_START, vt.T_END])
    offset = sgr_restricted(tr, th0[6:], drivers(tr, base), ends, lnL=lnL)[-1] - _live_rel(base)[-1, 6:]
    drv = drivers(tr, base, delay=DL)
    W = 1 / vt.TOL[6:]
    f = lambda s: (sgr_restricted(tr, s, drv, ends, mass_delay=DS, lnL=lnL)[-1] - offset - vt.TARGET[6:]) * W
    print('delays: LMC %.3f, Sgr %.3f; Sgr-delay-only start, predicted resid/TOL %s' %
          (DL, DS, np.array2string(f(th[6:]), precision=1)))
    res = least_squares(f, th[6:], x_scale=np.array([1, 1, 1, 10, 10, 10.]), diff_step=1e-4)
    print('restricted solve: predicted resid/TOL %s' % np.array2string(res.fun, precision=1))
    th[6:] = res.x
    return th
