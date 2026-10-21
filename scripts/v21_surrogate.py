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
SIGMA_PROFILE = None  # optional (r, sigma_1D(r)) of the MW halo; replaces SIGMA (set_sigma_profile)
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
def halo_sigma_profile(snapshot=os.path.join(vi.ROOT, 'output', 'v21_sim', 'ics', 'mw_triax_full', 'mw_halo.nemo'),
                       edges=(2, 5, 7.5, 10, 12.5, 15, 17.5, 20, 25, 30, 40, 50, 60, 70, 80, 100, 130, 170, 220)):
    """1D velocity dispersion of the MW halo particles in spherical shells (mass-weighted)."""
    xv, m = agama.readSnapshot(snapshot)
    r = np.linalg.norm(xv[:, :3], axis=1)
    mid, sig = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        s = (r > a) & (r < b)
        v = xv[s, 3:] - np.average(xv[s, 3:], axis=0, weights=m[s])
        mid.append(0.5 * (a + b))
        sig.append(np.sqrt(np.average(np.sum(v**2, axis=1), weights=m[s]) / 3))
    return np.array(mid), np.array(sig)


def set_sigma_profile(profile):
    """Use sigma(r) (tuple of arrays, e.g. from halo_sigma_profile) in the friction; None = fixed SIGMA."""
    global SIGMA_PROFILE
    SIGMA_PROFILE = profile


def _sigma(dx):
    if SIGMA_PROFILE is None or dx is None:
        return SIGMA
    return np.interp(np.linalg.norm(dx), *SIGMA_PROFILE)


def _friction(dx, dv, mass, lnL):
    """Chandrasekhar acceleration (V21 eq. 2) on a body of `mass` at dx, dv relative to the MW."""
    v = np.linalg.norm(dv)
    if lnL == 0 or v == 0:
        return np.zeros(3)
    X = v / (np.sqrt(2) * _sigma(dx))
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
    if 'runs' in info:
        return np.array(info['runs'][i], float)
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


# ------------------------------------------------------------------- evolving-field Sgr model
# The MW (halo: Multipole, disk+bulge: CylSpline, about the live MW centre) and the LMC (Multipole,
# about the live LMC centre) fitted to every snapshot of a live run, Sgr particles excluded. Between
# snapshots the centres follow the live tracks (cubic Hermite) and the shapes are interpolated
# linearly; Sgr moves in these fields as a test body with friction and its own MW reflex (as in
# sgr_restricted), so a changed Sgr orbit moves the MW centre consistently.
LMC_EXP = dict(type='Multipole', lmax=4, symmetry='n', gridSizeR=25, rmin=0.2, rmax=300.)


def expansion_dir(k, i=0):
    return os.path.join(vt.run_dir(k, i), 'expansions')


def build_expansions(k, i=0):
    """On OBS: expansions of every snapshot of run i of round k -> runKK/expansions/{halo,disk,lmc}_JJ.coef."""
    comps = json.load(open(vt.BASE_ICS + '.json'))['components']
    evol = os.path.join(vt.run_dir(k, i), 'evol.nemo')
    tr = np.load(os.path.join(vt.run_dir(k, i), 'centres.npz'))
    d = expansion_dir(k, i)
    os.makedirs(d, exist_ok=True)
    nst = vi.MW_COUNTS['nbody_stars']
    (a, b), (la, lb) = comps['mw'], comps['lmc']
    for j, t in enumerate(tr['times']):
        xv, m = rn.read_snapshot(evol, t)
        x = xv[a:b, :3] - tr['mw'][j, :3]
        agama.Potential(particles=(x[nst:], m[a + nst:b]), **rn.HALO_EXP).export(os.path.join(d, 'halo_%02d.coef' % j))
        agama.Potential(particles=(x[:nst], m[a:a + nst]), **rn.DISK_EXP).export(os.path.join(d, 'disk_%02d.coef' % j))
        agama.Potential(particles=(xv[la:lb, :3] - tr['lmc'][j, :3], m[la:lb]), **LMC_EXP).export(
            os.path.join(d, 'lmc_%02d.coef' % j))
        print('t = %6.3f expansions done' % t, flush=True)
    np.save(os.path.join(d, 'times.npy'), tr['times'])
    return d


class EvolvingField:
    """Time-interpolated MW and LMC fields of a live run (expansions from build_expansions)."""

    def __init__(self, d, run):
        from scipy.interpolate import CubicHermiteSpline
        self.times = np.load(os.path.join(d, 'times.npy'))
        n = len(self.times)
        self.halo = [agama.Potential(os.path.join(d, 'halo_%02d.coef' % j)) for j in range(n)]
        self.mw = [agama.Potential(self.halo[j], agama.Potential(os.path.join(d, 'disk_%02d.coef' % j)))
                   for j in range(n)]
        self.lmc = [agama.Potential(os.path.join(d, 'lmc_%02d.coef' % j)) for j in range(n)]
        self.xm = CubicHermiteSpline(run['times'], run['mw'][:, :3], run['mw'][:, 3:])
        self.xl = CubicHermiteSpline(run['times'], run['lmc'][:, :3], run['lmc'][:, 3:])

    def _w(self, t):
        j = int(np.clip(np.searchsorted(self.times, t) - 1, 0, len(self.times) - 2))
        return j, (t - self.times[j]) / (self.times[j + 1] - self.times[j])

    def force(self, x, t, xm):
        """Acceleration at absolute x for an MW centre at xm (the LMC follows its live track)."""
        j, w = self._w(t)
        dm, dl = x - xm, x - self.xl(t)
        return ((1 - w) * (self.mw[j].force(dm) + self.lmc[j].force(dl)) +
                w * (self.mw[j + 1].force(dm) + self.lmc[j + 1].force(dl)))

    def halo_density(self, d, t):
        j, w = self._w(t)
        return (1 - w) * self.halo[j].density(d) + w * self.halo[j + 1].density(d)


def _friction_rho(rho, dv, mass, lnL, dx=None):
    v = np.linalg.norm(dv)
    X = v / (np.sqrt(2) * _sigma(dx))
    return -4 * np.pi * rho * lnL * mass * dv / v**3 * (math.erf(X) - 2 / np.sqrt(np.pi) * X * np.exp(-X**2))


def sgr_evolving(ev, tr, run, th_sgr, t_eval, lnL=3., t0=vt.T_START, w0=None):
    """
    Sgr relative to the MW centre at t_eval, in the evolving field of `run`. The MW centre is the live
    track minus the live Sgr reflex plus the reflex to the model Sgr. Starts from th_sgr (relative,
    at T_START) or from an absolute Sgr state w0 at t0 (then the reflex starts at its live value).
    """
    from scipy.interpolate import CubicHermiteSpline
    Rl = sgr_reflex(tr, run)
    R_live = CubicHermiteSpline(run['times'], Rl[:, :3], Rl[:, 3:])
    vm_live = ev.xm.derivative()
    m = lambda t: np.interp(t, tr['sgr_t'], tr['sgr_m'])

    def rhs(t, w):
        x, v, R, VR = w[:3], w[3:6], w[6:9], w[9:]
        xm = ev.xm(t) - R_live(t) + R
        vm = vm_live(t) - R_live.derivative()(t) + VR
        d = x - xm
        ms = m(t)
        a = ev.force(x, t, xm) + _friction_rho(ev.halo_density(d, t), v - vm, ms, lnL, d)
        return np.hstack((v, a, VR, ms * d / (np.sum(d**2) + SGR_PLUMMER**2)**1.5))
    if w0 is None:
        w0 = np.hstack((th_sgr + np.hstack((ev.xm(t0), vm_live(t0))), np.zeros(6)))
    else:
        w0 = np.hstack((w0, R_live(t0), R_live.derivative()(t0)))
    t_eval = np.atleast_1d(t_eval)
    sol = solve_ivp(rhs, (t0, t_eval[-1] if t_eval[-1] != t0 else t_eval[0]), w0, t_eval=t_eval,
                    rtol=1e-9, atol=1e-8, method='DOP853')
    w = sol.y.T
    xm = ev.xm(sol.t) - R_live(sol.t) + w[:, 6:9]
    vm = vm_live(sol.t) - R_live.derivative()(sol.t) + w[:, 9:]
    return w[:, :6] - np.hstack((xm, vm))


def validate_evolving(ev, tr, lnL=3.):
    """Evolving-field Sgr vs the live round: base-run track and the Sgr block of the Jacobian."""
    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    info, base = tr['info'], tr['runs'][0]
    ts = base['times']
    a = sgr_evolving(ev, tr, base, np.array(info['theta'][6:]), ts, lnL)
    live = _live_rel(base)[:, 6:]
    print('   t   |dx| |dv|   (evolving model - live, base run)')
    for j in range(0, len(ts), 2):
        print('%6.3f %5.2f %6.2f' % (ts[j], np.linalg.norm(a[j, :3] - live[j, :3]), np.linalg.norm(a[j, 3:] - live[j, 3:])))
    Jm, Jl = [], []
    for i in range(7, 13):
        run, dl = tr['runs'][i], info['delta'][i - 1]
        Jm.append((sgr_evolving(ev, tr, base, run_theta(info, i)[6:], ts[-1:], lnL)[-1] - a[-1]) / dl)
        Jl.append((_live_rel(run)[-1, 6:] - live[-1]) / dl)
    Jm, Jl = np.array(Jm).T, np.array(Jl).T
    W = 1 / vt.TOL[6:]
    print('Sgr Jacobian column error (TOL-weighted):',
          np.linalg.norm(W[:, None] * (Jm - Jl), axis=0) / np.linalg.norm(W[:, None] * Jl, axis=0))
    print('endpoint (model - live)/TOL:', (a[-1] - live[-1]) * W)
    return a[-1] - live[-1], Jm, Jl


def evolving_start(ev, tr, theta_lmc, sgr_guess, offset, lnL=3.):
    """Sgr start reaching TARGET - offset in the evolving field (offset = model - live at the round's theta)."""
    base = tr['runs'][0]
    W = 1 / vt.TOL[6:]
    f = lambda s: (sgr_evolving(ev, tr, base, s, [vt.T_END], lnL)[-1] - offset - vt.TARGET[6:]) * W
    print('guess: predicted resid/TOL %s' % np.array2string(f(sgr_guess), precision=1))
    res = least_squares(f, sgr_guess, x_scale=np.array([0.1, 0.1, 0.1, 1, 1, 1.]), diff_step=1e-5)
    print('evolving-field solve: predicted resid/TOL %s' % np.array2string(res.fun, precision=1))
    return np.concatenate((theta_lmc, res.x))


def plot_orbits(ev, tr, starts, sgr_theta, fout, lnL=3.):
    """
    Sgr orbit from sgr_theta in the evolving field and the live LMC track of the round's base run,
    MW-centred (Galactocentric) x-y, x-z, y-z; starting points (t=-3) of each theta in `starts`
    ({label: theta}); present-day targets. Rows: Sgr scale, full LMC scale, zoom on the Sgr starts.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    base = tr['runs'][0]
    ts = np.linspace(vt.T_START, vt.T_END, 3001)
    sgr = sgr_evolving(ev, tr, base, np.asarray(sgr_theta), ts, lnL)
    lmc = _live_rel(base)[:, :6]
    from scipy.interpolate import CubicHermiteSpline
    lmc = CubicHermiteSpline(base['times'], lmc[:, :3], lmc[:, 3:])(ts)
    pairs = [(0, 1), (0, 2), (1, 2)]
    lab = 'xyz'
    marks = ['o', 's', '^', 'D', 'v']
    with plt.style.context('default'):
        fig, axes = plt.subplots(3, 3, figsize=(13, 13))
        for row, lim in ((0, 90), (1, 400)):
            for ax, (p, q) in zip(axes[row], pairs):
                ax.plot(sgr[:, p], sgr[:, q], '-', color='tab:red', lw=0.8, label='Sgr (evolving-field model)')
                ax.plot(lmc[:, p], lmc[:, q], '-', color='tab:blue', lw=1.2, label='LMC (round %d live)' % tr['info']['round'])
                ax.plot(0, 0, '+', color='k', ms=10)
                for (name, th), mk in zip(starts.items(), marks):
                    for s, c in ((6, 'tab:red'), (0, 'tab:blue')):
                        ax.plot(th[s + p], th[s + q], mk, mfc='none', mec=c, ms=7, label=name if s == 6 else None)
                for s, c in ((6, 'tab:red'), (0, 'tab:blue')):
                    ax.plot(vt.TARGET[s + p], vt.TARGET[s + q], '*', color=c, ms=12,
                            label='present day (target)' if s == 6 else None)
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
                ax.set_aspect('equal')
                ax.set_xlabel('%s [kpc]' % lab[p])
                ax.set_ylabel('%s [kpc]' % lab[q])
        # zoom on the Sgr starting points; arrows = starting velocity x 0.02 time units
        cen = np.mean([th[6:9] for th in starts.values()], axis=0)
        for ax, (p, q) in zip(axes[2], pairs):
            ax.plot(sgr[:, p], sgr[:, q], '-', color='tab:red', lw=0.8)
            for (name, th), mk in zip(starts.items(), marks):
                ax.plot(th[6 + p], th[6 + q], mk, mfc='none', mec='tab:red', ms=8)
                ax.annotate('', xy=(th[6 + p] + 0.02 * th[9 + p], th[6 + q] + 0.02 * th[9 + q]),
                            xytext=(th[6 + p], th[6 + q]), arrowprops=dict(arrowstyle='->', color='0.4', lw=0.8))
            ax.set_xlim(cen[p] - 4, cen[p] + 4)
            ax.set_ylim(cen[q] - 4, cen[q] + 4)
            ax.set_aspect('equal')
            ax.set_xlabel('%s [kpc]' % lab[p])
            ax.set_ylabel('%s [kpc]' % lab[q])
            ax.set_title('Sgr starting points (t = %g); arrows: v x 0.02' % vt.T_START, fontsize='small')
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc='upper center', ncol=4, frameon=False, fontsize='small')
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(fout, dpi=150)
    print('saved', fout)
