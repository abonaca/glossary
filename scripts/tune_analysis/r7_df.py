"""
Sgr's dynamical friction in round 7's base run, measured in the evolving MW+LMC expansions, and the
start shift that the measured model asks for to fix Sgr's present-day vx.

Model: Sgr as a test body in the time-interpolated MW (halo+disk) and LMC expansions of the live run
(v21_surrogate.EvolvingField), with its own MW reflex (as in vs.sgr_evolving) and Chandrasekhar
friction scaled per 122-Myr snapshot interval: a_DF = f_j * Chandrasekhar(lnL=1, rho_halo(t), M_Sgr(t)).

    import r7_df as rd
    m = rd.load()
    f, seg = rd.fit_friction(m)                     # f_j per interval + friction dv and misfits
    rd.forward_backward(m, f)                       # model from the live start / live end vs the live track
    rd.vx_shift(m, f)                               # start change for target vx (and full target)
"""
import sys
import numpy as np
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import v21_surrogate as vs
import v21_tune as vt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.interpolate import CubicHermiteSpline

D = '/home/ana/projects/glossary/output/v21_sim/tune/round07'
TU = 977.8                 # Myr per time unit
np.set_printoptions(precision=3, suppress=True, linewidth=160)


class CubicField(vs.EvolvingField):
    """EvolvingField with the expansions interpolated in time by Catmull-Rom (cubic, 4 snapshots)."""

    def _cw(self, t):
        j, u = self._w(t)
        n = len(self.times)
        if j == 0 or j == n - 2:                 # end intervals: quadratic through 3 snapshots
            k = [0, 1, 2] if j == 0 else [n - 3, n - 2, n - 1]
            s = u if j == 0 else u + 1
            return k, [(s - 1) * (s - 2) / 2, -s * (s - 2), s * (s - 1) / 2]
        u2, u3 = u * u, u * u * u
        return [j - 1, j, j + 1, j + 2], [(-u3 + 2 * u2 - u) / 2, (3 * u3 - 5 * u2 + 2) / 2,
                                           (-3 * u3 + 4 * u2 + u) / 2, (u3 - u2) / 2]

    def force(self, x, t, xm):
        k, c = self._cw(t)
        dm, dl = x - xm, x - self.xl(t)
        return sum(ci * (self.mw[ki].force(dm) + self.lmc[ki].force(dl)) for ki, ci in zip(k, c))

    def halo_density(self, d, t):
        k, c = self._cw(t)
        return sum(ci * self.halo[ki].density(d) for ki, ci in zip(k, c))


def load(d=D, cubic=False, mass=None, exp='expansions'):
    """mass: None = M(<5 kpc) from sgr_mass.npz; 'bound' or a radius (10., 20.) from run00/sgr_bound_mass.npz."""
    run = dict(np.load(d + '/run00/centres.npz'))
    sm = np.load(d + '/sgr_mass.npz')
    tr = dict(sgr_t=sm['times'], sgr_m=sm['mass'])
    if mass is not None:
        b = np.load(d + '/run00/sgr_bound_mass.npz')
        if mass in ('bound', 'bound_held'):
            tr['sgr_m'] = b['bound'].copy()
            if mass == 'bound_held':                 # t=0 value held at the previous snapshot's
                tr['sgr_m'][-1] = tr['sgr_m'][-2]
        else:
            tr['sgr_m'] = b['within'][:, list(b['radii']).index(mass)]
        tr['sgr_t'] = b['times']
    ev = (CubicField if cubic else vs.EvolvingField)(d + '/run00/' + exp, run)
    Rl = vs.sgr_reflex(tr, run)
    R = CubicHermiteSpline(run['times'], Rl[:, :3], Rl[:, 3:])
    info = vs.json.load(open(d + '/round.json'))
    return dict(run=run, tr=tr, ev=ev, R=R, vm=ev.xm.derivative(), dR=R.derivative(),
                theta=np.array(info['runs'][0]), times=run['times'],
                live=run['sgr'] - run['mw'])


def _seg(m, t):
    return int(np.clip(np.searchsorted(m['times'], t, side='right') - 1, 0, len(m['times']) - 2))


def _frame(d, dv):
    e1 = dv / np.linalg.norm(dv)
    e2 = d - np.dot(d, e1) * e1
    e2 /= np.linalg.norm(e2)
    return np.array([e1, e2, np.cross(e1, e2)])


def integrate(m, f, t0, t1, w0_rel, t_eval=None, reflex0=None, res=None, kind='time'):
    """
    Sgr from the MW-relative state w0_rel at t0 to t1 (either direction) with friction factors f
    (lnL: a scalar, one per interval, or one per snapshot interpolated linearly in time), plus an optional residual acceleration res[j] = (2, 3): in interval j,
    a = (res[j,0] + u res[j,1]) . (v_hat, r_perp_hat, normal), u = 0..1 across the interval, with v and
    r relative to the model MW centre. kind='chandra': the v_hat component is instead
    (res[j,0,0] + u res[j,1,0]) x the Chandrasekhar acceleration with lnL=1 (so it follows the orbit's
    own density and velocity); the other two components are as above. The model MW centre = live MW track - live Sgr reflex + model reflex;
    the model reflex starts at its live value at t0 (or reflex0). Returns (t, MW-relative states,
    friction acceleration along the solution).
    """
    ev, R, dR, vm_live = m['ev'], m['R'], m['dR'], m['vm']
    ms = lambda t: np.interp(t, m['tr']['sgr_t'], m['tr']['sgr_m'])
    if np.isscalar(f):
        fac = lambda t: f
    elif len(f) == len(m['times']):                 # values at the snapshots, linear in between
        fac = lambda t: np.interp(t, m['times'], f)
    else:                                           # one value per interval
        fac = lambda t: f[_seg(m, t)]

    def mw(t, w):
        return (ev.xm(t) - R(t) + w[6:9], vm_live(t) - dR(t) + w[9:])

    def df(t, w):
        xm, vmm = mw(t, w)
        d = w[:3] - xm
        ch = vs._friction_rho(ev.halo_density(d, t), w[3:6] - vmm, ms(t), 1., d)
        a = ch * fac(t)
        if res is not None:
            j = _seg(m, t)
            u = (t - m['times'][j]) / (m['times'][j + 1] - m['times'][j])
            c = res[j, 0] + u * res[j, 1]
            if kind == 'chandra':
                a = a + c[0] * ch + c[1:] @ _frame(d, w[3:6] - vmm)[1:]
            else:
                a = a + c @ _frame(d, w[3:6] - vmm)
        return a

    def rhs(t, w):
        xm, _ = mw(t, w)
        d = w[:3] - xm
        return np.hstack((w[3:6], ev.force(w[:3], t, xm) + df(t, w),
                          w[9:], ms(t) * d / (np.sum(d**2) + vs.SGR_PLUMMER**2)**1.5))
    r0 = np.hstack((R(t0), dR(t0))) if reflex0 is None else reflex0
    xm0 = np.hstack((ev.xm(t0), vm_live(t0))) - np.hstack((R(t0), dR(t0))) + r0
    w0 = np.hstack((np.asarray(w0_rel) + xm0, r0))
    t_eval = np.array([t1]) if t_eval is None else np.asarray(t_eval)
    sol = solve_ivp(rhs, (t0, t1), w0, t_eval=t_eval, rtol=1e-10, atol=1e-9, method='DOP853')
    w = sol.y.T
    rel = np.array([wi[:6] - np.hstack(mw(t, wi)) for t, wi in zip(sol.t, w)])
    adf = np.array([df(t, wi) for t, wi in zip(sol.t, w)])
    return sol.t, rel, adf


W = 1 / vt.TOL[6:]          # Sgr tolerances: 0.1 kpc, 1 km/s


def fit_friction(m, verbose=True):
    """Per interval: f_j that takes the live state at t_j onto the live state at t_j+1."""
    t, live = m['times'], m['live']
    f, rows = [], []
    for j in range(len(t) - 1):
        end = lambda q: integrate(m, q[0], t[j], t[j + 1], live[j])[1][-1]
        res = least_squares(lambda q: (end(q) - live[j + 1]) * W, [3.], diff_step=1e-3)
        fj = res.x[0]
        miss0 = end([0.]) - live[j + 1]
        miss = end([fj]) - live[j + 1]
        ts = np.linspace(t[j], t[j + 1], 41)
        _, rel, adf = integrate(m, fj, t[j], t[j + 1], live[j], t_eval=ts)
        along = np.sum(adf * rel[:, 3:], axis=1) / np.linalg.norm(rel[:, 3:], axis=1)
        dv = np.trapz(along, ts)                                   # km/s, along the velocity
        rmin = np.linalg.norm(rel[:, :3], axis=1).min()
        f.append(fj)
        rows.append(dict(t=t[j], f=fj, dv=dv, rmin=rmin, miss_nodf=miss0, miss=miss))
        if verbose:
            print('%6.0f Myr  r_min %5.1f kpc  f = %6.2f  dv_DF %6.2f km/s | miss f=0: %5.2f kpc %5.2f km/s'
                  '  fitted: %5.2f kpc %5.2f km/s' % (t[j] * TU, rmin, fj, dv, np.linalg.norm(miss0[:3]),
                                                       np.linalg.norm(miss0[3:]), np.linalg.norm(miss[:3]),
                                                       np.linalg.norm(miss[3:])), flush=True)
    return np.array(f), rows


def fit_residual(m, kind='time', verbose=True):
    """
    Per interval, the linear-in-time residual acceleration (no Chandrasekhar term) that takes the
    live state at t_j exactly onto the live state at t_j+1. Returns res (n_int, 2, 3) in
    (km/s)/time unit and per-interval dv (km/s) along v_hat, r_perp_hat, normal.
    """
    t, live = m['times'], m['live']
    n = len(t) - 1
    res = np.zeros((n, 2, 3))
    dvs = []
    for j in range(n):
        def miss(q):
            r = res.copy()
            r[j] = q.reshape(2, 3)
            return (integrate(m, 0., t[j], t[j + 1], live[j], res=r, kind=kind)[1][-1] - live[j + 1]) * W
        q, h = np.zeros(6), 1.            # Newton with a fixed finite-difference step [(km/s)/time unit]
        for _ in range(6):
            r0 = miss(q)
            if np.max(np.abs(r0)) < 1e-5:
                break
            J = np.array([(miss(q + h * e) - r0) / h for e in np.eye(6)]).T
            q = q - np.linalg.solve(J, r0)
        sol = type('S', (), dict(fun=miss(q)))
        res[j] = q.reshape(2, 3)
        ts = np.linspace(t[j], t[j + 1], 41)
        acc = components(m, res, kind, ts, live[j])[1]
        dv = np.trapz(acc, ts, axis=0)
        dvs.append(dv)
        if verbose:
            r = np.linalg.norm(live[j:j + 2, :3], axis=1)
            print('%6.0f Myr  r %5.1f-%5.1f kpc  dv drag %6.2f  in-plane %6.2f  normal %6.2f km/s  | left %.4f kpc %.4f km/s'
                  % (t[j] * TU, r[0], r[1], *dv, np.linalg.norm(sol.fun[:3] / W[:3]),
                     np.linalg.norm(sol.fun[3:] / W[3:])), flush=True)
    return res, np.array(dvs)


def components(m, res, kind, ts, w0_rel):
    """Non-conservative acceleration along an orbit from w0_rel at ts[0]: (rel states, (n, 3) components
    along v_hat, r_perp_hat, normal, in (km/s)/time unit)."""
    _, rel, adf = integrate(m, 0., ts[0], ts[-1], w0_rel, t_eval=ts, res=res, kind=kind)
    comp = np.array([_frame(r[:3], r[3:]) @ a for r, a in zip(rel, adf)])
    return rel, comp


def plot_drag(m, fits, fout='/home/ana/projects/glossary/plots/tune_r7_sgr_drag.png', lnL_ref=3.):
    """
    Drag components along the base run's orbit vs time, for each fit in `fits` ({label: (res, kind)}).
    Panels: Sgr distance; along-velocity (drag); in-plane perpendicular; normal; for 'chandra' fits the
    drag factor (effective lnL). Accelerations in km/s per Gyr.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.style.use('default')
    t = m['times']
    ts = np.concatenate([np.linspace(t[j], t[j + 1], 31)[:-1] for j in range(len(t) - 1)] + [t[-1:]])
    fig, ax = plt.subplots(5, 1, figsize=(8, 12), sharex=True)
    gyr = TU / 1e3
    for n, (label, (res, kind)) in enumerate(fits.items()):
        col = 'C%d' % n
        rel, comp = [], []
        for j in range(len(t) - 1):                    # each interval from the live state, as fitted
            sel = (ts >= t[j]) & (ts <= t[j + 1])
            r, c = components(m, res, kind, ts[sel], m['live'][j])
            rel.append(r); comp.append(c)
        rel, comp = np.vstack(rel), np.vstack(comp)
        tt = np.concatenate([ts[(ts >= t[j]) & (ts <= t[j + 1])] for j in range(len(t) - 1)]) * gyr
        for k in range(3):
            ax[k + 1].plot(tt, comp[:, k] / gyr, color=col, ls='-' if n == 0 else '--', label=label)
        if kind == 'chandra':
            u = np.concatenate([(ts[(ts >= t[j]) & (ts <= t[j + 1])] - t[j]) / (t[j + 1] - t[j]) for j in range(len(t) - 1)])
            jj = np.concatenate([np.full(((ts >= t[j]) & (ts <= t[j + 1])).sum(), j) for j in range(len(t) - 1)])
            ax[4].plot(tt, res[jj, 0, 0] + u * res[jj, 1, 0], color=col, label=label)
            ax[4].axhline(3, color='0.5', ls=':', label='V21 ln$\\Lambda$ = 3')
            ax[4].legend()
    ms = np.interp(tt / gyr, m['tr']['sgr_t'], m['tr']['sgr_m'])
    ch = np.array([vs._friction_rho(m['ev'].halo_density(r[:3], ti), r[3:], mi, lnL_ref, r[:3]) @ r[3:] / np.linalg.norm(r[3:])
                   for r, ti, mi in zip(rel, tt / gyr, ms)])
    ax[1].plot(tt, ch / gyr, 'k:', label='Chandrasekhar, ln$\\Lambda$ = %g (V21)' % lnL_ref)
    ax[1].legend()
    ax[0].plot(tt, np.linalg.norm(rel[:, :3], axis=1), 'k-')
    for tj in t * gyr:
        for a in ax:
            a.axvline(tj, color='0.85', lw=0.5, zorder=0)
    for a in ax[1:4]:
        a.axhline(0, color='0.5', lw=0.8)
    ax[0].set_ylabel('Sgr distance [kpc]')
    ax[1].set_ylabel('drag, along $v$\n[km/s/Gyr]')
    ax[2].set_ylabel('in-plane, $\\perp v$\n[km/s/Gyr]')
    ax[3].set_ylabel('normal to orbit\n[km/s/Gyr]')
    ax[4].set_ylabel('effective ln$\\Lambda$')
    ax[4].set_xlabel('time [Gyr]')
    fig.tight_layout()
    fig.savefig(fout, dpi=200)
    plt.close(fig)
    return fout


def forward_backward(m, f, res=None):
    """Model forwards from the live t=-3 start and backwards from the live t=0 state, vs the live track."""
    t, live = m['times'], m['live']
    _, fw, _ = integrate(m, f, t[0], t[-1], m['theta'][6:], t_eval=t, res=res)
    _, bw, _ = integrate(m, f, t[-1], t[0], live[-1], t_eval=t[::-1], res=res)
    bw = bw[::-1]
    print('   t[Myr]  forward - live: kpc  km/s | backward - live: kpc  km/s')
    for j in range(0, len(t), 2):
        print('%7.0f   %6.3f %6.2f | %6.3f %6.2f' % (t[j] * TU, np.linalg.norm(fw[j, :3] - live[j, :3]),
              np.linalg.norm(fw[j, 3:] - live[j, 3:]), np.linalg.norm(bw[j, :3] - live[j, :3]),
              np.linalg.norm(bw[j, 3:] - live[j, 3:])))
    print('forward end - live end:', fw[-1] - live[-1])
    print('backward start - theta start:', bw[0] - m['theta'][6:])
    return fw, bw


def vx_shift(m, f, res=None, dvx=None):
    """
    Start change from integrating back (a) the live end with vx at its target and (b) the full target,
    each relative to integrating back the live end itself. Then each shifted start is checked forwards.
    """
    t, live = m['times'], m['live']
    tgt = vt.TARGET[6:]
    end_vx = live[-1].copy()
    end_vx[3] = tgt[3] if dvx is None else end_vx[3] + dvx
    out = {}
    back = lambda e: integrate(m, f, t[-1], t[0], e, res=res)[1][-1]
    b0 = back(live[-1])
    for name, e in (('vx only', end_vx), ('full target', tgt)):
        d = back(e) - b0
        new = m['theta'][6:] + d
        fw0 = integrate(m, f, t[0], t[-1], m['theta'][6:], res=res)[1][-1]
        fw1 = integrate(m, f, t[0], t[-1], new, res=res)[1][-1]
        print('%-12s end change %s -> start change %s kpc, km/s' % (name, e - live[-1], d))
        print('             model forwards from shifted start: end change %s' % (fw1 - fw0))
        out[name] = d
    return out


def other_runs(m, f, res=None, d=D):
    """
    Model endpoint change (run i - base, from each run's start with the base run's friction/residual
    history) vs the live change from round.json, for runs 1-12. Returns (model, live), each (12, 6).
    """
    info = vs.json.load(open(d + '/round.json'))
    runs, misses = np.array(info['runs']), np.array(info['misses'])
    t = m['times']
    e0 = integrate(m, f, t[0], t[-1], runs[0][6:], res=res)[1][-1]
    mod = np.array([integrate(m, f, t[0], t[-1], runs[i][6:], res=res)[1][-1] - e0 for i in range(1, 13)])
    live = misses[1:, 6:] - misses[0, 6:]
    print('run   start change (kpc, km/s)            | live end change             | model - live')
    for i in range(12):
        print('%2d  %s | %s | %s' % (i + 1, runs[i + 1][6:] - runs[0][6:], live[i], mod[i] - live[i]))
    return mod, live


def other_tracks(m, f, res, runs, thetas):
    """
    Whole-orbit test: for each run (dict with times, mw, sgr centre tracks) started from its theta,
    the model's predicted change from the base vs the live change, at every snapshot. Prints the
    end changes and the largest model - live error along the orbit. Returns list of (model, live).
    """
    t = m['times']
    base_mod = integrate(m, f, t[0], t[-1], m['theta'][6:], t_eval=t, res=res)[1]
    out = []
    print('run  start change (kpc, km/s)              | live end change              | model - live at end         | max model-live: kpc km/s')
    for run, th in zip(runs, thetas):
        live = run['sgr'] - run['mw']
        mod = integrate(m, f, t[0], t[-1], np.asarray(th)[6:], t_eval=t, res=res)[1]
        dl, dm = live - m['live'], mod - base_mod
        err = dm - dl
        print('     %s | %s | %s | %.3f %.2f' % (np.asarray(th)[6:] - m['theta'][6:], dl[-1], err[-1],
              np.linalg.norm(err[:, :3], axis=1).max(), np.linalg.norm(err[:, 3:], axis=1).max()))
        out.append((dm, dl))
    return out
