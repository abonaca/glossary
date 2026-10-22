"""Restricted Sgr model with the MW's Sgr-induced reflex removed from the live MW track and recomputed."""
import sys, numpy as np
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import v21_surrogate as vs, v21_tune as vt
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicHermiteSpline as H
from scipy.optimize import least_squares
np.set_printoptions(precision=2, suppress=True, linewidth=200)
tr = vs.load_tracks(0); info = tr['info']; base = tr['runs'][0]
th0 = np.array(info['theta']); A2 = vs.SGR_PLUMMER**2; LNL = 3.

def sgr_reflex(run):
    """MW reflex (position, velocity) induced by the live Sgr (Plummer of its bound mass), from rest at T_START."""
    rel = H(run['times'], (run['sgr'] - run['mw'])[:, :3], (run['sgr'] - run['mw'])[:, 3:])
    m = lambda t: np.interp(t, tr['sgr_t'], tr['sgr_m'])
    f = lambda t, w: np.hstack((w[3:], m(t) * rel(t) / (np.sum(rel(t)**2) + A2)**1.5))
    w = solve_ivp(f, (vt.T_START, vt.T_END), np.zeros(6), t_eval=run['times'], rtol=1e-9, atol=1e-9).y.T
    return w

def drivers(run, delay=0.):
    """MW (with the Sgr reflex removed) and LMC absolute tracks of a live run, delayed by `delay`."""
    t = run['times'] + delay
    mw = run['mw'] - sgr_reflex(run); lmc = run['lmc'].copy()
    if delay > 0:
        def two(tt, w):
            xm, vm, xl, vl = w[:3], w[3:6], w[6:9], w[9:]
            return np.concatenate((vm, vs._lmc.force(xm - xl), vl, vs._mw.force(xl - xm)))
        tb = np.linspace(vt.T_START + delay, vt.T_START, 9)
        w = solve_ivp(two, (tb[0], tb[-1]), np.concatenate((mw[0], lmc[0])), t_eval=tb, rtol=1e-10, atol=1e-10).y.T[::-1][:-1]
        t = np.concatenate((tb[::-1][:-1], t)); mw = np.vstack((w[:, :6], mw)); lmc = np.vstack((w[:, 6:], lmc))
        mw0 = mw[0].copy(); dt = t - t[0]
        for a in (mw, lmc):
            a[:, :3] -= mw0[:3] + mw0[3:] * dt[:, None]; a[:, 3:] -= mw0[3:]
    return [H(t, a[:, :3], a[:, 3:]) for a in (mw, lmc)]

def sgr_orbit(th, drv, t_eval, mdelay=0., lnL=LNL):
    """Sgr + its MW reflex R; MW centre = driver + R. Returns Sgr relative to the MW centre."""
    xb, xl = drv; vb = xb.derivative()
    m = lambda t: np.interp(t - mdelay, tr['sgr_t'], tr['sgr_m'])
    def rhs(t, w):
        x, v, R, VR = w[:3], w[3:6], w[6:9], w[9:]
        xm, vm = xb(t) + R, vb(t) + VR
        d = x - xm
        aR = m(t) * d / (np.sum(d**2) + A2)**1.5
        return np.hstack((v, vs._mw.force(d) + vs._lmc.force(x - xl(t)) + vs._friction(d, v - vm, m(t), lnL), VR, aR))
    w0 = np.hstack((th + np.hstack((xb(vt.T_START), vb(vt.T_START))), np.zeros(6)))
    w = solve_ivp(rhs, (vt.T_START, t_eval[-1]), w0, t_eval=t_eval, rtol=1e-9, atol=1e-8, method='DOP853').y.T
    return w[:, :6] - np.hstack((xb(t_eval), vb(t_eval))) - w[:, 6:]

if __name__ == '__main__':
    live = base['sgr'] - base['mw']; ts = base['times']
    R0 = sgr_reflex(base)
    print('live Sgr-induced MW reflex: max |dx| %.2f kpc, max |dv| %.2f km/s' % (np.abs(R0[:, :3]).max(), np.abs(R0[:, 3:]).max()))
    a = sgr_orbit(th0[6:], drivers(base), ts)
    print('  t   |dx| |dv|  (model - live, round 0)')
    for j in range(0, 25, 3): print('%6.3f %5.2f %6.2f' % (ts[j], *[np.linalg.norm(a[j, s:s+3] - live[j, s:s+3]) for s in (0, 3)]))
    # Sgr block of the Jacobian
    Jr, Jl = [], []
    for i in range(7, 13):
        run = tr['runs'][i]; dl = info['delta'][i - 1]
        Jr.append((sgr_orbit(vs.run_theta(info, i)[6:], drivers(run), ts)[-1] - a[-1]) / dl)
        Jl.append(((run['sgr'] - run['mw'])[-1] - live[-1]) / dl)
    Jr, Jl = np.array(Jr).T, np.array(Jl).T; W = 1 / vt.TOL[6:]
    print('Sgr Jacobian column error:', np.linalg.norm(W[:, None] * (Jr - Jl), axis=0) / np.linalg.norm(W[:, None] * Jl, axis=0))
    # delays
    d = vs.best_delay(tr); DL, DS = -d['lmc'][0], -d['sgr'][0]
    th_ind = vs.delayed_theta(th0, DL, DS, dict(lnL_sgr=LNL), tr['sgr_m'][0])
    drv = drivers(base, delay=DL)
    b = sgr_orbit(th_ind[6:], drv, ts, mdelay=DS)
    print('  t   r(LMC delayed %.3f, Sgr delayed %.3f)  r(round0 model at t-%.3f)' % (DL, DS, DS))
    ra = H(ts, a[:, :3], a[:, 3:])
    for j in range(0, 25, 3): print('%6.3f %7.1f %7.1f' % (ts[j], np.linalg.norm(b[j, :3]), np.linalg.norm(ra(ts[j] - DS)) if ts[j] - DS >= -3 else np.nan))
    offset = a[-1] - live[-1]; TS = vt.TARGET[6:]
    f = lambda th: sgr_orbit(th, drv, np.array([vt.T_START, vt.T_END]), mdelay=DS)[-1]
    print('independent-delay Sgr start, predicted live (offset-corrected) resid/TOL:', (f(th_ind[6:]) - offset - TS) * W)
    r = least_squares(lambda th: (f(th) - offset - TS) * W, th_ind[6:], x_scale=np.array([1, 1, 1, 10, 10, 10.]), diff_step=1e-4)
    print('solved: max|resid|/TOL %.2f\n Sgr theta %s\n shift vs independent delay %s' % (np.max(np.abs(r.fun)), r.x, r.x - th_ind[6:]))
    np.save(sys.argv[1] if len(sys.argv) > 1 else '/dev/null', np.concatenate((th_ind[:6], r.x)))
