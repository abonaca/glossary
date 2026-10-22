"""Sgr as a test body in rigid MW/LMC potentials moving along the live round-0 MW and LMC tracks."""
import sys, numpy as np
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import v21_surrogate as vs, v21_tune as vt
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicHermiteSpline as H
from scipy.optimize import least_squares
np.set_printoptions(precision=2, suppress=True, linewidth=200)
tr = vs.load_tracks(0)
info = tr['info']; base = tr['runs'][0]

def drivers(run, delay=0.):
    """Absolute MW and LMC centre trajectories (splines) of a live run, delayed by `delay`. Before
    the run's start the bodies follow the rigid two-body orbit backwards."""
    t = run['times'] + delay
    mw, lmc = run['mw'], run['lmc']
    if delay > 0:
        def two(tt, w):
            xm, vm, xl, vl = w[:3], w[3:6], w[6:9], w[9:]
            return np.concatenate((vm, vs._lmc.force(xm - xl), vl, vs._mw.force(xl - xm)))
        tb = np.linspace(vt.T_START + delay, vt.T_START, 9)
        sol = solve_ivp(two, (tb[0], tb[-1]), np.concatenate((mw[0], lmc[0])), t_eval=tb, rtol=1e-10, atol=1e-10)
        w = sol.y.T[::-1][:-1]
        t = np.concatenate((tb[::-1][:-1], t)); mw = np.vstack((w[:, :6], mw)); lmc = np.vstack((w[:, 6:], lmc))
        # shift to the frame where the MW starts at rest at the origin
        mw0 = mw[0].copy(); dt = t - t[0]
        for a in (mw, lmc):
            a[:, :3] -= mw0[:3] + mw0[3:] * dt[:, None]; a[:, 3:] -= mw0[3:]
    return [H(t, a[:, :3], a[:, 3:]) for a in (mw, lmc)] + [t]

def sgr_orbit(th, drv, lnL, sgr_t, sgr_m, t_eval, delay=0.):
    xm, xl, td = drv; vm = xm.derivative()
    msgr = lambda t: np.interp(t - delay, sgr_t, sgr_m)
    def rhs(t, w):
        x, v = w[:3], w[3:]
        d = x - xm(t)
        return np.hstack((v, vs._mw.force(d) + vs._lmc.force(x - xl(t)) + vs._friction(d, v - vm(t), msgr(t), lnL)))
    w0 = th + np.hstack((xm(vt.T_START), vm(vt.T_START)))
    w = solve_ivp(rhs, (vt.T_START, t_eval[-1]), w0, t_eval=t_eval, rtol=1e-9, atol=1e-8, method='DOP853').y.T
    return w - np.hstack((xm(t_eval), vm(t_eval)))

# 1) calibrate lnL on the base run, check the track
drv0 = drivers(base)
live = base['sgr'] - base['mw']
def res(q):
    rel = sgr_orbit(np.array(info['theta'][6:]), drv0, q[0], tr['sgr_t'], tr['sgr_m'], base['times'])
    return (rel[:, :3] - live[:, :3]).ravel()
q = least_squares(res, [3.], bounds=([0], [20])).x[0]
rel = sgr_orbit(np.array(info['theta'][6:]), drv0, q, tr['sgr_t'], tr['sgr_m'], base['times'])
print('lnL_sgr = %.2f' % q)
print('   t   |dx| |dv|   (restricted - live, base run)')
for j in range(0, 25, 2):
    print('%6.3f %5.2f %6.2f' % (base['times'][j], np.linalg.norm(rel[j, :3] - live[j, :3]), np.linalg.norm(rel[j, 3:] - live[j, 3:])))
# 2) Sgr block of the Jacobian (runs 7-12 perturb Sgr; each run's own MW/LMC tracks as drivers)
F0 = rel[-1]
Jr, Jl = [], []
for i in range(7, 13):
    run = tr['runs'][i]
    th = vs.run_theta(info, i)[6:]
    r = sgr_orbit(th, drivers(run), q, tr['sgr_t'], tr['sgr_m'], run['times'])
    Jr.append((r[-1] - F0) / info['delta'][i - 1])
    Jl.append(((run['sgr'] - run['mw'])[-1] - live[-1]) / info['delta'][i - 1])
Jr, Jl = np.array(Jr).T, np.array(Jl).T
W = 1 / vt.TOL[6:]
print('Sgr block column error |J_r - J_live| / |J_live| (TOL-weighted):',
      np.linalg.norm(W[:, None] * (Jr - Jl), axis=0) / np.linalg.norm(W[:, None] * Jl, axis=0))
print('endpoint (restricted - live)/TOL:', (F0 - live[-1]) / vt.TOL[6:])
np.save('/home/ana/projects/glossary/scripts/tune_analysis/lnL_sgr.npy', q)
