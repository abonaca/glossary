"""Round-2 candidate starts, step 1 of 2 (writes round01/round2_candidates.pkl): evolving-field lnL scan on the round-1
base track; LMC damped (lambda=0.3) LMC-block step from round 1 (theta_lmc); Sgr delay guess, evolving-field and rigid
restricted solves. Step 2 is solve_ev_sigma.py, which adds the halo sigma(r) evolving-field solve (ev_sigma) used as
the round-2 start: v21_tune.new_round(2, c['ev_sigma'], note=...)."""
import sys, pickle, numpy as np
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import v21_surrogate as vs, v21_tune as vt
from scipy.optimize import least_squares
np.set_printoptions(precision=3, suppress=True, linewidth=200)
O = '/home/ana/projects/glossary/output/v21_sim/tune/'


def candidates(out=O + 'round01/round2_candidates.pkl'):
    tr = vs.load_tracks(1); info = tr['info']; base = tr['runs'][0]; th1 = np.array(info['theta'])
    ev = vs.EvolvingField(O + 'round01/expansions', base)
    ts = base['times']; live = vs._live_rel(base)[:, 6:]; W = 1 / vt.TOL[6:]
    best = None
    for lnL in np.arange(1.7, 2.61, 0.1):
        a = vs.sgr_evolving(ev, tr, base, th1[6:], ts, lnL=lnL); d = np.linalg.norm(a[:, :3] - live[:, :3], axis=1)
        rms = np.sqrt(np.mean(d**2))
        print('  evolving lnL %.1f: rms %.2f, t=0 |dx| %.2f |dv| %.1f' % (lnL, rms, d[-1], np.linalg.norm(a[-1, 3:] - live[-1, 3:])))
        if best is None or rms < best[1]: best = (lnL, rms)
    lnL_ev = best[0]; print('evolving lnL =', lnL_ev)
    # LMC: damped (lam=0.3) LMC-block step from round 1
    J = np.array(info['jacobian']); r = np.array(info['residual']); Wf = 1 / vt.TOL; D = np.array(info['delta'])
    A = Wf[:6, None] * J[:6, :6]; M = np.vstack((A, 0.3 * np.diag(1 / D[:6])))
    dl = np.linalg.lstsq(M, np.concatenate((-Wf[:6] * r[:6], np.zeros(6))), rcond=None)[0]
    th_lmc = th1[:6] + dl
    guess = vs.delayed_theta(th1, 0., 0.034, dict(lnL_sgr=3.), tr['sgr_m'][0])[6:]
    c = dict(theta1=th1, theta_lmc=th_lmc, sgr_delay=guess, lnL_ev=lnL_ev)
    # evolving-field solve
    off_ev = vs.sgr_evolving(ev, tr, base, th1[6:], [0.], lnL=lnL_ev)[-1] - live[-1]
    c['ev'] = vs.evolving_start(ev, tr, th_lmc, guess, off_ev, lnL=lnL_ev)
    # rigid restricted solve (round 1 tracks, no delay)
    drv = vs.drivers(tr, base); ends = np.array([-3., 0.])
    off_r = vs.sgr_restricted(tr, th1[6:], drv, ends)[-1] - live[-1]
    f = lambda s: (vs.sgr_restricted(tr, s, drv, ends)[-1] - off_r - vt.TARGET[6:]) * W
    res = least_squares(f, guess, x_scale=np.array([0.1, 0.1, 0.1, 1, 1, 1.]), diff_step=1e-5)
    print('rigid solve: predicted resid/TOL', res.fun)
    c['rigid'] = np.concatenate((th_lmc, res.x))
    if out:
        pickle.dump(c, open(out, 'wb'))
    return c


if __name__ == '__main__':
    candidates()
