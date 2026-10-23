"""Round-3 start (theta3_submitted = theta2 + theta3 step rounded to 3 decimals, passed to v21_tune.new_round(3, ...)): lnL
recalibration on the round-2 base, offset-corrected evolving-field Sgr solve, damped (lambda=0.3) LMC-block step, iterated
with the live Jacobian cross-terms for NPASS passes. Overwrites round3_candidates.pkl (later keys theta3_pass2, sgr_far are
added by other scripts)."""
import sys, json, pickle, numpy as np
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import v21_surrogate as vs, v21_tune as vt
from scipy.optimize import least_squares, minimize_scalar
np.set_printoptions(precision=3, suppress=True, linewidth=200)
O = '/home/ana/projects/glossary/output/v21_sim/tune/'
vs.set_sigma_profile(tuple(np.load('/home/ana/projects/glossary/output/v21_sim/ics/mw_triax_full/halo_sigma.npy')))
tr = vs.load_tracks(2); info = tr['info']; base = tr['runs'][0]; th2 = np.array(info['theta'])
ev = vs.EvolvingField(O + 'round02/expansions', base)
ts = base['times']; live = vs._live_rel(base)[:, 6:]; W = 1 / vt.TOL[6:]
f = lambda s, lnL, t, run=base: vs.sgr_evolving(ev, tr, base, s, t, lnL=lnL)

# 1. recalibrate lnL on the round-2 base track
rms = lambda lnL: np.sqrt(np.mean(np.sum((f(th2[6:], lnL, ts)[:, :3] - live[:, :3])**2, axis=1)))
L = minimize_scalar(rms, bounds=(0.5, 5), method='bounded', options=dict(xatol=0.01)).x
a = f(th2[6:], L, ts); d = np.linalg.norm(a[:, :3] - live[:, :3], axis=1)
off = a[-1] - live[-1]
print('lnL %.2f  track rms %.2f kpc  t=0 model-live: %s' % (L, np.sqrt(np.mean(d**2)), off), flush=True)
print('  |dx| along track:', d[::3])
# 2. model Sgr Jacobian vs the live round-2 Jacobian
Jm, Jl = [], []
for i in range(7, 13):
    run, dl = tr['runs'][i], info['delta'][i - 1]
    Jm.append((f(vs.run_theta(info, i)[6:], L, ts[-1:], run)[-1] - a[-1]) / dl)
    Jl.append((vs._live_rel(run)[-1, 6:] - live[-1]) / dl)
Jm, Jl = np.array(Jm).T, np.array(Jl).T
print('  Sgr Jacobian column error:', np.linalg.norm(W[:, None] * (Jm - Jl), axis=0) / np.linalg.norm(W[:, None] * Jl, axis=0), flush=True)
# 3. Sgr: offset-corrected solve from the round-2 start
g = lambda s: (f(s, L, [vt.T_END])[-1] - off - vt.TARGET[6:]) * W
print('round-2 start: predicted resid/TOL', g(th2[6:]), '(live', (live[-1] - vt.TARGET[6:]) * W, ')')
res = least_squares(g, th2[6:], x_scale=np.array([0.1, 0.1, 0.1, 1, 1, 1.]), diff_step=1e-5)
s = res.x
print('Sgr solve: predicted resid/TOL', res.fun, '\n  start', s, '\n  minus round 2', s - th2[6:], flush=True)
# 4. LMC: damped GN step from the live round-2 Jacobian, including the Sgr change's effect on the LMC
J = np.array(info['jacobian']); r = np.array(info['residual'])
TL = vt.TOL[:6]; Jw = J[:6, :6] / TL[:, None]; rw = (r[:6] + J[:6, 6:] @ (s - th2[6:])) / TL
A = Jw.T @ Jw; lam = 0.3
dl = -np.linalg.solve(A + lam * np.diag(np.diag(A)), Jw.T @ rw)
print('LMC resid/TOL now', r[:6] / TL, '\n  with Sgr change', rw, '\n  damped step', dl,
      '\n  predicted after step', rw + Jw @ dl, flush=True)
th3a = np.concatenate((th2[:6] + dl, s))
print('round-3 candidate (Sgr->LMC cross-terms only)', th3a, flush=True)
# 5. both directions: the model keeps the LMC on its round-2 live track, so shift the Sgr target by the
#    live Jacobian's estimate of the LMC step's effect on Sgr, re-solve, recompute the LMC step; repeat
sb, db = s, dl
NPASS = 2       # round 3 started from pass 1 (2 passes); 3 passes give theta3_pass2 in the saved pickle
for it in range(NPASS):
    eff = J[6:, :6] @ db
    gb = lambda q: (f(q, L, [vt.T_END])[-1] - off + eff - vt.TARGET[6:]) * W
    rb = least_squares(gb, sb, x_scale=np.array([0.1, 0.1, 0.1, 1, 1, 1.]), diff_step=1e-5)
    sb = rb.x
    rw = (r[:6] + J[:6, 6:] @ (sb - th2[6:])) / TL
    db = -np.linalg.solve(A + lam * np.diag(np.diag(A)), Jw.T @ rw)
    print('pass %d: LMC effect on Sgr /TOL %s\n  Sgr predicted resid/TOL %s\n  Sgr minus round 2 %s\n  LMC step %s, predicted LMC resid/TOL %s'
          % (it, eff / vt.TOL[6:], rb.fun, sb - th2[6:], db, rw + Jw @ db), flush=True)
th3 = np.concatenate((th2[:6] + db, sb))
print('round-3 candidate (both directions)', th3, '\n  minus one-way candidate', th3 - th3a)
theta3_submitted = th2 + np.round(th3 - th2, 3)   # the round-3 start: step rounded to 3 decimals
c = dict(theta2=th2, theta3=th3, theta3_submitted=theta3_submitted, theta3_oneway=th3a, lnL=L, offset=off, sgr_fun=rb.fun, lmc_step=db,
         auto_gn=np.array(info['theta_next']))
pickle.dump(c, open(O + 'round02/round3_candidates.pkl', 'wb'))
th0 = np.array(json.load(open(O + 'round00/round.json'))['theta'])
th1 = np.array(json.load(open(O + 'round01/round.json'))['theta'])
starts = {'round 0 start': th0, 'round 1 start': th1, 'round 2 start': th2, 'round 3 candidate': th3}
vs.plot_orbits(ev, tr, starts, sb, '/home/ana/projects/glossary/plots/tune_r3_orbits.png', lnL=L)
