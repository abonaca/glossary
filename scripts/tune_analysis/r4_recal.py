import sys, json, pickle, time, numpy as np
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import v21_surrogate as vs, v21_tune as vt
from scipy.optimize import least_squares
np.set_printoptions(precision=2, suppress=True, linewidth=200)
O = '/home/ana/projects/glossary/output/v21_sim/tune/'
W = 1 / vt.TOL; B = slice(0, 6)
tr4 = vs.load_tracks(4)
m2, m4 = np.load(O + 'round02/sgr_mass.npz'), tr4
print('Sgr mass (1e8 Msun) r2:', np.interp([-3, -2, -1, -0.5, 0], m2['times'], m2['mass']) * 2.325e5 / 1e8,
      '\n                     r4:', np.interp([-3, -2, -1, -0.5, 0], tr4['sgr_t'], tr4['sgr_m']) * 2.325e5 / 1e8)
I = {k: json.load(open(O + 'round%02d/round.json' % k)) for k in (3, 4)}
th = {k: np.array(I[k]['theta']) for k in I}; r = {k: np.array(I[k]['residual']) for k in I}; J4 = np.array(I[4]['jacobian'])
live = {k: r[k] + vt.TARGET for k in I}
p_old = pickle.load(open(O + 'round00/surrogate_calib.pkl', 'rb'))['p']
out = {}
for lab, sig in (('fixed sigma', None), ('sigma(r)', tuple(np.load('/home/ana/projects/glossary/output/v21_sim/ics/mw_triax_full/halo_sigma.npy')))):
    vs.set_sigma_profile(sig)
    t = time.time(); p = vs.calibrate(tr4, p0=(p_old['lnL_lmc'], p_old['lnL_sgr'])); print('  (%s, %.0f s)' % (lab, time.time() - t))
    out[lab] = p
cases = [('old: r0 lnL, r2 Sgr mass, fixed sigma', p_old, m2['times'], m2['mass'], None),
         ('new: r4 lnL, r4 Sgr mass, fixed sigma', out['fixed sigma'], tr4['sgr_t'], tr4['sgr_m'], None),
         ('new: r4 lnL, r4 Sgr mass, sigma(r)', out['sigma(r)'], tr4['sgr_t'], tr4['sgr_m'],
          tuple(np.load('/home/ana/projects/glossary/output/v21_sim/ics/mw_triax_full/halo_sigma.npy')))]
res = {}
for lab, p, st, smm, sig in cases:
    vs.set_sigma_profile(sig)
    E = lambda q: vs.endpoint(q, p, st, smm)
    F3, F4 = E(th[3]), E(th[4])
    F4b, Jr = vs.jacobian(th[4], p, st, smm, delta=np.array(I[4]['delta']))
    print('\n== %s  (lnL_lmc %.2f, lnL_sgr %.2f)' % (lab, p['lnL_lmc'], p['lnL_sgr']))
    print('  rigid - live at r4 start /TOL', (F4 - live[4]) * W)
    print('  r3->r4 change  rigid', (F4 - F3) * W, '\n                 live ', (live[4] - live[3]) * W)
    rd = np.linalg.norm(W[:, None] * (Jr - J4), axis=0) / np.linalg.norm(W[:, None] * J4, axis=0)
    print('  Jacobian rel diff vs J4 per column', rd)
    print('  LMC vy row: J4/rigid for y0, vy0, vz0', (J4 / Jr)[4, [1, 4, 5]])
    full = lambda d: W * (E(th[4] + np.concatenate((d, np.zeros(6)))) - F4)
    g = lambda d: r[4][B] * W[B] + full(d)[B]
    x0 = -np.linalg.solve(W[B, None] * J4[B, B], r[4][B] * W[B])
    s = least_squares(g, x0, diff_step=1e-3, x_scale=np.array([1, 1, 1, .3, .3, .3]), xtol=1e-10, ftol=1e-10, max_nfev=300)
    print('  LMC solve from r4: step', s.x, 'miss', s.fun, '\n     J4-linear miss for that step', r[4][B] * W[B] + W[B] * (J4[B, B] @ s.x), '\n     effect on Sgr', full(s.x)[6:])
    res[lab] = dict(p=p, F4=F4, J=Jr, step=s.x)
pickle.dump(dict(calib=out, res=res), open(O + 'round04/rigid_recal.pkl', 'wb'))
