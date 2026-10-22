import sys, pickle, numpy as np
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import v21_surrogate as vs, v21_tune as vt
from scipy.optimize import minimize_scalar
np.set_printoptions(precision=2, suppress=True, linewidth=200)
tr = vs.load_tracks(1); info = tr['info']; base = tr['runs'][0]; th = np.array(info['theta'])[6:]
ev = vs.EvolvingField('/home/ana/projects/glossary/output/v21_sim/tune/round01/expansions', base)
ts = base['times']; live = vs._live_rel(base)[:, 6:]; W = 1 / vt.TOL[6:]
prof = tuple(np.load('/home/ana/projects/glossary/output/v21_sim/ics/mw_triax_full/halo_sigma.npy'))
models = dict(rigid=lambda s, lnL, t, run=base: vs.sgr_restricted(tr, s, vs.drivers(tr, run), t, lnL=lnL),
              evolving=lambda s, lnL, t, run=base: vs.sgr_evolving(ev, tr, base, s, t, lnL=lnL))
out = {}
for name, f in models.items():
    for sig in ('fixed 120', 'sigma(r)'):
        vs.set_sigma_profile(None if sig == 'fixed 120' else prof)
        def rms(lnL):
            d = np.linalg.norm(f(th, lnL, ts)[:, :3] - live[:, :3], axis=1)
            return np.sqrt(np.mean(d**2))
        res = minimize_scalar(rms, bounds=(0.5, 8), method='bounded', options=dict(xatol=0.01))
        lnL = res.x
        a = f(th, lnL, ts); d = np.linalg.norm(a[:, :3] - live[:, :3], axis=1)
        Jm, Jl = [], []
        for i in range(7, 13):
            run, dl = tr['runs'][i], info['delta'][i - 1]
            Jm.append((f(vs.run_theta(info, i)[6:], lnL, ts[-1:], run)[-1] - a[-1]) / dl)
            Jl.append((vs._live_rel(run)[-1, 6:] - live[-1]) / dl)
        Jm, Jl = np.array(Jm).T, np.array(Jl).T
        jerr = np.linalg.norm(W[:, None] * (Jm - Jl), axis=0) / np.linalg.norm(W[:, None] * Jl, axis=0)
        # sensitivity of the t=0 endpoint to lnL
        b = f(th, lnL + 0.1, ts[-1:])[-1]
        print('%-8s %-9s lnL %.2f  rms %.2f  max %.2f  t=0 |dx| %.2f |dv| %.1f  | dlnL=+0.1 -> %.2f kpc %.1f km/s' % (
            name, sig, lnL, d.std() * 0 + np.sqrt(np.mean(d**2)), d.max(), d[-1], np.linalg.norm(a[-1, 3:] - live[-1, 3:]),
            np.linalg.norm(b[:3] - a[-1, :3]), np.linalg.norm(b[3:] - a[-1, 3:])), flush=True)
        print('         |dx| along track:', d[::3])
        print('         Sgr Jacobian column error:', jerr, flush=True)
        out[(name, sig)] = dict(lnL=lnL, d=d, jerr=jerr, end=a[-1] - live[-1])
vs.set_sigma_profile(None)
pickle.dump(out, open('/home/ana/projects/glossary/output/v21_sim/tune/round01/sigma_test.pkl', 'wb'))
