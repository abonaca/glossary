"""
How precisely are the present-day MW, LMC and Sgr centres measured in the tuning runs (v21_tune)?

For the final (t=0) snapshot of every run of a round, re-measure the centres with several aperture
sizes (the velocity aperture of the MW and LMC shrinking-sphere centres, the radius of the Sgr median
centre), and bootstrap the particles of the base run to get the error of each estimator. The
round's Jacobian can then be recomputed for each aperture and compared with the bootstrap error.

Usage (on OBS, ~1-2 h for 13 runs; obs/centre_check.sbatch ROUND):
    import v21_centre_check as cc
    res = cc.measure_round(8)          # -> TUNE/round08/centre_check.npz

The MW velocity is also measured from the MW dark matter (R_VEL_MWDM) around the stellar MW centre, as a check of
the star-based MW velocity.
"""
import os
import time
import numpy as np
import v21_tune as vt
import r0_noise as rn

R_VEL_MW = (1., 2., 3., 5., 8.)      # kpc, aperture of the mean star velocity around the MW centre (v21_tune: 3)
R_VEL_MWDM = (3., 5., 10., 20.)       # kpc, same with the MW dark matter around the (stellar) MW centre
R_VEL_LMC = (2., 3., 5., 8.)          # kpc, same for the LMC (v21_tune: 5)
R_SGR = (2., 3., 5., 7.)              # kpc, radius of the Sgr median centre (V21 and v21_tune use 5)


def final_snapshot(k, i):
    """Path of the evolved snapshot file of run i of round k (before 822b296, perturbed runs were moved to SCRAP)."""
    f = os.path.join(vt.run_dir(k, i), 'evol.nemo')
    if os.path.exists(f):
        return f
    return os.path.join(vt.SCRAP, 'round%02d' % k, 'run%02d_evol.nemo' % i)


def _shrink(x, m, guess, r_start, r_stop=1., shrink=0.8):
    c, r = np.asarray(guess, float), r_start
    while r > r_stop:
        sel = np.sum((x - c)**2, axis=1) < r**2
        if sel.sum() < 50:
            break
        c = np.average(x[sel], axis=0, weights=m[sel])
        r *= shrink
    return c


def _mean_vel(xv, m, c, r):
    sel = np.sum((xv[:, :3] - c)**2, axis=1) < r**2
    return np.average(xv[sel, 3:6], axis=0, weights=m[sel]), int(sel.sum())


def _median(x, v, guess, radius, niter=20):
    c = np.asarray(guess, float)
    for _ in range(niter):
        sel = np.sum((x - c)**2, axis=1) < radius**2
        cnew = np.median(x[sel], axis=0)
        done = np.linalg.norm(cnew - c) < 1e-4
        c = cnew
        if done:
            break
    sel = np.sum((x - c)**2, axis=1) < radius**2
    return c, np.median(v[sel], axis=0), int(sel.sum())


def centres(xv, m, comps, guess):
    """
    MW, LMC and Sgr centres of one snapshot for every aperture. guess = the run's v21_tune centres at
    this time (dict mw, lmc, sgr of 6-vectors). Returns dict of arrays (n_aperture, 6) and counts.
    """
    nst = vt.vi.MW_COUNTS['nbody_stars']
    a = comps['mw'][0]
    xm, mm = xv[a:a + nst], m[a:a + nst]
    cm = _shrink(xm[:, :3], mm, guess['mw'][:3], 30.)
    out = dict(mw=[], mwdm=[], lmc=[], sgr=[], n_mw=[], n_mwdm=[], n_lmc=[], n_sgr=[])
    for r in R_VEL_MW:
        v, n = _mean_vel(xm, mm, cm, r)
        out['mw'].append(np.hstack((cm, v)))
        out['n_mw'].append(n)
    xd, md = xv[a + nst:comps['mw'][1]], m[a + nst:comps['mw'][1]]
    for r in R_VEL_MWDM:
        v, n = _mean_vel(xd, md, cm, r)
        out['mwdm'].append(np.hstack((cm, v)))
        out['n_mwdm'].append(n)
    a, b = comps['lmc']
    xl, ml = xv[a:b], m[a:b]
    cl = _shrink(xl[:, :3], ml, guess['lmc'][:3], 20.)
    for r in R_VEL_LMC:
        v, n = _mean_vel(xl, ml, cl, r)
        out['lmc'].append(np.hstack((cl, v)))
        out['n_lmc'].append(n)
    a = comps['sgr'][0]
    xs = xv[a:a + vt.NSGR_STARS]
    for r in R_SGR:
        c, v, n = _median(xs[:, :3], xs[:, 3:6], guess['sgr'][:3], r)
        out['sgr'].append(np.hstack((c, v)))
        out['n_sgr'].append(n)
    return {key: np.array(val) for key, val in out.items()}


def bootstrap(xv, m, comps, guess, nboot=100, seed=1):
    """
    Bootstrap the centres of one snapshot: resample (with replacement) the particles of each body
    within a generous radius of its centre and re-run the estimators. Returns dict of arrays
    (nboot, n_aperture, 6).
    """
    rng = np.random.default_rng(seed)
    nst = vt.vi.MW_COUNTS['nbody_stars']
    pick = {}
    for name, (a, b), rad in (('mw', (comps['mw'][0], comps['mw'][0] + nst), 40.),
                              ('lmc', tuple(comps['lmc']), 30.),
                              ('sgr', (comps['sgr'][0], comps['sgr'][0] + vt.NSGR_STARS), 15.)):
        idx = a + np.flatnonzero(np.sum((xv[a:b, :3] - guess[name][:3])**2, axis=1) < rad**2)
        pick[name] = idx
    out = dict(mw=[], lmc=[], sgr=[])
    for _ in range(nboot):
        sub = {name: rng.choice(idx, len(idx)) for name, idx in pick.items()}
        xm, mm = xv[sub['mw']], m[sub['mw']]
        cm = _shrink(xm[:, :3], mm, guess['mw'][:3], 30.)
        out['mw'].append([np.hstack((cm, _mean_vel(xm, mm, cm, r)[0])) for r in R_VEL_MW])
        xl, ml = xv[sub['lmc']], m[sub['lmc']]
        cl = _shrink(xl[:, :3], ml, guess['lmc'][:3], 20.)
        out['lmc'].append([np.hstack((cl, _mean_vel(xl, ml, cl, r)[0])) for r in R_VEL_LMC])
        xs = xv[sub['sgr']]
        out['sgr'].append([np.hstack(_median(xs[:, :3], xs[:, 3:6], guess['sgr'][:3], r)[:2]) for r in R_SGR])
    return {key: np.array(val) for key, val in out.items()}


def run_ids(k):
    """Runs of round k with tracked centres (centres.npz) and an evolved snapshot."""
    ids, i = [], 0
    while os.path.isdir(vt.run_dir(k, i)):
        if os.path.exists(os.path.join(vt.run_dir(k, i), 'centres.npz')) and os.path.exists(final_snapshot(k, i)):
            ids.append(i)
        i += 1
    return ids


def measure_round(k, nboot=100):
    """Centres of the final snapshot of every run of round k (run_ids), plus the bootstrap of the base run."""
    comps = vt.json.load(open(vt.BASE_ICS + '.json'))['components']
    res = {'run': np.array(run_ids(k))}
    for i in res['run']:
        t0 = time.time()
        tr = np.load(os.path.join(vt.run_dir(k, i), 'centres.npz'))
        guess = {name: tr[name][-1] for name in ('mw', 'lmc', 'sgr')}
        xv, m = rn.read_snapshot(final_snapshot(k, i), vt.T_END, tmpdir=vt.run_dir(k, i))
        c = centres(xv, m, comps, guess)
        for key, val in c.items():
            res.setdefault(key, []).append(val)
        res.setdefault('tune_endpoint', []).append(vt.endpoint(tr))
        if i == 0:
            b = bootstrap(xv, m, comps, guess, nboot=nboot)
            for key, val in b.items():
                res['boot_' + key] = val
        print('run %2d: %.0f s, n_lmc %s, n_sgr %s' % (i, time.time() - t0, c['n_lmc'], c['n_sgr']), flush=True)
        del xv, m
    res = {key: np.array(val) for key, val in res.items()}
    res.update(r_vel_mw=np.array(R_VEL_MW), r_vel_mwdm=np.array(R_VEL_MWDM), r_vel_lmc=np.array(R_VEL_LMC),
               r_sgr=np.array(R_SGR))
    out = os.path.join(vt.round_dir(k), 'centre_check.npz')
    np.savez(out, **res)
    print('saved', out)
    return res
