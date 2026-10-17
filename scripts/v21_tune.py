"""
Joint tuning of the LMC and Sgr initial conditions in the live MW+LMC+Sgr run (R2).

Unknowns: theta = (LMC x, y, z, vx, vy, vz, Sgr x, y, z, vx, vy, vz) at t=-3 in the frame where the
Milky Way starts at rest at the origin. Targets: present-day (t=0) LMC and Sgr positions/velocities
relative to the Milky Way centre (V21 Section 2 / 3.2). Each round runs the base theta plus one run
per unknown nudged by +DELTA (13 gyrfalcON runs), measures the endpoints from the snapshots and
takes a Gauss-Newton step with the finite-difference Jacobian.

Centres: Milky Way - shrinking sphere on its stars; LMC - shrinking sphere on its particles,
followed from snapshot to snapshot; Sgr - V21's definition, iterated median position/velocity of
Sgr star particles within 5 kpc.

Layout on OBS:  TUNE/roundNN/round.json         theta, deltas, (later) endpoints and next theta
                TUNE/roundNN/runKK/ics.nemo     run 0 = base, run k = theta + DELTA_k e_(k-1)
                TUNE/roundNN/runKK/evol.nemo    snapshots every 1/8 time unit, -3 ... 0

Usage:
    import v21_tune as vt
    vt.new_round(0)                    # round 0 from V21's initial conditions (login node is fine)
    # sbatch --array=0-12 tune_round.sbatch 0   -> each task: vt.build_ics(0, k) + gyrfalcON
    # sbatch tune_endpoints.sbatch 0            -> vt.finish_round(0): endpoints, step, round 1
"""
import os
import json
import shutil
import numpy as np
import agama
import v21_ics as vi
import r0_noise as rn

TUNE = '/carnegie/nobackup/users/abonaca/v21_sim/tune'
SCRAP = '/carnegie/nobackup/users/abonaca/v21_sim/scrap/tune'
BASE_ICS = '/carnegie/nobackup/users/abonaca/v21_sim/ics/runs_mw_triax_full/R2'   # .nemo and .json
T_START, T_END, SNAP_STEP = -3.0, 0.0, 0.125
NAMES = ['lmc_x', 'lmc_y', 'lmc_z', 'lmc_vx', 'lmc_vy', 'lmc_vz',
         'sgr_x', 'sgr_y', 'sgr_z', 'sgr_vx', 'sgr_vy', 'sgr_vz']
# present-day targets relative to the Milky Way centre (V21)
TARGET = np.array([-0.6, -41.3, -27.1, -63.9, -213.8, 206.6,
                   17.9, 2.6, -6.6, 239.5, -29.6, 213.5])
TOL = np.array([0.2] * 3 + [1.] * 3 + [0.1] * 3 + [1.] * 3)
DELTA = np.array([1.] * 3 + [1.] * 3 + [0.3] * 3 + [3.] * 3)
NSGR_STARS = vi.SGR_COUNTS['nstar']          # Sgr star particles come first within the Sgr block


def round_dir(k):
    return os.path.join(TUNE, 'round%02d' % k)


def run_dir(k, i):
    return os.path.join(round_dir(k), 'run%02d' % i)


def theta_v21():
    """V21's initial conditions: LMC from trajlmc.txt, Sgr from Sgr_snapshot/trajectory.txt (t=-3)."""
    return np.concatenate([np.concatenate(vi.initial_phase(c, T_START)) for c in ('lmc', 'sgr')])


def new_round(k, theta=None, delta=DELTA, note=''):
    """Write roundNN/round.json for the base theta (default: V21's) and the perturbation sizes."""
    theta = theta_v21() if theta is None else np.asarray(theta, float)
    d = round_dir(k)
    os.makedirs(d, exist_ok=True)
    info = dict(round=k, names=NAMES, theta=theta.tolist(), delta=np.asarray(delta, float).tolist(),
                target=TARGET.tolist(), tol=TOL.tolist(), note=note)
    json.dump(info, open(os.path.join(d, 'round.json'), 'w'), indent=1)
    print('round %d: theta = %s' % (k, np.array2string(theta, precision=3)))
    return info


def run_theta(k, i):
    """theta of run i in round k (0 = base, i>0 = base + delta along unknown i-1)."""
    info = json.load(open(os.path.join(round_dir(k), 'round.json')))
    theta = np.array(info['theta'])
    if i > 0:
        theta[i - 1] += info['delta'][i - 1]
    return theta


def build_ics(k, i):
    """
    Initial snapshot of run i of round k: R2's particles with the LMC and Sgr moved from V21's t=-3
    centres (where R2 placed them) to the run's theta.
    """
    comps = json.load(open(BASE_ICS + '.json'))['components']
    xv, m = agama.readSnapshot(BASE_ICS + '.nemo')
    shift = run_theta(k, i) - theta_v21()
    for j, c in enumerate(('lmc', 'sgr')):
        a, b = comps[c]
        xv[a:b] += shift[6 * j:6 * j + 6]
    d = run_dir(k, i)
    os.makedirs(d, exist_ok=True)
    tmp, f = os.path.join(d, 'ics_t0.nemo'), os.path.join(d, 'ics.nemo')
    for g in (tmp, f):
        if os.path.exists(g):
            os.remove(g)
    agama.writeSnapshot(tmp, (xv, m), 'nemo')
    vi._nemo('s2s %s %s time=%g' % (tmp, f, T_START))
    os.remove(tmp)
    print('round %d run %d: theta = %s' % (k, i, np.array2string(run_theta(k, i), precision=3)))
    return f


# ------------------------------------------------------------------------------------------ centres
def shrinking_centre(xv, m, guess, r_start=30., r_stop=1., shrink=0.8, r_vel=1.):
    """Shrinking-sphere centre starting at `guess`; velocity = mass-weighted mean within r_vel."""
    c, r = np.asarray(guess, float), r_start
    while r > r_stop:
        sel = np.sum((xv[:, :3] - c)**2, axis=1) < r**2
        if sel.sum() < 50:
            break
        c = np.average(xv[sel, :3], axis=0, weights=m[sel])
        r *= shrink
    sel = np.sum((xv[:, :3] - c)**2, axis=1) < r_vel**2
    return c, np.average(xv[sel, 3:6], axis=0, weights=m[sel])


def median_centre(xv, guess, radius=5., niter=20):
    """V21's Sgr centre: iterated median position and velocity of particles within `radius`."""
    c = np.asarray(guess, float)
    for _ in range(niter):
        sel = np.sum((xv[:, :3] - c)**2, axis=1) < radius**2
        cnew = np.median(xv[sel, :3], axis=0)
        if np.linalg.norm(cnew - c) < 1e-4:
            c = cnew
            break
        c = cnew
    sel = np.sum((xv[:, :3] - c)**2, axis=1) < radius**2
    return c, np.median(xv[sel, 3:6], axis=0), int(sel.sum())


def track_centres(evol, comps, theta):
    """
    Centres of the Milky Way (stars), LMC and Sgr in every snapshot of a run. Each search starts at
    the previous centre advanced by v dt. Returns dict of arrays (time, 6) plus Sgr bound-star counts.
    """
    times = rn.snapshot_times(evol)
    out = dict(times=times, mw=[], lmc=[], sgr=[], sgr_n=[])
    mw_g, lmc_g, sgr_g = np.zeros(6), np.array(theta[:6]), np.array(theta[6:])
    tprev = times[0]
    for t in times:
        xv, m = rn.read_snapshot(evol, t)
        dt = t - tprev
        a, b = comps['mw']
        c, v = shrinking_centre(xv[a:a + vi.MW_COUNTS['nbody_stars']], m[a:a + vi.MW_COUNTS['nbody_stars']],
                                mw_g[:3] + mw_g[3:] * dt)
        mw_g = np.hstack((c, v))
        a, b = comps['lmc']
        c, v = shrinking_centre(xv[a:b], m[a:b], lmc_g[:3] + lmc_g[3:] * dt, r_start=20.)
        lmc_g = np.hstack((c, v))
        a, b = comps['sgr']
        c, v, n = median_centre(xv[a:a + NSGR_STARS], sgr_g[:3] + sgr_g[3:] * dt)
        sgr_g = np.hstack((c, v))
        for key, val in (('mw', mw_g), ('lmc', lmc_g), ('sgr', sgr_g), ('sgr_n', n)):
            out[key].append(val)
        tprev = t
    for key in ('mw', 'lmc', 'sgr', 'sgr_n'):
        out[key] = np.array(out[key])
    return out


def endpoint(tr):
    """Present-day (last snapshot) LMC and Sgr phase-space coordinates relative to the Milky Way centre."""
    if abs(tr['times'][-1] - T_END) > 1e-6:
        raise ValueError('run did not reach t=%g (last snapshot %g)' % (T_END, tr['times'][-1]))
    return np.concatenate((tr['lmc'][-1] - tr['mw'][-1], tr['sgr'][-1] - tr['mw'][-1]))


# ------------------------------------------------------------------------------------ Gauss-Newton
def gauss_newton(F0, Fk, delta, block=False):
    """
    Step dtheta solving J dtheta = TARGET - F0, with J[:, j] = (Fk[j] - F0) / delta[j]; rows scaled by
    the tolerances. block=True keeps only the LMC-LMC and Sgr-Sgr blocks of J.
    """
    J = (np.asarray(Fk) - F0).T / np.asarray(delta)[None, :]
    if block:
        J[:6, 6:] = 0
        J[6:, :6] = 0
    W = 1 / TOL
    dtheta = np.linalg.solve(W[:, None] * J, W * (TARGET - F0))
    return dtheta, J


def finish_round(k, block=False, move_perturbed=True):
    """
    After all runs of round k: track centres, endpoints, Gauss-Newton step; write the results into
    round.json and set up round k+1 unless the base run is within tolerance. Snapshots of the
    perturbed runs go to SCRAP (the base run's are kept).
    """
    d = round_dir(k)
    info = json.load(open(os.path.join(d, 'round.json')))
    comps = json.load(open(BASE_ICS + '.json'))['components']
    F = []
    for i in range(13):
        evol = os.path.join(run_dir(k, i), 'evol.nemo')
        tr = track_centres(evol, comps, run_theta(k, i))
        np.savez(os.path.join(run_dir(k, i), 'centres.npz'), **tr)
        F.append(endpoint(tr))
        print('run %2d endpoint %s' % (i, np.array2string(F[-1], precision=2)), flush=True)
    F = np.array(F)
    resid = F[0] - TARGET
    ok = bool(np.all(np.abs(resid) < TOL))
    dtheta, J = gauss_newton(F[0], F[1:], info['delta'], block=block)
    info.update(endpoints=F.tolist(), residual=resid.tolist(), converged=ok, jacobian=J.tolist(),
                dtheta=dtheta.tolist(), theta_next=(np.array(info['theta']) + dtheta).tolist(), block=block)
    json.dump(info, open(os.path.join(d, 'round.json'), 'w'), indent=1)
    print('residual (base - target): %s' % np.array2string(resid, precision=2))
    print('in units of the tolerance: %s' % np.array2string(resid / TOL, precision=1))
    print('converged' if ok else 'step: %s' % np.array2string(dtheta, precision=3))
    if move_perturbed:
        for i in range(1, 13):
            src = os.path.join(run_dir(k, i), 'evol.nemo')
            dst = os.path.join(SCRAP, 'round%02d' % k, 'run%02d_evol.nemo' % i)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
    if not ok:
        new_round(k + 1, info['theta_next'], info['delta'], note='Gauss-Newton step from round %d' % k)
    return info
