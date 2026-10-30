"""Round 21: Levenberg-Marquardt steps from round 20's f = 0.40 run (run 7) with the Jacobian J20 from round 20's
nudge runs 19-30 (one start component each, by round 20's NUDGE), at damping MU = 0.3, 1, 3 (3 runs), LMC and Sgr
moved together, Sgr rescaled by 1.03.

Step: minimise |W (m7 + J20 dtheta)|^2 + mu |dtheta / NUDGE|^2, W = 1 / tolerance, m7 = run 7's t=0 misses, i.e. the
damping measures each start change against its own nudge size. Run 7 is out of tolerance only in LMC vy (+1.14
km/s), Sgr z (+0.101 kpc) and Sgr vy (+1.19 km/s). Along the round 18 -> 19 line J20 overpredicts the Sgr responses
1.6-3x, hence mu = 0.3 (larger step) besides 1; mu = 3 in case J20 is too small in the direction of the step. Every
run is scored on its own (v21_tune kind 'sgr_offsets'); offsets are the 12-component start changes from run 7.

    setup()     # on OBS: TUNE/round21/round.json
    check()     # on OBS: round.json reproduces
    steps()     # locally or on OBS: steps and linear forecasts (physical units)
Submit:  sbatch --array=0-2 ~/projects/glossary/scripts/obs/tune_round.sbatch 21
"""
import sys
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
sys.path.insert(0, '/home/abonaca/projects/glossary/scripts')
import json, os
import numpy as np

K = 21
MU = (0.3, 1., 3.)
NOTE = ('Levenberg-Marquardt steps from round 20 run 7 (f=0.40) with J20 (round 20 runs 19-30), damping mu = %s '
        'in nudge units (r21_lm.py)' % (MU,))


def _r20():
    """Round 20's round.json: on OBS from the tune directory, locally from output/v21_sim/tune."""
    import v21_tune as vt, sgr_timing as st
    d = vt.round_dir(20) if os.path.isdir(vt.round_dir(20)) else os.path.join(st.T, 'round20')
    return json.load(open(os.path.join(d, 'round.json')))


def lm_steps(r20, mu=MU):
    """Start changes from round 20's base run (12 components, kpc and km/s) at each damping, the Jacobian and the
    base run's misses."""
    import v21_tune as vt
    F = np.array(r20['endpoints'])
    b0, nu = r20['nudge_base'], np.array(r20['nudge'])
    n = len(r20['fracs'])
    J = (F[n:n + 12] - F[b0]).T / nu
    m0 = F[b0] - vt.TARGET
    A = (J / vt.TOL[:, None]) * nu
    b = -m0 / vt.TOL
    dth = [np.linalg.solve(A.T @ A + m * np.eye(12), A.T @ b) * nu for m in mu]
    return np.array(dth), J, m0


def _info():
    import v21_tune as vt
    r20 = _r20()
    dth, _, _ = lm_steps(r20)
    th0 = np.array(r20['runs'][r20['nudge_base']])
    return dict(round=K, kind='sgr_offsets', names=vt.NAMES, theta=th0.tolist(), offsets=dth.tolist(),
                labels=['LM mu=%g' % m for m in MU], runs=[(th0 + o).tolist() for o in dth], mu=list(MU),
                target=vt.TARGET.tolist(), tol=vt.TOL.tolist(), note=NOTE, base_ics=r20['base_ics'],
                base_theta=r20['base_theta'], sgr_rscale=[r20['sgr_rscale'][0]] * len(MU))


def setup():
    import v21_tune as vt
    d = vt.round_dir(K)
    if os.path.exists(os.path.join(d, 'round.json')):
        raise RuntimeError('round %d exists; move it to scrap first' % K)
    os.makedirs(d, exist_ok=True)
    info = _info()
    json.dump(info, open(os.path.join(d, 'round.json'), 'w'), indent=1)
    print('round %d: %d LM steps from round 20 run %d, mu = %s, Sgr rescaled by %g'
          % (K, len(MU), _r20()['nudge_base'], MU, info['sgr_rscale'][0]))
    return info


def check():
    import v21_tune as vt
    d = json.load(open(os.path.join(vt.round_dir(K), 'round.json')))
    ref = _info()
    ok = all(d[key] == ref[key] for key in ref)
    print('round %d reproduces: %s; sgr_rscale: %s; theta == round 20 run %d: %s'
          % (K, ok, set(d['sgr_rscale']), _r20()['nudge_base'], d['theta'] == _r20()['runs'][_r20()['nudge_base']]))
    steps(d=d)


def steps(d=None):
    """Print each run's start change from round 20 run 7 and the linear forecast of its misses (J20)."""
    import sgr_timing as st
    mu = d['mu'] if d is not None else MU
    dth, J, m0 = lm_steps(_r20(), mu)
    for m, s in zip(mu, dth):
        p = m0 + J @ s
        print('mu %-4g start change %s' % (m, np.array2string(s, precision=3, suppress_small=True)))
        print('        forecast miss %s, arrival %+.2f Myr early' % (np.array2string(p, precision=2, suppress_small=True),
                                                                    st.arrival(p)))


if __name__ == '__main__':
    setup() if len(sys.argv) > 1 and sys.argv[1] == 'setup' else check()
