"""Round 16: Sgr-only round (LMC fixed at round 15's start): base = round 15 + a Levenberg-Marquardt step on the Sgr
block of J15, lambda = 1e-2 (r14_lm.lm), plus 6 Sgr nudge runs. 7 runs.

Round 15 (full-J14 LM, lambda 3e-3) brought the LMC within tolerance but left Sgr 1.79 Myr early along its orbit
(Sgr +0.47, -0.09, +0.35 kpc; -6.79, -0.87, +2.38 km/s). lambda = 1e-2 keeps every Sgr component within its nudge
and needs the least cancellation between components in Sgr's predicted timing change; J15 forecasts every miss within
tolerance except Sgr vz (+1.14 km/s). Built, like rounds 11-15, from v21_sim/adopted/r09run02_prograde/ics.nemo
(MW disc rotating as in the MW). Sgr nudges as rounds 7-15. round.json has kind 'sgr' (v21_tune._finish_sgr_round:
Sgr-block Jacobian from runs 1-6, no next round) with no timing scan; v21_tune.new_sgr_round needs 13 runs, so the
round file is written here. The round 16 that finish_round(15) set up with the full J15 step is in
SCRAP/round16_fullstep_J15.

    setup()     # on OBS: TUNE/round16/round.json
    check()     # on OBS: round.json reproduces
Submit:  sbatch --array=0-6 ~/projects/glossary/scripts/obs/tune_round.sbatch 16
"""
import sys
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
sys.path.insert(0, '/home/abonaca/projects/glossary/scripts')
import json, os
import numpy as np

K = 16
LAM = 1e-2
S = slice(6, 12)
NOTE = ('Sgr only, LMC fixed at round 15: round 15 base + Levenberg-Marquardt step on the J15 Sgr block, lambda = %g '
        '(r14_lm.lm), and 6 Sgr nudge runs' % LAM)


def runs():
    import v21_tune as vt
    from r14_lm import lm
    prev = json.load(open(os.path.join(vt.round_dir(K - 1), 'round.json')))
    F, delta = np.array(prev['endpoints']), np.array(prev['delta'])
    J = (F[1:] - F[0]).T / delta[None, :]
    theta = np.array(prev['theta'])
    theta[S] += lm(J[S, S], (F[0] - vt.TARGET)[S], delta[S], vt.TOL[S], LAM)
    out = [theta.copy()]
    for j in range(6):
        th = theta.copy()
        th[6 + j] += delta[6 + j]
        out.append(th)
    return theta, delta[S], out, prev


def setup():
    import v21_tune as vt
    d = vt.round_dir(K)
    if os.path.exists(os.path.join(d, 'round.json')):
        raise RuntimeError('round %d exists; move it to scrap first' % K)
    theta, sgr_delta, rr, prev = runs()
    os.makedirs(d, exist_ok=True)
    info = dict(round=K, kind='sgr', names=vt.NAMES, theta=theta.tolist(), sgr_delta=sgr_delta.tolist(), scan_dt=[],
                runs=[r.tolist() for r in rr], target=vt.TARGET.tolist(), tol=vt.TOL.tolist(), note=NOTE,
                base_ics=prev['base_ics'], base_theta=prev['base_theta'])
    json.dump(info, open(os.path.join(d, 'round.json'), 'w'), indent=1)
    print('round %d (Sgr only): theta = %s' % (K, np.array2string(theta, precision=3)))
    return info


def check():
    import v21_tune as vt
    d = json.load(open(os.path.join(vt.round_dir(K), 'round.json')))
    theta, sgr_delta, rr, prev = runs()
    print('theta: %s; runs: %s; LMC == r%d: %s; sgr_delta: %s; base_ics == r%d: %s; base_theta == r%d: %s'
          % (np.array_equal(d['theta'], theta), np.array_equal(d['runs'], rr), K - 1,
             np.array_equal(theta[:6], prev['theta'][:6]), sgr_delta.tolist(), K - 1,
             d['base_ics'] == prev['base_ics'], K - 1, d['base_theta'] == prev['base_theta']))
    print('Sgr step:', np.array2string(theta[S] - np.array(prev['theta'])[S], precision=4))


if __name__ == '__main__':
    setup() if len(sys.argv) > 1 and sys.argv[1] == 'setup' else check()
