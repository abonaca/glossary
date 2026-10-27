"""Round 15: Jacobian round (base + 12 nudges) around round 14's base plus a full-J14 Levenberg-Marquardt step,
lambda = 3e-3 (r14_lm.lm: (A^T A + lam diag(A^T A)) u = -A^T b in tolerance-per-nudge units).

Round 14 (0.25 x the J13 block step) left LMC y +0.23 kpc, LMC vz -1.33 km/s and Sgr vx -1.58 km/s outside the
tolerances. The undamped J14 step needs LMC z -5.0 kpc, a poorly determined direction; LM with lambda = 3e-3 keeps
every component within about its nudge (LMC z -0.52 kpc) and forecasts every miss within tolerance. Built, like rounds
11-14, from v21_sim/adopted/r09run02_prograde/ics.nemo (MW disc rotating as in the MW), with the same nudges. The
round 15 that finish_round(14) set up with the full J14 step is in SCRAP/round15_fullstep_J14.

    setup()     # on OBS: TUNE/round15/round.json
    check()     # on OBS: round.json reproduces
Submit:  sbatch --array=0-12 ~/projects/glossary/scripts/obs/tune_round.sbatch 15
"""
import sys
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
sys.path.insert(0, '/home/abonaca/projects/glossary/scripts')
import json, os
import numpy as np

K = 15
LAM = 3e-3
NOTE = 'Round 14 base + full-J14 Levenberg-Marquardt step, lambda = %g (r14_lm.lm)' % LAM


def theta():
    import v21_tune as vt
    from r14_lm import lm
    prev = json.load(open(os.path.join(vt.round_dir(K - 1), 'round.json')))
    F, delta = np.array(prev['endpoints']), np.array(prev['delta'])
    J = (F[1:] - F[0]).T / delta[None, :]
    return np.array(prev['theta']) + lm(J, F[0] - vt.TARGET, delta, vt.TOL, LAM), prev


def setup():
    import v21_tune as vt
    if os.path.exists(os.path.join(vt.round_dir(K), 'round.json')):
        raise RuntimeError('round %d exists; move it to scrap first' % K)
    th, prev = theta()
    return vt.new_round(K, th, prev['delta'], note=NOTE, base_ics=prev['base_ics'], base_theta=prev['base_theta'])


def check():
    import v21_tune as vt
    d = json.load(open(os.path.join(vt.round_dir(K), 'round.json')))
    th, prev = theta()
    print('theta == r%d + LM(%g) step: %s; delta == r%d: %s; base_ics == r%d: %s; base_theta == r%d: %s'
          % (K - 1, LAM, np.array_equal(d['theta'], th), K - 1, d['delta'] == prev['delta'], K - 1,
             d['base_ics'] == prev['base_ics'], K - 1, d['base_theta'] == prev['base_theta']))
    print('step:', np.array2string(th - np.array(prev['theta']), precision=3))


if __name__ == '__main__':
    setup() if len(sys.argv) > 1 and sys.argv[1] == 'setup' else check()
