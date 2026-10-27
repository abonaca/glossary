"""Round 14: Jacobian round (base + 12 nudges) around round 13's base plus 0.25 of round 13's block Gauss-Newton step.

Round 13 (0.1 x the J12 step) left LMC y +0.25 kpc, LMC vz -1.04 km/s, Sgr vx -1.69 km/s and Sgr vy +1.43 km/s
outside the tolerances. The full J13 step leans on large, cancelling LMC <-> Sgr cross-terms (LMC z -3.28 kpc), which
have changed by 75-270% between rounds, so this round uses the block step (J13 LMC-LMC and Sgr-Sgr blocks only,
v21_tune.gauss_newton(block=True)), damped by 0.25. Built, like rounds 11-13, from
v21_sim/adopted/r09run02_prograde/ics.nemo (MW disc rotating as in the MW), with the same nudges. The round 14 that
finish_round(13) set up with the full J13 step is in SCRAP/round14_fullstep_J13_from_r13damp0.1.

    setup()     # on OBS: TUNE/round14/round.json
    check()     # on OBS: round.json reproduces
Submit:  sbatch --array=0-12 ~/projects/glossary/scripts/obs/tune_round.sbatch 14
"""
import sys
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
sys.path.insert(0, '/home/abonaca/projects/glossary/scripts')
import json, os
import numpy as np

K = 14
DAMP = 0.25
NOTE = 'Round 13 base + %g x round 13 block Gauss-Newton step (J13 LMC-LMC and Sgr-Sgr blocks)' % DAMP


def theta():
    import v21_tune as vt
    prev = json.load(open(os.path.join(vt.round_dir(K - 1), 'round.json')))
    F = np.array(prev['endpoints'])
    step, _ = vt.gauss_newton(F[0], F[1:], prev['delta'], block=True)
    return np.array(prev['theta']) + DAMP * step, prev


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
    print('theta == r%d + %g block step: %s; delta == r%d: %s; base_ics == r%d: %s; base_theta == r%d: %s'
          % (K - 1, DAMP, np.array_equal(d['theta'], th), K - 1, d['delta'] == prev['delta'], K - 1,
             d['base_ics'] == prev['base_ics'], K - 1, d['base_theta'] == prev['base_theta']))
    print('step:', np.array2string(th - np.array(prev['theta']), precision=3))


if __name__ == '__main__':
    setup() if len(sys.argv) > 1 and sys.argv[1] == 'setup' else check()
