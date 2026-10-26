"""Round 13 (redo): Jacobian round (base + 12 nudges) around round 12's base plus 0.1 of round 12's Gauss-Newton step.

Round 12 (0.2 x the J11 step) brought Sgr to 0.35 Myr late along its orbit and every miss within ~2x the tolerance.
J12 (full 12x12, condition ~350 in tolerance/nudge units) is better behaved than J11 (~3000), but the full J11 step
had overshot Sgr's timing ~5x. The first round 13 took 0.5 of the J12 step (commit 983033f) and went backwards: Sgr
moved from 0.35 Myr late to 0.89 Myr early, against J12's prediction, so this redo takes 0.1 of the J12 step. Built,
like rounds 11-12, from v21_sim/adopted/r09run02_prograde/ics.nemo (MW disc rotating as in the MW), with the same
nudges. The round 13 that finish_round(12) set up with the full step is in SCRAP/round13_fullstep_J12, the 0.5 round
13 in SCRAP/round13_damp0.5 and the round 14 that finish_round(13) set up from it in SCRAP/round14_fullstep_J13.

    setup()     # on OBS: TUNE/round13/round.json
    check()     # on OBS: round.json reproduces
Submit:  sbatch --array=0-12 ~/projects/glossary/scripts/obs/tune_round.sbatch 13
"""
import sys
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
sys.path.insert(0, '/home/abonaca/projects/glossary/scripts')
import json, os
import numpy as np

K = 13
DAMP = 0.1
NOTE = 'Round 12 base + %g x round 12 Gauss-Newton step (J12, full matrix)' % DAMP


def theta():
    import v21_tune as vt
    prev = json.load(open(os.path.join(vt.round_dir(K - 1), 'round.json')))
    return np.array(prev['theta']) + DAMP * np.array(prev['dtheta']), prev


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
    print('theta == r%d + %g step: %s; delta == r%d: %s; base_ics == r%d: %s; base_theta == r%d: %s'
          % (K - 1, DAMP, np.array_equal(d['theta'], th), K - 1, d['delta'] == prev['delta'], K - 1,
             d['base_ics'] == prev['base_ics'], K - 1, d['base_theta'] == prev['base_theta']))
    print('step:', np.array2string(th - np.array(prev['theta']), precision=3))


if __name__ == '__main__':
    setup() if len(sys.argv) > 1 and sys.argv[1] == 'setup' else check()
