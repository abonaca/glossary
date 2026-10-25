"""Round 12 (redo): Jacobian round (base + 12 nudges) around round 11's base plus 0.2 of round 11's Gauss-Newton step.

The first round 12 took the full round-11 step (J11, full 12x12, undamped) and overshot: Sgr went from 1.87 Myr early
to 7.85 Myr late along its orbit. Interpolating run 0's misses between the two puts Sgr's timing at zero near 0.19 of
the step, so this round takes 0.2 of it. Built, like round 11, from v21_sim/adopted/r09run02_prograde/ics.nemo (MW
disc rotating as in the MW), with round 11's nudges. The first round 12 and the round 13 set up from it are in
SCRAP/round12_fullstep and SCRAP/round13_fullstep.

    setup()     # on OBS: TUNE/round12/round.json
    check()     # on OBS: round.json reproduces
Submit:  sbatch --array=0-12 ~/projects/glossary/scripts/obs/tune_round.sbatch 12
"""
import sys
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
sys.path.insert(0, '/home/abonaca/projects/glossary/scripts')
import json, os
import numpy as np

K = 12
DAMP = 0.2
NOTE = ('Round 11 base + %g x round 11 Gauss-Newton step (J11, full matrix); redo of round 12 after the full step '
        'overshot Sgr by ~9.7 Myr along its orbit' % DAMP)


def theta():
    import v21_tune as vt
    r11 = json.load(open(os.path.join(vt.round_dir(11), 'round.json')))
    return np.array(r11['theta']) + DAMP * np.array(r11['dtheta']), r11


def setup():
    import v21_tune as vt
    if os.path.exists(os.path.join(vt.round_dir(K), 'round.json')):
        raise RuntimeError('round %d exists; move it to scrap first' % K)
    th, r11 = theta()
    return vt.new_round(K, th, r11['delta'], note=NOTE, base_ics=r11['base_ics'], base_theta=r11['base_theta'])


def check():
    import v21_tune as vt
    d = json.load(open(os.path.join(vt.round_dir(K), 'round.json')))
    th, r11 = theta()
    print('theta == r11 + %g step: %s; delta == r11: %s; base_ics == r11: %s; base_theta == r11: %s'
          % (DAMP, np.array_equal(d['theta'], th), d['delta'] == r11['delta'], d['base_ics'] == r11['base_ics'],
             d['base_theta'] == r11['base_theta']))
    print('step:', np.array2string(th - np.array(r11['theta']), precision=3))


if __name__ == '__main__':
    setup() if len(sys.argv) > 1 and sys.argv[1] == 'setup' else check()
