"""Round 11: Jacobian round (base + 12 nudges) around the adopted model with the MW disc rotating as in the MW.

Base = round 9 run 2's theta, built from v21_sim/adopted/r09run02_prograde/ics.nemo (round 9 run 2's initial
conditions with the MW stars' velocities reversed, mw_prograde.build). Nudges: the latest adopted sizes, LMC as in
rounds 3-6 (3, 1, 3 kpc; 3, 2, 3 km/s) and Sgr as in rounds 7-9 (0.05, 0.25, 0.05 kpc; 0.25, 0.25, 0.25 km/s).

    setup()     # on OBS: TUNE/round11/round.json
    check()     # on OBS: round.json reproduces; run 0's theta is round 9 run 2's
Submit:  sbatch --array=0-12 ~/projects/glossary/scripts/obs/tune_round.sbatch 11
"""
import sys
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
sys.path.insert(0, '/home/abonaca/projects/glossary/scripts')
import json, os
import numpy as np

K = 11
BASE_ICS = '/carnegie/nobackup/users/abonaca/v21_sim/adopted/r09run02_prograde/ics.nemo'
DELTA = np.array([3., 1., 3., 3., 2., 3.] + [0.05, 0.25, 0.05, 0.25, 0.25, 0.25])
NOTE = ('Jacobian round around round 9 run 2 with the MW disc reversed to rotate as in the MW (mw_prograde, '
        'adopted/r09run02_prograde/ics.nemo); LMC nudges as rounds 3-6, Sgr nudges as rounds 7-9')


def setup():
    import v21_tune as vt
    theta = vt.run_theta(9, 2)
    return vt.new_round(K, theta, DELTA, note=NOTE, base_ics=BASE_ICS, base_theta=theta)


def check():
    import v21_tune as vt
    d = json.load(open(os.path.join(vt.round_dir(K), 'round.json')))
    theta = vt.run_theta(9, 2)
    print('theta == r9 run 2: %s; delta matches: %s; base_ics %s; base_theta == theta: %s'
          % (np.array_equal(d['theta'], theta), np.array_equal(d['delta'], DELTA), d['base_ics'] == BASE_ICS,
             np.array_equal(d['base_theta'], theta)))


if __name__ == '__main__':
    setup() if len(sys.argv) > 1 and sys.argv[1] == 'setup' else check()
