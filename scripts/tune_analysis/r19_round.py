"""Round 19: Jacobian round (base + 12 nudges, LMC and Sgr) at round 18's base plus the full Gauss-Newton step from
J18 (v21_tune.gauss_newton, undamped, full matrix), Sgr rescaled by round 18's 1.03 (present-day M(<5 kpc) 3.3-3.5e8
Msun in round 18, within V21's (3-4)e8). This is the round 19 that finish_round(18) set up; nudges as round 18.

    setup()     # on OBS: TUNE/round19/round.json (if finish_round(18) has not written it)
    check()     # on OBS: round.json reproduces
Submit:  sbatch --array=0-12 ~/projects/glossary/scripts/obs/tune_round.sbatch 19
"""
import sys
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
sys.path.insert(0, '/home/abonaca/projects/glossary/scripts')
import json, os
import numpy as np

K = 19


def _info():
    import v21_tune as vt
    prev = json.load(open(os.path.join(vt.round_dir(K - 1), 'round.json')))
    F = np.array(prev['endpoints'])
    dtheta, _ = vt.gauss_newton(F[0], F[1:], prev['delta'], block=False)
    theta = np.array(prev['theta']) + dtheta
    return theta, prev


def setup():
    import v21_tune as vt
    if os.path.exists(os.path.join(vt.round_dir(K), 'round.json')):
        raise RuntimeError('round %d exists; move it to scrap first' % K)
    theta, prev = _info()
    return vt.new_round(K, theta, prev['delta'], note='Gauss-Newton step from round %d' % (K - 1),
                        base_ics=prev['base_ics'], base_theta=prev['base_theta'], sgr_rscale=prev['sgr_rscale'][0])


def check():
    import v21_tune as vt
    d = json.load(open(os.path.join(vt.round_dir(K), 'round.json')))
    theta, prev = _info()
    print('theta == r%d + full GN step: %s; == r%d theta_next: %s; delta == r%d: %s; sgr_rscale: %s; '
          'base_ics == r%d: %s; base_theta == r%d: %s'
          % (K - 1, np.array_equal(d['theta'], theta), K - 1, d['theta'] == prev['theta_next'], K - 1,
             d['delta'] == prev['delta'], set(d['sgr_rscale']), K - 1, d['base_ics'] == prev['base_ics'], K - 1,
             d['base_theta'] == prev['base_theta']))
    print('step:', np.array2string(theta - np.array(prev['theta']), precision=3))


if __name__ == '__main__':
    setup() if len(sys.argv) > 1 and sys.argv[1] == 'setup' else check()
