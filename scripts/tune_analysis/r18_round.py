"""Round 18: Jacobian round (base + 12 nudges, LMC and Sgr) at round 16's best start, with Sgr rescaled by LAM = 1.03.

Round 16's best run (run 2, 0.75 x the 98-run Sgr-only solution) has the LMC within tolerance and Sgr at (-0.07,
+0.02, +0.14) kpc, (-1.05, +0.69, +1.19) km/s, but Sgr ends with M(<5 kpc) = 5.8e8 Msun against V21's (3-4)e8. The
round 17 size scan (r17_rscale.py) puts M(<5 kpc) = 3.56e8 at lam = 1.03, mid-range. Every run here rescales Sgr
about its centre of mass by LAM (v21_tune.build_ics, sgr_rscale) before the LMC and Sgr are moved to the run's theta;
nudges as round 15's (round 16 had none).

    setup()     # on OBS: TUNE/round18/round.json
    check()     # on OBS: round.json reproduces
Submit:  sbatch --array=0-12 ~/projects/glossary/scripts/obs/tune_round.sbatch 18
"""
import sys
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
sys.path.insert(0, '/home/abonaca/projects/glossary/scripts')
import json, os
import numpy as np

K = 18
LAM = 1.03
RUN16 = 2
NOTE = 'Round 16 run %d start (98-run x0.75), Sgr rescaled by %g; nudges as round 15 (r18_round.py)' % (RUN16, LAM)


def _info():
    import v21_tune as vt
    r16 = json.load(open(os.path.join(vt.round_dir(16), 'round.json')))
    r15 = json.load(open(os.path.join(vt.round_dir(15), 'round.json')))
    return dict(round=K, names=vt.NAMES, theta=r16['runs'][RUN16], delta=r15['delta'], target=vt.TARGET.tolist(),
                tol=vt.TOL.tolist(), note=NOTE, base_ics=r16['base_ics'], base_theta=r16['base_theta'],
                sgr_rscale=[LAM] * 13)


def setup():
    import v21_tune as vt
    d = vt.round_dir(K)
    if os.path.exists(os.path.join(d, 'round.json')):
        raise RuntimeError('round %d exists; move it to scrap first' % K)
    os.makedirs(d, exist_ok=True)
    info = _info()
    json.dump(info, open(os.path.join(d, 'round.json'), 'w'), indent=1)
    print('round %d: theta = %s, Sgr rescaled by %g' % (K, np.array2string(np.array(info['theta']), precision=3), LAM))
    return info


def check():
    import v21_tune as vt
    d = json.load(open(os.path.join(vt.round_dir(K), 'round.json')))
    ref = _info()
    print('round.json reproduces: %s' % all(d[key] == ref[key] for key in ref))
    print('theta == round 16 run %d: %s; sgr_rscale: %s' % (RUN16, d['theta'] == ref['theta'], set(d['sgr_rscale'])))


if __name__ == '__main__':
    setup() if len(sys.argv) > 1 and sys.argv[1] == 'setup' else check()
