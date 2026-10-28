"""Round 17 (redo, fine): Sgr progenitor size scan at round 15's starts (LMC and Sgr orbit unchanged).

All reversed-disc runs (rounds 11-15) end with 5.7-6.1e8 Msun within 5 kpc of Sgr's centre, against V21's present-day
(3-4)e8 (sgr_mass.py), and V21 tuned the progenitor's initial radius to get that mass. Each run here rescales Sgr about
its centre of mass by lam (positions x lam, velocities / sqrt(lam): same mass, an equilibrium lam times larger;
v21_tune.build_ics, round.json sgr_rescale), at round 15's 12 starting values (round 15 itself is lam = 1). The first,
coarse round 17 (lam 1.0-1.5, commit abbea85, now SCRAP/round17_coarse) gave M(<5 kpc) = 5.8e8 at lam 1, 1.2e8 at
1.1 and ~3e7 (Sgr disrupted) from 1.2 up, putting V21's (3-4)e8 at lam ~1.02-1.04; this redo steps lam by 0.005.
Scored per run (kind 'sgr_offsets'); then sgr_mass.measure(17).

    setup()     # on OBS: TUNE/round17/round.json
    check()     # on OBS: round.json reproduces
Submit:  sbatch --array=0-9 ~/projects/glossary/scripts/obs/tune_round.sbatch 17
"""
import sys
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
sys.path.insert(0, '/home/abonaca/projects/glossary/scripts')
import json, os
import numpy as np

K = 17
LAMS = tuple(round(1 + 0.005 * j, 3) for j in range(1, 11))     # 1.005 ... 1.05
NOTE = 'Sgr size scan at round 15 starts: Sgr rescaled about its centre of mass by %s (r17_rscale.py)' % (LAMS,)


def _info():
    import v21_tune as vt
    prev = json.load(open(os.path.join(vt.round_dir(15), 'round.json')))
    theta = np.array(prev['theta'])
    return dict(round=K, kind='sgr_offsets', names=vt.NAMES, theta=theta.tolist(),
                offsets=np.zeros((len(LAMS), 6)).tolist(), labels=['Sgr size x%g' % l for l in LAMS],
                runs=[theta.tolist()] * len(LAMS), sgr_rscale=list(LAMS), target=vt.TARGET.tolist(),
                tol=vt.TOL.tolist(), note=NOTE, base_ics=prev['base_ics'], base_theta=prev['base_theta'])


def setup():
    import v21_tune as vt
    d = vt.round_dir(K)
    if os.path.exists(os.path.join(d, 'round.json')):
        raise RuntimeError('round %d exists; move it to scrap first' % K)
    os.makedirs(d, exist_ok=True)
    info = _info()
    json.dump(info, open(os.path.join(d, 'round.json'), 'w'), indent=1)
    print('round %d: Sgr size scan %s at round 15 theta %s' % (K, LAMS, np.array2string(np.array(info['theta']), precision=3)))
    return info


def check():
    import v21_tune as vt
    d = json.load(open(os.path.join(vt.round_dir(K), 'round.json')))
    ref = _info()
    print('round.json reproduces: %s' % all(d[key] == ref[key] for key in ref))
    print('sgr_rscale: %s; all runs at round 15 theta: %s' % (d['sgr_rscale'], all(r == ref['theta'] for r in d['runs'])))


if __name__ == '__main__':
    setup() if len(sys.argv) > 1 and sys.argv[1] == 'setup' else check()
