"""Round 16 (redo): scan along the exact Sgr-only solutions of two multi-run Jacobians, from round 15's start.

The first round 16 (Sgr-block Levenberg-Marquardt, lambda 1e-2, now SCRAP/round16_lm) did not move Sgr's timing.
Over rounds 11-16 the Sgr velocity misses at the right arrival time stay at about (-2.5, +0.6, +0.6) km/s, whatever
the centre apertures (aperture_offsets.py); small linear steps don't reach it. A linear fit of all endpoints on all
starts over many runs (sgr_timing.fit_jacobian) gives an exact Sgr-only solution (LMC fixed), but it depends on which
runs enter the fit: with all 98 runs of rounds 11-16 (incl. the redone round12_fullstep, round13_damp0.5) it is
mostly Sgr z0 -0.41 kpc, y0 +0.38 kpc and vz0 -0.67 km/s; with the 72 runs of the kept rounds it is mostly
y0 +0.37 kpc, vx0 -0.59 km/s and z0 -0.11 kpc. This round scans both directions; every run is scored on its own
(v21_tune kind 'sgr_offsets').

    offsets()       # locally: run 0 = round 15 repeat, runs 1-6 / 7-12 = FRACS x the 98-run / 72-run step
                    #          -> r16_scan_offsets.json (committed)
    setup()         # on OBS: TUNE/round16/round.json from the json, base_ics/base_theta as round 15
    check()         # on OBS: round.json reproduces
Submit:  sbatch --array=0-12 ~/projects/glossary/scripts/obs/tune_round.sbatch 16
"""
import sys
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
sys.path.insert(0, '/home/abonaca/projects/glossary/scripts')
import json, os
import numpy as np

K = 16
FRACS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'r16_scan_offsets.json')
NOTE = ('Sgr-only scan from round 15 (LMC fixed): run 0 round 15 repeat; runs 1-6 and 7-12 at %s of the exact Sgr-only '
        'solutions of the multi-run Jacobians over 98 and 72 runs (r16_scan.py, sgr_timing.fit_jacobian)' % (FRACS,))


def offsets(out=OUT):
    import sgr_timing as st, v21_tune as vt
    d = json.load(open(os.path.join(st.T, 'round15', 'round.json')))
    r = np.array(d['endpoints'][0]) - vt.TARGET
    delta = np.array(d['delta'])
    offs, labs = [np.zeros(6)], ['round 15 repeat']
    for name, rounds in (('98-run', st.ROUNDS), ('72-run', st.CORE)):
        J, _ = st.fit_jacobian(rounds)
        s = st.step(r, J, cols=list(range(6, 12)), delta=delta)[6:]
        for f in FRACS:
            offs.append(f * s)
            labs.append('%s x%g' % (name, f))
    json.dump(dict(offsets=np.array(offs).tolist(), labels=labs, rounds98=list(st.ROUNDS), rounds72=list(st.CORE)),
              open(out, 'w'), indent=1)
    for lab, o in zip(labs, offs):
        print('%-16s %s' % (lab, np.array2string(np.asarray(o), precision=4)))
    return offs, labs


def _runs():
    import v21_tune as vt
    prev = json.load(open(os.path.join(vt.round_dir(K - 1), 'round.json')))
    o = json.load(open(OUT))
    theta = np.array(prev['theta'])
    runs = []
    for off in np.array(o['offsets']):
        th = theta.copy()
        th[6:] += off
        runs.append(th)
    return theta, o, runs, prev


def setup():
    import v21_tune as vt
    d = vt.round_dir(K)
    if os.path.exists(os.path.join(d, 'round.json')):
        raise RuntimeError('round %d exists; move it to scrap first' % K)
    theta, o, runs, prev = _runs()
    os.makedirs(d, exist_ok=True)
    info = dict(round=K, kind='sgr_offsets', names=vt.NAMES, theta=theta.tolist(), offsets=o['offsets'],
                labels=o['labels'], runs=[r.tolist() for r in runs], target=vt.TARGET.tolist(), tol=vt.TOL.tolist(),
                note=NOTE, base_ics=prev['base_ics'], base_theta=prev['base_theta'])
    json.dump(info, open(os.path.join(d, 'round.json'), 'w'), indent=1)
    print('round %d (Sgr offsets, %d runs) from round %d theta %s' % (K, len(runs), K - 1,
                                                                     np.array2string(theta, precision=3)))
    return info


def check():
    import v21_tune as vt
    d = json.load(open(os.path.join(vt.round_dir(K), 'round.json')))
    theta, o, runs, prev = _runs()
    print('runs: %s; theta == r%d: %s; LMC fixed in all runs: %s; base_ics == r%d: %s; base_theta == r%d: %s'
          % (np.array_equal(d['runs'], runs), K - 1, np.array_equal(d['theta'], prev['theta']),
             all(np.array_equal(r[:6], prev['theta'][:6]) for r in d['runs']), K - 1, d['base_ics'] == prev['base_ics'],
             K - 1, d['base_theta'] == prev['base_theta']))
    for lab, off in zip(d['labels'], d['offsets']):
        print('  %-16s %s' % (lab, np.array2string(np.array(off), precision=4)))


if __name__ == '__main__':
    setup() if len(sys.argv) > 1 and sys.argv[1] == 'setup' else check()
