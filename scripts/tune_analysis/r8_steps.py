"""Round-8 Sgr start offsets (relative to the round-7 base; LMC fixed), as passed to v21_tune.new_offset_round(8, ...)
and v21_tune.extend_offset_round(8, ...):

    lm_offsets()          # runs 0-6: Marquardt-damped steps with the round-7 Sgr Jacobian J7 -> r8_offsets_lm.json
    nudge_offsets(sign)   # runs 7-12 (sign=+1) and 13-18 (sign=-1): run 3 +/- one round-7 nudge per start component
    check()               # compare with round08/round.json

Round 9 is set up from J8 by r8_jacobian.r9_offsets.
"""
import json, numpy as np
from r8_jacobian import lm_steps, DELTA, NAMES
T = '/home/ana/projects/glossary/output/v21_sim/tune/'
OUT = '/home/ana/projects/glossary/scripts/tune_analysis/r8_offsets_lm.json'
LAMS = (0, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1)


def lm_offsets(out=OUT, lams=LAMS):
    i7 = json.load(open(T + 'round07/round.json'))
    J7 = np.array(i7['jacobian_sgr'])
    base = np.array(i7['runs'][0])
    r = np.array(i7['endpoints'][0])[6:] - np.array(i7['target'])[6:]
    offs = [s for _, s, _ in lm_steps(J7, r, lams, damping='marquardt')]
    d = dict(base=base.tolist(), offsets=np.array(offs).tolist(), labels=['J7 LM lambda=%g' % lam for lam in lams],
             note='Gauss-Newton/LM steps from the round-7 base Sgr miss with the round-7 Sgr Jacobian '
                  '(tolerance-weighted, nudge-scaled, diag damping)')
    if out:
        json.dump(d, open(out, 'w'), indent=1)
    return d


def nudge_offsets(sign=1, base_run=3):
    """Offsets (relative to the round's reference theta) of round-8 run `base_run` +/- DELTA in each start component."""
    o3 = np.array(lm_offsets(out=None)['offsets'][base_run])
    offs = [o3 + sign * DELTA[i] * np.eye(6)[i] for i in range(6)]
    labs = ['run%d %sd %s' % (base_run, '+' if sign > 0 else '-', n) for n in NAMES]
    return np.array(offs), labs


def check():
    d = json.load(open(T + 'round08/round.json'))
    offs = np.vstack([lm_offsets(out=None)['offsets'], nudge_offsets(1)[0], nudge_offsets(-1)[0]])
    labs = lm_offsets(out=None)['labels'] + nudge_offsets(1)[1] + nudge_offsets(-1)[1]
    print('max |offset - round.json| = %.2e; labels match: %s' % (np.abs(offs - np.array(d['offsets'])).max(),
                                                                labs == d['labels']))


if __name__ == '__main__':
    check()
