"""Sgr and LMC present-day misses of chosen round-9/10 runs in heliocentric observables (V21's solar frame, r8_obs),
relative to the V21 Galactocentric targets converted to observables, and to V21's stated observed values."""
import json, numpy as np
from r8_obs import to_obs, from_obs, obs as SGR_OBS
T = '/home/ana/projects/glossary/output/v21_sim/tune/'
# V21 section 3: LMC bar photometric centre (Vasiliev 2018), D = 50 kpc (Freedman+2001), vlos = 260 (vdM&K 2014)
LMC_OBS = dict(ra=81., dec=-69.75, d=50., pmra=1.8, pmdec=0.35, vlos=260.)
TGT = dict(lmc=np.array([-0.6, -41.3, -27.1, -63.9, -213.8, 206.6]), sgr=np.array([17.9, 2.6, -6.6, 239.5, -29.6, 213.5]))
COLS = ('dRA[deg]', 'dDec[deg]', 'dD[kpc]', 'dpmra', 'dpmdec', 'dvlos')
RUNS = [(10, 2), (10, 4), (9, 2), (8, 3)]


def report(runs=RUNS):
    for body, o, sl in (('sgr', SGR_OBS, slice(6, 12)), ('lmc', LMC_OBS, slice(0, 6))):
        ot = to_obs(TGT[body])[0]
        print('\n%s: target -> observables %s' % (body.upper(), np.round(ot, 4)))
        print('     V21 stated  - target obs %s' % np.round(np.array([o[k] for k in ('ra', 'dec', 'd', 'pmra', 'pmdec', 'vlos')]) - ot, 4))
        print('%-12s' % '' + ''.join('%11s' % c for c in COLS))
        for k, i in runs:
            m = np.array(json.load(open(T + 'round%02d/round.json' % k))['misses'][i])[sl]
            d = to_obs(TGT[body] + m)[0] - ot
            print('%-12s' % ('r%d run %d' % (k, i)) + ''.join('%11.4f' % x for x in d))


if __name__ == '__main__':
    report()
