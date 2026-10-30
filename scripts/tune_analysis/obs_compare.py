"""Present-day LMC and Sgr of chosen runs in heliocentric observables (V21's solar frame, r8_obs.to_obs), minus the
V21 targets converted the same way, against observational uncertainties of the centres.

Uncertainties (1 sigma) of the centres V21 adopted:
  Sgr (VB20 = Vasiliev & Belokurov 2020, papers/vb20.txt): mean PM <~ 0.05 mas/yr (Gaia DR2 systematics, VB20 sec. 2);
      vlos 1-2 km/s (systematic offsets between spectroscopic samples, VB20 sec. 2); distance 1.3 kpc (systematic,
      Hamanowicz et al. 2016 RR Lyrae, quoted in VB20 sec. 3; VB20 adopt 26.5, V21 27 kpc).
  LMC: PM 0.020 / 0.047 mas/yr (Kallivayalil et al. 2013 HST centre-of-mass PM, mu_W / mu_N); distance 0.54 kpc
      (Pietrzynski et al. 2019, 49.59 +- 0.09 stat +- 0.54 sys; V21 use 50, Freedman et al. 2001); vlos 3.4 km/s
      (van der Marel & Kallivayalil 2014, 262.2 +- 3.4; V21 use 260).
Centre positions on the sky: no uncertainty stated; reported in arcmin for scale.

    table([(20, 7)])
"""
import json, os
import numpy as np
from r8_obs import to_obs
import sgr_timing as st
import v21_tune as vt

COLS = ('RA', 'Dec', 'D', 'pmra', 'pmdec', 'vlos')
UNITS = ('arcmin', 'arcmin', 'kpc', 'mas/yr', 'mas/yr', 'km/s')
SIGMA = dict(sgr=(np.nan, np.nan, 1.3, 0.05, 0.05, 2.), lmc=(np.nan, np.nan, 0.54, 0.020, 0.047, 3.4))


def offsets(k, i):
    """Observable offsets (run - target) of round k run i for 'lmc' and 'sgr'; RA offset times cos(Dec)."""
    m = np.array(json.load(open(os.path.join(st.T, 'round%02d' % k, 'round.json')))['misses'][i])
    out = {}
    for body, sl in (('lmc', slice(0, 6)), ('sgr', slice(6, 12))):
        t = vt.TARGET[sl]
        o, ot = to_obs(t + m[sl])[0], to_obs(t)[0]
        d = o - ot
        d[0] = ((d[0] + 180) % 360 - 180) * np.cos(np.radians(ot[1])) * 60
        d[1] *= 60
        out[body] = d
    return out


def table(runs=((20, 7),)):
    for k, i in runs:
        o = offsets(k, i)
        print('round %d run %d (run - target)' % (k, i))
        for body in ('lmc', 'sgr'):
            print('  %s: ' % body.upper() + '; '.join(
                '%s %+.3g %s' % (c, x, u) + ('' if np.isnan(s) else ' (sigma %g)' % s)
                for c, x, u, s in zip(COLS, o[body], UNITS, SIGMA[body])))
    return o
