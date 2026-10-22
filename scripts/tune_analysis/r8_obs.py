"""Sgr present-day states (round 7 base, round 8) converted to heliocentric observables with V21's solar parameters."""
import json, numpy as np, astropy.units as u, astropy.coordinates as coord
T = '/home/ana/projects/glossary/output/v21_sim/tune/'
fr = coord.Galactocentric(galcen_distance=np.hypot(8.1, 0.02) * u.kpc, z_sun=20 * u.pc,
                          galcen_v_sun=coord.CartesianDifferential([12.9, 245.6, 7.8] * u.km / u.s))
obs = dict(ra=283.76, dec=-30.48, d=27., pmra=-2.7, pmdec=-1.35, vlos=142.)


def to_obs(w):
    w = np.atleast_2d(w)
    g = coord.SkyCoord(x=w[:, 0] * u.kpc, y=w[:, 1] * u.kpc, z=w[:, 2] * u.kpc, v_x=w[:, 3] * u.km / u.s,
                       v_y=w[:, 4] * u.km / u.s, v_z=w[:, 5] * u.km / u.s, frame=fr).icrs
    return np.column_stack((g.ra.deg, g.dec.deg, g.distance.kpc, g.pm_ra_cosdec.value, g.pm_dec.value,
                            g.radial_velocity.value))


def from_obs(o):
    c = coord.SkyCoord(ra=o['ra'] * u.deg, dec=o['dec'] * u.deg, distance=o['d'] * u.kpc, pm_ra_cosdec=o['pmra'] * u.mas / u.yr,
                       pm_dec=o['pmdec'] * u.mas / u.yr, radial_velocity=o['vlos'] * u.km / u.s).transform_to(fr)
    return np.r_[c.x.value, c.y.value, c.z.value, c.v_x.value, c.v_y.value, c.v_z.value]


if __name__ == '__main__':
    print('V21 observables -> Galactocentric:', np.round(from_obs(obs), 3))
    tgt = np.array([17.9, 2.6, -6.6, 239.5, -29.6, 213.5])
    o_t = to_obs(tgt)[0]
    print('V21 target -> observables:', np.round(o_t, 4))
    info7 = json.load(open(T + 'round07/round.json')); info8 = json.load(open(T + 'round08/round.json'))
    rows = [('r7 base', np.array(info7['misses'][0])[6:])] + \
           [('r8 ' + l.replace('J7 LM lambda=', 'lam='), np.array(m)[6:]) for l, m in zip(info8['labels'], info8['misses'])]
    print('%-14s %9s %9s %8s %9s %9s %8s' % ('', 'dRA[deg]', 'dDec[deg]', 'dD[kpc]', 'dpmra', 'dpmdec', 'dvlos'))
    for lab, m in rows:
        d = to_obs(tgt + m)[0] - o_t
        print('%-14s %9.4f %9.4f %8.3f %9.4f %9.4f %8.2f' % (lab, *d))
    # sensitivity: what 1 km/s or 0.1 kpc in each Galactocentric axis means observationally
    for k, n in enumerate(['x 0.1kpc', 'y 0.1kpc', 'z 0.1kpc', 'vx 1km/s', 'vy 1km/s', 'vz 1km/s']):
        e = np.zeros(6); e[k] = 0.1 if k < 3 else 1.
        print('%-9s' % n, np.round(to_obs(tgt + e)[0] - o_t, 4))
