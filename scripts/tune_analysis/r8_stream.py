"""Round-8 run-3 Sgr particles at t=0 (relative to the MW centre) vs the VBE21 stream catalogue, in
Galactocentric Cartesian projections (V21 solar parameters)."""
import numpy as np, astropy.units as u, astropy.coordinates as coord
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('default')
fr = coord.Galactocentric(galcen_distance=np.hypot(8.1, 0.02) * u.kpc, z_sun=20 * u.pc,
                          galcen_v_sun=coord.CartesianDifferential([12.9, 245.6, 7.8] * u.km / u.s))


def load_obs(f='/home/ana/projects/glossary/data/vbe21_catalogue/catalogue.txt'):
    d = np.genfromtxt(f, names=True)
    c = coord.SkyCoord(ra=d['ra'] * u.deg, dec=d['dec'] * u.deg, distance=d['dist'] * u.kpc).transform_to(fr)
    return np.column_stack((c.x.value, c.y.value, c.z.value)), d


def load_sim(f='/home/ana/projects/glossary/output/v21_sim/tune/round08/run03/sgr_t0.npz'):
    s = np.load(f)
    return s['xv'], s['star'], s['sgr'] - s['mw']


def plot(out='/home/ana/projects/glossary/plots/tune_r8_run03_sgr_stream_xyz.png'):
    xo, d = load_obs()
    xv, star, c = load_sim()
    tgt = np.array([17.9, 2.6, -6.6])
    fig, ax = plt.subplots(2, 3, figsize=(16, 10.5))
    for p, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
        for row in range(2):
            a = ax[row, p]
            if row == 0:
                a.plot(xv[~star, i], xv[~star, j], ',', color='0.75', alpha=0.3, rasterized=True)
                a.plot(xv[star, i], xv[star, j], ',', color='k', alpha=0.3, rasterized=True)
            else:
                a.plot(xv[star, i], xv[star, j], ',', color='0.6', alpha=0.3, rasterized=True)
                a.plot(xo[:, i], xo[:, j], '.', color='tab:red', ms=1, alpha=0.4, rasterized=True)
            a.plot(tgt[i], tgt[j], '*', color='tab:orange', mec='k', ms=12, zorder=5)
            a.plot(0, 0, '+', color='tab:blue', ms=10, mew=2, zorder=5)
            a.set_xlabel('%s [kpc]' % 'xyz'[i]); a.set_ylabel('%s [kpc]' % 'xyz'[j])
            a.set_xlim(-90, 60); a.set_ylim(-60, 60); a.set_aspect('equal')
    ax[0, 0].text(0.02, 0.98, 'r8 run 3, t = 0: Sgr stars (black), Sgr DM (grey)', transform=ax[0, 0].transAxes,
                  va='top', fontsize=8)
    ax[1, 0].text(0.02, 0.98, 'r8 run 3 Sgr stars (grey) vs VBE21 catalogue, %d stars (red)\n'
                  'star: V21 Sgr position; +: MW centre' % len(xo), transform=ax[1, 0].transAxes,
                  va='top', fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(out)
    print('sim centre - target [kpc]:', np.round(c[:3] - tgt, 3))
    print('obs x range', np.round(xo.min(0), 1), np.round(xo.max(0), 1))


if __name__ == '__main__':
    plot()
