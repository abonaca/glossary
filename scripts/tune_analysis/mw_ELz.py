"""Energy and angular momentum of the MW star particles in a snapshot of the adopted model (round 9 run 2; the dense
rerun in v21_sim/adopted/r09run02 is identical), in the MW's own potential at that time.

    compute(t=0.)    # on OBS: snapshot -> MW-centred stars, E in the MW (halo + stars) expansions -> mw_ELz_t*.npz
    plot()           # locally: E-Lz density -> plots/adopted_mw_stars_ELz.png
    plot_sky()       # locally: stars with |z| > 2 kpc and |b| > 30 deg seen from the Sun -> plots/adopted_mw_stars_ELz_zb_sun.png
    plot_sky(circle=True)   # ... seen from anywhere on the solar circle -> plots/adopted_mw_stars_ELz_zb_circle.png

Frame: positions and velocities relative to the MW centre (shrinking sphere on the MW stars with a 3 kpc velocity
aperture, v21_tune.track_centres -> centres.npz). Lz = x vy - y vx in V21's Galactocentric axes. The model disc
rotates with Lz > 0 (Agama QuasiIsothermal DF, as in V21's create_mw.py, which does not flip it), i.e. opposite to
the Sun's motion in the astropy frame V21 use for the satellites (Sun at x = -8.1 kpc, vy = +245.6 km/s -> Lz < 0).
E excludes the LMC and Sgr potentials.
"""
import sys, os
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
sys.path.insert(0, '/home/abonaca/projects/glossary/scripts')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('default')
NSTARS, NMW = 1000000, 5000000          # MW stars first, then MW dark matter (R2.json)
LOCAL = '/home/ana/projects/glossary/output/v21_sim/adopted/'


def compute(t=0., k=9, i=2, outdir=None):
    import agama, v21_tune as vt, r0_noise as rn
    agama.setUnits(length=1, velocity=1, mass=vt.vi.MASS_UNIT)
    d = vt.run_dir(k, i)
    c = np.load(os.path.join(d, 'centres.npz'))
    j = int(np.argmin(np.abs(c['times'] - t)))
    xv, m = rn.read_snapshot(os.path.join(d, 'evol.nemo'), t)
    xv, m = xv[:NMW].copy(), m[:NMW]
    xv -= c['mw'][j]
    pot = agama.Potential(rn.fit(xv[NSTARS:], m[NSTARS:], 'halo'), rn.fit(xv[:NSTARS], m[:NSTARS], 'disk'))
    s = xv[:NSTARS]
    E = 0.5 * np.sum(s[:, 3:]**2, 1) + pot.potential(s[:, :3])
    L = np.cross(s[:, :3], s[:, 3:])
    out = os.path.join(outdir or d, 'mw_ELz_t%+.3f.npz' % t)
    np.savez(out, E=E.astype(np.float32), L=L.astype(np.float32), R=np.hypot(s[:, 0], s[:, 1]).astype(np.float32),
             z=s[:, 2].astype(np.float32), x=s[:, :3].astype(np.float32), t=t, centre=c['mw'][j])
    print(out)


def plot(f=LOCAL + 'mw_ELz_t+0.000.npz', out='/home/ana/projects/glossary/plots/adopted_mw_stars_ELz.png'):
    from matplotlib.colors import LogNorm
    d = np.load(f)
    E, Lz = d['E'] / 1e4, d['L'][:, 2] / 1e3
    fig, ax = plt.subplots(figsize=(7.5, 6))
    h = ax.hist2d(Lz, E, bins=(400, 300), range=((-1., 5.5), (-26, -5)), norm=LogNorm(), cmap='Greys', rasterized=True)
    fig.colorbar(h[3], ax=ax, label='stars per bin')
    ax.set_xlabel('$L_z$ [$10^3$ kpc km/s]')
    ax.set_ylabel('E (MW potential) [$10^4$ (km/s)$^2$]')
    ax.text(0.02, 0.98, 'adopted model (r9 run 2), MW stars, t = 0', transform=ax.transAxes, va='top', fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(out)


SUN = np.array([-8.1, 0., 0.])          # V21's astropy frame, solar radius 8.1 kpc, in the plane


def sky_selection(x, zmin=2., bmin=30., sun=SUN):
    """|z| > zmin [kpc] from the MW plane and |b| > bmin [deg] as seen from the Sun."""
    d = x - sun
    b = np.degrees(np.arcsin(d[:, 2] / np.linalg.norm(d, axis=1)))
    return (np.abs(x[:, 2]) > zmin) & (np.abs(b) > bmin)


def circle_fraction(x, zmin=2., bmin=30., rsun=-SUN[0], nphi=360):
    """Fraction of nphi observers spaced evenly around the solar circle (radius rsun, in the plane) for whom the
    star is at |b| > bmin; zero for stars with |z| <= zmin."""
    f = np.zeros(len(x))
    s = np.abs(x[:, 2]) > zmin
    for phi in 2 * np.pi * np.arange(nphi) / nphi:
        f[s] += sky_selection(x[s], zmin, bmin, rsun * np.array([np.cos(phi), np.sin(phi), 0.]))
    return f / nphi


def plot_sky(f=LOCAL + 'mw_ELz_t+0.000.npz', circle=False, zmin=2., bmin=30., out=None):
    """E-Lz points of the stars with |z| > zmin and |b| > bmin, seen from the Sun (circle=False) or from anywhere on
    the solar circle (circle=True: stars visible to at least one observer, coloured by the fraction of observers)."""
    d = np.load(f)
    out = out or '/home/ana/projects/glossary/plots/adopted_mw_stars_ELz_zb_%s.png' % ('circle' if circle else 'sun')
    if circle:
        frac = circle_fraction(d['x'], zmin, bmin)
        s = frac > 0
        where = 'from the solar circle'
    else:
        s = sky_selection(d['x'], zmin, bmin)
        where = 'from the Sun'
    E, Lz = d['E'][s] / 1e4, d['L'][s, 2] / 1e3
    print('%d of %d stars selected' % (s.sum(), len(s)))
    fig, ax = plt.subplots(figsize=(7.5 if circle else 6.5, 6))
    if circle:
        o = np.argsort(frac[s])
        p = ax.scatter(Lz[o], E[o], c=frac[s][o], s=1, lw=0, cmap='viridis', vmin=0, rasterized=True)
        fig.colorbar(p, ax=ax, label='fraction of solar-circle observers with |b| > %g$^\\circ$' % bmin)
    else:
        ax.plot(Lz, E, 'k.', ms=2, mew=0, rasterized=True)
    ax.set_xlim(-1.5, 3.5); ax.set_ylim(-18, -6)
    ax.set_xlabel('$L_z$ [$10^3$ kpc km/s]  (model disc: $L_z > 0$)')
    ax.set_ylabel('E (MW potential) [$10^4$ (km/s)$^2$]')
    ax.text(0.02, 0.98, 'adopted model (r9 run 2), MW stars, t = 0\n|z| > %g kpc, |b| > %g$^\\circ$ %s (%d stars)'
            % (zmin, bmin, where, s.sum()), transform=ax.transAxes, va='top', fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(out)

if __name__ == '__main__':
    compute(float(sys.argv[1]) if len(sys.argv) > 1 else 0.)
