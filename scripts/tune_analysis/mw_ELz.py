"""Energy and angular momentum of the MW star particles in a snapshot of the adopted model (round 9 run 2; the dense
rerun in v21_sim/adopted/r09run02 is identical), in the MW's own potential at that time.

    compute(t=0.)    # on OBS: snapshot -> MW-centred stars, E in the MW (halo + stars) expansions -> mw_ELz_t*.npz
    plot()           # locally: E-Lz density -> plots/adopted_mw_stars_ELz.png
    plot_sky()       # locally: stars with |z| > 2 kpc and |b| > 30 deg seen from the Sun -> plots/adopted_mw_stars_ELz_zb_sun.png
    plot_sky(circle=True)   # ... seen from anywhere on the solar circle -> plots/adopted_mw_stars_ELz_zb_circle.png
    plot_circle_density()   # same selection, density at t = -3 and 0 -> plots/adopted_mw_stars_ELz_zb_circle_density.png
    periods(t)       # on OBS: radial periods of the solar-circle selection and of Sgr -> mw_periods_t*.npz
    plot_periods()   # locally: period histograms at t = -3 and 0 with Sgr's n:1 resonances in range -> plots/adopted_mw_stars_Tr_zb_circle.png
    plot_circle_Ehist()     # same selection, energy histograms at t = -3 and 0 -> plots/adopted_mw_stars_E_zb_circle.png

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
GYR = 0.9778                            # Gyr per model time unit
LOCAL = '/home/ana/projects/glossary/output/v21_sim/adopted/'


def _snapshot_potential(t, k=9, i=2):
    """MW-centred MW particles at time t and the MW (halo + stars) expansions; also the centres file and index."""
    import agama, v21_tune as vt, r0_noise as rn
    agama.setUnits(length=1, velocity=1, mass=vt.vi.MASS_UNIT)
    d = vt.run_dir(k, i)
    c = np.load(os.path.join(d, 'centres.npz'))
    j = int(np.argmin(np.abs(c['times'] - t)))
    xv, m = rn.read_snapshot(os.path.join(d, 'evol.nemo'), t)
    xv, m = xv[:NMW].copy(), m[:NMW]
    xv -= c['mw'][j]
    pot = agama.Potential(rn.fit(xv[NSTARS:], m[NSTARS:], 'halo'), rn.fit(xv[:NSTARS], m[:NSTARS], 'disk'))
    return xv, pot, c, j, d


def compute(t=0., k=9, i=2, outdir=None):
    xv, pot, c, j, d = _snapshot_potential(t, k, i)
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

def plot_circle_density(times=(-3., 0.), zmin=2., bmin=30.,
                        out='/home/ana/projects/glossary/plots/adopted_mw_stars_ELz_zb_circle_density.png'):
    """E-Lz density of the stars passing the solar-circle selection (as plot_sky(circle=True)), applied separately at
    each time: the same criteria, not the same particles. Shared colour scale."""
    from matplotlib.colors import LogNorm
    fig, ax = plt.subplots(1, len(times), figsize=(5.5 * len(times) + 1.5, 6), sharex=True, sharey=True)
    rng, bins = ((-1.5, 3.5), (-18, -6)), (100, 120)
    for a, t in zip(ax, times):
        d = np.load(LOCAL + 'mw_ELz_t%+.3f.npz' % t)
        s = circle_fraction(d['x'], zmin, bmin) > 0
        print('t = %g: %d of %d stars selected' % (t, s.sum(), len(s)))
        h = a.hist2d(d['L'][s, 2] / 1e3, d['E'][s] / 1e4, bins=bins, range=rng, norm=LogNorm(vmin=1, vmax=60),
                     cmap='Greys', rasterized=True)
        a.set_xlabel('$L_z$ [$10^3$ kpc km/s]  (model disc: $L_z > 0$)')
        a.text(0.02, 0.98, 't = %+.2f Gyr\n|z| > %g kpc, |b| > %g$^\\circ$ from the solar circle (%d stars)'
               % (t * 0.9778, zmin, bmin, s.sum()), transform=a.transAxes, va='top', fontsize=9)
    ax[0].set_ylabel('E (MW potential) [$10^4$ (km/s)$^2$]')
    fig.colorbar(h[3], ax=ax, label='stars per bin', fraction=0.03)
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(out)


def plot_circle_Ehist(times=(-3., 0.), zmin=2., bmin=30., bins=np.linspace(-16, -8, 70),
                      out='/home/ana/projects/glossary/plots/adopted_mw_stars_E_zb_circle.png'):
    """Energy histograms of the stars passing the solar-circle selection (as plot_sky(circle=True)), applied
    separately at each time (top to bottom); bin edges in 10^4 (km/s)^2."""
    fig, ax = plt.subplots(len(times), 1, figsize=(6.5, 3.2 * len(times)), sharex=True)
    for a, t in zip(ax, times):
        d = np.load(LOCAL + 'mw_ELz_t%+.3f.npz' % t)
        s = circle_fraction(d['x'], zmin, bmin) > 0
        n = a.hist(d['E'][s] / 1e4, bins=bins, histtype='step', color='k')[0]
        a.set_ylim(0, 1.25 * n.max())
        a.set_ylabel('stars per %.3g $\\times 10^4$ (km/s)$^2$' % (bins[1] - bins[0]))
        a.text(0.02, 0.96, 't = %+.2f Gyr\n|z| > %g kpc, |b| > %g$^\\circ$ from the solar circle (%d stars)'
               % (t * 0.9778, zmin, bmin, s.sum()), transform=a.transAxes, va='top', fontsize=9)
    ax[-1].set_xlabel('E (MW potential) [$10^4$ (km/s)$^2$]')
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(out)

def radial_periods(pot, ic, tint, dt=0.001, chunk=2000):
    """Radial periods [model time units] of the orbits from ic in the frozen potential pot: mean interval between
    successive pericentres (local minima of r on a grid of step dt) over an integration of length tint. Also the
    mean azimuthal period 2 pi tint / |total change of phi|. NaN where fewer than 2 pericentres."""
    import agama
    n = int(round(tint / dt)) + 1
    Tr, Tphi = np.full(len(ic), np.nan), np.full(len(ic), np.nan)
    for a in range(0, len(ic), chunk):
        orb = agama.orbit(potential=pot, ic=ic[a:a + chunk], time=tint, trajsize=n)
        for q, (tt, w) in enumerate(orb):
            r = np.linalg.norm(w[:, :3], axis=1)
            p = np.where((r[1:-1] < r[:-2]) & (r[1:-1] <= r[2:]))[0] + 1
            if len(p) > 1:
                Tr[a + q] = (tt[p[-1]] - tt[p[0]]) / (len(p) - 1)
            phi = np.unwrap(np.arctan2(w[:, 1], w[:, 0]))
            Tphi[a + q] = 2 * np.pi * tt[-1] / abs(phi[-1] - phi[0])
    return Tr, Tphi


def periods(t=0., k=9, i=2, tint_stars=3., tint_sgr=12., zmin=2., bmin=30.):
    """On OBS: radial (and azimuthal) periods, in the frozen MW potential at time t, of the stars passing the
    solar-circle selection at t and of Sgr (its centre relative to the MW centre, as a test particle: no mass loss,
    no dynamical friction, no LMC). Times in model units (x 0.9778 Gyr)."""
    xv, pot, c, j, d = _snapshot_potential(t, k, i)
    s = np.where(circle_fraction(xv[:NSTARS, :3], zmin, bmin) > 0)[0]
    Tr, Tphi = radial_periods(pot, xv[s], tint_stars)
    sgr = c['sgr'][j] - c['mw'][j]
    Tr_sgr, Tphi_sgr = radial_periods(pot, sgr[None], tint_sgr)
    out = os.path.join(d, 'mw_periods_t%+.3f.npz' % t)
    np.savez(out, index=s, Tr=Tr, Tphi=Tphi, Tr_sgr=Tr_sgr[0], Tphi_sgr=Tphi_sgr[0], sgr=sgr, t=t)
    print(out, '%d stars, %d without 2 pericentres; Sgr Tr = %.3f Gyr, Tphi = %.3f Gyr'
          % (len(s), np.isnan(Tr).sum(), Tr_sgr[0] * GYR, Tphi_sgr[0] * GYR))


def plot_periods(times=(-3., 0.), bins=np.linspace(0.05, 0.25, 70),
                 out='/home/ana/projects/glossary/plots/adopted_mw_stars_Tr_zb_circle.png'):
    """Histograms of the stars' radial periods [Gyr] (periods()), one panel per time, with Sgr's n:1 resonances at
    that time (T_Sgr / n, T_Sgr = Sgr's radial period) that fall within the range of the stars' periods."""
    fig, ax = plt.subplots(len(times), 1, figsize=(7.5, 3.4 * len(times)), sharex=True)
    for a, t in zip(ax, times):
        d = np.load(LOCAL + 'mw_periods_t%+.3f.npz' % t)
        Tr, T = d['Tr'] * GYR, d['Tr_sgr'] * GYR
        n = a.hist(Tr, bins=bins, histtype='step', color='k')[0]
        a.set_ylim(0, 1.3 * n.max())
        lo, hi = max(Tr.min(), bins[0]), min(Tr.max(), bins[-1])
        for q in range(2, 100):
            if T / q < lo:
                break
            if T / q <= hi:
                a.axvline(T / q, color='C3', lw=0.8, alpha=0.7)
                if T / q - T / (q + 1) > 0.004 * (bins[-1] - bins[0]) / 0.31 * 1.5:   # label only where they fit
                    a.text(T / q, 1.27 * n.max(), '%d:1' % q, color='C3', fontsize=7, ha='center', va='top', rotation=90,
                           bbox=dict(fc='w', ec='none', pad=0.5))
        a.set_ylabel('stars per %.3g Myr' % (1e3 * (bins[1] - bins[0])))
        a.text(0.98, 0.80, 't = %+.2f Gyr, %d stars\n|z| > 2 kpc, |b| > 30$^\\circ$ from the solar circle\n'
               'Sgr $T_r$ = %.3f Gyr; red lines: $T_r$ / n' % (t * GYR, len(Tr), T), transform=a.transAxes,
               va='top', ha='right', fontsize=9, bbox=dict(fc='w', ec='none'), zorder=5)
    ax[-1].set_xlim(bins[0], bins[-1])
    ax[-1].set_xlabel('radial period in the frozen MW potential [Gyr]')
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(out)

if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == 'periods':
        periods(float(sys.argv[2]))
    else:
        compute(float(sys.argv[1]) if len(sys.argv) > 1 else 0.)
