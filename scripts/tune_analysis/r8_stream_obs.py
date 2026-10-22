"""
Round-8 run-3 Sgr stars in V21's observed coordinates vs the VBE21 stream catalogue, with V21's own
fiducial snapshot (Zenodo 4038141) as a benchmark: B, distance, v_GSR, pmra, pmdec and the ratio of
reflex-corrected proper motions vs Lambda; binned medians and V21-style scores.
"""
import numpy as np, astropy.units as u, astropy.coordinates as coord
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('default')
XSUN = np.array([-8.1, 0., 0.02])
VSUN = np.array([12.9, 245.6, 7.8])
K = 4.740470463
CAT = '/home/ana/projects/glossary/data/vbe21_catalogue/catalogue.txt'
V21SNAP = '/home/ana/projects/glossary/data/tango/Sgr_snapshot/stars.txt'
RUN3 = '/home/ana/projects/glossary/output/v21_sim/tune/round08/run03/sgr_t0.npz'
SIG = dict(B=3., dist=0.1, vgsr=12.)       # V21's score uncertainties (distance: fractional)


def unit(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    return np.column_stack((np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)))


def sgr_rotation():
    """ICRS -> (Lambda, B) rotation, fitted (Kabsch) to the catalogue's own Lambda, Beta columns."""
    d = np.genfromtxt(CAT, names=True)
    a, b = unit(d['ra'], d['dec']), unit(d['Lambda'], d['Beta'])
    U, _, Vt = np.linalg.svd(b.T @ a)
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    return R, np.degrees(np.max(np.arccos(np.clip(np.sum((a @ R.T) * b, 1), -1, 1))))


R_SGR, _ = sgr_rotation()
fr = coord.Galactocentric(galcen_distance=np.hypot(*XSUN[[0, 2]]) * u.kpc, z_sun=XSUN[2] * 1e3 * u.pc,
                          galcen_v_sun=coord.CartesianDifferential(VSUN * u.km / u.s))


def observe(xv):
    """Galactocentric 6D -> dict of ra, dec, dist, pmra, pmdec, vlos, vgsr, Lambda, B, reflex-corrected pmL, pmB."""
    g = coord.SkyCoord(x=xv[:, 0] * u.kpc, y=xv[:, 1] * u.kpc, z=xv[:, 2] * u.kpc, v_x=xv[:, 3] * u.km / u.s,
                       v_y=xv[:, 4] * u.km / u.s, v_z=xv[:, 5] * u.km / u.s, frame=fr).icrs
    o = dict(ra=g.ra.deg, dec=g.dec.deg, dist=g.distance.kpc, pmra=g.pm_ra_cosdec.value, pmdec=g.pm_dec.value,
             vlos=g.radial_velocity.value)
    return derived(o)


def derived(o):
    """Add vgsr, Lambda, B and reflex-corrected proper motions in (Lambda, B) to a dict of ICRS observables."""
    ra, dec = np.radians(o['ra']), np.radians(o['dec'])
    n = unit(o['ra'], o['dec'])
    ea = np.column_stack((-np.sin(ra), np.cos(ra), np.zeros_like(ra)))
    ed = np.column_stack((-np.sin(dec) * np.cos(ra), -np.sin(dec) * np.sin(ra), np.cos(dec)))
    # solar velocity in ICRS
    vs = coord.SkyCoord(x=0 * u.kpc, y=0 * u.kpc, z=0 * u.kpc, v_x=0 * u.km / u.s, v_y=0 * u.km / u.s,
                        v_z=0 * u.km / u.s, frame=fr).icrs.velocity.d_xyz.to(u.km / u.s).value
    vs = -np.asarray(vs).ravel()          # velocity of the Sun relative to the Galactic rest frame (ICRS axes)
    o['vgsr'] = o['vlos'] + n @ vs
    # reflex-corrected tangential velocity [mas/yr], ICRS components
    pa = o['pmra'] + (ea @ vs) / (K * o['dist'])
    pd = o['pmdec'] + (ed @ vs) / (K * o['dist'])
    ns = n @ R_SGR.T
    L, B = np.arctan2(ns[:, 1], ns[:, 0]), np.arcsin(np.clip(ns[:, 2], -1, 1))
    eL = np.column_stack((-np.sin(L), np.cos(L), np.zeros_like(L))) @ R_SGR
    eB = np.column_stack((-np.sin(B) * np.cos(L), -np.sin(B) * np.sin(L), np.cos(B))) @ R_SGR
    vt = pa[:, None] * ea + pd[:, None] * ed
    o.update(Lambda=np.degrees(L), B=np.degrees(B), pmL=np.sum(vt * eL, 1), pmB=np.sum(vt * eB, 1))
    return o


def load_cat():
    d = np.genfromtxt(CAT, names=True)
    return derived(dict(ra=d['ra'], dec=d['dec'], dist=d['dist'], pmra=d['pmra'], pmdec=d['pmdec'], vlos=d['vlos']))


def check_v21_snapshot():
    """Recompute V21's tabulated observables from their x, v: how well does the frame reproduce them?"""
    s = np.genfromtxt(V21SNAP, names=True, max_rows=20000)
    o = observe(np.column_stack([s[k] for k in ('x', 'y', 'z', 'vx', 'vy', 'vz')]))
    for k, c in (('dist', 'distance'), ('pmra', 'pmra'), ('pmdec', 'pmdec'), ('vlos', 'vlos'), ('Lambda', 'Lambda'),
                 ('B', 'Beta')):
        dd = o[k] - s[c]
        if k == 'Lambda':
            dd = (dd + 180) % 360 - 180
        print('  V21 snapshot %-6s recomputed - tabulated: median %+.4f, max |.| %.4f' % (k, np.median(dd), np.abs(dd).max()))


def load_model(which):
    if which == 'run3':
        s = np.load(RUN3)
        xv = s['xv'][s['star']].astype(float)
    else:
        s = np.genfromtxt(V21SNAP, names=True)
        xv = np.column_stack([s[k] for k in ('x', 'y', 'z', 'vx', 'vy', 'vz')])
    return observe(xv)


def binned(o, key, edges, sel=None, nmin=5):
    sel = np.isfinite(o[key]) if sel is None else sel & np.isfinite(o[key])
    idx = np.digitize(o['Lambda'][sel], edges) - 1
    v = o[key][sel]
    med = np.full(len(edges) - 1, np.nan)
    for i in range(len(edges) - 1):
        w = idx == i
        if w.sum() >= nmin:
            med[i] = np.median(v[w])
    return med


QUANT = [('B', 'B [deg]'), ('dist', 'distance [kpc]'), ('vgsr', '$v_{GSR}$ [km/s]'), ('pmra', '$\\mu_\\alpha$ [mas/yr]'),
         ('pmdec', '$\\mu_\\delta$ [mas/yr]'), ('ratio', '$\\mu^*_B/\\mu^*_\\Lambda$ (reflex-corr.)')]


def compare(edges=np.linspace(-150, 150, 31), bcut=12.):
    cat = load_cat()
    models = dict(run3=load_model('run3'), v21=load_model('v21'))
    for o in [cat] + list(models.values()):
        o['ratio'] = o['pmB'] / o['pmL']
    sky = lambda o: np.abs(o['B']) < bcut
    res = {}
    for name, o in models.items():
        res[name] = {}
        for k, _ in QUANT:
            if k == 'ratio':
                # ratio of binned mean reflex-corrected proper motions, as V21
                mo = binned(o, 'pmB', edges, sky(o)) / binned(o, 'pmL', edges, sky(o))
                ob = binned(cat, 'pmB', edges) / binned(cat, 'pmL', edges)
            else:
                mo, ob = binned(o, k, edges, sky(o)), binned(cat, k, edges)
            res[name][k] = (mo, ob)
    return cat, models, res, edges


def score(res, edges):
    print('binned medians in %d Lambda bins (%g..%g deg), model stars with |B| < 12 deg' % (len(edges) - 1, edges[0], edges[-1]))
    print('%-6s %-8s %6s %9s %9s %9s %10s' % ('model', 'quantity', 'nbins', 'mean d', 'rms d', 'max|d|', 'chi2/bin'))
    for name, r in res.items():
        tot, nb = 0., 0
        for k, _ in QUANT:
            mo, ob = r[k]
            w = np.isfinite(mo) & np.isfinite(ob)
            d = mo[w] - ob[w]
            line = '%-6s %-8s %6d %9.3f %9.3f %9.3f' % (name, k, w.sum(), d.mean(), np.sqrt(np.mean(d**2)), np.abs(d).max())
            if k in SIG:
                s = SIG[k] * ob[w] if k == 'dist' else SIG[k]
                c = np.sum((d / s)**2)
                tot += c; nb += w.sum()
                line += ' %10.3f' % (c / w.sum())
            print(line)
        print('%-6s V21-style score (B 3 deg, distance 10%%, v 12 km/s): chi2 %.1f over %d bins = %.3f per bin' % (name, tot, nb, tot / nb))


def plot(cat, models, res, edges, out='/home/ana/projects/glossary/plots/tune_r8_run03_sgr_stream_obs.png'):
    lc = 0.5 * (edges[1:] + edges[:-1])
    fig, ax = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    ax = ax.ravel()
    o = models['run3']
    sel = np.abs(o['B']) < 12.
    for p, (k, lab) in enumerate(QUANT):
        a = ax[p]
        if k != 'ratio':
            a.plot(o['Lambda'][sel], o[k][sel], ',', color='0.6', alpha=0.4, rasterized=True)
            a.plot(cat['Lambda'], cat[k], '.', color='tab:red', ms=1, alpha=0.25, rasterized=True)
        for name, col, ls in (('run3', 'k', '-'), ('v21', 'tab:blue', '--')):
            a.plot(lc, res[name][k][0], ls, color=col, lw=1.8, label='r8 run 3' if name == 'run3' else 'V21 fiducial model')
        a.plot(lc, res['run3'][k][1], 'o-', color='tab:red', ms=3, lw=1, label='VBE21 catalogue')
        a.set_ylabel(lab)
        lim = dict(B=(-15, 15), dist=(0, 80), vgsr=(-250, 250), pmra=(-5, 3), pmdec=(-5, 3), ratio=(-0.6, 0.6))[k]
        a.set_ylim(*lim)
    ax[0].legend(fontsize=8, loc='lower left', markerscale=3)
    ax[0].text(0.02, 0.98, 'grey: r8 run 3 stars (|B| < 12$^\\circ$); red dots: catalogue; lines: binned medians',
               transform=ax[0].transAxes, va='top', fontsize=8)
    for a in ax[-2:]:
        a.set_xlabel('$\\Lambda$ [deg] (leading arm > 0)')
        a.set_xlim(-160, 160)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(out)


if __name__ == '__main__':
    R, err = sgr_rotation()
    print('Sgr frame fitted to catalogue Lambda/Beta: max mismatch %.2e deg' % err)
    check_v21_snapshot()
    cat, models, res, edges = compare()
    score(res, edges)
    plot(cat, models, res, edges)
