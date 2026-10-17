"""
Structural evolution of the live Milky Way model in run R0 (MW alone): stellar density maps near the
disk plane and time series of the expansion coefficients of the disk, bulge and halo.

Particle order in R0 (v21_ics.assemble_runs): stars first (NSTARS), then the halo. Within the stars,
agama.sampleOrbitLibrary emits particles grouped by parent orbit and the bulge orbits came first,
so the first NBULGE particles are the bulge (non-rotating) and the rest the disk; the switch in
Lz/L happens at index ~212000 (bulge 1.31e10 Msun, disk 4.89e10 vs the model's 1.2e10 and 5.0e10;
the boundary is fuzzy at the 0.2% level). gyrfalcON preserves the particle order.

Expansions (about the stellar centre): halo and bulge Multipole (lmax=4), disk CylSpline (mmax=4).
Coefficient amplitudes for each (l, |m|) or |m| are sqrt(c_m^2 + c_-m^2) of the potential
coefficients, normalised by the monopole (l=0 / m=0) at the same radius; for l=0 / m=0 the
relative change since the first snapshot is shown instead.

Usage (ipython, on OBS where the snapshots are):
    import r0_structure as rs
    maps = rs.stellar_maps(EVOL)                 # ~0.5 Gyr apart
    coefs = rs.coefficient_series(EVOL)          # every snapshot
    rs.plot_maps(maps); rs.plot_coefficients(coefs)
"""
import os
import numpy as np
import agama
import r0_noise as rn

ROOT = rn.ROOT
EVOL = '/carnegie/nobackup/users/abonaca/v21_sim/runs/R0/R0_evol.nemo'
NSTARS = rn.NSTARS
NBULGE = 212000
GYR = 0.9778                       # Gyr per time unit
EXP = dict(halo=dict(type='Multipole', lmax=4, symmetry='n', gridSizeR=25, rmin=0.38, rmax=262.6),
           bulge=dict(type='Multipole', lmax=4, symmetry='n', gridSizeR=20, rmin=0.05, rmax=20.),
           disk=dict(type='CylSpline', mmax=4, symmetry='n', gridSizeR=20, gridSizez=20,
                     Rmin=0.2, Rmax=60., zmin=0.05, zmax=20.))
SLICES = dict(bulge=slice(0, NBULGE), disk=slice(NBULGE, NSTARS), halo=slice(NSTARS, None))
# radii (kpc) at which coefficient time series are shown
RADII = dict(halo=(5., 20., 50., 120.), bulge=(0.3, 0.7, 1.5, 3.), disk=(2., 4., 8., 12.))


def times_every(evol, dt_gyr=0.5):
    """Snapshot times closest to multiples of dt_gyr since the start, plus the last snapshot."""
    t = rn.snapshot_times(evol)
    want = np.arange(t[0], t[-1], dt_gyr / GYR)
    sel = sorted(set([int(np.argmin(abs(t - w))) for w in want] + [len(t) - 1]))
    return t[sel]


def _load(evol, t):
    xv, m = rn.read_snapshot(evol, t)
    c, v = rn.centre(xv[:NSTARS], m[:NSTARS])
    xv[:, :3] -= c
    xv[:, 3:6] -= v
    return xv, m, np.hstack((c, v))


# ------------------------------------------------------------------------------------- density maps
def stellar_maps(evol=EVOL, times=None, extent=12., nbin=240, zcut=0.5, zext=4., nbinz=80):
    """
    Face-on surface density of disk+bulge particles with |z| < zcut, the same for bulge particles
    only, and the edge-on (x-z) surface density of all stars, at `times` (default every 0.5 Gyr).
    Densities in Msun/kpc^2.
    """
    times = times_every(evol) if times is None else np.atleast_1d(times)
    edges = np.linspace(-extent, extent, nbin + 1)
    zedges = np.linspace(-zext, zext, nbinz + 1)
    pix, zpix = (edges[1] - edges[0])**2, (edges[1] - edges[0]) * (zedges[1] - zedges[0])
    out = dict(times=times, edges=edges, zedges=zedges, zcut=zcut, face=[], face_bulge=[], edge=[], centre=[])
    for t in times:
        print('maps t = %.4f' % t, flush=True)
        xv, m, cen = _load(evol, t)
        s, mm = xv[:NSTARS], m[:NSTARS] * 232500.
        near = abs(s[:, 2]) < zcut
        out['face'].append(np.histogram2d(s[near, 0], s[near, 1], [edges, edges], weights=mm[near])[0] / pix)
        nb = near & (np.arange(NSTARS) < NBULGE)
        out['face_bulge'].append(np.histogram2d(s[nb, 0], s[nb, 1], [edges, edges], weights=mm[nb])[0] / pix)
        out['edge'].append(np.histogram2d(s[:, 0], s[:, 2], [edges, zedges], weights=mm)[0] / zpix)
        out['centre'].append(cen)
    for k in ('face', 'face_bulge', 'edge', 'centre'):
        out[k] = np.array(out[k])
    return out


def azimuthal_residual(face, edges, nr=30, rebin=4):
    """Sigma / <Sigma>_phi - 1 for a face-on map rebinned by `rebin` (azimuthal mean in radial rings)."""
    n = face.shape[0] // rebin
    face = face[:n * rebin, :n * rebin].reshape(n, rebin, n, rebin).mean(axis=(1, 3))
    edges = edges[::rebin][:n + 1]
    xc = 0.5 * (edges[1:] + edges[:-1])
    R = np.hypot(*np.meshgrid(xc, xc, indexing='ij'))
    rb = np.linspace(0, edges[-1], nr + 1)
    ring = np.digitize(R, rb) - 1
    mean = np.array([face[ring == i].mean() if np.any(ring == i) else np.nan for i in range(nr + 1)])
    with np.errstate(invalid='ignore', divide='ignore'):
        res = face / mean[ring] - 1
    res[R > edges[-1]] = np.nan
    return res


def plot_maps(maps, fname=os.path.join(ROOT, 'plots', 'r0_stellar_maps.png')):
    """Rows: face-on stars (|z|<zcut), m!=0 residual, face-on bulge particles, edge-on stars."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    n = len(maps['times'])
    e, ze = maps['edges'], maps['zedges']
    fig, ax = plt.subplots(4, n, figsize=(2.3 * n, 2.3 * 3 + 1.2), squeeze=False,
                           gridspec_kw=dict(height_ratios=[1, 1, 1, 0.45]))
    vmax = np.percentile(maps['face'][0], 99.9)
    vb = np.percentile(maps['face_bulge'][0], 99.9)
    ve = np.percentile(maps['edge'][0], 99.9)
    for j, t in enumerate(maps['times']):
        ext = [e[0], e[-1], e[0], e[-1]]
        ax[0, j].imshow(maps['face'][j].T, origin='lower', extent=ext, norm=LogNorm(vmax * 1e-3, vmax), cmap='magma')
        ax[1, j].imshow(azimuthal_residual(maps['face'][j], e).T, origin='lower', extent=ext, vmin=-0.5, vmax=0.5,
                        cmap='RdBu_r')
        ax[2, j].imshow(maps['face_bulge'][j].T, origin='lower', extent=ext, norm=LogNorm(vb * 1e-3, vb), cmap='magma')
        ax[3, j].imshow(maps['edge'][j].T, origin='lower', extent=[e[0], e[-1], ze[0], ze[-1]],
                        norm=LogNorm(ve * 1e-4, ve), cmap='magma', aspect='auto')
        ax[0, j].set_title('%.2f Gyr (t=%.3f)' % ((t - maps['times'][0]) * GYR, t), fontsize=9)
        for a in ax[:, j]:
            a.tick_params(labelsize=7)
    lab = ['stars, |z|<%g kpc' % maps['zcut'], r'$\Sigma/\langle\Sigma\rangle_\phi - 1$',
           'bulge particles, |z|<%g' % maps['zcut'], 'stars, edge-on (x-z)']
    for i, l in enumerate(lab):
        ax[i, 0].set_ylabel(l, fontsize=8)
    fig.tight_layout()
    fig.savefig(fname, dpi=110)
    print('saved', fname)
    return fig


# ------------------------------------------------------------------------------ coefficient series
def parse_coefs(pot, tmp):
    """Potential coefficients of a Multipole (array radius x (l,m)) or CylSpline (dict m -> R x z)."""
    pot.export(tmp)
    lines = open(tmp).read().split('\n')
    os.remove(tmp)
    i = lines.index('#Phi')
    if 'type=Multipole' in lines:
        rows = []
        for l in lines[i + 2:]:
            if not l.strip():
                break
            rows.append([float(x) for x in l.split()])
        rows = np.array(rows)
        return dict(r=rows[:, 0], c=rows[:, 1:])
    out, j = {}, i + 1
    while j < len(lines) and lines[j].strip() and not lines[j].startswith('#d'):
        m = int(lines[j].split()[0])
        z = np.array([float(x) for x in lines[j + 1].split()[1:]])
        rows, j = [], j + 2
        while j < len(lines) and lines[j].strip() and not lines[j].startswith('#') and '#m' not in lines[j]:
            rows.append([float(x) for x in lines[j].split()])
            j += 1
        rows = np.array(rows)
        out[m] = rows[:, 1:]
        R = rows[:, 0]
        while j < len(lines) and not lines[j].strip():
            j += 1
            if j < len(lines) and lines[j].startswith('#d'):
                break
        if j < len(lines) and lines[j].startswith('#d'):
            break
    return dict(R=R, z=z, c=out)


def coefficient_series(evol=EVOL, times=None, split_seed=0):
    """
    Expansion coefficients of halo, bulge and disk for every snapshot (or `times`), plus a half-split
    noise estimate of each coefficient from the first snapshot.
    """
    times = rn.snapshot_times(evol) if times is None else np.atleast_1d(times)
    tmp = os.path.join(os.path.dirname(evol), '_coefs.ini')
    out = dict(times=times, centre=[], halo=[], bulge=[], disk=[])
    for k, t in enumerate(times):
        print('coefs t = %.4f' % t, flush=True)
        xv, m, cen = _load(evol, t)
        out['centre'].append(cen)
        for comp, sl in SLICES.items():
            out[comp].append(parse_coefs(agama.Potential(particles=(xv[sl, :3], m[sl]), **EXP[comp]), tmp))
        if k == 0:
            half = np.random.default_rng(split_seed).permutation(len(m)) % 2 == 0
            for comp, sl in SLICES.items():
                sel = np.zeros(len(m), bool)
                sel[sl] = True
                a = parse_coefs(agama.Potential(particles=(xv[sel & half, :3], 2 * m[sel & half]), **EXP[comp]), tmp)
                b = parse_coefs(agama.Potential(particles=(xv[sel & ~half, :3], 2 * m[sel & ~half]), **EXP[comp]), tmp)
                out[comp + '_split'] = (a, b)
    out['centre'] = np.array(out['centre'])
    return out


LM = [(l, m) for l in range(5) for m in range(-l, l + 1)]


def _multipole_amp(c, l, m):
    """sqrt(c_{l,m}^2 + c_{l,-m}^2) (or |c_{l,0}|) along radius."""
    i, j = LM.index((l, m)), LM.index((l, -m))
    return np.abs(c[:, i]) if m == 0 else np.hypot(c[:, i], c[:, j])


def _amp_series(coefs, comp, r):
    """Dict (l or m) -> array (time, |m| or l...) of normalised amplitudes at radius r."""
    t = coefs['times']
    if comp in ('halo', 'bulge'):
        rr = coefs[comp][0]['r']
        k = int(np.argmin(abs(np.log(rr / r))))
        c = np.array([s['c'][k] for s in coefs[comp]])[:, None, :]        # time x 1 x (l,m)
        mono = c[:, 0, 0]
        res = {0: (mono / mono[0] - 1)[:, None]}
        for l in range(1, 5):
            res[l] = np.array([[_multipole_amp(c[n], l, m)[0] / abs(mono[n]) for m in range(l + 1)]
                               for n in range(len(t))])
        return res, rr[k]
    R = coefs['disk'][0]['R']
    z = coefs['disk'][0]['z']
    k, kz = int(np.argmin(abs(R - r))), int(np.argmin(abs(z)))
    phi = lambda s, m: s['c'][m][k, kz] if m in s['c'] else 0.
    mono = np.array([phi(s, 0) for s in coefs['disk']])
    res = {0: (mono / mono[0] - 1)[:, None]}
    for m in range(1, 5):
        res[m] = np.array([np.hypot(phi(s, m), phi(s, -m)) / abs(phi(s, 0)) for s in coefs['disk']])[:, None]
    return res, R[k]


def _split_noise(coefs, comp, r):
    """Half-split noise of the normalised amplitudes at radius r (|a-b|/2), same layout as _amp_series."""
    a, b = coefs[comp + '_split']
    fake = lambda s: dict(coefs, times=coefs['times'][:1], **{comp: [s]})
    ra, _ = _amp_series(fake(a), comp, r)
    rb, _ = _amp_series(fake(b), comp, r)
    return {k: np.abs(ra[k][0] - rb[k][0]) / 2 for k in ra}


def disk_bar_phase(coefs, r=4.):
    """Phase (rad) of the m=2 disk coefficient at radius r in the plane, unwrapped; pattern speed from its slope."""
    R, z = coefs['disk'][0]['R'], coefs['disk'][0]['z']
    k, kz = int(np.argmin(abs(R - r))), int(np.argmin(abs(z)))
    ph = np.array([np.arctan2(s['c'][-2][k, kz], s['c'][2][k, kz]) / 2 for s in coefs['disk']])
    return np.unwrap(2 * ph) / 2


def plot_coefficients(coefs, comp, fname=None):
    """Rows: radii; columns: l (multipole) or |m| (disk); lines: |m| within l; dotted: half-split noise."""
    import matplotlib.pyplot as plt
    fname = fname or os.path.join(ROOT, 'plots', 'r0_coefs_%s.png' % comp)
    radii = RADII[comp]
    t = (coefs['times'] - coefs['times'][0]) * GYR
    fig, ax = plt.subplots(len(radii), 5, figsize=(16, 2.6 * len(radii)), sharex=True, squeeze=False)
    for i, r in enumerate(radii):
        res, rk = _amp_series(coefs, comp, r)
        noise = _split_noise(coefs, comp, r) if comp + '_split' in coefs else None
        for col in range(5):
            a = ax[i, col]
            y = res[col]
            for j in range(y.shape[1]):
                line, = a.plot(t, y[:, j], lw=1, label=('m=%d' % j) if comp != 'disk' else None)
                if noise is not None and col > 0:
                    a.axhline(noise[col][j], color=line.get_color(), ls=':', lw=0.8)
            if col > 0:
                a.set_yscale('log')
                a.yaxis.set_minor_formatter(plt.NullFormatter())
            if i == 0:
                a.set_title(('l=%d' if comp != 'disk' else 'm=%d') % col +
                            (' (relative change)' if col == 0 else ' / monopole'), fontsize=9)
            if col == 0:
                a.set_ylabel('%s r=%.2g kpc' % (comp, rk) if comp != 'disk' else 'disk R=%.2g, z=0' % rk, fontsize=8)
            if i == len(radii) - 1:
                a.set_xlabel('time since start [Gyr]')
            a.tick_params(labelsize=7)
            if comp != 'disk' and i == 0 and col > 0:
                a.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(fname, dpi=110)
    print('saved', fname)
    return fig
