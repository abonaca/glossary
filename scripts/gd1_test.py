"""GD-1 feasibility test: forward models of GD-1 under different perturbers.

Usage (ipython):
    import gd1_test as gt
    pot = gt.load_potential('mwlmc')          # slow, build once
    s = gt.Stream('mwlmc', pot=pot)
    s.base()                                  # unperturbed model, saved to output/gd1_test/base.h5
    gt.plot_base()                            # compare static and MW+LMC base models with the data
"""

import os
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import h5py

import astropy.units as u
import astropy.coordinates as coord
from astropy.table import Table

import gala.coordinates as gc

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import diffrax

import streamsculptor
from streamsculptor import potential
from streamsculptor import JaxCoords as jc
from streamsculptor.main import usys   # streamsculptor's unit system (kpc, Myr, Msun)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # this file lives in scripts/
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
OUT_DIR = os.path.join(PROJECT_DIR, 'output', 'gd1_test')
PLOT_DIR = os.path.join(PROJECT_DIR, 'plots')

# solar motion used to define v_gsr (astropy Galactocentric default, as used by DESI GD-1 papers) [km/s]
V_SUN_GSR = np.array([12.9, 245.6, 7.78])

# progenitor fitted to GD-1 in the MW+LMC potential (notebooks/gd1_subhalo_impacts.ipynb)
PROG_MWLMC = np.array([-11.88431447, 1.61212652, 7.14988315, -0.09, -0.205, -0.1])


# Data #

def load_members():
    """GD-1 members with membership probability >= 0.75 (Starkman+2024)"""

    return Table.read(os.path.join(DATA_DIR, 'gd1_members_0.75.fits'))

def n_members():
    """Number of GD-1 members in the catalog"""

    return len(load_members())

def members_gd1_frame():
    """GD-1 members in the GD-1 frame: phi1, phi2, distance, proper motions"""

    t = load_members()
    cg = t['coord'].transform_to(gc.GD1Koposov10())

    return dict(phi1=cg.phi1.wrap_at(180*u.deg).deg, phi2=cg.phi2.deg, dist=cg.distance.to(u.kpc).value,
                pm1=cg.pm_phi1_cosphi2.to(u.mas/u.yr).value, pm2=cg.pm_phi2.to(u.mas/u.yr).value)


def load_hectochelle(members_only=True):
    """Bonaca+2020 MMT/Hectochelle spectroscopy of GD-1 (github.com/abonaca/spur_rv, data/catalog.fits).
    Returns GD-1 frame coordinates, Gaia proper motions, and heliocentric radial velocities [km/s]"""

    t = Table.read(os.path.join(DATA_DIR, 'bonaca2020_hectochelle', 'catalog.fits'))
    if members_only:
        t = t[t['mem']]

    c = coord.ICRS(ra=np.array(t['ra'])*u.deg, dec=np.array(t['dec'])*u.deg,
                   pm_ra_cosdec=np.array(t['pmra'])*u.mas/u.yr, pm_dec=np.array(t['pmdec'])*u.mas/u.yr)
    cg = c.transform_to(gc.GD1Koposov10())

    return dict(phi1=cg.phi1.wrap_at(180*u.deg).deg, phi2=cg.phi2.deg,
                pm1=cg.pm_phi1_cosphi2.to(u.mas/u.yr).value, pm2=cg.pm_phi2.to(u.mas/u.yr).value,
                vr=np.array(t['Vrad']), vr_err=np.array(t['std_Vrad']))


def load_desi(p_min=0.5):
    """Jarvis+2026 DESI DR2 GD-1 members (Table 7, zenodo.org/records/19889980, version 3).
    Members have P_THIN + P_COCOON > p_min (p_min=0.5 gives the 608 members of the paper).
    Returns GD-1 frame coordinates, distances from DISTMOD, Gaia proper motions, heliocentric radial velocities [km/s],
    and the thin-stream and cocoon membership probabilities"""

    t = Table.read(os.path.join(DATA_DIR, 'jarvis2026_desi_dr2', 'Table7.fits'))
    t = t[(t['P_THIN'] + t['P_COCOON']) > p_min]

    c = coord.ICRS(ra=np.array(t['RA'])*u.deg, dec=np.array(t['Dec'])*u.deg,
                   pm_ra_cosdec=np.array(t['PM_RA'])*u.mas/u.yr, pm_dec=np.array(t['PM_DEC'])*u.mas/u.yr)
    cg = c.transform_to(gc.GD1Koposov10())

    return dict(source_id=np.array(t['SOURCE_ID']), phi1=cg.phi1.wrap_at(180*u.deg).deg, phi2=cg.phi2.deg,
                dist=10**(np.array(t['DISTMOD'])/5 - 2), pm1=cg.pm_phi1_cosphi2.to(u.mas/u.yr).value,
                pm2=cg.pm_phi2.to(u.mas/u.yr).value, vr=np.array(t['V_LOS']), vr_err=np.array(t['V_ERR']),
                p_thin=np.array(t['P_THIN']), p_cocoon=np.array(t['P_COCOON']))


# Setup #

def load_potential(kind='mwlmc'):
    """Base Milky Way potential: 'static' (gala MilkyWayPotential) or 'mwlmc' (time-dependent MW+LMC)"""

    if kind=='static':
        return potential.GalaMilkyWayPotential(units=usys)
    elif kind=='mwlmc':
        return potential.MW_LMC_Potential(units=usys)
    else:
        raise ValueError(f"unknown potential kind: {kind}")

def progenitor(kind='mwlmc'):
    """Present-day GD-1 progenitor [kpc, kpc/Myr], fitted separately in each potential"""

    if kind=='static':
        prog = np.load(os.path.join(DATA_DIR, 'GD1_progenitor.npy'), allow_pickle=True).item()
        pos = prog.pos.xyz.to(u.kpc).value
        vel = prog.vel.d_xyz.to(u.kpc/u.Myr).value
        return np.hstack([pos, vel])
    elif kind=='mwlmc':
        return PROG_MWLMC.copy()
    else:
        raise ValueError(f"unknown potential kind: {kind}")


# Coordinates #

def to_gd1_frame(w):
    """Simulation cartesian phase space [kpc, kpc/Myr] -> GD-1 frame observables"""

    w = jnp.asarray(w)
    ra, dec, d = jax.vmap(jc.simcart_to_icrs)(w[:,:3])
    pmra, pmdec, vr = jax.vmap(jc.simvel_to_ICRS)(w[:,:3], w[:,3:])

    ceq = coord.ICRS(ra=np.asarray(ra)*u.deg, dec=np.asarray(dec)*u.deg, distance=np.asarray(d)*u.kpc,
                     pm_ra_cosdec=np.asarray(pmra)*u.mas/u.yr, pm_dec=np.asarray(pmdec)*u.mas/u.yr,
                     radial_velocity=(np.asarray(vr)*u.kpc/u.Myr).to(u.km/u.s))
    cg = ceq.transform_to(gc.GD1Koposov10())

    # v_gsr: heliocentric radial velocity plus the projected solar motion
    cgal = ceq.transform_to(coord.Galactic())
    n = np.vstack([np.cos(cgal.b)*np.cos(cgal.l), np.cos(cgal.b)*np.sin(cgal.l), np.sin(cgal.b)]).T
    vr_kms = cg.radial_velocity.to(u.km/u.s).value

    return dict(phi1=cg.phi1.wrap_at(180*u.deg).deg, phi2=cg.phi2.deg, dist=cg.distance.to(u.kpc).value,
                pm1=cg.pm_phi1_cosphi2.to(u.mas/u.yr).value, pm2=cg.pm_phi2.to(u.mas/u.yr).value,
                vr=vr_kms, vgsr=vr_kms + n @ V_SUN_GSR)


# Stream models #

class Stream:
    """GD-1 stream model in a given base potential.

    kind: 'static' or 'mwlmc', sets the progenitor and (if pot not given) the base potential
    pot: base potential; pass a prebuilt one to avoid rebuilding the MW+LMC splines
    dt_release: time between releases [Myr]; each release adds one star at each Lagrange point
    """

    def __init__(self, kind='mwlmc', pot=None, dt_release=1., t_age=2500., t_dissolve=-300., Msat=1e4,
                 rtol=1e-6, atol=1e-6):
        self.kind = kind
        self.pot = load_potential(kind) if pot is None else pot
        self.prog_w0 = progenitor(kind)
        self.dt_release = dt_release
        self.t_age = t_age
        self.t_dissolve = t_dissolve
        self.Msat = Msat
        self.rtol = rtol
        self.atol = atol

        # stripping times from -t_age to t_dissolve every dt_release; gen_stream releases one leading
        # and one trailing star at each, and ts[-1] is the final integration time
        n_strip = int(round((t_dissolve + t_age) / dt_release)) + 1
        self.ts = jnp.hstack([jnp.linspace(-t_age, t_dissolve, n_strip), jnp.array([0.0])])
        self.n_stars = 2 * n_strip

        self.base_model = None

    def initial_conditions(self, pot=None):
        """Progenitor phase space at -t_age, integrated back in pot (default: base potential)"""

        pot = self.pot if pot is None else pot
        return pot.integrate_orbit(w0=jnp.asarray(self.prog_w0), t0=0.0, t1=-self.t_age,
                                   ts=jnp.array([-self.t_age])).ys[0]

    def generate(self, pot=None, seed=532):
        """Particle-spray stream in pot (default: base potential). Returns the model dict"""

        pot = self.pot if pot is None else pot
        ic = self.initial_conditions(pot=pot)
        lead, trail = pot.gen_stream_vmapped(prog_w0=ic, ts=self.ts, seed_num=seed, Msat=self.Msat,
                                             atol=self.atol, rtol=self.rtol, solver=diffrax.Dopri5())

        n = len(self.ts) - 1
        model = dict(w=np.vstack([np.asarray(lead), np.asarray(trail)]),
                     t_release=np.hstack([np.asarray(self.ts[:-1])]*2),
                     arm=np.hstack([np.ones(n, dtype=int), -np.ones(n, dtype=int)]),
                     meta=self.metadata(seed=seed))

        return model

    def base(self, seed=532, save=True):
        """Unperturbed stream in the base potential; cached on the object and optionally saved"""

        self.base_model = self.generate(seed=seed)
        self.base_model['meta']['perturber'] = 'none'
        if save:
            save_model(self.base_model, name='base', group=self.kind)

        return self.base_model

    def metadata(self, seed):
        """Setup needed to reproduce a model"""

        return dict(kind=self.kind, n_stars=self.n_stars, dt_release=self.dt_release, t_age=self.t_age,
                    t_dissolve=self.t_dissolve, Msat=self.Msat, seed=seed, rtol=self.rtol, atol=self.atol,
                    prog_w0=np.asarray(self.prog_w0), streamsculptor_commit=streamsculptor_commit(),
                    jax_version=jax.__version__)


# Storage #

def streamsculptor_commit():
    """Git commit of the streamsculptor checkout in use"""

    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                       cwd=os.path.dirname(streamsculptor.__file__), text=True).strip()
    except Exception:
        return 'unknown'

def model_path(name):
    return os.path.join(OUT_DIR, f'{name}.h5')

def save_model(model, name, group):
    """Write a model as a group in output/gd1_test/<name>.h5, replacing an existing group of that name.
    Datasets: w (N x 6, kpc and kpc/Myr), t_release (Myr), arm (+1 leading, -1 trailing); metadata as attributes"""

    os.makedirs(OUT_DIR, exist_ok=True)

    with h5py.File(model_path(name), 'a') as f:
        if group in f:
            del f[group]
        g = f.create_group(group)
        g.create_dataset('w', data=model['w'])
        g.create_dataset('t_release', data=model['t_release'])
        g.create_dataset('arm', data=model['arm'])
        for k, v in model['meta'].items():
            g.attrs[k] = v

def load_model(name, group):
    """Read a model saved with save_model"""

    with h5py.File(model_path(name), 'r') as f:
        g = f[group]
        model = dict(w=g['w'][:], t_release=g['t_release'][:], arm=g['arm'][:], meta=dict(g.attrs))

    return model

def list_models(name):
    """Groups stored in output/gd1_test/<name>.h5"""

    with h5py.File(model_path(name), 'r') as f:
        return list(f.keys())


# Base models #

def run_base(seed=532):
    """Generate and save the base models in both the static and MW+LMC potentials"""

    for kind in ['static', 'mwlmc']:
        Stream(kind).base(seed=seed)

def plot_base(save=True):
    """Static and MW+LMC base models compared with the GD-1 members"""

    data = members_gd1_frame()
    models = {kind: to_gd1_frame(load_model('base', kind)['w']) for kind in ['static', 'mwlmc']}
    colors = dict(static='tab:blue', mwlmc='tab:red')
    labels = dict(static='static MW', mwlmc='MW+LMC')

    ylist = ['phi2', 'dist', 'pm1', 'pm2', 'vr']
    ylabels = [r'$\phi_2$ [deg]', 'd [kpc]', r'$\mu_{\phi_1}$ [mas/yr]', r'$\mu_{\phi_2}$ [mas/yr]', '$V_r$ [km/s]']

    fig, ax = plt.subplots(len(ylist), 1, figsize=(10,11), sharex=True)
    for i, (k, label) in enumerate(zip(ylist, ylabels)):
        plt.sca(ax[i])
        if k in data:
            plt.plot(data['phi1'], data[k], 'ko', ms=2, mew=0, alpha=0.6, label='GD-1 members' if i==0 else None)
        for kind, m in models.items():
            plt.plot(m['phi1'], m[k], 'o', color=colors[kind], ms=1.5, mew=0, alpha=0.5, zorder=0,
                     label=labels[kind] if i==0 else None)
        plt.ylabel(label)

    ax[0].set_ylim(-6, 4)
    ax[1].set_ylim(5, 15)
    ax[0].legend(loc='lower left', markerscale=4, fontsize='small', ncol=3)
    plt.xlim(-100, 30)
    plt.xlabel(r'$\phi_1$ [deg]')
    plt.tight_layout(h_pad=0)

    if save:
        os.makedirs(PLOT_DIR, exist_ok=True)
        plt.savefig(os.path.join(PLOT_DIR, 'gd1_test_base.png'), dpi=150)

    return fig

def track(obs, key, width=2., step=0.1, phi1_range=(-110, 40), n_min=5):
    """Smooth track of coordinate key vs phi1: running median in overlapping phi1 windows of the given width,
    centered every step degrees, linearly interpolated. Windows with fewer than n_min stars are skipped.
    Returns a function of phi1 (NaN outside the covered range)"""

    isort = np.argsort(obs['phi1'])
    phi1, y = obs['phi1'][isort], obs[key][isort]

    grid = np.arange(phi1_range[0], phi1_range[1] + 0.5*step, step)
    i_lo = np.searchsorted(phi1, grid - 0.5*width, side='left')
    i_hi = np.searchsorted(phi1, grid + 0.5*width, side='right')

    centers, medians = [], []
    for c, lo, hi in zip(grid, i_lo, i_hi):
        if hi - lo >= n_min:
            centers.append(c)
            medians.append(np.median(y[lo:hi]))
    centers, medians = np.array(centers), np.array(medians)

    return lambda x: np.interp(x, centers, medians, left=np.nan, right=np.nan)

def plot_base_residuals(ref='mwlmc', save=True):
    """Static and MW+LMC base models, GD-1 members, Hectochelle (Bonaca+2020) and DESI DR2 (Jarvis+2026) members,
    relative to the smooth track of the reference model"""

    data = members_gd1_frame()
    spec = load_hectochelle()
    desi = load_desi()
    phi2_lim = (np.floor(np.min(spec['phi2'])*2)/2, np.ceil(np.max(spec['phi2'])*2)/2)
    models = {kind: to_gd1_frame(load_model('base', kind)['w']) for kind in ['static', 'mwlmc']}
    colors = dict(static='tab:blue', mwlmc='tab:red')
    labels = dict(static='static MW', mwlmc='MW+LMC')

    ylist = ['phi2', 'dist', 'pm1', 'pm2', 'vr']
    ylabels = [r'$\Delta\phi_2$ [deg]', r'$\Delta$d [kpc]', r'$\Delta\mu_{\phi_1}$ [mas/yr]',
               r'$\Delta\mu_{\phi_2}$ [mas/yr]', r'$\Delta V_r$ [km/s]']
    ylims = [(-2, 2), (-1.5, 1.5), (-1.5, 1.5), (-1, 1), (-20, 20)]

    fig, ax = plt.subplots(len(ylist), 1, figsize=(10,11), sharex=True)
    for i, (k, label) in enumerate(zip(ylist, ylabels)):
        plt.sca(ax[i])
        ref_track = track(models[ref], k)
        if k in data:
            plt.plot(data['phi1'], data[k] - ref_track(data['phi1']), 'ko', ms=2, mew=0, alpha=0.6,
                     label='GD-1 members' if i==0 else None)
        if k in spec:
            dy = spec[k] - ref_track(spec['phi1'])
            if k=='vr':
                plt.errorbar(spec['phi1'], dy, yerr=spec['vr_err'], fmt='none', ecolor='0.3', elinewidth=1, zorder=3)
            im = plt.scatter(spec['phi1'], dy, c=spec['phi2'], cmap='viridis', vmin=phi2_lim[0], vmax=phi2_lim[1],
                             marker='s', s=18, edgecolors='k', linewidths=0.5, zorder=4,
                             label='Hectochelle (Bonaca+2020)' if i==0 else None)
        if k in desi:
            thin = desi['p_thin'] > desi['p_cocoon']
            dy = desi[k] - ref_track(desi['phi1'])
            plt.plot(desi['phi1'][thin], dy[thin], 'D', color='w', mec='k', mew=0.7, ms=3.5, zorder=3,
                     label='DESI DR2 thin (Jarvis+2026)' if i==0 else None)
            plt.plot(desi['phi1'][~thin], dy[~thin], 'D', color='0.6', mec='0.6', mew=0.7, ms=2.5, zorder=2,
                     label='DESI DR2 cocoon' if i==0 else None)
        for kind, m in models.items():
            plt.plot(m['phi1'], m[k] - ref_track(m['phi1']), 'o', color=colors[kind], ms=1.5, mew=0, alpha=0.5,
                     zorder=0, label=labels[kind] if i==0 else None)
        plt.axhline(0, color='0.5', lw=0.5)
        plt.ylim(ylims[i])
        plt.ylabel(label)

    fig.legend(*ax[0].get_legend_handles_labels(), loc='upper center', ncol=3, markerscale=2, fontsize='small', frameon=False)
    plt.xlim(-100, 30)
    plt.xlabel(r'$\phi_1$ [deg]')
    plt.tight_layout(h_pad=0, rect=(0, 0, 1, 0.95))
    fig.colorbar(im, ax=ax, label=r'Hectochelle $\phi_2$ [deg]', fraction=0.02, pad=0.01)

    if save:
        os.makedirs(PLOT_DIR, exist_ok=True)
        plt.savefig(os.path.join(PLOT_DIR, f'gd1_test_base_residuals_{ref}.png'), dpi=150)

    return fig

def footprint_counts():
    """Number of model stars within the phi1 range of the catalog, for each base model"""

    data = members_gd1_frame()
    phi1_min, phi1_max = np.min(data['phi1']), np.max(data['phi1'])
    out = {}
    for kind in ['static', 'mwlmc']:
        m = to_gd1_frame(load_model('base', kind)['w'])
        out[kind] = int(np.sum((m['phi1']>=phi1_min) & (m['phi1']<=phi1_max)))

    return out
