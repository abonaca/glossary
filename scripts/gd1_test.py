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
import interpax
from typing import Any

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
                pm2=cg.pm_phi2.to(u.mas/u.yr).value, pm1_err=np.array(t['PM_PHI1_ERR']),
                pm2_err=np.array(t['PM_PHI2_ERR']), vr=np.array(t['V_LOS']), vr_err=np.array(t['V_ERR']),
                p_thin=np.array(t['P_THIN']), p_cocoon=np.array(t['P_COCOON']))


# Setup #

# MW and LMC models computed with AGAMA (scripts/mw_lmc_agama.py; Vasiliev, Belokurov & Erkal 2021 models)
AGAMA_MWLMC_FILE = os.path.join(DATA_DIR, 'mw_lmc_agama', 'mw_lmc_1.5e+11.npz')

class RadialTablePotential(potential.Potential):
    """Spherical potential tabulated in radius (monotonic cubic spline)"""
    spl: Any

    def __init__(self, r, phi, units=usys):
        super().__init__(units)
        self.spl = interpax.Interpolator1D(x=jnp.asarray(r), f=jnp.asarray(phi), method='monotonic')

    def potential(self, xyz, t):
        return self.spl(jnp.sqrt(jnp.sum(xyz**2)))

class AxisymTablePotential(potential.Potential):
    """Axisymmetric potential tabulated on a grid in u=asinh(R/a), v=asinh(z/b) (cubic spline);
    the table covers z >= 0 and is mirrored to z < 0"""
    spl: Any
    a: float
    b: float

    def __init__(self, u, v, phi, a, b, units=usys):
        super().__init__(units)
        v_full = np.hstack([-v[:0:-1], v])
        phi_full = np.hstack([phi[:,:0:-1], phi], )
        self.spl = interpax.Interpolator2D(x=jnp.asarray(u), y=jnp.asarray(v_full), f=jnp.asarray(phi_full),
                                           method='cubic2')
        self.a, self.b = a, b

    def potential(self, xyz, t):
        R = jnp.sqrt(xyz[0]**2 + xyz[1]**2)
        return self.spl(jnp.arcsinh(R/self.a), jnp.arcsinh(xyz[2]/self.b))

class FrameAcceleration(potential.Potential):
    """Uniform acceleration of the Milky Way-centered (non-inertial) frame, tabulated in time.
    gradient returns the MW center acceleration a_MW(t), so that test particles feel -a_MW"""
    ax: Any
    ay: Any
    az: Any

    def __init__(self, t, acc, units=usys):
        super().__init__(units)
        self.ax, self.ay, self.az = [interpax.Interpolator1D(x=jnp.asarray(t), f=jnp.asarray(acc[:,i]), method='cubic2')
                                     for i in range(3)]

    def potential(self, xyz, t):
        raise NotImplementedError("uniform acceleration has no scalar potential")

    def gradient(self, xyz, t):
        return jnp.array([self.ax(t), self.ay(t), self.az(t)]).reshape(3)

def agama_mw(tab):
    """Milky Way of Vasiliev, Belokurov & Erkal (2021) from the AGAMA tables: bulge, exponential disk, halo"""

    bulge = RadialTablePotential(tab['r'], tab['phi_bulge'])
    halo = RadialTablePotential(tab['r'], tab['phi_halo'])
    disk = AxisymTablePotential(tab['disk_u'], tab['disk_v'], tab['phi_disk'], *tab['disk_scales'])

    return [bulge, disk, halo]

def agama_lmc(tab):
    """Moving LMC and the reflex acceleration of the Milky Way frame from the AGAMA tables"""

    t, lmc_w = tab['t'], tab['lmc_w']
    cx, cy, cz = [interpax.Interpolator1D(x=jnp.asarray(t), f=jnp.asarray(lmc_w[:,i]), method='cubic2')
                  for i in range(3)]
    center = lambda time: jnp.array([cx(time), cy(time), cz(time)]).reshape(3)
    lmc = potential.TimeDepTranslatingPotential(pot=RadialTablePotential(tab['r'], tab['phi_lmc']), center_spl=center)

    return [lmc, FrameAcceleration(t, tab['mw_acc'])]

def load_potential(kind='v21_lmc'):
    """Base Milky Way potential:
    'v21': static Milky Way of Vasiliev, Belokurov & Erkal (2021), computed with AGAMA
    'v21_lmc': the same Milky Way plus the moving LMC (1.5e11 Msun) and the Milky Way reflex acceleration
    'static' (gala MilkyWayPotential) and 'mwlmc' (streamsculptor MW_LMC_Potential): earlier, mismatched pair"""

    if kind in ('v21', 'v21_lmc'):
        tab = dict(np.load(AGAMA_MWLMC_FILE))
        comps = agama_mw(tab) + (agama_lmc(tab) if kind=='v21_lmc' else [])
        return potential.Potential_Combine(comps, units=usys)
    elif kind=='static':
        return potential.GalaMilkyWayPotential(units=usys)
    elif kind=='mwlmc':
        return potential.MW_LMC_Potential(units=usys)
    else:
        raise ValueError(f"unknown potential kind: {kind}")

def progenitor(kind='mwlmc', source='fit'):
    """Present-day GD-1 progenitor [kpc, kpc/Myr] for the given potential.
    source: 'fit' (from fit_progenitor, output/gd1_test/progenitor.h5) or 'notebook' (earlier hand-tuned values)"""

    if source=='fit':
        with h5py.File(os.path.join(OUT_DIR, 'progenitor.h5'), 'r') as f:
            return f[kind]['w0'][:]
    elif source!='notebook':
        raise ValueError(f"unknown progenitor source: {source}")

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

def progenitor_from_obs(phi1, phi2, dist, pm1, pm2, vr):
    """GD-1 frame observables [deg, kpc, mas/yr, km/s] -> simulation cartesian phase space [kpc, kpc/Myr]"""

    cg = gc.GD1Koposov10(phi1=phi1*u.deg, phi2=phi2*u.deg, distance=dist*u.kpc, pm_phi1_cosphi2=pm1*u.mas/u.yr,
                         pm_phi2=pm2*u.mas/u.yr, radial_velocity=vr*u.km/u.s)
    ceq = cg.transform_to(coord.ICRS())
    x, v = jc.ICRS_to_simcart(ceq.ra.rad, ceq.dec.rad, dist, ceq.pm_ra_cosdec.to(u.mas/u.yr).value,
                              ceq.pm_dec.to(u.mas/u.yr).value, (vr*u.km/u.s).to(u.kpc/u.Myr).value)

    return np.hstack([np.asarray(x), np.asarray(v)])


# Stream models #

class Stream:
    """GD-1 stream model in a given base potential.

    kind: 'static' or 'mwlmc', sets the progenitor and (if pot not given) the base potential
    pot: base potential; pass a prebuilt one to avoid rebuilding the MW+LMC splines
    dt_release: time between releases [Myr]; each release adds one star at each Lagrange point
    prog: progenitor source passed to progenitor() ('fit' or 'notebook'), or a phase-space array
    """

    def __init__(self, kind='mwlmc', pot=None, prog='fit', dt_release=1., t_age=2500., t_dissolve=-300., Msat=1e4,
                 rtol=1e-6, atol=1e-6):
        self.kind = kind
        self.pot = load_potential(kind) if pot is None else pot
        self.prog_w0 = progenitor(kind, source=prog) if isinstance(prog, str) else np.asarray(prog)
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

    def initial_conditions(self, pot=None, prog_w0=None):
        """Progenitor phase space at -t_age, integrated back in pot (default: base potential)
        from prog_w0 (default: the stream's progenitor)"""

        pot = self.pot if pot is None else pot
        prog_w0 = self.prog_w0 if prog_w0 is None else prog_w0
        return pot.integrate_orbit(w0=jnp.asarray(prog_w0), t0=0.0, t1=-self.t_age,
                                   ts=jnp.array([-self.t_age])).ys[0]

    def generate(self, pot=None, seed=532, prog_w0=None):
        """Particle-spray stream in pot (default: base potential) from prog_w0 (default: the stream's progenitor).
        Returns the model dict"""

        pot = self.pot if pot is None else pot
        ic = self.initial_conditions(pot=pot, prog_w0=prog_w0)
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


# Progenitor fit #

PHI1_PROG = -21.   # progenitor phi1 [deg], fixed in the fit

# fitted progenitor parameters (GD-1 frame at PHI1_PROG); the velocity zero-point offsets of DESI and Hectochelle
# are profiled out (optimized for every stream model)
FIT_PARAMS = ['phi2', 'dist', 'pm1', 'pm2', 'vr']
FIT_SCALES = np.array([0.1, 0.2, 0.1, 0.1, 2.])     # characteristic step sizes for the optimizer
OFFSET_NAMES = ['dv_desi', 'dv_hecto']

# Gaussian priors on the velocity zero-point offsets [km/s]: DESI from comparisons to Gaia/APOGEE
# (Koposov+2024, 2025: 0.2-0.9 km/s), Hectochelle/MINESweeper from twilight spectra (Conroy+2019: 0.5 km/s)
DV_PRIOR = dict(desi=1.0, hecto=0.5)

# radial velocity error model: systematic floors added in quadrature [km/s] (DESI DR1 dark program floor,
# Koposov+2025; Hectochelle, Conroy+2019) and the Hectochelle scaling of formal errors from repeats (Conroy+2019)
VR_FLOOR = dict(desi=1.6, hecto=0.5)
HECTO_ERR_SCALE = 2.

# intrinsic stream widths added in quadrature to measurement errors
INTRINSIC = dict(phi2=0.25, pm1=0.1, pm2=0.1, vr=2., dm=0.)
DM_ERR = 0.1   # assumed uncertainty of the DESI distance-modulus track [mag]

def load_fit_data():
    """Data for the progenitor fit, one entry per observable with phi1, values, errors, and dataset labels.
    phi2: Gaia members (Starkman+2024), excluding stars more likely in the spur than the stream
    pm1, pm2: DESI DR2 thin-stream members
    vr: DESI DR2 thin-stream members (label 0) and Hectochelle main-stream members (label 1; spur stars with
        phi2 > 0.6 deg excluded), with errors following VR_FLOOR and HECTO_ERR_SCALE
    dm: DESI DR2 distance-modulus track (Jarvis+2026 Table 1)"""

    t = load_members()
    t = t[t['stream (MLE)'] > t['spur (MLE)']]
    cg = t['coord'].transform_to(gc.GD1Koposov10())
    phi1_g, phi2_g = cg.phi1.wrap_at(180*u.deg).deg, cg.phi2.deg

    desi = load_desi()
    thin = desi['p_thin'] > desi['p_cocoon']
    hecto = load_hectochelle()
    main = hecto['phi2'] < 0.6

    dtrack = Table.read(os.path.join(DATA_DIR, 'jarvis2026_desi_dr2', 'Table1.fits'))

    data = dict(
        phi2=dict(phi1=phi1_g, y=phi2_g, err=np.zeros(len(phi1_g)), label=np.zeros(len(phi1_g), dtype=int)),
        pm1=dict(phi1=desi['phi1'][thin], y=desi['pm1'][thin], err=desi['pm1_err'][thin],
                 label=np.zeros(np.sum(thin), dtype=int)),
        pm2=dict(phi1=desi['phi1'][thin], y=desi['pm2'][thin], err=desi['pm2_err'][thin],
                 label=np.zeros(np.sum(thin), dtype=int)),
        vr=dict(phi1=np.hstack([desi['phi1'][thin], hecto['phi1'][main]]),
                y=np.hstack([desi['vr'][thin], hecto['vr'][main]]),
                err=np.hstack([np.sqrt(desi['vr_err'][thin]**2 + VR_FLOOR['desi']**2),
                               np.sqrt((HECTO_ERR_SCALE*hecto['vr_err'][main])**2 + VR_FLOOR['hecto']**2)]),
                label=np.hstack([np.zeros(np.sum(thin), dtype=int), np.ones(np.sum(main), dtype=int)])),
        dm=dict(phi1=np.array(dtrack['phi1'], dtype=float), y=np.array(dtrack['m-M'], dtype=float),
                err=np.full(len(dtrack), DM_ERR), label=np.zeros(len(dtrack), dtype=int)),
    )

    return data

def smooth_track(phi1, y, x, sigma=1., n_min=3.):
    """Gaussian-kernel weighted mean of y vs phi1, evaluated at x (NaN where the effective count < n_min)"""

    w = np.exp(-0.5*((x[:,None] - phi1[None,:])/sigma)**2)
    norm = np.sum(w, axis=1)
    out = np.full(len(x), np.nan)
    ok = norm >= n_min
    out[ok] = (w[ok] @ y) / norm[ok]

    return out

def obs_from_params(p):
    """Progenitor phase space from fit parameters (GD-1 frame at PHI1_PROG)"""

    return progenitor_from_obs(PHI1_PROG, p[0], p[1], p[2], p[3], p[4])

def params_from_w0(w0):
    """Fit parameters from a progenitor phase space"""

    o = to_gd1_frame(np.atleast_2d(w0))
    return np.array([o['phi2'][0], o['dist'][0], o['pm1'][0], o['pm2'][0], o['vr'][0]])

def soft_l1(chi):
    """Robust soft-L1 loss, ~chi^2 for small chi and ~2|chi| for large chi"""

    chi = chi[np.isfinite(chi)]
    return np.sum(2*(np.sqrt(1 + chi**2) - 1))

def vr_cost(dv, d, y_model):
    """Radial velocity misfit plus Gaussian priors, for zero-point offsets dv = (dv_desi, dv_hecto);
    data = truth + offset, so each survey's offset is subtracted from its velocities"""

    y = d['y'] - dv[0]*(d['label']==0) - dv[1]*(d['label']==1)
    chi = (y - y_model) / np.sqrt(d['err']**2 + INTRINSIC['vr']**2)

    return soft_l1(chi) + (dv[0]/DV_PRIOR['desi'])**2 + (dv[1]/DV_PRIOR['hecto'])**2

def best_offsets(d, y_model):
    """Velocity zero-point offsets minimizing vr_cost for a given model track (profile likelihood)"""

    from scipy.optimize import minimize
    res = minimize(vr_cost, np.zeros(2), args=(d, y_model), method='L-BFGS-B')

    return res.x

def fit_cost(p, s, data, seed=532, return_terms=False):
    """Robust (soft-L1) misfit between data and the smoothed tracks of a stream generated from progenitor
    parameters p. The radial velocity term includes the priors and is minimized over the velocity zero-point offsets"""

    model = s.generate(seed=seed, prog_w0=obs_from_params(p))
    obs = to_gd1_frame(model['w'])
    obs['dm'] = 5*np.log10(obs['dist']*100)

    terms = {}
    for k, d in data.items():
        y_model = smooth_track(obs['phi1'], obs[k], d['phi1'])
        if k=='vr':
            offsets = best_offsets(d, y_model)
            terms[k] = vr_cost(offsets, d, y_model)
        else:
            terms[k] = soft_l1((d['y'] - y_model) / np.sqrt(d['err']**2 + INTRINSIC[k]**2))

    cost = np.sum(list(terms.values()))
    if return_terms:
        return cost, terms, dict(zip(OFFSET_NAMES, offsets))

    return cost

def fit_progenitor(kind='mwlmc', pot=None, prog='notebook', dt_release=5., maxiter=600, seed=532, save=True):
    """Fit the progenitor (phi2, distance, proper motions, radial velocity at PHI1_PROG) to the GD-1 data, using
    Nelder-Mead on a coarsely sampled stream, with the DESI and Hectochelle velocity offsets profiled out.
    prog: starting progenitor ('notebook', 'fit', or an array of fit parameters)"""

    from scipy.optimize import minimize

    p0 = np.asarray(prog, dtype=float) if not isinstance(prog, str) else params_from_w0(progenitor(kind, source=prog))
    s = Stream(kind, pot=pot, prog=obs_from_params(p0), dt_release=dt_release)
    data = load_fit_data()
    print(f'start: {dict(zip(FIT_PARAMS, np.round(p0, 4)))}  cost={fit_cost(p0, s, data, seed=seed):.1f}')

    history = []
    def f(x):
        c = fit_cost(p0 + x*FIT_SCALES, s, data, seed=seed)
        history.append(c)
        if len(history)%25==0:
            print(f'  {len(history)} evaluations, best cost {np.min(history):.1f}', flush=True)
        return c

    # initial simplex with one FIT_SCALES step per parameter (scipy's default steps are tiny around x=0)
    n = len(p0)
    simplex = np.vstack([np.zeros(n), np.eye(n)])
    res = minimize(f, np.zeros(n), method='Nelder-Mead',
                   options=dict(maxiter=maxiter, xatol=1e-2, fatol=0.1, adaptive=True, initial_simplex=simplex))
    p_best = p0 + res.x*FIT_SCALES
    cost, terms, offsets = fit_cost(p_best, s, data, seed=seed, return_terms=True)
    print(f'best: {dict(zip(FIT_PARAMS, np.round(p_best, 4)))}  offsets={ {k: round(v, 2) for k, v in offsets.items()} }')
    print(f'  cost={cost:.1f}  terms={ {k: round(v, 1) for k, v in terms.items()} }')
    print(f'  {res.nfev} evaluations, converged: {res.success} ({res.message})')

    if save:
        save_progenitor_fit(kind, p_best, offsets, cost, terms, dict(dt_release=dt_release, seed=seed, nfev=res.nfev))

    return p_best, offsets, cost, terms

def save_progenitor_fit(kind, p, offsets, cost, terms, info):
    """Store the fitted progenitor in output/gd1_test/progenitor.h5 (group per potential)"""

    os.makedirs(OUT_DIR, exist_ok=True)
    with h5py.File(os.path.join(OUT_DIR, 'progenitor.h5'), 'a') as f:
        if kind in f:
            del f[kind]
        g = f.create_group(kind)
        g.create_dataset('w0', data=obs_from_params(p))
        g.create_dataset('params', data=p)
        g.attrs['param_names'] = FIT_PARAMS
        g.attrs['phi1_prog'] = PHI1_PROG
        g.attrs['cost'] = cost
        for k, v in offsets.items():
            g.attrs[k] = v
        for k, v in terms.items():
            g.attrs[f'cost_{k}'] = v
        for k, v in info.items():
            g.attrs[k] = v

def load_progenitor_fit(kind):
    """Fitted progenitor parameters and costs"""

    with h5py.File(os.path.join(OUT_DIR, 'progenitor.h5'), 'r') as f:
        g = f[kind]
        return dict(params=dict(zip(FIT_PARAMS, g['params'][:])), w0=g['w0'][:],
                    offsets={k: g.attrs[k] for k in OFFSET_NAMES if k in g.attrs}, attrs=dict(g.attrs))


# Base models #

# styles of the base models in comparison plots
KIND_COLORS = dict(static='tab:blue', mwlmc='tab:red', v21='tab:blue', v21_lmc='tab:red')
KIND_LABELS = dict(static='static MW (gala)', mwlmc='MW+LMC (streamsculptor)', v21='static MW (V21)',
                   v21_lmc='MW+LMC (V21)')

def run_base(seed=532, prog='fit', kinds=('v21', 'v21_lmc')):
    """Generate and save the base models in the given potentials"""

    for kind in kinds:
        Stream(kind, prog=prog).base(seed=seed)

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

def plot_base_residuals(ref='v21_lmc', kinds=('v21', 'v21_lmc'), save=True):
    """Base models, GD-1 members, Hectochelle (Bonaca+2020) and DESI DR2 (Jarvis+2026) members,
    relative to the smooth track of the reference model"""

    data = members_gd1_frame()
    spec = load_hectochelle()
    desi = load_desi()
    phi2_lim = (np.floor(np.min(spec['phi2'])*2)/2, np.ceil(np.max(spec['phi2'])*2)/2)
    models = {kind: to_gd1_frame(load_model('base', kind)['w']) for kind in kinds}
    colors, labels = KIND_COLORS, KIND_LABELS

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
