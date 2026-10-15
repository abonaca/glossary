"""Past trajectories of the Milky Way and the LMC computed with AGAMA, for use in streamsculptor.

Follows AGAMA's py/example_lmc_mw_interaction.py (commit c507fc3), which is based on the Milky Way and LMC models
of Vasiliev, Belokurov & Erkal (2021): rigid potentials of both galaxies under mutual gravity, with Chandrasekhar
dynamical friction on the LMC (distance-dependent Coulomb logarithm calibrated on N-body simulations).
All components are identical to the example. For streamsculptor, the spherical components are tabulated in radius
and the exponential disk is tabulated on an (R, z) grid uniform in asinh(R/R_SCALE) and asinh(|z|/Z_SCALE).

Run with a Python that has agama installed, e.g.
    ~/.pyenv/versions/3.10.16/bin/python mw_lmc_agama.py [massLMC]
Output: data/mw_lmc_agama/mw_lmc_<massLMC>.npz in kpc, Myr, Msun units
"""

import os
import sys
import json

import numpy as np
import scipy.integrate
import scipy.special
import agama

agama.setUnits(length=1, velocity=1, mass=1)   # kpc, km/s, Msun
TIME_UNIT_MYR = agama.getUnits()['time']       # AGAMA time unit in Myr (kpc / (km/s) ~ 977.8 Myr)
KMS_TO_KPCMYR = 1.0227121650537077e-3

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_DIR, 'data', 'mw_lmc_agama')

# present-day LMC: PM from Luri+2021, distance from Pietrzynski+2019, center and velocity from van der Marel+2002
LMC_OBS = dict(ra=81.28, dec=-69.78, dist=49.6, pmra=1.858, pmdec=0.385, vlos=262.2)

PARAM_BULGE = dict(type='Spheroid', mass=1.2e10, scaleRadius=0.2, outerCutoffRadius=1.8, gamma=0.0, beta=1.8)
PARAM_DISK = dict(type='Disk', mass=5.0e10, scaleRadius=3.0, scaleHeight=-0.4)   # exponential, isothermal (sech^2) in z

# (R, z) grid of the tabulated disk potential: uniform in asinh(R/R_SCALE), asinh(|z|/Z_SCALE) out to DISK_RMAX [kpc]
R_SCALE, Z_SCALE, DISK_RMAX, N_DISK_R, N_DISK_Z = 1.0, 0.1, 500., 400, 400
PARAM_HALO = dict(type='Spheroid', densityNorm=1.35e7, scaleRadius=14, outerCutoffRadius=300, cutoffStrength=4,
                  gamma=1, beta=3)

def param_lmc(mass):
    """Truncated NFW LMC with the mass-radius relation of Vasiliev, Belokurov & Erkal (2021), Fig. 3"""

    radius = (mass/1e11)**0.6 * 8.5
    return dict(type='Spheroid', mass=mass, scaleRadius=radius, outerCutoffRadius=radius*10, gamma=1, beta=3)

def lmc_present():
    """Present-day Galactocentric position and velocity of the LMC [kpc, km/s] (AGAMA's solar parameters)"""

    l, b, pml, pmb = agama.transformCelestialCoords(agama.fromICRStoGalactic, LMC_OBS['ra']*np.pi/180,
                                                    LMC_OBS['dec']*np.pi/180, LMC_OBS['pmra'], LMC_OBS['pmdec'])
    return np.array(agama.getGalactocentricFromGalactic(l, b, LMC_OBS['dist'], pml*4.74, pmb*4.74, LMC_OBS['vlos']))

def integrate(mass_lmc=1.5e11, t_rewind=-4000., dt=2.):
    """Integrate the MW and LMC back in time (rigid potentials, dynamical friction on the LMC), as in the AGAMA example.
    t_rewind, dt in Myr. Returns a dict with the time grid [Myr], trajectories, and the MW acceleration"""

    pot_mw = agama.Potential(PARAM_BULGE, PARAM_DISK, PARAM_HALO)
    dens_halo = agama.Density(PARAM_HALO)
    plmc = param_lmc(mass_lmc)
    pot_lmc = agama.Potential(plmc)
    bmin_coulog = plmc['scaleRadius'] * 2.0

    # velocity dispersion of an isotropic halo in the sphericalized MW potential, for dynamical friction
    pot_mw_sph = agama.Potential(type='Multipole', potential=pot_mw, lmax=0, rmin=0.01, rmax=1000)
    gm_halo = agama.GalaxyModel(pot_mw_sph, agama.DistributionFunction(type='quasispherical', density=dens_halo,
                                                                       potential=pot_mw_sph))
    rgrid = np.logspace(1, 3, 16)
    xyz = np.column_stack([rgrid, rgrid*0, rgrid*0])
    sigmafnc = agama.Spline(rgrid, gm_halo.moments(xyz, dens=False, vel=False, vel2=True)[:,0]**0.5)

    def difeq(w, t):
        x0, v0, x1, v1 = w[0:3], w[3:6], w[6:9], w[9:12]
        dx, dv = x1 - x0, v1 - v0
        dist, vmag = np.sum(dx**2)**0.5, np.sum(dv**2)**0.5
        f0 = pot_lmc.force(-dx)
        f1 = pot_mw.force(dx)
        rho = pot_mw.density(dx)
        sigma = sigmafnc(dist)
        coulog = max(0, np.log(dist / bmin_coulog)**0.5)
        X = vmag / (sigma * 2**.5)
        drag = -(4*np.pi * rho * dv / vmag * (scipy.special.erf(X) - 2/np.pi**.5 * X * np.exp(-X*X)) *
                 mass_lmc * agama.G**2 / vmag**2 * coulog)
        return np.hstack((v0, f0, v1, f1 + drag))

    # time grid in AGAMA units
    n = int(round(-t_rewind/dt)) + 1
    tgrid = np.linspace(t_rewind, 0., n) / TIME_UNIT_MYR
    ic = np.hstack((np.zeros(6), lmc_present()))
    sol = scipy.integrate.odeint(difeq, ic, tgrid[::-1], rtol=1e-10, atol=1e-10)[::-1]

    # MW acceleration in the inertial frame: second derivative of its trajectory
    acc = np.column_stack([agama.Spline(tgrid, sol[:,i], der=sol[:,i+3])(tgrid, 2) for i in range(3)])

    return dict(t=tgrid*TIME_UNIT_MYR,
                lmc_w=np.column_stack([sol[:,6:9] - sol[:,0:3], (sol[:,9:12] - sol[:,3:6])*KMS_TO_KPCMYR]),
                mw_w=np.column_stack([sol[:,0:3], sol[:,3:6]*KMS_TO_KPCMYR]),
                mw_acc=acc * KMS_TO_KPCMYR / TIME_UNIT_MYR,
                params=dict(bulge=PARAM_BULGE, disk=PARAM_DISK, halo=PARAM_HALO, lmc=plmc, lmc_obs=LMC_OBS,
                            bmin_coulog=bmin_coulog, t_rewind=t_rewind, dt=dt, agama_example_commit='c507fc3'))

def radial_tables(mass_lmc=1.5e11, rgrid=np.logspace(-4, 3, 2000)):
    """Potentials [kpc^2/Myr^2] of the spherical components (bulge, halo, LMC) tabulated in radius [kpc]"""

    xyz = np.column_stack([rgrid, rgrid*0, rgrid*0])
    conv = KMS_TO_KPCMYR**2
    return dict(r=rgrid,
                phi_bulge=agama.Potential(PARAM_BULGE).potential(xyz) * conv,
                phi_halo=agama.Potential(PARAM_HALO).potential(xyz) * conv,
                phi_lmc=agama.Potential(param_lmc(mass_lmc)).potential(xyz) * conv)

def disk_table():
    """Disk potential [kpc^2/Myr^2] on a grid in u=asinh(R/R_SCALE), v=asinh(|z|/Z_SCALE)"""

    u = np.linspace(0, np.arcsinh(DISK_RMAX/R_SCALE), N_DISK_R)
    v = np.linspace(0, np.arcsinh(DISK_RMAX/Z_SCALE), N_DISK_Z)
    R, z = R_SCALE*np.sinh(u), Z_SCALE*np.sinh(v)
    RR, zz = np.meshgrid(R, z, indexing='ij')
    xyz = np.column_stack([RR.ravel(), RR.ravel()*0, zz.ravel()])
    phi = agama.Potential(PARAM_DISK).potential(xyz).reshape(RR.shape) * KMS_TO_KPCMYR**2

    return dict(disk_u=u, disk_v=v, phi_disk=phi, disk_scales=np.array([R_SCALE, Z_SCALE]))

def run(mass_lmc=1.5e11):
    """Compute and save trajectories and potential tables to data/mw_lmc_agama/mw_lmc_<mass>.npz"""

    traj = integrate(mass_lmc=mass_lmc)
    tab = radial_tables(mass_lmc=mass_lmc)
    dtab = disk_table()
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f'mw_lmc_{mass_lmc:.2g}.npz')
    np.savez(path, t=traj['t'], lmc_w=traj['lmc_w'], mw_w=traj['mw_w'], mw_acc=traj['mw_acc'],
             r=tab['r'], phi_bulge=tab['phi_bulge'], phi_halo=tab['phi_halo'], phi_lmc=tab['phi_lmc'], **dtab,
             params=json.dumps(traj['params']), agama_version=getattr(agama, '__version__', 'unknown'))
    print(f'saved {path}')
    print(f"LMC today: x={traj['lmc_w'][-1,:3].round(2)} kpc, |x|={np.linalg.norm(traj['lmc_w'][-1,:3]):.1f} kpc")
    print(f"LMC pericenter: {np.min(np.linalg.norm(traj['lmc_w'][:,:3], axis=1)):.1f} kpc; "
          f"distance 4 Gyr ago: {np.linalg.norm(traj['lmc_w'][0,:3]):.0f} kpc")

    return path

if __name__=='__main__':
    run(float(sys.argv[1]) if len(sys.argv)>1 else 1.5e11)
