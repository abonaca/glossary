#!/usr/bin/env python

'''
Create a simple spherical model for the LMC with a gradually truncated NFW profile
'''
import agama,sys

if len(sys.argv)<=1:
    Mlmc = 1.5e11  # mass in physical units [Msun]
else:
    Mlmc = float(sys.argv[1])

# internally use N-body units:  length=1 kpc, velocity=1 km/s, mass=232500 Msun (G=1).
Rlmc = (Mlmc*1e-11)**0.6 * 8.5  # scale radius in kpc
Rcut = Rlmc*10                  # outer cutoff radius
pot  = agama.Potential(type='spheroid', gamma=1, beta=3, alpha=1,  # power-law indices of the NFW profile
    scaleradius=Rlmc, mass=Mlmc/232500, outercutoffradius=Rcut)
df   = agama.DistributionFunction(type='QuasiSpherical', density=pot, potential=pot)  # Eddington DF
Nbody= 1000000
snap = agama.GalaxyModel(pot, df).sample(Nbody)
format = 'nemo'  # can export in 'text', 'nemo' and possibly 'gadget' formats
agama.writeSnapshot('LMC.dat', snap, format)
