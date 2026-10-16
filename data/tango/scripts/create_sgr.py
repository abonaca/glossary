#!/usr/bin/env python

'''
Create a two-component spherical model for the Sagittarius galaxy progenitor
'''
import agama,numpy
numpy.random.seed(42)

# units:  length=1 kpc, velocity=1 km/s, mass=232500 Msun (G=1).
# note that the length and mass scales have some fiducial values,
# but the actual model used in the simulation is further scaled by
# Mscale=0.86 and rscale=0.80 (hence vscale=1.03)
comp1 = agama.Potential(type='king', mass=1000, scaleradius=1, w0=4)   # stars
comp2 = agama.Potential(type='spheroid', mass=18000, scaleradius=8,    # halo
    gamma=0, beta=3, alpha=1, outercutoffradius=4.5, cutoffstrength=2)
pot = agama.Potential(comp1, comp2)
df1 = agama.DistributionFunction(type='QuasiSpherical', density=comp1, potential=pot)
df2 = agama.DistributionFunction(type='QuasiSpherical', density=comp2, potential=pot)
Nstar = 200000
Nhalo = 500000
starp, starm = agama.GalaxyModel(pot, df1).sample(Nstar)
halop, halom = agama.GalaxyModel(pot, df2).sample(Nhalo)
# mass refinement for halo particles based on energy
Es = pot.potential(starp[:,0:3]) + 0.5*numpy.sum(starp[:,3:6]**2, axis=1)
starp = starp[numpy.argsort(Es)]
Eh = pot.potential(halop[:,0:3]) + 0.5*numpy.sum(halop[:,3:6]**2, axis=1)
order = numpy.argsort(Eh)
halop = halop[order]
# input range => output range;  mscale  count
# 0    - 0.20    0    - 0.20    1       50%
# 0.20 - 0.60    0.20 - 0.36    2.5     40%
# 0.60 - 1.00    0.36 - 0.40    10      10%
ind1 = Nhalo//5
ind2 = Nhalo*3//5
ind3 = Nhalo
cnt1 = (ind2-ind1)*2//5
cnt2 = (ind3-ind2)//10
halop = numpy.vstack((halop[0:ind1],
    halop[ind1:ind2][numpy.random.choice(ind2-ind1, cnt1, replace=False)],
    halop[ind2:ind3][numpy.random.choice(ind3-ind2, cnt2, replace=False)]))
halom = numpy.hstack((numpy.ones(ind1), numpy.ones(cnt1)*2.5, numpy.ones(cnt2)*10))*halom[0]
# sort halo particles in energy once again
Eh = pot.potential(halop[:,0:3]) + 0.5*numpy.sum(halop[:,3:6]**2, axis=1)
halop = halop[numpy.argsort(Eh)]
# export the snapshot
format = 'nemo'  # can export in 'text', 'nemo' and possibly 'gadget' formats
agama.writeSnapshot('Sgr_init.dat', (numpy.vstack((starp, halop)), numpy.hstack((starm, halom))), format)
