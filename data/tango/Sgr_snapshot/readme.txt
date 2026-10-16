This folder contains the final snapshot (at the current time) of the simulation
of the disrupting Sagittarius galaxy in the evolving potential of the Milky Way
(with a triaxial halo) and LMC of mass 1.5e11 Msun.
The time-dependent potentials of the Milky Way and the LMC are stored in
../potentials_triax/potential_evolving.ini
The snapshot is split into two components -- stars and dark matter, each one
contains 2e5 particles; DM particles have varying mass -- initially more
tightly bound ones are more numerous but less massive. Particles are sorted
by the initial binding energy (more tightly bound ones come first).
Each of the two files (stars.txt, darkmatter.txt) contains the following columns:
1-3:   x,y,z - position in the right-handed Galactocentric cartesian coordinate system [kpc], in which the Sun sits at x=-8 kpc
4-6:   vx,vy,vz - velocity [km/s]
7-8:   ra, dec - ICRS celestial coordinates [deg]
9:     heliocentric distance [kpc]
10-11: proper motions pmra pmdec [mas/yr]
12:    heliocentric line-of-sight velocity [km/s] (*not* GSR)
13-14: Lambda Beta - celestial coordinates in the right-handed system aligned with the stream (leading arm has Lambda>0; the range is greater than 360 deg and takes into account multiple wraps)
15:    Zeta - angular coordinate similar to Lambda, but measured from the Galactic centre vangage point
16:    stripping time [Gyr] (0 - still bound, negative values indicate the most recent time that the particle separated by more than 5 kpc from the remnant)
17:    particle mass [Msun]

The file  trajectory.txt  contains the trajectory and other time-dependent properties
of the Sgr galaxy in the simulation, with the following columns:
1:     time [Gyr], starting at -3 and ending slightly after the current time (t=0)
2-4:   centre position of the remnant [kpc]
5-7:   its velocity [km/s]
8:     total mass within 5 kpc from the centre [Msun]
9:     stellar mass within this radius [Msun]
10:    estimate of the tidal radius [kpc]
11:    energy of the remnant centre-of-mass in the Milky Way potential [(km/s)^2]
12:    gravitational potential at the centre of the remnant (sourced only by the Sgr particles, excluding Milky Way) [(km/s)^2]
