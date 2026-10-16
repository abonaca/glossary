#!/bin/sh

# Launch a GyrfalcON simulation of a disrupting Sgr galaxy in the evolving MW+LMC potentials.
# The result is slightly different from the simulation presented in the paper, since that one
# additionally used a dynamical friction runtime manipulator, but the difference is minor.
# Due to chaotic nature of the N-body problem and a near-impossibility to produce bitwise
# identical results in the simulations run on different machines, the final position/velocity
# may differ by a few tenths of kpc and a few km/s -- much larger than the maximum error
# limit in our simulations; nevertheless, the stream features should be well reproducible.
# The initial conditions for the Sgr galaxy are created by a separate script, then scaled.
python create_sgr.py
rm -f Sgr_evol.dat
snapscale Sgr_init.dat - mscale=0.86 rscale=0.805 vscale=1.0336 | \
snapshift - - rshift=56.8,6.25,-32.37 vshift=54.1,-46.79,77.42 | \
s2s - - time=-3.0 | \
gyrfalcON - Sgr_evol.dat eps=0.1 kmax=8 Nlev=3 fea=0.2 tstop=0.0 step=0.03125 \
accname=agama accfile=../potentials_triax/potential_evolving.ini
