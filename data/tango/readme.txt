The data in this archive are from the simulations of the disrupting Sagittarius galaxy
in the combined potential of the Milky Way and the Large Magellanic Cloud.
These simulations are described in the paper [Vasiliev,Belokurov&Erkal 2020].

Sgr_snapshot
contains the final (present-day) snapshot from the fiducial simulation with a triaxial
Milky Way halo and M_LMC=1.5e11 Msun (see the readme file in that folder for details).

potentials_triax
contains the initial and subsequently evolving potentials of both the Milky Way and LMC,
represented by multipole expansions, as well as the trajectory of the LMC and the
reflex motion-induced acceleration of the Milky Way -- everything that is needed to
study the dynamics of the Sgr stream and other objects in a time-dependent potential
of the interacting Milky Way and LMC. This model corresponds to the stream simulation
in the previous folder.

potentials_axisym
contains the same data, but for another Milky Way halo model, which is axisymmetric
rather than triaxial (note that in either case, its axis ratios vary with radius).
It may be more convenient in certain applications, and produces a stream that fits
the observations almost as well as the triaxial model.

scripts
contains the Python scripts illustrating how to integrate orbits in a time-dependent
potential, and how to construct initial conditions for these simulations
(the parameter files for various choices of Milky Way halo potentials are also included).
These scripts use the Agama framework, available at http://agama.software
