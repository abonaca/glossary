#!/usr/bin/env python

'''
Create a self-consistent equilibrium model of the Milky Way,
using the Schwarzschild orbit-superposition method.
The parameters of several variants of models are contained in mw_model_***.ini;
the ini filename should be provided as a command-line argument for this script.
Only models with a triaxial symmetry can be constructed with this method;
however, the routine 'create_halo' can be used to compute the potential of
a more general (twisted) halo density profile via multipole expansion.
'''
import agama, numpy, sys
try:
    from ConfigParser import RawConfigParser  # python 2
except ImportError:
    from configparser import RawConfigParser  # python 3

def create_halo(scaleradius, gamma, beta, alpha, q_in, q_out, p_out, alpha_q, beta_q, gamma_q, shaperadius,
    outerCutoffRadius=200.0, fiducialRadius=8.0, fiducialCircularVelocity=155.0):
    '''
    Construct a twisted halo density model
    '''
    # first create a spherical halo profile with unit normalization, and then rescale it
    params = dict(Type='Spheroid', outerCutoffRadius=outerCutoffRadius, cutoffStrength=2,
        gamma=gamma, beta=beta, alpha=alpha, scaleRadius=scaleradius, densityNorm=1)
    pot0  = agama.Potential(**params)
    # rescale the halo density to provide the desired circular velocity at the fiducial radius
    params['densityNorm'] = fiducialCircularVelocity**2 / (-fiducialRadius * pot0.force(fiducialRadius,0,0)[0])
    dens  = agama.Density(params)
    ca,sa = numpy.cos(alpha_q), numpy.sin(alpha_q)
    cg,sg = numpy.cos(gamma_q), numpy.sin(gamma_q)
    # the actual halo density is non-spherical and is described by this function
    def twisteddens(x):
        r2 = numpy.sum(x**2, axis=1)
        ch = 1 / (1 + r2 / shaperadius**2)  # changeover fnc: 1 at small radii, 0 at large radii
        q  = q_out + (q_in-q_out) * ch
        p  = p_out + ( 1  -p_out) * ch
        cb,sb = numpy.cos(beta_q*(1-ch)), numpy.sin(beta_q*(1-ch))
        mat = numpy.array( [
            [ ca*cg - sa*cb*sg,  sa*cg + ca*cb*sg, sb*sg],
            [-ca*sg - sa*cb*cg, -sa*sg + ca*cb*cg, sb*cg],
            [         sa*sb   ,         -ca*sb   , cb   ] ])
        xp, yp, zp = numpy.einsum('ijk,kj->ik', mat, x)
        s = (xp**2 + (yp/p)**2 + (zp/q)**2)**0.5 * (p*q)**(1./3)
        return dens.density(numpy.column_stack((s, s*0, s*0)))
    return twisteddens


# read parameters from the INI file
if len(sys.argv) <= 1:
    print('Provide the name of the INI file with model parameters')
    exit(1)
iniFileName = sys.argv[1]
ini = RawConfigParser()
ini.read(iniFileName)
iniPotenHalo  = dict((key, float(val)) for key, val in ini.items('Density halo'))
iniPotenBulge = dict(ini.items('Potential bulge'))
iniPotenDisk  = dict(ini.items('Potential disk'))
iniDFDisk     = dict(ini.items('DF disk'))

print('Computing halo potential')
# create initial density profiles of all components
denDisk  = agama.Density(iniPotenDisk)
denBulge = agama.Density(iniPotenBulge)
denStars = agama.Density(denDisk, denBulge)
potStars = agama.Potential(iniPotenDisk, iniPotenBulge)
denHalo  = create_halo(**iniPotenHalo)
potHalo  = agama.Potential(type='Multipole', density=denHalo, lmax=4, mmax=4, symmetry='r', rmin=1e-2, rmax=1e3)
potHalo.export(iniFileName.replace('.ini', '_halo.ini'))

# if the halo is twisted (i.e. not symmetric w.r.t flipping the sign of z),
# no equilibrium model can be constructed
if not numpy.isclose(potHalo.density(50,0,50), potHalo.density(50,0,-50)):
    print('Halo density is not triaxial, exiting...')
    exit()

# if the halo is triaxial with nonzero rotation angle alpha_q, we memorize this angle,
# then unset it, since Schwarzschild models need a triaxial density profile aligned with x,y,z,
# and finally rotate the resulting snapshot by this angle when saving it
alpha_q  = iniPotenHalo['alpha_q']
iniPotenHalo['alpha_q'] = 0
denHaloT = create_halo(**iniPotenHalo)   # aligned (triaxial) density
potHaloT = agama.Potential(type='Multipole', density=denHaloT, lmax=4, mmax=4, symmetry='t', rmin=1e-2, rmax=1e3)

# yet another variation of halo potential is forced to be axisymmetric, and used for creating initial conditions
potHaloA = agama.Potential(type='Multipole', density=denHaloT, lmax=4, mmax=0, symmetry='t', rmin=1e-2, rmax=1e3)

# total potential (triaxial and axisymmetric variants)
potTotalT = agama.Potential(potStars, potHaloT)
potTotalA = agama.Potential(potStars, potHaloA)

# utility function for rotating an N-body snapshot by angle alpha_q
def rotate(snap):
    ca,sa = numpy.cos(alpha_q), numpy.sin(alpha_q)
    x  = snap[0][:,0] * ca - snap[0][:,1] * sa
    y  = snap[0][:,1] * ca + snap[0][:,0] * sa
    snap[0][:,0] = x
    snap[0][:,1] = y
    vx = snap[0][:,3] * ca - snap[0][:,4] * sa
    vy = snap[0][:,4] * ca + snap[0][:,3] * sa
    snap[0][:,3] = vx
    snap[0][:,4] = vy

print('Creating initial conditions')
# number of orbits in disk+bulge and halo components
Ndisk  = 100000
Nbulge = 25000
Nhalo  = 125000
# construct the DF of the disk component, using the axisymmetrized potential (otherwise the method won't work)
dfDisk = agama.DistributionFunction(potential=potTotalA, **iniDFDisk)
icDisk = agama.GalaxyModel(potTotalA, dfDisk).sample(Ndisk)[0]
# same for the bulge component, but use Eddington inversion, not a disk-type DF
icBulge= denBulge.sample(Nbulge, potTotalT)[0]
icStars= numpy.vstack((icBulge, icDisk))  # treat bulge+disk together as the stellar component
# same for the halo
icHalo = potHaloT.sample(Nhalo, potTotalT)[0]

# define 'Target' objects for Schwarzschild modelling, which specify density and kinematic constraints
targetStarsDen = agama.Target(type='DensityCylindricalLinear',
    gridr=agama.nonuniformGrid(25, 0.1, 20.0), gridz=agama.nonuniformGrid(20, 0.05, 2.5), mmax=4)

targetHaloDen = agama.Target(type='DensitySphHarm',
    gridr=agama.nonuniformGrid(40, 0.5, 250.0), lmax=4, mmax=4)

targetHaloKin = agama.Target(type='KinemShell', gridR=agama.nonuniformGrid(40, 0.5, 300.0), degree=1)

consStarsDen = targetStarsDen(denStars)   # values of density constraints for stars
consHaloDen  = targetHaloDen (denHaloT)   # and for the halo

intTimeStars = 100*potTotalA.Tcirc(icStars)  # integration time is 100 orbital periods
intTimeHalo  = 100*potTotalA.Tcirc(icHalo)

print('Vcirc at 8kpc=%g, at 20kpc=%g, at 50kpc=%g, at 100kpc=%g' % (
    (-potTotalA.force( 8,0,0)[0]* 8)**0.5, (-potTotalA.force( 20,0,0)[0]* 20)**0.5,
    (-potTotalA.force(50,0,0)[0]*50)**0.5, (-potTotalA.force(100,0,0)[0]*100)**0.5))
print('Mdisk=%g (df: %g), bulge=%g, halo=%g' % (
    denDisk.totalMass(), dfDisk.totalMass(), denBulge.totalMass(), potHaloT.totalMass()))

print('Constructing stellar component')
dataStarsDen, trajStars = agama.orbit(potential=potTotalT, ic=icStars, time=intTimeStars,
    trajsize=101, targets=targetStarsDen)
consStarsMass = denStars.totalMass()
dataStarsMass = numpy.ones((len(icStars), 1))
weightStars = agama.solveOpt(matrix=(dataStarsMass.T, dataStarsDen.T), rhs=([consStarsMass], consStarsDen),
    rpenl=([numpy.inf], numpy.ones(len(consStarsDen))*numpy.inf), xpenq=numpy.ones(len(icStars))*0.1 )

# create the N-body snapshot from the trajectories sampled during orbit integration in proportion to weights
snapshotFormat = 'nemo'   # can use 'text', 'nemo' and possibly 'gadget'
nbodyStars = 1000000
status,result = agama.sampleOrbitLibrary(nbodyStars, trajStars, weightStars)
if not status:
    # this may occur if there was not enough recorded trajectory points for some high-weight orbits:
    # in this case their indices and the required numbers of points are returned in the result tuple
    indices,trajsizes = result
    print('reintegrating %i orbits; max # of sampling points is %i' % (len(indices), max(trajsizes)))
    trajStars[indices] = agama.orbit(potential=potTotalT, ic=icStars[indices],
        time=intTimeStars[indices], trajsize=trajsizes)
    status,result = agama.sampleOrbitLibrary(nbodyStars, trajStars[:,1], weightStars)
    if not status: print('Failed to produce output N-body model')

rotate(result)
agama.writeSnapshot(iniFileName.replace('.ini', '_stars.dat'), result, snapshotFormat)

print('Constructing halo component')
dataHaloDen, dataHaloKin, trajHalo = agama.orbit(potential=potTotalT, ic=icHalo, time=intTimeHalo,
    trajsize=101, targets=(targetHaloDen, targetHaloKin))
consHaloMass = agama.Density(denHalo).totalMass()
dataHaloMass = numpy.ones((len(icHalo), 1))
# dataHaloKin is kinematic data: for each orbit (row) it contains
# Ngrid values of rho sigma_r^2, then the same number of values of rho sigma_t^2;
# we combine them into a single array of Ngrid values that enforce isotropy (sigma_t^2 - 2 sigma_r^2 = 0)
beta        = 0.  # desired value of the velocity anisotropy coefficient
Ngrid       = len(targetHaloKin) // 2
dataHaloKin = 2 * (1-beta) * dataHaloKin[:,:Ngrid] - dataHaloKin[:,Ngrid:]
consHaloKin = numpy.zeros(Ngrid)

weightHalo = agama.solveOpt(matrix=(dataHaloMass.T, dataHaloDen.T, dataHaloKin.T),
    rhs=([consHaloMass], consHaloDen, consHaloKin),
    rpenl=([numpy.inf], numpy.ones(len(consHaloDen))*numpy.inf, numpy.ones(len(consHaloKin))*numpy.inf),
    xpenq=numpy.ones(len(icStars))*0.1 )

nbodyHalo = 4000000
status,result = agama.sampleOrbitLibrary(nbodyHalo, trajHalo, weightHalo)
if not status:
    indices,trajsizes = result
    print('reintegrating %i orbits; max # of sampling points is %i' % (len(indices), max(trajsizes)))
    trajHalo[indices] = agama.orbit(potential=potTotalT, ic=icHalo[indices],
        time=intTimeHalo[indices], trajsize=trajsizes+1)
    status,result = agama.sampleOrbitLibrary(nbodyHalo, trajHalo[:,1], weightHalo)
    if not status: print('Failed to produce output N-body model')

rotate(result)
agama.writeSnapshot(iniFileName.replace('.ini', '_halo.dat'), result, snapshotFormat)
