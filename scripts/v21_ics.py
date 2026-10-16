"""
Initial conditions for rerunning the Vasiliev, Belokurov & Erkal (2021) simulations.

Ports of V21's create_mw.py, create_lmc.py and create_sgr.py (data/tango/scripts/) as
functions with the particle/orbit numbers as arguments, timing every stage, plus the
scaling/shift of the Sgr progenitor done in run_sim.sh (snapscale + snapshift + s2s).

Units: kpc, km/s, mass unit 232500 Msun (G=1); times in kpc/(km/s) = 0.978 Gyr.

Usage (ipython):
    import v21_ics as v
    t = v.make_mw(scale=0.1)          # 1/10 of V21's orbit and particle numbers
    v.compare_halo(t['halo_ini'])     # regenerated halo potential vs the published one
    v.make_lmc(nbody=100000)
    v.make_sgr(scale=0.1)
"""
import os
import time
import numpy as np
import agama
from configparser import RawConfigParser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V21_SCRIPTS = os.path.join(ROOT, 'data', 'tango', 'scripts')
V21_POT = os.path.join(ROOT, 'data', 'tango', 'potentials_triax')
OUT = os.path.join(ROOT, 'output', 'v21_sim', 'ics')

MASS_UNIT = 232500.          # Msun
# V21 numbers (create_mw.py, create_lmc.py, create_sgr.py)
MW_COUNTS = dict(ndisk_orb=100000, nbulge_orb=25000, nhalo_orb=125000,
                 nbody_stars=1000000, nbody_halo=4000000)
LMC_NBODY = 1000000
SGR_COUNTS = dict(nstar=200000, nhalo=500000)
# run_sim.sh: progenitor scaling and its position/velocity at the start time
SGR_SCALE = dict(mscale=0.86, rscale=0.805, vscale=1.0336)
SGR_SHIFT = dict(r=(56.8, 6.25, -32.37), v=(54.1, -46.79, 77.42), time=-3.0)


class Timer:
    """Collects wall-clock times of named stages."""
    def __init__(self, verbose=True):
        self.times, self.verbose, self._t0 = {}, verbose, None

    def start(self, name):
        self._name, self._t0 = name, time.perf_counter()
        if self.verbose:
            print('%s ...' % name, flush=True)

    def stop(self):
        dt = time.perf_counter() - self._t0
        self.times[self._name] = dt
        if self.verbose:
            print('    %s: %.1f s' % (self._name, dt), flush=True)
        return dt


def _outdir(tag):
    d = os.path.join(OUT, tag)
    os.makedirs(d, exist_ok=True)
    return d


def create_halo(scaleradius, gamma, beta, alpha, q_in, q_out, p_out, alpha_q, beta_q, gamma_q,
                shaperadius, outerCutoffRadius=200.0, fiducialRadius=8.0, fiducialCircularVelocity=155.0):
    """Twisted halo density with radially varying axis ratios (verbatim from V21 create_mw.py)."""
    params = dict(Type='Spheroid', outerCutoffRadius=outerCutoffRadius, cutoffStrength=2,
                  gamma=gamma, beta=beta, alpha=alpha, scaleRadius=scaleradius, densityNorm=1)
    pot0 = agama.Potential(**params)
    params['densityNorm'] = fiducialCircularVelocity**2 / (-fiducialRadius * pot0.force(fiducialRadius, 0, 0)[0])
    dens = agama.Density(params)
    ca, sa = np.cos(alpha_q), np.sin(alpha_q)
    cg, sg = np.cos(gamma_q), np.sin(gamma_q)

    def twisteddens(x):
        r2 = np.sum(x**2, axis=1)
        ch = 1 / (1 + r2 / shaperadius**2)
        q = q_out + (q_in - q_out) * ch
        p = p_out + (1 - p_out) * ch
        cb, sb = np.cos(beta_q * (1 - ch)), np.sin(beta_q * (1 - ch))
        mat = np.array([
            [ ca*cg - sa*cb*sg,  sa*cg + ca*cb*sg, sb*sg],
            [-ca*sg - sa*cb*cg, -sa*sg + ca*cb*cg, sb*cg],
            [          sa*sb   ,          -ca*sb   , cb   ]])
        xp, yp, zp = np.einsum('ijk,kj->ik', mat, x)
        s = (xp**2 + (yp/p)**2 + (zp/q)**2)**0.5 * (p*q)**(1./3)
        return dens.density(np.column_stack((s, s*0, s*0)))
    return twisteddens


def _rotate(snap, alpha_q):
    """Rotate a snapshot (posvel, mass) by alpha_q about the z axis, in place."""
    ca, sa = np.cos(alpha_q), np.sin(alpha_q)
    xv = snap[0]
    x, y = xv[:, 0]*ca - xv[:, 1]*sa, xv[:, 1]*ca + xv[:, 0]*sa
    vx, vy = xv[:, 3]*ca - xv[:, 4]*sa, xv[:, 4]*ca + xv[:, 3]*sa
    xv[:, 0], xv[:, 1], xv[:, 3], xv[:, 4] = x, y, vx, vy


def _sample_library(n, traj, weights, pot, ic, inttime, timer):
    """agama.sampleOrbitLibrary with V21's re-integration of under-sampled orbits."""
    status, result = agama.sampleOrbitLibrary(n, traj, weights)
    if not status:
        indices, trajsizes = result
        print('    reintegrating %i orbits; max # of sampling points is %i' % (len(indices), max(trajsizes)))
        traj[indices] = agama.orbit(potential=pot, ic=ic[indices], time=inttime[indices],
                                    trajsize=trajsizes + 1)
        status, result = agama.sampleOrbitLibrary(n, traj[:, 1], weights)
        if not status:
            raise RuntimeError('failed to produce the N-body model')
    return result


def make_mw(variant='triax', scale=None, seed=1, tag=None, write=True, **counts):
    """
    Schwarzschild model of the Milky Way, as in V21 create_mw.py.
    scale multiplies all of V21's orbit and particle numbers (MW_COUNTS); individual counts
    can be overridden by keyword. Returns a dict with timings, counts and output files.
    """
    n = dict(MW_COUNTS)
    if scale is not None:
        n = {k: int(round(v * scale)) for k, v in n.items()}
    n.update(counts)
    tag = tag or 'mw_%s_x%g' % (variant, n['nbody_halo'] / MW_COUNTS['nbody_halo'])
    out = _outdir(tag)
    agama.setRandomSeed(seed)
    np.random.seed(seed)
    T = Timer()

    ini = RawConfigParser()
    ini.optionxform = str
    ini.read(os.path.join(V21_SCRIPTS, 'mw_model_%s.ini' % variant))
    lower = lambda sec: dict((k.lower(), v) for k, v in ini.items(sec))
    iniHalo = dict((k, float(v)) for k, v in lower('Density halo').items())
    iniBulge, iniDisk, iniDFDisk = lower('Potential bulge'), lower('Potential disk'), lower('DF disk')

    T.start('halo potential')
    denDisk, denBulge = agama.Density(**iniDisk), agama.Density(**iniBulge)
    denStars = agama.Density(denDisk, denBulge)
    potStars = agama.Potential(agama.Potential(**iniDisk), agama.Potential(**iniBulge))
    halo_args = dict(iniHalo)
    denHalo = create_halo(**halo_args)
    potHalo = agama.Potential(type='Multipole', density=denHalo, lmax=4, mmax=4, symmetry='r',
                              rmin=1e-2, rmax=1e3)
    halo_ini = os.path.join(out, 'mw_model_%s_halo.ini' % variant)
    potHalo.export(halo_ini)
    if not np.isclose(potHalo.density(50, 0, 50), potHalo.density(50, 0, -50)):
        raise ValueError('halo density is not triaxial: no equilibrium model can be built')
    alpha_q = halo_args['alpha_q']
    halo_args['alpha_q'] = 0
    denHaloT = create_halo(**halo_args)
    potHaloT = agama.Potential(type='Multipole', density=denHaloT, lmax=4, mmax=4, symmetry='t',
                               rmin=1e-2, rmax=1e3)
    potHaloA = agama.Potential(type='Multipole', density=denHaloT, lmax=4, mmax=0, symmetry='t',
                               rmin=1e-2, rmax=1e3)
    potTotalT = agama.Potential(potStars, potHaloT)
    potTotalA = agama.Potential(potStars, potHaloA)
    T.stop()

    T.start('orbit initial conditions')
    dfDisk = agama.DistributionFunction(potential=potTotalA, **iniDFDisk)
    icDisk = agama.GalaxyModel(potTotalA, dfDisk).sample(n['ndisk_orb'])[0]
    icBulge = denBulge.sample(n['nbulge_orb'], potTotalT)[0]
    icStars = np.vstack((icBulge, icDisk))
    icHalo = potHaloT.sample(n['nhalo_orb'], potTotalT)[0]
    T.stop()

    targetStarsDen = agama.Target(type='DensityCylindricalLinear',
        gridr=agama.nonuniformGrid(25, 0.1, 20.0), gridz=agama.nonuniformGrid(20, 0.05, 2.5), mmax=4)
    targetHaloDen = agama.Target(type='DensitySphHarm', gridr=agama.nonuniformGrid(40, 0.5, 250.0),
                                 lmax=4, mmax=4)
    targetHaloKin = agama.Target(type='KinemShell', gridR=agama.nonuniformGrid(40, 0.5, 300.0), degree=1)
    consStarsDen = targetStarsDen(denStars)
    consHaloDen = targetHaloDen(denHaloT)
    # newer Agama needs the symmetry of a user-defined density for totalMass (V21 omitted it)
    massHalo = agama.Density(denHalo, symmetry='r').totalMass()
    intTimeStars = 100 * potTotalA.Tcirc(icStars)
    intTimeHalo = 100 * potTotalA.Tcirc(icHalo)
    vc = lambda r: (-potTotalA.force(r, 0, 0)[0] * r)**0.5
    print('    Vcirc at 8kpc=%g, 20kpc=%g, 50kpc=%g, 100kpc=%g' % (vc(8), vc(20), vc(50), vc(100)))

    T.start('stellar orbits (%d)' % len(icStars))
    dataStarsDen, trajStars = agama.orbit(potential=potTotalT, ic=icStars, time=intTimeStars,
                                          trajsize=101, targets=targetStarsDen)
    T.stop()
    T.start('stellar weights (solveOpt)')
    weightStars = agama.solveOpt(matrix=(np.ones((len(icStars), 1)).T, dataStarsDen.T),
        rhs=([denStars.totalMass()], consStarsDen),
        rpenl=([np.inf], np.ones(len(consStarsDen)) * np.inf), xpenq=np.ones(len(icStars)) * 0.1)
    T.stop()
    T.start('stellar N-body sampling (%d)' % n['nbody_stars'])
    stars = _sample_library(n['nbody_stars'], trajStars, weightStars, potTotalT, icStars, intTimeStars, T)
    _rotate(stars, alpha_q)
    T.stop()

    T.start('halo orbits (%d)' % len(icHalo))
    dataHaloDen, dataHaloKin, trajHalo = agama.orbit(potential=potTotalT, ic=icHalo, time=intTimeHalo,
        trajsize=101, targets=(targetHaloDen, targetHaloKin))
    T.stop()
    T.start('halo weights (solveOpt)')
    beta = 0.   # isotropic: sigma_t^2 - 2 sigma_r^2 = 0
    ngrid = len(targetHaloKin) // 2
    dataHaloKin = 2 * (1 - beta) * dataHaloKin[:, :ngrid] - dataHaloKin[:, ngrid:]
    # V21 used len(icStars) for xpenq, which equals len(icHalo) only for their numbers
    weightHalo = agama.solveOpt(matrix=(np.ones((len(icHalo), 1)).T, dataHaloDen.T, dataHaloKin.T),
        rhs=([massHalo], consHaloDen, np.zeros(ngrid)),
        rpenl=([np.inf], np.ones(len(consHaloDen)) * np.inf, np.ones(ngrid) * np.inf),
        xpenq=np.ones(len(icHalo)) * 0.1)
    T.stop()
    T.start('halo N-body sampling (%d)' % n['nbody_halo'])
    halo = _sample_library(n['nbody_halo'], trajHalo, weightHalo, potTotalT, icHalo, intTimeHalo, T)
    _rotate(halo, alpha_q)
    T.stop()

    files = {}
    if write:
        T.start('writing snapshots')
        files['stars'] = os.path.join(out, 'mw_stars.nemo')
        files['halo'] = os.path.join(out, 'mw_halo.nemo')
        agama.writeSnapshot(files['stars'], stars, 'nemo')
        agama.writeSnapshot(files['halo'], halo, 'nemo')
        T.stop()
    print('total %.1f s' % sum(T.times.values()))
    return dict(times=T.times, counts=n, halo_ini=halo_ini, files=files, out=out,
                stars=stars, halo=halo, weights=dict(stars=weightStars, halo=weightHalo))


def compare_halo(halo_ini, npoints=2000, seed=0):
    """
    Forces of a regenerated halo potential vs V21's shipped mw_model_triax_halo.ini and vs mw00.pot
    (the initial halo of the evolving potential), at random points with 1 < r < 300 kpc.
    """
    rng = np.random.default_rng(seed)
    r = 10**rng.uniform(0, np.log10(300), npoints)
    u = rng.normal(size=(npoints, 3))
    x = u / np.linalg.norm(u, axis=1)[:, None] * r[:, None]
    new = agama.Potential(halo_ini)
    res = {}
    for name, f in [('shipped halo ini', os.path.join(V21_SCRIPTS, 'mw_model_triax_halo.ini')),
                    ('mw00.pot', os.path.join(V21_POT, 'mw00.pot'))]:
        ref = agama.Potential(f)
        fn, fr = new.force(x), ref.force(x)
        rel = np.linalg.norm(fn - fr, axis=1) / np.linalg.norm(fr, axis=1)
        res[name] = rel
        print('%-17s |dF|/F median %.1e, 99%% %.1e, max %.1e' % (name, np.median(rel),
              np.percentile(rel, 99), rel.max()))
    return res


def make_lmc(mass=1.5e11, nbody=LMC_NBODY, seed=1, tag=None, write=True):
    """Truncated NFW LMC sampled from its Eddington DF, as in V21 create_lmc.py."""
    agama.setRandomSeed(seed)
    T = Timer()
    rlmc = (mass * 1e-11)**0.6 * 8.5
    T.start('LMC model and DF')
    pot = agama.Potential(type='spheroid', gamma=1, beta=3, alpha=1, scaleradius=rlmc,
                          mass=mass / MASS_UNIT, outercutoffradius=rlmc * 10)
    df = agama.DistributionFunction(type='QuasiSpherical', density=pot, potential=pot)
    T.stop()
    T.start('LMC sampling (%d)' % nbody)
    snap = agama.GalaxyModel(pot, df).sample(nbody)
    T.stop()
    files = {}
    if write:
        out = _outdir(tag or 'lmc_x%g' % (nbody / LMC_NBODY))
        files['lmc'] = os.path.join(out, 'lmc.nemo')
        agama.writeSnapshot(files['lmc'], snap, 'nemo')
    return dict(times=T.times, nbody=nbody, files=files, snap=snap, pot=pot)


def make_sgr(scale=None, seed=42, tag=None, write=True, nstar=None, nhalo=None, shift=True):
    """
    Two-component Sgr progenitor with energy-dependent mass refinement of the halo
    (V21 create_sgr.py), then scaled and placed at its t=-3 position/velocity (run_sim.sh).
    """
    n = dict(SGR_COUNTS)
    if scale is not None:
        n = {k: int(round(v * scale)) for k, v in n.items()}
    if nstar: n['nstar'] = nstar
    if nhalo: n['nhalo'] = nhalo
    agama.setRandomSeed(seed)
    np.random.seed(seed)
    T = Timer()
    T.start('Sgr model and DFs')
    comp1 = agama.Potential(type='king', mass=1000, scaleradius=1, w0=4)
    comp2 = agama.Potential(type='spheroid', mass=18000, scaleradius=8, gamma=0, beta=3, alpha=1,
                            outercutoffradius=4.5, cutoffstrength=2)
    pot = agama.Potential(comp1, comp2)
    df1 = agama.DistributionFunction(type='QuasiSpherical', density=comp1, potential=pot)
    df2 = agama.DistributionFunction(type='QuasiSpherical', density=comp2, potential=pot)
    T.stop()
    T.start('Sgr sampling (%d stars, %d halo before refinement)' % (n['nstar'], n['nhalo']))
    starp, starm = agama.GalaxyModel(pot, df1).sample(n['nstar'])
    halop, halom = agama.GalaxyModel(pot, df2).sample(n['nhalo'])
    energy = lambda p: pot.potential(p[:, 0:3]) + 0.5 * np.sum(p[:, 3:6]**2, axis=1)
    starp = starp[np.argsort(energy(starp))]
    halop = halop[np.argsort(energy(halop))]
    # input range => output range; mscale; count:  0-0.2 => 0-0.2 x1 (50%), 0.2-0.6 => 0.2-0.36
    # x2.5 (40%), 0.6-1 => 0.36-0.4 x10 (10%)
    nh = n['nhalo']
    i1, i2 = nh // 5, nh * 3 // 5
    c1, c2 = (i2 - i1) * 2 // 5, (nh - i2) // 10
    halop = np.vstack((halop[:i1],
                       halop[i1:i2][np.random.choice(i2 - i1, c1, replace=False)],
                       halop[i2:][np.random.choice(nh - i2, c2, replace=False)]))
    halom = np.hstack((np.ones(i1), np.ones(c1) * 2.5, np.ones(c2) * 10)) * halom[0]
    halop = halop[np.argsort(energy(halop))]
    T.stop()
    xv = np.vstack((starp, halop))
    m = np.hstack((starm, halom))
    if shift:
        s = SGR_SCALE
        xv[:, 0:3] *= s['rscale']
        xv[:, 3:6] *= s['vscale']
        m = m * s['mscale']
        xv[:, 0:3] += SGR_SHIFT['r']
        xv[:, 3:6] += SGR_SHIFT['v']
    print('    Sgr: %d particles (%d stars), total mass %.3g Msun' % (len(m), len(starm), m.sum() * MASS_UNIT))
    files = {}
    if write:
        out = _outdir(tag or 'sgr_x%g' % (n['nstar'] / SGR_COUNTS['nstar']))
        files['sgr'] = os.path.join(out, 'sgr_init.nemo')
        agama.writeSnapshot(files['sgr'], (xv, m), 'nemo')
    return dict(times=T.times, counts=n, nstars=len(starm), files=files, snap=(xv, m),
                time=SGR_SHIFT['time'] if shift else 0.0)


def extrapolate_mw_times(times, counts, target=MW_COUNTS):
    """Linear-in-N extrapolation of make_mw stage timings to V21's numbers."""
    ratio = {
        'orbits': target['nhalo_orb'] / counts['nhalo_orb'],
        'stars': (target['ndisk_orb'] + target['nbulge_orb']) / (counts['ndisk_orb'] + counts['nbulge_orb']),
        'nb_stars': target['nbody_stars'] / counts['nbody_stars'],
        'nb_halo': target['nbody_halo'] / counts['nbody_halo'],
    }
    est = {}
    for k, t in times.items():
        if k.startswith('stellar orbits') or k.startswith('stellar weights'):
            f = ratio['stars']
        elif k.startswith('halo orbits') or k.startswith('halo weights') or k.startswith('orbit initial'):
            f = ratio['orbits']
        elif k.startswith('stellar N-body'):
            f = ratio['nb_stars']
        elif k.startswith('halo N-body') or k.startswith('writing'):
            f = ratio['nb_halo']
        else:
            f = 1.
        est[k] = t * f
        print('%-36s %8.1f s  -> x%-6g %9.0f s' % (k, t, f, est[k]))
    print('%-36s %8.1f s  ->         %9.0f s (%.1f h)' % ('total', sum(times.values()),
          sum(est.values()), sum(est.values()) / 3600))
    return est


# ---------------------------------------------------------------- placing and merging the galaxies
# V21's trajectories are in the Milky Way-centred frame; at t=-3 the (live) Milky Way sits at rest
# at the origin, as in their MW+LMC run, so these rows are the initial phase-space coordinates.
T_START = -3.0
NEMO_START = os.path.expanduser('~/projects/python/nemo/nemo_start.sh')
RUNS = dict(R0=('mw',), R1=('mw', 'lmc'), R2=('mw', 'lmc', 'sgr'), R3=('mw', 'sgr'))
GYRFALCON = dict(eps=0.1, kmax=8, Nlev=3, fea=0.2)    # run_sim.sh


def initial_phase(component, t=T_START):
    """Centre position/velocity at time t of 'lmc' (trajlmc.txt) or 'sgr' (Sgr_snapshot/trajectory.txt)."""
    f = dict(lmc=os.path.join(V21_POT, 'trajlmc.txt'),
             sgr=os.path.join(ROOT, 'data', 'tango', 'Sgr_snapshot', 'trajectory.txt'))[component]
    tab = np.loadtxt(f)
    row = tab[np.argmin(abs(tab[:, 0] - t))]
    if abs(row[0] - t) > 1e-6:
        raise ValueError('%s has no row at t=%g (nearest %g)' % (f, t, row[0]))
    return row[1:4], row[4:7]


def _nemo(cmd):
    import subprocess
    return subprocess.run(['bash', '-c', 'set +u; source %s >/dev/null 2>&1; %s' % (NEMO_START, cmd)],
                          check=True, capture_output=True, text=True)


def assemble_runs(runs=('R0', 'R1', 'R2', 'R3'), mw_tag='mw_triax_full', lmc_tag='lmc_x1',
                  sgr_tag='sgr_x1', t=T_START, tag=None):
    """
    Initial snapshots of the live runs: Milky Way at rest at the origin, the LMC and Sgr shifted to
    their t=-3 centres. Components are stacked in the order of RUNS[run]; the index ranges go to
    <run>.json. Returns {run: snapshot file}.
    """
    import json
    snaps = dict(mw=[os.path.join(OUT, mw_tag, 'mw_stars.nemo'), os.path.join(OUT, mw_tag, 'mw_halo.nemo')],
                 lmc=[os.path.join(OUT, lmc_tag, 'lmc.nemo')], sgr=[os.path.join(OUT, sgr_tag, 'sgr_init.nemo')])
    parts = {}
    for comp, files in snaps.items():
        xv, m = map(np.concatenate, zip(*[agama.readSnapshot(f) for f in files]))
        if comp == 'sgr':   # make_sgr placed it at the run_sim.sh position; undo that first
            xv[:, 0:3] -= SGR_SHIFT['r']
            xv[:, 3:6] -= SGR_SHIFT['v']
        if comp != 'mw':
            r, v = initial_phase(comp, t)
            xv[:, 0:3] += r
            xv[:, 3:6] += v
        parts[comp] = (xv, m)
        print('%-4s %8d particles, %.4g Msun, centre (%s) kpc' % (comp, len(m), m.sum() * MASS_UNIT,
              ', '.join('%.2f' % c for c in np.median(xv[:, 0:3], axis=0))))
    out = _outdir(tag or 'runs_' + mw_tag)
    files = {}
    for run in runs:
        comps = RUNS[run]
        xv = np.vstack([parts[c][0] for c in comps])
        m = np.hstack([parts[c][1] for c in comps])
        tmp = os.path.join(out, run + '_t0.nemo')
        files[run] = os.path.join(out, run + '.nemo')
        agama.writeSnapshot(tmp, (xv, m), 'nemo')
        for f in (files[run],):
            if os.path.exists(f):
                os.remove(f)
        _nemo('s2s %s %s time=%g' % (tmp, files[run], t))
        os.remove(tmp)
        i, ranges = 0, {}
        for c in comps:
            ranges[c] = [i, i + len(parts[c][1])]
            i += len(parts[c][1])
        json.dump(dict(time=t, components=ranges, mw=mw_tag, lmc=lmc_tag, sgr=sgr_tag),
                  open(os.path.join(out, run + '.json'), 'w'), indent=1)
        print('%s: %s, %d particles' % (run, '+'.join(comps), len(m)))
    return files


def time_gyrfalcon(snapshot, nsteps=4, label=None, **params):
    """
    Evolve `snapshot` with gyrfalcON for nsteps of the longest block step (2^-(kmax-Nlev+1)),
    writing only the log. Returns wall time per unit of model time and per Gyr.
    """
    p = dict(GYRFALCON)
    p.update(params)
    dtmax = 0.5**(p['kmax'] - p['Nlev'] + 1)
    t0 = float(_nemo("tsf %s | grep -m1 'double Time'" % snapshot).stdout.split()[-1])
    tstop = t0 + nsteps * dtmax
    base = os.path.splitext(snapshot)[0] + '_' + (label or 'timing')
    for f in (base + '.log', base + '.out'):
        if os.path.exists(f):
            os.remove(f)
    args = ' '.join('%s=%s' % kv for kv in p.items())
    T = Timer()
    T.start('gyrfalcON %s, %d x %g' % (os.path.basename(snapshot), nsteps, dtmax))
    _nemo('gyrfalcON %s %s tstop=%g step=%g startout=f logfile=%s %s' %
          (snapshot, base + '.out', tstop, tstop - t0, base + '.log', args))
    wall = T.stop()
    per_unit = wall / (tstop - t0)
    print('    %.0f s per time unit, %.2f h per Gyr, %.1f h for 3 time units' %
          (per_unit, per_unit / 0.9778 / 3600, per_unit * 3 / 3600))
    return dict(wall=wall, per_unit=per_unit, log=base + '.log', params=p, nsteps=nsteps)
