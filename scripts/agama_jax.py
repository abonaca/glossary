"""JAX versions of AGAMA's time-dependent potentials, for use in streamsculptor.

Built for the evolving Milky Way + LMC of Vasiliev, Belokurov & Erkal (2021, "Tango"; Zenodo 4038141, in data/tango/),
with Sagittarius added as a moving Plummer sphere whose mass follows the N-body simulation.

Multipole snapshots (.pot coefficient files) are evaluated as in AGAMA's MultipoleInterp1d: for every (l,m) a quintic
spline in ln r through the stored Phi_lm and dPhi_lm/dr (l=0 log-scaled, l>0 scaled by l=0), power-law extrapolation
beyond the radial grid, and AGAMA's real spherical harmonics. For lmax > 2 AGAMA itself interpolates with 2d splines in
(ln r, theta), so the two agree exactly at the grid radii and differ slightly between them (tested in test_agama_jax.py).
Time dependence follows AGAMA's Evolving (nearest or linearly interpolated snapshots, clamped outside the time range),
moving centers and UniformAcceleration (cubic splines in time, linear extrapolation outside).

Units: input files are in kpc, km/s, and time in kpc/(km/s) (TIME_UNIT_MYR); potentials here take t in Myr and return
kpc^2/Myr^2, as in streamsculptor.

Usage (ipython):
    import agama_jax as aj
    aj.make_tango_tables('triax')      # once, needs agama: tabulates bulge+disk and the initial LMC
    pot = aj.tango_potential('triax', sgr=True, sgr_reflex=False)
"""

import os
import re
import tempfile

import numpy as np
import scipy.special
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import equinox as eqx
import interpax
from typing import Any

from streamsculptor.main import Potential, usys
from streamsculptor import potential

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # this file lives in scripts/
TANGO_DIR = os.path.join(PROJECT_DIR, 'data', 'tango')

KMS_TO_KPCMYR = 1.0227121650537077e-3
TIME_UNIT_MYR = 1 / KMS_TO_KPCMYR              # AGAMA time unit kpc/(km/s) in Myr
G_KMS = 4.300917270038e-6                      # G in kpc (km/s)^2 / Msun
G = G_KMS * KMS_TO_KPCMYR**2                   # G in kpc^3 / Myr^2 / Msun

# AGAMA constants (potential_multipole.cpp)
GRID_SAFETY_FACTOR = np.finfo(float).eps**(1/3)
LMAX_EXTRAPOLATION = 8


# Splines #

@jax.custom_jvp
def safe_log(x):
    """Natural log computed as log1p(m-1) + e ln2 with x = m 2^e. Works around an XLA CPU bug in jax 0.4.38, where
    some jitted scalar jnp.log calls recurse into themselves and segfault (fixed by
    XLA_FLAGS=--xla_cpu_use_thunk_runtime=false as well). m-1 is exact for m in [0.5, 1), so no precision is lost"""

    m, e = jnp.frexp(x)
    return jnp.log1p(m - 1.) + e * np.log(2.)

@safe_log.defjvp
def _safe_log_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    return safe_log(x), dx / x

def quintic_node_derivs2(x, f, df):
    """Second derivatives at the nodes of AGAMA's 'standard' quintic spline through values f and first derivatives df:
    continuous 3rd derivative at interior nodes, zero 4th derivative at the endpoints (math_spline.cpp)"""

    n = len(x)
    mat, rhs = np.zeros((n, n)), np.zeros(n)
    for i in range(n):
        if i > 0:
            hi = 1 / (x[i] - x[i-1])
            mat[i,i] += 3*hi
            mat[i-1,i] = -hi
            rhs[i] -= (20*(f[i] - f[i-1])*hi - 12*df[i] - 8*df[i-1]) * hi**2
        if i < n-1:
            hi = 1 / (x[i+1] - x[i])
            mat[i,i] += 3*hi
            mat[i+1,i] = -hi
            rhs[i] += (20*(f[i+1] - f[i])*hi - 12*df[i] - 8*df[i+1]) * hi**2
    hi = 1 / (x[1] - x[0])
    mat[0,1] = -2*hi
    rhs[0] = (30*(f[1] - f[0])*hi - 14*df[1] - 16*df[0]) * hi**2
    hi = 1 / (x[-1] - x[-2])
    mat[-1,-2] = -2*hi
    rhs[-1] = (-30*(f[-1] - f[-2])*hi + 14*df[-2] + 16*df[-1]) * hi**2

    return np.linalg.solve(mat, rhs)

def quintic_eval(x, xg, n, f, df, d2f):
    """Quintic Hermite interpolation at scalar x on nodes xg[:n] (xg may be padded beyond n);
    f, df, d2f: (..., len(xg)) values and derivatives at the nodes. x must lie within the grid"""

    i = jnp.clip(jnp.searchsorted(xg, x, side='right') - 1, 0, n - 2)
    h = xg[i+1] - xg[i]
    t = (x - xg[i]) / h
    t2, t3 = t*t, t*t*t
    h0 = 1 - t3*(10 - t*(15 - 6*t))
    h1 = t - t3*(6 - t*(8 - 3*t))
    h2 = 0.5*t2 - t3*(1.5 - t*(1.5 - 0.5*t))
    h5 = t3*(10 - t*(15 - 6*t))
    h4 = -t3*(4 - t*(7 - 3*t))
    h3 = t3*(0.5 - t*(1 - 0.5*t))

    return (f[...,i]*h0 + h*df[...,i]*h1 + h*h*d2f[...,i]*h2 +
            f[...,i+1]*h5 + h*df[...,i+1]*h4 + h*h*d2f[...,i+1]*h3)

class HermiteSpline(eqx.Module):
    """Cubic Hermite interpolation in time of K quantities from values and derivatives at the nodes,
    linearly extrapolated outside the nodes (AGAMA's CubicSpline). Called with t in Myr"""
    t: jnp.ndarray       # nodes in AGAMA time units
    f: jnp.ndarray       # (n, K)
    df: jnp.ndarray      # (n, K) derivatives w.r.t. AGAMA time

    def __init__(self, t, f, df):
        self.t, self.f, self.df = jnp.asarray(t), jnp.asarray(f), jnp.asarray(df)

    def __call__(self, t_myr):
        tt = t_myr / TIME_UNIT_MYR
        n = self.t.shape[0]
        i = jnp.clip(jnp.searchsorted(self.t, tt, side='right') - 1, 0, n - 2)
        h = self.t[i+1] - self.t[i]
        s = jnp.clip((tt - self.t[i]) / h, 0., 1.)
        interior = (self.f[i]*(1 + 2*s)*(1 - s)**2 + h*self.df[i]*s*(1 - s)**2 +
                    self.f[i+1]*s**2*(3 - 2*s) - h*self.df[i+1]*s**2*(1 - s))
        left = self.f[0] + self.df[0]*(tt - self.t[0])
        right = self.f[-1] + self.df[-1]*(tt - self.t[-1])
        return jnp.where(tt < self.t[0], left, jnp.where(tt > self.t[-1], right, interior))

def time_spline(path, value_unit=1.):
    """Spline of a time-dependent array from a file in AGAMA's format (readTimeDependentArray):
    columns t, f1..fK give a natural cubic spline with AGAMA's regularization filter, columns t, f1..fK, df1..dfK a
    Hermite spline. Node derivatives of the natural spline are taken from agama itself"""

    d = np.loadtxt(path, ndmin=2)
    if d.shape[1] == 7:
        t, f, df = d[:,0], d[:,1:4], d[:,4:7]
    else:
        import agama
        t, f = d[:,0], d[:,1:]
        df = np.column_stack([agama.Spline(t, f[:,k], reg=True)(t, der=1) for k in range(f.shape[1])])

    return HermiteSpline(t, f*value_unit, df*value_unit)


# Multipole snapshots #

def read_pot_file(path):
    """Parse an AGAMA potential file: a Multipole coefficient table, or the parameters of an analytic potential.
    Returns dict(type, params) and, for Multipole, radii [kpc] and Phi, dPhi [(km/s)^2] of shape (ncoef, nr)"""

    with open(path) as f:
        lines = f.read().splitlines()
    params, i = {}, 0
    while i < len(lines) and not lines[i].startswith('#Phi'):
        m = re.match(r'\s*(\w+)\s*=\s*(.+)', lines[i])
        if m:
            params[m.group(1).lower()] = m.group(2).strip()
        i += 1
    out = dict(type=params.get('type', '').lower(), params=params)
    if out['type'] != 'multipole':
        return out

    def table(start):
        rows = []
        for line in lines[start:]:
            if not line.strip() or line.startswith('#'):
                if rows:
                    break
                continue
            rows.append([float(v) for v in line.split()])
        return np.array(rows)

    phi = table(i + 1)
    j = next(k for k in range(i + 1, len(lines)) if lines[k].startswith('#dPhi'))
    dphi = table(j + 1)
    out.update(radii=phi[:,0], Phi=phi[:,1:].T, dPhi=dphi[:,1:].T, lmax=int(params['lmax']))

    return out

def lm_index(c):
    """(l, m) of the combined coefficient index c = l(l+1)+m"""

    l = int(np.floor(np.sqrt(c)))
    return l, c - l*(l + 1)

def extrapolation_coefs(Phi1, Phi2, dPhi1, dPhi2, r1, r2, v, s0):
    """AGAMA computeExtrapolationCoefs: Phi(r) = U (r/r1)^s + W (r/r1)^v (times ln(r/r1) for U if s==v)"""

    lnr = np.log(r2 / r1)
    num1, num2, den1, den2 = r1*dPhi1, v*Phi1, Phi1, Phi2*np.exp(-v*lnr)
    with np.errstate(divide='ignore', invalid='ignore'):
        A = lnr * (num1 - num2) / (den1 - den2)
    eps = 100 * np.finfo(float).eps
    roundoff = (abs(num1 - num2) < max(abs(num1), abs(num2))*eps or abs(den1 - den2) < max(abs(den1), abs(den2))*eps)
    if not np.isfinite(A) or A >= 0 or roundoff:
        return 0., 0., Phi1
    if abs(A + 1) < np.sqrt(np.finfo(float).eps):
        s = float(v)
    else:
        s = v + (A - scipy.special.lambertw(A*np.exp(A), k=-1 if A > -1 else 0).real) / lnr
    if v >= 0 and not np.isfinite(s):
        s = 2.
    if v < 0 and (not np.isfinite(s) or s >= 0):
        s = -2.
    if v > 0:
        s = max(s, s0)
    if v < -1:
        s = min(s, s0)
    if s != v:
        U = (r1*dPhi1 - v*Phi1) / (s - v)
        W = (r1*dPhi1 - s*Phi1) / (v - s)
    else:
        U = r1*dPhi1 - v*Phi1
        W = Phi1
    if v == 0:
        dPhi2a = U * s * np.exp(s*lnr) / r2
        dPhi2b = r2/r1 * (6*r1*(Phi2 - Phi1)/(r2 - r1) - dPhi1*(2*r1 + r2)) / (2*r2 + r1)
        if abs(dPhi2 - dPhi2b) < abs(dPhi2 - dPhi2a):
            s = 2.
            U = 0.5 * r1 * dPhi1
            W = Phi1 - U
    return s, U, W

def asymptotic_coefs(radii, Phi, dPhi, inner):
    """Power-law extrapolation coefficients S, U, W for all harmonics (AGAMA initAsympt)"""

    nc = Phi.shape[0]
    S, U, W = np.zeros(nc), np.zeros(nc), np.zeros(nc)
    i1, i2 = (0, 1) if inner else (-1, -2)
    for c in range(min(nc, (LMAX_EXTRAPOLATION + 1)**2)):
        l, _ = lm_index(c)
        S[c], U[c], W[c] = extrapolation_coefs(Phi[c,i1], Phi[c,i2], dPhi[c,i1], dPhi[c,i2], radii[i1], radii[i2],
                                               l if inner else -l-1, S[0])
    return S, U, W

def prepare_multipole(radii, Phi, dPhi):
    """Radial splines (AGAMA MultipoleInterp1d) and extrapolation coefficients of one Multipole snapshot.
    Phi, dPhi: (ncoef, nr) in (km/s)^2 and (km/s)^2/kpc. Returns a dict of numpy arrays"""

    nc, nr = Phi.shape
    s, U, W = extrapolation_coefs(Phi[0,0], Phi[0,1], dPhi[0,0], dPhi[0,1], radii[0], radii[1], 0, np.nan)
    inv_phi0 = 1/W if s > 0 else 0.
    log_scaling = bool(np.all(Phi[0] < 0))
    if np.any(Phi[0]*inv_phi0 >= 1):
        inv_phi0 = 0.

    f, df = np.zeros((nc, nr)), np.zeros((nc, nr))
    for c in range(nc):
        if log_scaling and c == 0:
            f[c] = np.log(inv_phi0 - 1/Phi[0])
            df[c] = radii * dPhi[0] / (Phi[0] * (inv_phi0*Phi[0] - 1))
        elif log_scaling:
            f[c] = Phi[c] / Phi[0]
            df[c] = (dPhi[c] - f[c]*dPhi[0]) * radii / Phi[0]
        else:
            f[c] = Phi[c]
            df[c] = radii * dPhi[c]
    logr = np.log(radii)
    d2f = np.array([quintic_node_derivs2(logr, f[c], df[c]) if np.any(f[c]!=0) or np.any(df[c]!=0) else np.zeros(nr)
                    for c in range(nc)])

    inner, outer = asymptotic_coefs(radii, Phi, dPhi, True), asymptotic_coefs(radii, Phi, dPhi, False)
    return dict(logr=logr, f=f, df=df, d2f=d2f, log_scaling=log_scaling, inv_phi0=inv_phi0,
                rmin=radii[0], rmax=radii[-1], s_in=inner[0], u_in=inner[1], w_in=inner[2],
                s_out=outer[0], u_out=outer[1], w_out=outer[2])

def stack_multipoles(snaps):
    """Pad a list of prepare_multipole outputs to a common number of harmonics and radii and stack them"""

    nc = max(s['f'].shape[0] for s in snaps)
    nr = max(s['f'].shape[1] for s in snaps)
    out = {}
    for key in ['f', 'df', 'd2f']:
        out[key] = np.stack([np.pad(s[key], ((0, nc - s[key].shape[0]), (0, nr - s[key].shape[1]))) for s in snaps])
    # pad the radial grids with increasing values beyond the last node (never used for interpolation)
    out['logr'] = np.stack([np.concatenate([s['logr'], s['logr'][-1] + 1 + np.arange(nr - len(s['logr']))])
                            for s in snaps])
    out['nr'] = np.array([len(s['logr']) for s in snaps])
    for key in ['s_in', 'u_in', 'w_in', 's_out', 'u_out', 'w_out']:
        out[key] = np.stack([np.pad(s[key], (0, nc - len(s[key]))) for s in snaps])
    for key in ['log_scaling', 'inv_phi0', 'rmin', 'rmax']:
        out[key] = np.array([s[key] for s in snaps])
    out['lmax'] = int(np.sqrt(nc)) - 1

    return out

def legendre_q(lmax, ct):
    """Q_lm(cos theta) such that AGAMA's normalized associated Legendre function is P_lm = Q_lm sin^m(theta),
    including the normalization and the (-1)^m phase. Returns dict {(l, m): Q_lm} for 0 <= m <= l <= lmax"""

    q = {}
    for m in range(lmax + 1):
        # unnormalized: Q_mm = (2m-1)!!, Q_{m+1,m} = (2m+1) ct Q_mm, then the standard recurrence
        qmm = float(scipy.special.factorial2(2*m - 1)) if m > 0 else 1.
        prev2, prev1 = None, qmm*jnp.ones_like(ct)
        for l in range(m, lmax + 1):
            if l == m:
                cur = prev1
            elif l == m + 1:
                cur = ct * (2*m + 1) * prev1
            else:
                cur = (ct * (2*l - 1) * prev1 - (l + m - 1) * prev2) / (l - m)
            norm = np.sqrt((2*l + 1)/(4*np.pi) * scipy.special.factorial(l - m)/scipy.special.factorial(l + m))
            q[(l, m)] = cur * norm * (-1)**m
            if l > m:
                prev2, prev1 = prev1, cur
    return q

def sph_harm_real(lmax, xyz, r):
    """AGAMA's real spherical harmonics (times 2 sqrt(pi), so that Phi = sum_c Phi_c(r) Y_c) for all c = l(l+1)+m,
    written as polynomials in x/r, y/r, z/r so that autodiff is regular on the z axis"""

    x, y, z = xyz[0]/r, xyz[1]/r, xyz[2]/r
    q = legendre_q(lmax, z)
    # Re and Im of (x + iy)^m = sin^m(theta) cos(m phi), sin^m(theta) sin(m phi)
    cm, sm = [jnp.ones_like(x)], [jnp.zeros_like(x)]
    for m in range(1, lmax + 1):
        c_prev, s_prev = cm[-1], sm[-1]
        cm.append(x*c_prev - y*s_prev)
        sm.append(x*s_prev + y*c_prev)
    y_c = []
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            a = abs(m)
            if m == 0:
                y_c.append(2*np.sqrt(np.pi) * q[(l, 0)])
            else:
                y_c.append(2*np.sqrt(2*np.pi) * q[(l, a)] * (cm[a] if m > 0 else sm[a]))
    return jnp.stack(y_c)

def _radial_coefs(st, k, logr_x):
    """Phi_c(r) of snapshot k at ln r = logr_x, inside the grid (AGAMA MultipoleInterp1d::evalCyl)"""

    c = quintic_eval(logr_x, st['logr'][k], st['nr'][k], st['f'][k], st['df'][k], st['d2f'][k])
    expx = jnp.exp(c[0])
    phi0 = jnp.where(st['log_scaling'][k], 1 / (st['inv_phi0'][k] - expx), c[0])
    rest = jnp.where(st['log_scaling'][k], c[1:] * phi0, c[1:])
    return jnp.concatenate([phi0[None], rest])

def _power_law_coefs(s, u, w, v, dlogr):
    """Phi_c(r) of AGAMA PowerLawMultipole; dlogr = ln(r/r0)"""

    rv = jnp.exp(dlogr * v)
    rs = jnp.where(s != v, jnp.exp(dlogr * s), rv)
    urs = u * rs * jnp.where((s != v) | (u == 0), 1., dlogr)
    return urs + w * rv

class MultipoleSnapshots(Potential):
    """One or more AGAMA Multipole potentials (stacked snapshots); potential(xyz, t, k) evaluates snapshot k.
    Values in kpc^2/Myr^2"""
    st: dict
    l_of_c: jnp.ndarray
    lmax: int = eqx.field(static=True)

    def __init__(self, snaps, units=usys):
        super().__init__(units)
        st = stack_multipoles(snaps)
        self.lmax = st.pop('lmax')
        self.st = {key: jnp.asarray(val) for key, val in st.items()}
        self.l_of_c = jnp.array([lm_index(c)[0] for c in range((self.lmax + 1)**2)], dtype=float)

    def snapshot_potential(self, xyz, k):
        st = self.st
        r = jnp.sqrt(jnp.sum(xyz**2))
        logr = safe_log(r)
        rmin, rmax = st['rmin'][k], st['rmax'][k]
        logrmin, logrmax = st['logr'][k,0], st['logr'][k,st['nr'][k]-1]
        in_inner = r < rmin * (1 + GRID_SAFETY_FACTOR)
        in_outer = r > rmax * (1 - GRID_SAFETY_FACTOR)
        logr_clip = jnp.clip(logr, logrmin, logrmax)
        c_mid = _radial_coefs(st, k, logr_clip)
        c_in = _power_law_coefs(st['s_in'][k], st['u_in'][k], st['w_in'][k], self.l_of_c, logr - logrmin)
        c_out = _power_law_coefs(st['s_out'][k], st['u_out'][k], st['w_out'][k], -self.l_of_c - 1, logr - logrmax)
        coefs = jnp.where(in_inner, c_in, jnp.where(in_outer, c_out, c_mid))
        return jnp.sum(coefs * sph_harm_real(self.lmax, xyz, r)) * KMS_TO_KPCMYR**2

    def potential(self, xyz, t, k=0):
        return self.snapshot_potential(xyz, k)

class EvolvingMultipole(Potential):
    """AGAMA Evolving potential made of Multipole snapshots at the given times [AGAMA time units]:
    linear interpolation between snapshots (linear=True) or the nearest snapshot (AGAMA's default),
    clamped to the first/last snapshot outside the time range. t in Myr"""
    snaps: MultipoleSnapshots
    times: jnp.ndarray
    linear: bool = eqx.field(static=True)

    def __init__(self, times, snaps, linear=True, units=usys):
        super().__init__(units)
        self.times = jnp.asarray(times)
        self.snaps = MultipoleSnapshots(snaps, units=units)
        self.linear = linear

    def weights(self, t):
        """Index of the left snapshot and its weight (AGAMA searchInterp)"""

        tt = t / TIME_UNIT_MYR
        n = self.times.shape[0]
        i = jnp.searchsorted(self.times, tt, side='right') - 1
        inside = (i >= 0) & (i < n - 1)
        i = jnp.clip(i, 0, n - 2)
        w_left = (self.times[i+1] - tt) / (self.times[i+1] - self.times[i])
        if not self.linear:
            i = jnp.where(self.times[i+1] - tt < tt - self.times[i], i + 1, i)
            w_left = 1.
        # outside the time range: the first or the last snapshot
        i = jnp.where(inside, i, jnp.where(tt < self.times[0], 0, n - 1))
        w_left = jnp.where(inside, w_left, 1.)
        return i, w_left

    def potential(self, xyz, t):
        n = self.times.shape[0]
        if n == 1:
            return self.snaps.snapshot_potential(xyz, 0)
        i, w = self.weights(t)
        phi_left = self.snaps.snapshot_potential(xyz, i)
        phi_right = self.snaps.snapshot_potential(xyz, jnp.minimum(i + 1, n - 1))
        return w * phi_left + (1 - w) * phi_right

def load_evolving(ini, section, linear=True, extra=None):
    """EvolvingMultipole from a [section] with 'Timestamps' in an AGAMA ini file.
    extra: dict {filename: prepare_multipole output} replacing snapshots that are not Multipole files"""

    folder = os.path.dirname(ini)
    times, files = [], []
    with open(ini) as f:
        lines = f.read().splitlines()
    i = next(k for k, line in enumerate(lines) if line.strip().lower() == f'[potential {section}]'.lower())
    j = next(k for k in range(i, len(lines)) if lines[k].strip().lower() == 'timestamps')
    for line in lines[j+1:]:
        parts = line.split()
        if len(parts) != 2 or line.startswith('['):
            break
        times.append(float(parts[0]))
        files.append(parts[1])
    snaps = []
    for fname in files:
        if extra is not None and fname in extra:
            snaps.append(extra[fname])
            continue
        d = read_pot_file(os.path.join(folder, fname))
        if d['type'] != 'multipole':
            raise ValueError(f"{fname} is not a Multipole file; supply it via 'extra'")
        snaps.append(prepare_multipole(d['radii'], d['Phi'], d['dPhi']))

    return EvolvingMultipole(np.array(times), snaps, linear=linear)


# Other components #

class AxisymTablePotential(Potential):
    """Axisymmetric potential tabulated on a grid in u=asinh(R/a), v=asinh(z/b) (cubic spline);
    the table covers z >= 0 and is mirrored to z < 0 (as in gd1_test)"""
    spl: Any
    a: float
    b: float

    def __init__(self, u, v, phi, a, b, units=usys):
        super().__init__(units)
        v_full = np.hstack([-v[:0:-1], v])
        phi_full = np.hstack([phi[:,:0:-1], phi])
        self.spl = interpax.Interpolator2D(x=jnp.asarray(u), y=jnp.asarray(v_full), f=jnp.asarray(phi_full),
                                           method='cubic2')
        self.a, self.b = a, b

    def potential(self, xyz, t):
        R = jnp.sqrt(xyz[0]**2 + xyz[1]**2)
        return self.spl(jnp.arcsinh(R/self.a), jnp.arcsinh(xyz[2]/self.b))

class UniformAcceleration(Potential):
    """AGAMA UniformAcceleration: Phi = -a(t).x, so that every particle feels the acceleration a(t) from the file
    (minus the acceleration of the Milky Way center). acc: HermiteSpline in kpc/Myr^2"""
    acc: HermiteSpline

    def __init__(self, acc, units=usys):
        super().__init__(units)
        self.acc = acc

    def potential(self, xyz, t):
        return -jnp.sum(self.acc(t) * xyz)

class EvolvingPlummer(Potential):
    """Plummer sphere moving along a trajectory (HermiteSpline, kpc) with mass linearly interpolated in time
    (clamped outside the tabulated range, as an AGAMA Evolving potential with linearInterp)"""
    center: HermiteSpline
    t_mass: jnp.ndarray     # AGAMA time units
    mass: jnp.ndarray       # Msun
    b: float

    def __init__(self, center, t_mass, mass, b, units=usys):
        super().__init__(units)
        self.center, self.t_mass, self.mass, self.b = center, jnp.asarray(t_mass), jnp.asarray(mass), b

    def mass_at(self, t):
        return jnp.interp(t / TIME_UNIT_MYR, self.t_mass, self.mass)

    def potential(self, xyz, t):
        dx = xyz - self.center(t)
        return -G * self.mass_at(t) / jnp.sqrt(jnp.sum(dx**2) + self.b**2)

class PlummerReflex(Potential):
    """Reflex acceleration of a rigid Milky Way pulled by an EvolvingPlummer: the frame centered on the Milky Way
    accelerates by a_MW = -grad Phi_plummer(0), so particles feel -a_MW; Phi = a_MW(t).x"""
    plummer: EvolvingPlummer

    def __init__(self, plummer, units=usys):
        super().__init__(units)
        self.plummer = plummer

    def a_mw(self, t):
        c = self.plummer.center(t)
        return G * self.plummer.mass_at(t) * c / (jnp.sum(c**2) + self.plummer.b**2)**1.5

    def potential(self, xyz, t):
        return jnp.sum(self.a_mw(t) * xyz)


# Tango model #

SGR_SCALE_RADIUS = 1.5    # kpc (Vasiliev+2021 approximate models)
TANGO_N_DISK = 400        # (R, z) grid of the bulge+disk table, uniform in asinh(R/1 kpc), asinh(|z|/0.1 kpc)
TANGO_RMAX = 500.

def tango_dir(variant='triax'):
    return os.path.join(TANGO_DIR, f'potentials_{variant}')

def tango_tables_path(variant='triax'):
    return os.path.join(TANGO_DIR, f'tables_{variant}.npz')

def tango_ini(variant='triax', which='evolving'):
    return os.path.join(tango_dir(variant), f'potential_{which}.ini')

def read_ini_sections(path):
    """Sections of an AGAMA ini file as {name: {key: value}} (lower-case keys; 'Timestamps' blocks skipped)"""

    out, cur = {}, None
    with open(path) as f:
        for line in f:
            line = line.split('#')[0].strip()
            m = re.match(r'\[potential\s*(\w*)\]', line, re.I)
            if m:
                cur = m.group(1).lower()
                out[cur] = {}
            elif cur is not None and '=' in line:
                k, v = [s.strip() for s in line.split('=', 1)]
                out[cur][k.lower()] = v
    return out

def make_tango_tables(variant='triax', save=True):
    """Tabulate with agama the components that are not Multipole files (needs agama):
    the bulge + disk potential on an (R, z) grid, and the initial analytic LMC (lmc00) as a fine l=0 Multipole.
    Saved to data/tango/tables_<variant>.npz"""

    import agama
    sec = read_ini_sections(tango_ini(variant))
    bulge_disk = agama.Potential(sec['bulge'], sec['disk'])   # N-body units (G=1, kpc, km/s), as in the ini

    u = np.linspace(0, np.arcsinh(TANGO_RMAX/1.), TANGO_N_DISK)
    v = np.linspace(0, np.arcsinh(TANGO_RMAX/0.1), TANGO_N_DISK)
    R, z = np.sinh(u), 0.1*np.sinh(v)
    RR, zz = np.meshgrid(R, z, indexing='ij')
    xyz = np.column_stack([RR.ravel(), RR.ravel()*0, zz.ravel()])
    phi = bulge_disk.potential(xyz).reshape(RR.shape) * KMS_TO_KPCMYR**2

    lmc00 = read_pot_file(os.path.join(tango_dir(variant), 'lmc00.pot'))
    lmc0 = agama.Potential(**lmc00['params'])
    radii = np.logspace(-3, 4, 200)
    pts = np.column_stack([radii, radii*0, radii*0])
    lmc_phi = lmc0.potential(pts)
    lmc_dphi = -lmc0.force(pts)[:,0]

    out = dict(disk_u=u, disk_v=v, phi_bulge_disk=phi, disk_scales=np.array([1., 0.1]),
               lmc0_radii=radii, lmc0_phi=lmc_phi, lmc0_dphi=lmc_dphi)
    if save:
        np.savez(tango_tables_path(variant), **out)
        print(f'saved {tango_tables_path(variant)}')
    return out

def load_tango_tables(variant='triax'):
    return dict(np.load(tango_tables_path(variant)))

def tango_components(variant='triax', linear=True, static=False):
    """Components of the Tango Milky Way + LMC: dict with 'bulge_disk', 'halo', and unless static, 'lmc', 'acc'.
    static=True: bulge, disk, and the initial halo (mw00) only, i.e. the unperturbed Milky Way"""

    tab = load_tango_tables(variant)
    folder = tango_dir(variant)
    ini = tango_ini(variant)
    comps = dict(bulge_disk=AxisymTablePotential(tab['disk_u'], tab['disk_v'], tab['phi_bulge_disk'],
                                                *tab['disk_scales']))
    if static:
        d = read_pot_file(os.path.join(folder, 'mw00.pot'))
        comps['halo'] = EvolvingMultipole([0.], [prepare_multipole(d['radii'], d['Phi'], d['dPhi'])])
        return comps

    comps['halo'] = load_evolving(ini, 'halo', linear=linear)
    lmc0 = prepare_multipole(tab['lmc0_radii'], tab['lmc0_phi'][None], tab['lmc0_dphi'][None])
    lmc = load_evolving(ini, 'lmc', linear=linear, extra={'lmc00.pot': lmc0})
    # trajlmc.txt: t, x, v -> Hermite spline; velocities in km/s are derivatives w.r.t. AGAMA time
    comps['lmc'] = potential.TimeDepTranslatingPotential(pot=lmc, center_spl=time_spline(os.path.join(folder,
                                                                                                    'trajlmc.txt')))
    comps['acc'] = UniformAcceleration(time_spline(os.path.join(folder, 'accel.txt'), value_unit=KMS_TO_KPCMYR**2))
    return comps

def sgr_plummer(b=SGR_SCALE_RADIUS):
    """Sagittarius as a Plummer sphere on its N-body trajectory (data/tango/Sgr_snapshot/trajectory.txt), with
    the mass within 5 kpc of the remnant (column 8) as its time-dependent mass"""

    d = np.loadtxt(os.path.join(TANGO_DIR, 'Sgr_snapshot', 'trajectory.txt'))
    return EvolvingPlummer(HermiteSpline(d[:,0], d[:,1:4], d[:,4:7]), d[:,0], d[:,7], b)

def tango_potential(variant='triax', linear=True, static=False, sgr=True, sgr_reflex=False, b_sgr=SGR_SCALE_RADIUS):
    """Tango Milky Way + LMC (+ Sgr) as a streamsculptor potential. See tango_components; sgr adds the Sgr Plummer,
    sgr_reflex the Milky Way reflex acceleration due to Sgr (both ignored if static). The reflex is off by default,
    consistent with the N-body run that produced the Sgr trajectory, where the Milky Way did not respond to Sgr"""

    comps = list(tango_components(variant, linear=linear, static=static).values())
    if sgr and not static:
        p = sgr_plummer(b_sgr)
        comps.append(p)
        if sgr_reflex:
            comps.append(PlummerReflex(p))
    return potential.Potential_Combine(comps, units=usys)
