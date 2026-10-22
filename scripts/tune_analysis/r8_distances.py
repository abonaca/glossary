"""Round-8 run-3 MW, LMC and Sgr centres vs time: distance from the t=0 MW centre, and the MW centre's shift from its t=-3
position (it starts at rest) compared with the shifts expected from Sgr and the LMC alone (point masses, with
Sgr's measured mass within 20 kpc and the LMC's NFW mass enclosed within its distance to the MW)."""
import numpy as np
from scipy.interpolate import CubicHermiteSpline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('default')
C = '/home/ana/projects/glossary/output/v21_sim/tune/round08/run03/centres.npz'
SGR_M = '/home/ana/projects/glossary/output/v21_sim/tune/round07/run00/sgr_bound_mass.npz'   # same Sgr model, r7 base
MU = 232500.                    # Msun per model mass unit (G=1, kpc, km/s)
M_LMC = 1.499e11 / MU
RS_LMC = 1.5**0.6 * 8.5         # v21_ics.make_lmc scale radius [kpc], cutoff at 10 rs
COL = dict(mw='tab:orange', lmc='tab:blue', sgr='tab:green')


def load():
    c = np.load(C)
    return c['times'], {k: c[k] for k in ('mw', 'lmc', 'sgr')}


def lmc_enclosed(r):
    m = lambda x: np.log(1 + x) - x / (1 + x)
    return M_LMC * np.minimum(m(r / RS_LMC) / m(10.), 1.)


def kicks(t, w, nfine=4000):
    """Velocity change of the MW centre from each perturber as a point mass, along the measured orbits
    (cubic Hermite interpolation between snapshots using the tracked velocities)."""
    tf = np.linspace(t[0], t[-1], nfine)
    pos = lambda k: CubicHermiteSpline(t, w[k][:, :3], w[k][:, 3:])(tf)
    xmw = pos('mw')
    s = np.load(SGR_M)
    msgr = np.interp(tf, s['times'], s['within'][:, 2])
    out = {}
    for k, mass in (('sgr', lambda r: msgr), ('lmc', lmc_enclosed)):
        d = pos(k) - xmw
        r = np.linalg.norm(d, axis=1)
        a = (mass(r) / r**3)[:, None] * d
        out[k] = np.vstack((np.zeros(3), np.cumsum(0.5 * (a[1:] + a[:-1]) * np.diff(tf)[:, None], axis=0)))
    return tf, out


def plot(out='/home/ana/projects/glossary/plots/tune_r8_run03_distances.png'):
    t, w = load()
    ref = w['mw'][-1, :3]
    tf, dv = kicks(t, w)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    for k, lab in (('mw', 'MW'), ('lmc', 'LMC'), ('sgr', 'Sgr')):
        ax[0].semilogy(t * 1e3, np.linalg.norm(w[k][:, :3] - ref, axis=1), 'o-', color=COL[k], ms=4, label=lab)
    ax[0].set_ylabel('distance from the t = 0 MW centre [kpc]')
    ax[0].legend(fontsize=9)
    shift = lambda a: np.vstack((np.zeros(3), np.cumsum(0.5 * (a[1:] + a[:-1]) * np.diff(tf)[:, None], axis=0)))
    dx = {k: shift(dv[k]) for k in dv}
    ax[1].plot(t * 1e3, np.linalg.norm(w['mw'][:, :3] - w['mw'][0, :3], axis=1), 'o', color=COL['mw'], ms=5,
               label='MW centre (measured)')
    ax[1].plot(tf * 1e3, np.linalg.norm(dx['sgr'] + dx['lmc'], axis=1), '-', color='k', lw=1.5,
               label='Sgr + LMC (point-mass estimate)')
    ax[1].plot(tf * 1e3, np.linalg.norm(dx['sgr'], axis=1), '-', color=COL['sgr'], lw=1.5, label='Sgr only (M(<20 kpc))')
    ax[1].plot(tf * 1e3, np.linalg.norm(dx['lmc'], axis=1), '-', color=COL['lmc'], lw=1.5,
               label='LMC only (enclosed NFW mass)')
    ax[1].set_ylabel('MW centre shift from its t = -3 position [kpc]')
    ax[1].legend(fontsize=8)
    for a in ax:
        a.set_xlabel('t [Myr]')
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(out)
    for tt in (-2.5, -2.0, -1.5, -1.0, -0.5, 0.0):
        j = np.argmin(np.abs(tf - tt)); jj = np.argmin(np.abs(t - tt))
        print('t %5.0f Myr  MW shift measured %5.1f  Sgr %4.1f  LMC %5.1f  Sgr+LMC %5.1f kpc' % (tt * 1e3,
              np.linalg.norm(w['mw'][jj, :3] - w['mw'][0, :3]), np.linalg.norm(dx['sgr'][j]),
              np.linalg.norm(dx['lmc'][j]), np.linalg.norm(dx['sgr'][j] + dx['lmc'][j])))

if __name__ == '__main__':
    plot()
