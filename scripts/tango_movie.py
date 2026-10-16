"""Diagnostics of the time evolution of the Tango potential (agama_jax): a movie of the Milky Way halo and LMC
density response, and tests for interpolation artifacts in time.

Usage (ipython):
    import tango_movie as tm
    tm.kink_table()          # force at fixed points: smoothness across snapshot times
    tm.make_movie()          # plots/tango_evolution.mp4
"""

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import jax
import jax.numpy as jnp

import agama_jax as aj

PLOT_DIR = os.path.join(aj.PROJECT_DIR, 'plots')
SNAP_TIMES_MYR = np.arange(-1.5, 0.25 + 1e-9, 0.0625) * aj.TIME_UNIT_MYR

# fixed probe points [kpc]: Galactocentric radii along a GD-1-like direction (MW halo), offsets from the LMC center
PROBE_DIR = np.array([-0.4, 0.5, 0.77]) / np.linalg.norm([-0.4, 0.5, 0.77])
MW_PROBE_R = [15., 30., 60., 120.]
LMC_PROBE_R = [2., 5., 10., 20.]


# Evaluation #

def density_fn(pot):
    """Density [Msun/kpc^3] from the Laplacian of a potential, vectorized over points at one time [Myr]"""

    lap = lambda x, t: jnp.trace(jax.hessian(pot.potential)(x, t))
    return jax.jit(jax.vmap(lambda x, t: lap(x, t) / (4*np.pi*aj.G), in_axes=(0, None)))

def force_fn(pot):
    """Force [kpc/Myr^2] at fixed points as a function of time, vectorized over times"""

    return jax.jit(jax.vmap(lambda x, t: -jax.vmap(pot.gradient, in_axes=(0, None))(x, t), in_axes=(None, 0)))

def components(linear=True):
    comps = aj.tango_components('triax', linear=linear)
    return dict(halo=comps['halo'], lmc=comps['lmc'].pot, lmc_center=comps['lmc'].center_spl)

def plane_grid(extent, n, plane):
    """Points on an n x n grid in the x-z or x-y plane"""

    u = np.linspace(-extent, extent, n)
    U, V = np.meshgrid(u, u, indexing='xy')
    zero = np.zeros_like(U)
    xyz = np.stack([U, V, zero], -1) if plane == 'xy' else np.stack([U, zero, V], -1)
    return u, xyz.reshape(-1, 3)


# Interpolation artifacts in time #

def probe_forces(times, linear=True):
    """Forces of the MW halo and the LMC (in its own frame) at the probe points, at the given times [Myr]"""

    c = components(linear)
    xm = jnp.asarray(np.outer(MW_PROBE_R, PROBE_DIR))
    xl = jnp.asarray(np.outer(LMC_PROBE_R, PROBE_DIR))
    t = jnp.asarray(times)
    return np.asarray(force_fn(c['halo'])(xm, t)), np.asarray(force_fn(c['lmc'])(xl, t))

def kink_table(verbose=True):
    """Smoothness of the force at the probe points across snapshot times. With linear interpolation the force is
    piecewise linear in time; the change of slope at each snapshot measures how much the snapshot-to-snapshot
    evolution departs from a smooth trend (N-body noise or genuine change). Returns per probe:
    step: |F_{k+1} - F_k| / |F| (the jump in nearest mode), kink: |slope_k - slope_{k-1}| / rms slope"""

    f_mw, f_lmc = probe_forces(SNAP_TIMES_MYR)
    out = {}
    for name, f, radii in [('MW halo', f_mw, MW_PROBE_R), ('LMC', f_lmc, LMC_PROBE_R)]:
        for j, r in enumerate(radii):
            fj = f[:,j]
            step = np.linalg.norm(np.diff(fj, axis=0), axis=1) / np.linalg.norm(fj[:-1], axis=1)
            slope = np.diff(fj, axis=0)
            kink = np.linalg.norm(np.diff(slope, axis=0), axis=1) / np.sqrt(np.mean(np.sum(slope[1:]**2, axis=1)))
            out[(name, r)] = dict(step_first=step[0], step_median=np.median(step[1:]), step_max=np.max(step[1:]),
                                  kink_median=np.median(kink[1:]), kink_max=np.max(kink[1:]), kink_first=kink[0])
            if verbose:
                v = out[(name, r)]
                print(f'{name:8s} r={r:5.0f} kpc  nearest-mode jump |dF|/F: first {v["step_first"]:.1e}, '
                      f'median {v["step_median"]:.1e}, max {v["step_max"]:.1e}   '
                      f'slope change / rms slope: median {v["kink_median"]:.2f}, max {v["kink_max"]:.2f}, '
                      f'first {v["kink_first"]:.2f}')
    return out

def plot_probe_forces(dt=2., save=True):
    """Force relative to its initial value at the probe points vs time, linear and nearest interpolation"""

    t = np.arange(SNAP_TIMES_MYR[0] - 100, SNAP_TIMES_MYR[-1] + 50, dt)
    res = {lin: probe_forces(t, linear=lin) for lin in (True, False)}
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for i, (name, radii) in enumerate([('MW halo', MW_PROBE_R), ('LMC (own frame)', LMC_PROBE_R)]):
        plt.sca(ax[i])
        for j, r in enumerate(radii):
            for lin, ls in [(True, '-'), (False, ':')]:
                f = res[lin][i][:,j]
                dfrac = np.linalg.norm(f - f[0], axis=1) / np.linalg.norm(f[0])
                plt.plot(t/1e3, dfrac, ls, color=f'C{j}', lw=1.2,
                         label=f'r={r:.0f} kpc' if lin else None)
        for ts in SNAP_TIMES_MYR:
            plt.axvline(ts/1e3, color='0.85', lw=0.5, zorder=0)
        plt.ylabel(r'$|F(t) - F_0| / |F_0|$')
        plt.title(f'{name}: linear (solid) and nearest (dotted) interpolation', fontsize='medium')
        plt.legend(fontsize='small', ncol=4)
    plt.xlabel('t [Gyr]')
    plt.tight_layout()
    if save:
        os.makedirs(PLOT_DIR, exist_ok=True)
        plt.savefig(os.path.join(PLOT_DIR, 'tango_probe_forces.png'), dpi=150)
    return fig


# Movie #

def make_movie(dt=5., n_grid=121, mw_extent=100., lmc_extent=30., fps=15, linear=True, fname='tango_evolution.mp4'):
    """Movie of the density contrast of the Milky Way halo (x-z and x-y planes, relative to the initial halo) and of
    the LMC in its own frame (x-z plane, relative to the initial LMC), with the LMC and Sgr positions and the force
    at the probe points vs time"""

    c = components(linear)
    sgr = aj.sgr_plummer()
    t_frames = np.arange(SNAP_TIMES_MYR[0] - 50, SNAP_TIMES_MYR[-1] + 1e-6, dt)

    u_mw, x_xz = plane_grid(mw_extent, n_grid, 'xz')
    _, x_xy = plane_grid(mw_extent, n_grid, 'xy')
    u_lmc, x_lmc = plane_grid(lmc_extent, n_grid, 'xz')
    rho_mw, rho_lmc = density_fn(c['halo']), density_fn(c['lmc'])
    t0 = SNAP_TIMES_MYR[0] - 1.
    ref = dict(xz=rho_mw(jnp.asarray(x_xz), t0), xy=rho_mw(jnp.asarray(x_xy), t0), lmc=rho_lmc(jnp.asarray(x_lmc), t0))

    def frame_maps(t):
        return dict(xz=np.asarray(rho_mw(jnp.asarray(x_xz), t) / ref['xz'] - 1).reshape(n_grid, n_grid),
                    xy=np.asarray(rho_mw(jnp.asarray(x_xy), t) / ref['xy'] - 1).reshape(n_grid, n_grid),
                    lmc=np.asarray(rho_lmc(jnp.asarray(x_lmc), t) / ref['lmc'] - 1).reshape(n_grid, n_grid))

    t_track = np.arange(SNAP_TIMES_MYR[0] - 1000, SNAP_TIMES_MYR[-1] + 1e-6, 5.)
    lmc_track = np.asarray(jax.vmap(c['lmc_center'])(jnp.asarray(t_track)))
    sgr_track = np.asarray(jax.vmap(sgr.center)(jnp.asarray(t_track)))
    t_series = np.arange(t_frames[0], t_frames[-1] + 1e-6, 2.)
    f_mw, f_lmc = probe_forces(t_series, linear=linear)

    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.6, 1])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_t = fig.add_subplot(gs[1, :])
    kw = dict(cmap='RdBu_r', vmin=-0.5, vmax=0.5, origin='lower', interpolation='nearest')
    m0 = frame_maps(t_frames[0])
    ims = [axes[0].imshow(m0['xz'], extent=[-mw_extent, mw_extent]*2, **kw),
           axes[1].imshow(m0['xy'], extent=[-mw_extent, mw_extent]*2, **kw),
           axes[2].imshow(m0['lmc'], extent=[-lmc_extent, lmc_extent]*2, **kw)]
    titles = ['MW halo, x-z', 'MW halo, x-y', 'LMC (own frame), x-z']
    labels = [('x [kpc]', 'z [kpc]'), ('x [kpc]', 'y [kpc]'), ('x [kpc]', 'z [kpc]')]
    for a, ti, (xl, yl) in zip(axes, titles, labels):
        a.set_title(ti, fontsize='medium')
        a.set_xlabel(xl)
        a.set_ylabel(yl)
    fig.colorbar(ims[2], ax=axes, label=r'$\rho(t)/\rho_{\rm initial} - 1$', fraction=0.015, pad=0.01)

    trails, dots = [], []
    for k, (a, (i, j)) in enumerate(zip(axes[:2], [(0, 2), (0, 1)])):
        for track, col, lab in [(lmc_track, 'k', 'LMC'), (sgr_track, 'm', 'Sgr')]:
            tr, = a.plot([], [], '-', color=col, lw=1)
            d, = a.plot([], [], 'o', color=col, ms=5, label=lab if k == 0 else None)
            trails.append((tr, track, i, j))
            dots.append((d, track, i, j))
        a.set_xlim(-mw_extent, mw_extent)
        a.set_ylim(-mw_extent, mw_extent)
    axes[0].legend(loc='lower left', fontsize='small')

    for j, r in enumerate(MW_PROBE_R):
        dfrac = np.linalg.norm(f_mw[:,j] - f_mw[0,j], axis=1) / np.linalg.norm(f_mw[0,j])
        ax_t.plot(t_series/1e3, dfrac, '-', color=f'C{j}', lw=1.2, label=f'MW halo r={r:.0f} kpc')
    for j, r in enumerate(LMC_PROBE_R):
        dfrac = np.linalg.norm(f_lmc[:,j] - f_lmc[0,j], axis=1) / np.linalg.norm(f_lmc[0,j])
        ax_t.plot(t_series/1e3, dfrac, '--', color=f'C{j+4}', lw=1.2, label=f'LMC {r:.0f} kpc from center')
    for ts in SNAP_TIMES_MYR:
        ax_t.axvline(ts/1e3, color='0.85', lw=0.5, zorder=0)
    ax_t.set_yscale('log')
    ax_t.set_ylim(1e-4, 2)
    ax_t.set_xlabel('t [Gyr]')
    ax_t.set_ylabel(r'$|F(t) - F_0| / |F_0|$')
    ax_t.legend(fontsize='x-small', ncol=4, loc='upper left')
    marker = ax_t.axvline(t_frames[0]/1e3, color='k', lw=1)
    suptitle = fig.suptitle('')

    def update(n):
        t = t_frames[n]
        m = frame_maps(t)
        for im, key in zip(ims, ['xz', 'xy', 'lmc']):
            im.set_data(m[key])
        sel = (t_track <= t) & (t_track > t - 300)
        for tr, track, i, j in trails:
            tr.set_data(track[sel, i], track[sel, j])
        k = np.searchsorted(t_track, t) - 1
        for d, track, i, j in dots:
            d.set_data([track[k, i]], [track[k, j]])
        marker.set_xdata([t/1e3, t/1e3])
        mode = 'linear' if linear else 'nearest'
        suptitle.set_text(f'Tango potential ({mode} interpolation), t = {t/1e3:+.3f} Gyr')
        return ims

    anim = animation.FuncAnimation(fig, update, frames=len(t_frames), blit=False)
    os.makedirs(PLOT_DIR, exist_ok=True)
    path = os.path.join(PLOT_DIR, fname)
    anim.save(path, writer=animation.FFMpegWriter(fps=fps, bitrate=4000), dpi=100)
    plt.close(fig)
    print(f'saved {path}')
    return path
