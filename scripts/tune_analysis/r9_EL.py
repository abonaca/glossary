"""Sgr energy and angular momentum for an offset round (default round 9) with round-8 run 3 as the reference:

    full(9)     # along each run, t=-3..0, in the round-7 base run's evolving fields (as r8_EL)
                #   -> plots/tune_r9_sgr_EL.png
    window(9)   # each run's t=0 state integrated +-20 Myr in round 7's t=0 fields (as r7_EL_window), with the
                #   stretch inside the position tolerance and the best-velocity moment -> plots/tune_r9_sgr_EL_window.png
"""
import sys
sys.path.insert(0, "/home/ana/projects/glossary/scripts")
import json, numpy as np, agama, v21_tune as vt
from itertools import product
from scipy.spatial import ConvexHull
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('default')
agama.setUnits(length=1, velocity=1, mass=232500)
T = '/home/ana/projects/glossary/output/v21_sim/tune/'
D = T + 'round07/run00/expansions_hi'
P = '/home/ana/projects/glossary/plots/'
TGT = vt.TARGET[6:]
TU = 977.8
REF = (8, 3)                    # reference run: round 8 run 3


def runs_of(k):
    """[(label, colour, linestyle, centres file)]: the reference run, then every run of round k."""
    info = json.load(open(T + 'round%02d/round.json' % k))
    out = [('r%d run %d (ref.)' % REF, 'k', '-', T + 'round%02d/run%02d/centres.npz' % REF)]
    nlm = sum(l.startswith('J8 LM') for l in info['labels'])
    for i, lab in enumerate(info['labels']):
        if lab.startswith('J8 LM'):
            col, ls, lab = plt.cm.viridis(i / max(nlm - 1, 1) * 0.9), '-', lab.replace('J8 LM lambda', '$\\lambda$')
        else:
            col, ls = plt.cm.tab10(i - nlm), '--'
        out.append(('r%d %d: %s' % (k, i, lab), col, ls, T + 'round%02d/run%02d/centres.npz' % (k, i)))
    return out


def tolerance_box(E_L):
    box = np.array(list(product(*[(-1, 1)] * 6)), float)
    box = np.vstack((box, np.random.default_rng(1).uniform(-1, 1, (20000, 6)))) * vt.TOL[6:] + TGT
    return E_L(box)


def draw_region(a, Lb, Eb, k, label):
    pts = np.column_stack((Lb[:, k], Eb))
    h = ConvexHull(pts).vertices
    a.fill(pts[h, 0], pts[h, 1], color='r', alpha=0.15, lw=0, zorder=0, label=label)
    a.plot(np.r_[pts[h, 0], pts[h[0], 0]], np.r_[pts[h, 1], pts[h[0], 1]], 'r-', lw=0.8, zorder=0)


def full(k=9, out=None):
    base = dict(np.load(T + 'round07/run00/centres.npz'))
    n = len(base['times'])
    mw = [agama.Potential(agama.Potential(D + '/halo_%02d.coef' % j), agama.Potential(D + '/disk_%02d.coef' % j))
          for j in range(n)]
    lmc = [agama.Potential(D + '/lmc_%02d.coef' % j) for j in range(n)]
    xl = base['lmc'][:, :3] - base['mw'][:, :3]

    def EL(s, j):
        s = np.atleast_2d(s)
        return 0.5 * np.sum(s[:, 3:]**2, 1) + mw[j].potential(s[:, :3]) + lmc[j].potential(s[:, :3] - xl[j]), \
            np.cross(s[:, :3], s[:, 3:])

    runs = []
    for lab, col, ls, f in runs_of(k):
        r = dict(np.load(f))
        rel = r['sgr'] - r['mw']
        E, L = zip(*[EL(rel[j], j) for j in range(n)])
        runs.append((lab, col, ls, np.ravel(E), np.vstack(L)))
    Et, Lt = EL(TGT, n - 1)
    Eb, Lb = tolerance_box(lambda b: EL(b, n - 1))
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    for row in range(2):
        sel = slice(None) if row == 0 else slice(-2, None)
        for p, c in enumerate((2, 1, 0)):
            a = ax[row, p]
            for lab, col, ls, E, L in runs:
                ref = lab.endswith('(ref.)')
                a.plot(L[sel, c], E[sel], ls, color=col, lw=2 if ref else 0.9, label=lab, zorder=3 if ref else 2)
                a.plot(L[-1, c], E[-1], 'o', color=col, ms=6, mec='k', mew=0.5, zorder=4)
                if row == 0:
                    a.plot(L[0, c], E[0], 's', color=col, ms=4, zorder=4)
            draw_region(a, Lb, Eb, c, 'tolerance region')
            a.plot(Lt[0, c], Et[0], '*', color='r', ms=16, mec='k', zorder=5, label='target, t=0')
            a.set_xlabel('$L_%s$ [kpc km/s]' % 'xyz'[c])
            a.set_ylabel('E (MW + LMC) [(km/s)$^2$]')
            if row == 1:
                Ee = np.r_[[E[-1] for *_, E, L in runs], Eb]
                Le = np.r_[[L[-1, c] for *_, E, L in runs], Lb[:, c]]
                pe, pl = 0.08 * np.ptp(Ee), 0.08 * np.ptp(Le)
                a.set_xlim(Le.min() - pl, Le.max() + pl); a.set_ylim(Ee.min() - pe, Ee.max() + pe)
    ax[0, 2].legend(fontsize=7, loc='best')
    ax[0, 0].text(0.02, 0.97, 'squares: t = -3, circles: t = 0', transform=ax[0, 0].transAxes, va='top', fontsize=8)
    ax[1, 0].text(0.02, 0.97, 'zoom: last snapshot interval (t = -0.12 to 0)', transform=ax[1, 0].transAxes,
                  va='top', fontsize=8)
    fig.tight_layout()
    out = out or P + 'tune_r%d_sgr_EL.png' % k
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(out)


def window(k=9, out=None, span=20.):
    E0 = T + 'round07/run00/expansions_hi'
    mw = agama.Potential(agama.Potential(E0 + '/halo_24.coef'), agama.Potential(E0 + '/disk_24.coef'))
    lmc = agama.Potential(E0 + '/lmc_24.coef')
    ts = np.linspace(-span, span, 8001) / TU
    i0 = 4000
    ts[i0] = 0.

    def orbit(s, l):
        xl = lambda t: l[:3] + l[3:] * t
        f = lambda t, w: np.hstack((w[3:], mw.force(w[:3]) + lmc.force(w[:3] - xl(t)) - lmc.force(-xl(t))))
        up = solve_ivp(f, (0, ts[-1]), s, t_eval=ts[i0:], rtol=1e-11, atol=1e-11, method='DOP853').y.T
        dn = solve_ivp(f, (0, ts[0]), s, t_eval=ts[i0::-1], rtol=1e-11, atol=1e-11, method='DOP853').y.T[::-1]
        W = np.vstack((dn[:-1], up))
        xls = l[:3] + l[3:] * ts[:, None]
        E = 0.5 * np.sum(W[:, 3:]**2, 1) + mw.potential(W[:, :3]) + lmc.potential(W[:, :3] - xls)
        return W, E, np.cross(W[:, :3], W[:, 3:])

    runs = runs_of(k)
    lt = None
    fig, ax = plt.subplots(1, 3, figsize=(17, 6))
    for lab, col, ls, f in runs:
        r = np.load(f)
        s, l = r['sgr'][-1] - r['mw'][-1], r['lmc'][-1] - r['mw'][-1]
        if lab.endswith('(ref.)'):
            lt = l[:3]
        W, E, L = orbit(s, l)
        o = W - TGT
        ok = np.all(np.abs(o[:, :3]) <= vt.TOL[6:9], axis=1)
        for p, c in enumerate((2, 1, 0)):
            a = ax[p]
            a.plot(L[:, c], E, ls, color=col, lw=1.5 if lab.endswith('(ref.)') else 0.8, label=lab if p == 0 else None)
            a.plot(L[i0, c], E[i0], 'o', color=col, ms=4)
            if ok.any():
                idx = np.flatnonzero(ok)
                a.plot(L[idx, c], E[idx], '-', color=col, lw=4, alpha=0.9, solid_capstyle='round')
                kb = idx[np.argmin(np.linalg.norm(o[idx, 3:], axis=1))]
                a.plot(L[kb, c], E[kb], 'D', color=col, mec='k', ms=7, zorder=5)
    EL = lambda b: (0.5 * np.sum(b[:, 3:]**2, 1) + mw.potential(b[:, :3]) + lmc.potential(b[:, :3] - lt),
                    np.cross(b[:, :3], b[:, 3:]))
    Eb, Lb = tolerance_box(EL)
    Et, Lt = EL(TGT[None])
    for p, c in enumerate((2, 1, 0)):
        draw_region(ax[p], Lb, Eb, c, 'tolerance region' if p == 0 else None)
        ax[p].plot(Lt[0, c], Et[0], '*', color='r', mec='k', ms=18, zorder=6, label='target' if p == 0 else None)
        ax[p].set_xlabel('$L_%s$ [kpc km/s]' % 'xyz'[c])
        ax[p].set_ylabel('E (MW + LMC) [(km/s)$^2$]')
    ax[0].text(0.02, 0.98, 'lines: $\\pm$%g Myr around t = 0 (circles); thick: all $|\\Delta x_i| \\leq$ 0.1 kpc;\n'
               'diamonds: smallest $|\\Delta v|$ within that; solid: damped steps, dashed: nudges;\n'
               'red region: states within 0.1 kpc and 1 km/s of the target in every component' % span,
               transform=ax[0].transAxes, va='top', fontsize=7)
    fig.legend(loc='center right', fontsize=7)
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    out = out or P + 'tune_r%d_sgr_EL_window.png' % k
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(out)


if __name__ == '__main__':
    full(9)
    window(9)
