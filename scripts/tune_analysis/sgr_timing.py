"""Sgr arrival time vs the LMC and Sgr t=-3 starts, from every run of rounds 11-16 (reversed, MW-like disc), including
the redone rounds (round12_fullstep, round13_damp0.5).

Arrival: r9_analysis.timing_split of Sgr's t=0 miss, in Myr (+ = early). A linear fit of the arrival on all 12 start
coordinates (offsets from round 11's base) gives each coordinate's slope and partial correlation; the partial-residual
plot shows the arrival with every other coordinate's fitted contribution removed.

    collect()        # starts (n, 12), t=0 misses (n, 12), arrival (n,), round labels
    fit()            # slopes, partial correlations, residual scatter
    plot_partial()   # plots/sgr_arrival_vs_start.png
    x_fix(k=15)      # misses after fixing round k's Sgr timing with an Sgr x0 shift, with each round's J x-column
"""
import sys
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
sys.path.insert(0, '/home/abonaca/projects/glossary/scripts')
import json, os
import numpy as np
import v21_tune as vt
from r9_analysis import timing_split

T = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../output/v21_sim/tune/')
ROUNDS = ('round11', 'round12_fullstep', 'round12', 'round13_damp0.5', 'round13', 'round14', 'round15', 'round16_lm')
PLOTS = '/home/ana/projects/glossary/plots/'
UNITS = ['kpc'] * 3 + ['km/s'] * 3


def _run_thetas(d):
    if 'runs' in d:
        return np.array(d['runs'])
    th, de = np.array(d['theta']), np.array(d['delta'])
    return np.array([th] + [th + np.eye(12)[j] * de[j] for j in range(12)])


def arrival(miss):
    return timing_split(np.asarray(miss)[6:], vt.TARGET[9:])[0]


def collect(rounds=ROUNDS):
    th, miss, lab = [], [], []
    for r in rounds:
        d = json.load(open(os.path.join(T, r, 'round.json')))
        F = np.array(d['endpoints'])
        th.append(_run_thetas(d)[:len(F)])
        miss.append(F - vt.TARGET)
        lab += [r] * len(F)
    th, miss = np.vstack(th), np.vstack(miss)
    return th, miss, np.array([arrival(m) for m in miss]), np.array(lab)


def fit(rounds=ROUNDS):
    th, miss, a, lab = collect(rounds)
    ref = th[0]
    X = np.hstack((np.ones((len(a), 1)), th - ref))
    b, *_ = np.linalg.lstsq(X, a, rcond=None)
    res = a - X @ b
    pr = []
    for j in range(12):
        part = res + b[j + 1] * (th[:, j] - ref[j])
        pr.append(np.corrcoef(th[:, j] - ref[j], part)[0, 1])
    return dict(th=th, miss=miss, a=a, lab=lab, ref=ref, b=b, res=res, partial_r=np.array(pr),
                rms=np.std(res, ddof=13))


def plot_partial(out=PLOTS + 'sgr_arrival_vs_start.png'):
    import matplotlib.pyplot as plt
    plt.style.use('default')
    f = fit()
    de = np.array(json.load(open(os.path.join(T, 'round11', 'round.json')))['delta'])
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    rounds = list(dict.fromkeys(f['lab']))
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(rounds)))
    for j, ax in zip(range(6, 12), axes.ravel()):
        dx = f['th'][:, j] - f['ref'][j]
        part = f['res'] + f['b'][j + 1] * dx
        for r, c in zip(rounds, cols):
            s = f['lab'] == r
            ax.plot(dx[s], part[s], 'o', ms=4, color=c, label=r.replace('round', 'r'))
        xx = np.linspace(dx.min(), dx.max(), 2)
        ax.plot(xx, f['b'][j + 1] * xx, 'k-', lw=1)
        ax.text(0.03, 0.04, 'partial r = %+.2f\n%+.1f Myr per %g %s (nudge)' % (
            f['partial_r'][j], f['b'][j + 1] * de[j], de[j], UNITS[j % 6]), transform=ax.transAxes, fontsize=9)
        ax.set_xlabel(r'$\Delta$ Sgr %s$_0$ [%s]' % (vt.NAMES[j][4:], UNITS[j % 6]))
    for ax in axes[:, 0]:
        ax.set_ylabel('Sgr arrival, partial residual [Myr]\n(+ = early)')
    axes[0, 0].legend(fontsize=7, ncol=2, loc='upper right')
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    return f


def x_fix(k=15, rounds=('round11', 'round12', 'round13', 'round14', 'round15', 'round16_lm')):
    """Round k's base misses after an Sgr x0 shift that zeroes its arrival offset, with each round's Sgr x0 column."""
    d = json.load(open(os.path.join(T, 'round%02d' % k, 'round.json')))
    r = np.array(d['endpoints'][0]) - vt.TARGET
    out = []
    for rr in rounds:
        e = json.load(open(os.path.join(T, rr, 'round.json')))
        F = np.array(e['endpoints'])
        if 'runs' in e:
            col = (F[1] - F[0]) / e['sgr_delta'][0]
        else:
            col = (F[7] - F[0]) / e['delta'][6]
        dx = -arrival(r) / arrival(col)
        out.append((rr, dx, r + col * dx))
    return r, out


if __name__ == '__main__':
    f = plot_partial()
    np.set_printoptions(precision=3, suppress=True, linewidth=180)
    print('runs %d, rms residual %.2f Myr' % (len(f['a']), f['rms']))
    for j in range(12):
        print('%-7s slope %+8.2f Myr per %s, partial r %+.2f' % (vt.NAMES[j], f['b'][j + 1], UNITS[j % 6], f['partial_r'][j]))
    r, out = x_fix()
    print('round 15 miss', r, 'arrival %+.2f' % arrival(r))
    for rr, dx, m in out:
        print('%-10s dx %+.4f kpc -> %s' % (rr, dx, m))


SGR5 = [6, 8, 9, 10, 11]          # Sgr x, z, vx, vy, vz (y left fixed)
CORE = ('round11', 'round12', 'round13', 'round14', 'round15', 'round16_lm')


def fit_jacobian(rounds=ROUNDS):
    """Linear fit of all 12 t=0 endpoints on all 12 t=-3 starts over every run of the given rounds: a Jacobian that
    averages over rounds (and over run-to-run scatter). Returns J (12 outputs x 12 starts) and each output's rms
    residual."""
    th, miss, a, lab = collect(rounds)
    X = np.hstack((np.ones((len(th), 1)), th - th[0]))
    B, *_ = np.linalg.lstsq(X, miss, rcond=None)
    res = miss - X @ B
    return B[1:].T, np.sqrt(np.sum(res ** 2, axis=0) / (len(th) - 13))


def round_jacobian(name):
    d = json.load(open(os.path.join(T, name, 'round.json')))
    F = np.array(d['endpoints'])
    if 'runs' in d:
        J = np.full((12, 12), np.nan)
        J[:, 6:] = (F[1:7] - F[0]).T / np.array(d['sgr_delta'])[None, :]
        return J
    return (F[1:] - F[0]).T / np.array(d['delta'])[None, :]


def step(r, J, cols=SGR5, lam=0., delta=None):
    """Step in the start coordinates `cols` (others fixed) minimising |W (r + J step)|^2, W = 1/TOL over all 12
    endpoint components; Marquardt damping lam in tolerance-per-nudge units."""
    delta = np.asarray(delta if delta is not None else json.load(open(os.path.join(T, 'round11', 'round.json')))['delta'])
    A = J[:, cols] / vt.TOL[:, None] * delta[cols][None, :]
    M = A.T @ A
    u = -np.linalg.solve(M + lam * np.diag(np.diag(M)), A.T @ (r / vt.TOL))
    s = np.zeros(12)
    s[cols] = u * delta[cols]
    return s
