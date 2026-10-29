"""How large must a nudge be to measure a Jacobian column above the non-smooth part of the response, and how small to
stay linear? From the round 20 line (round 18 base f=0 ... round 19 base f=1, 21 runs, LMC and Sgr starts moved
together) and the nudge columns of rounds 18 and 19 (same nudges, Sgr rescaled by 1.03 in all three rounds).

    plot_pairs()    # |change of each t=0 miss| vs |f_i - f_j| over all pairs of line runs, with the straight-line
                    #   response between f=0 and f=1 -> plots/tune_step_size_pairs.png
    floors()        # each miss's scatter about a smooth (cubic) fit along the line (physical units)
    plot_grid()     # 12 x 12: each t=0 miss of the line runs vs each start's difference from round 19's start
                    #   -> plots/tune_r20_grid.png
    table()         # per nudge: start change, response in each miss for J18 and J19, and the smallest nudge whose
                    #   response clears 3x the floor (largest-response miss), in physical units
"""
import sys
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
sys.path.insert(0, '/home/abonaca/projects/glossary/scripts')
import json, os
import numpy as np
import sgr_timing as st
import v21_tune as vt

T = st.T
UNITS = ['kpc'] * 3 + ['km/s'] * 3
PLOTS = '/home/ana/projects/glossary/plots/'


def line():
    """f, start (12) and misses (12) plus Sgr arrival [Myr early] of the 21 runs on the round 18-19 line."""
    d = json.load(open(os.path.join(T, 'round20', 'round.json')))
    r = [json.load(open(os.path.join(T, k, 'round.json'))) for k in ('round18', 'round19')]
    f = np.r_[0, d['fracs'], 1]
    th = np.vstack((r[0]['theta'], d['runs'], r[1]['theta']))
    E = np.vstack((np.array(r[0]['endpoints'][0]) - vt.TARGET, d['misses'], np.array(r[1]['endpoints'][0]) - vt.TARGET))
    E = np.column_stack((E, [st.arrival(e) for e in E]))
    return f, th, E


def floors(deg=3):
    """rms scatter of each miss (and arrival) about a degree-`deg` polynomial in f along the line."""
    f, th, E = line()
    res = E - np.column_stack([np.polyval(np.polyfit(f, E[:, k], deg), f) for k in range(E.shape[1])])
    return np.sqrt(np.sum(res ** 2, axis=0) / (len(f) - deg - 1))


def plot_pairs(out=PLOTS + 'tune_step_size_pairs.png'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.style.use('default')
    f, th, E = line()
    i, j = np.triu_indices(len(f), 1)
    df = np.abs(f[j] - f[i])
    slope = np.abs(E[-1] - E[0])                                 # straight-line response per unit f
    fl = floors()
    names = vt.NAMES + ['sgr_arrival']
    units = UNITS * 2 + ['Myr']
    tol = np.r_[vt.TOL, np.nan]
    fig, ax = plt.subplots(3, 5, figsize=(20, 11.5), sharex=True)
    ax = ax.ravel()
    x = np.logspace(np.log10(0.05), 0, 50)
    for k in range(13):
        a = ax[k]
        a.loglog(df, np.abs(E[j, k] - E[i, k]), 'k.', ms=3, alpha=0.5)
        a.loglog(x, slope[k] * x, 'r-', lw=1, label='straight line f=0 to 1')
        a.axhline(fl[k], color='b', ls='--', lw=1, label='scatter about cubic fit')
        if np.isfinite(tol[k]):
            a.axhline(tol[k], color='0.5', ls=':', lw=1, label='tolerance')
        a.set_ylabel('|change of %s miss| [%s]' % (names[k], units[k]))
        a.set_ylim(1e-3 if units[k] == 'kpc' else 1e-2, None)
    ax[0].legend(fontsize=8, loc='lower right')
    for a in ax[13:]:
        a.axis('off')
    for a in ax[8:13]:
        a.set_xlabel('|f difference| (f=1: round 19 - round 18 start)')
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    return out


def table():
    r = [json.load(open(os.path.join(T, k, 'round.json'))) for k in ('round18', 'round19')]
    delta = np.array(r[0]['delta'])
    fl = floors()
    resp = []                                    # per round: (13 outputs incl. arrival) x 12 nudges
    for d in r:
        F = np.array(d['endpoints']) - vt.TARGET
        A = np.column_stack((F, [st.arrival(e) for e in F]))
        resp.append((A[1:] - A[0]).T)
    names = vt.NAMES + ['arrival']
    units = UNITS * 2 + ['Myr']
    print('floors (scatter about a cubic along the line): ' +
          ', '.join('%s %.2f %s' % (n, s, u) for n, s, u in zip(names, fl, units)))
    for c in range(12):
        snr = np.max([np.abs(R[:, c]) / fl for R in resp], axis=0)
        k = np.argsort(snr)[::-1][:3]
        print('nudge %-6s %5.2f %-4s: ' % (vt.NAMES[c], delta[c], UNITS[c % 6]) +
              '; '.join('%s %+.2f/%+.2f %s (%.0fx floor)' % (names[q], resp[0][q, c], resp[1][q, c], units[q], snr[q])
                        for q in k) +
              '; 3x floor at %.3g %s' % (3 * delta[c] / snr[k[0]], UNITS[c % 6]))
    return resp, fl


def plot_grid(out=PLOTS + 'tune_r20_grid.png'):
    """Rows: t=0 misses; columns: start minus round 19's start; round 20 runs coloured by f, round 18 (f=0) and
    round 19 (f=1) as open squares; shaded bands: tolerances."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.style.use('default')
    f, th, E = line()
    D = th - th[-1]
    r20 = slice(1, -1)
    fig, ax = plt.subplots(12, 12, figsize=(30, 28), sharex='col', sharey='row')
    for i in range(12):
        for j in range(12):
            a = ax[i, j]
            a.axhspan(-vt.TOL[i], vt.TOL[i], color='0.85', lw=0)
            a.axhline(0, color='0.5', lw=0.5)
            a.axvline(0, color='0.5', lw=0.5)
            sc = a.scatter(D[r20, j], E[r20, i], c=f[r20], cmap='viridis', vmin=0, vmax=1, s=10, zorder=3)
            a.plot(D[[0, -1], j], E[[0, -1], i], 's', mfc='none', mec='r', ms=4, zorder=4)
            a.tick_params(labelsize=6)
            if i == 11:
                a.set_xlabel('%s start - r19 [%s]' % (vt.NAMES[j], UNITS[j % 6]), fontsize=8)
            if j == 0:
                a.set_ylabel('%s miss [%s]' % (vt.NAMES[i], UNITS[i % 6]), fontsize=8)
    fig.tight_layout(rect=(0, 0, 0.97, 1))
    cb = fig.colorbar(sc, ax=ax, fraction=0.01, pad=0.01)
    cb.set_label('f (round 20 runs; red squares: round 18 f=0, round 19 f=1)')
    fig.savefig(out, dpi=200)
    return out
