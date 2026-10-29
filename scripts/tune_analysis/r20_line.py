"""Round 20: line search between round 18's base (f = 0) and round 19's base (f = 1, round 18 + the full J18
Gauss-Newton step), at f = 0.05, 0.10, ..., 0.95 (19 runs), LMC and Sgr moved together, Sgr rescaled by 1.03.

Round 18 arrived 5.0 Myr early and round 19 5.8 Myr late; 10 of the 12 misses change sign between them, but cross
zero at different f if linear (Sgr x, y, z, vx at 0.45-0.48, Sgr vy, vz at 0.26-0.31, LMC at 0.17-0.42) and LMC vx,
vz do not change sign. Nothing guarantees the misses are linear in f, hence the fine sampling. Every run is scored on
its own (v21_tune kind 'sgr_offsets'); offsets are the 12-component start changes from round 18's base. The round 20
that finish_round(19) set up (full J19 step) is in SCRAP/round20_GN_J19.

    setup()     # on OBS: TUNE/round20/round.json
    check()     # on OBS: round.json reproduces
Submit:  sbatch --array=0-18 ~/projects/glossary/scripts/obs/tune_round.sbatch 20
"""
import sys
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
sys.path.insert(0, '/home/abonaca/projects/glossary/scripts')
import json, os
import numpy as np

K = 20
FRACS = tuple(round(0.05 * j, 2) for j in range(1, 20))      # 0.05 ... 0.95
NOTE = ('Line search between round 18 base (f=0) and round 19 base (f=1) at f = %s, LMC and Sgr together '
        '(r20_line.py)' % (FRACS,))


def _info():
    import v21_tune as vt
    r18 = json.load(open(os.path.join(vt.round_dir(18), 'round.json')))
    r19 = json.load(open(os.path.join(vt.round_dir(19), 'round.json')))
    th0, th1 = np.array(r18['theta']), np.array(r19['theta'])
    offs = [f * (th1 - th0) for f in FRACS]
    return dict(round=K, kind='sgr_offsets', names=vt.NAMES, theta=th0.tolist(), offsets=[o.tolist() for o in offs],
                labels=['f=%.2f' % f for f in FRACS], runs=[(th0 + o).tolist() for o in offs], fracs=list(FRACS),
                target=vt.TARGET.tolist(), tol=vt.TOL.tolist(), note=NOTE, base_ics=r19['base_ics'],
                base_theta=r19['base_theta'], sgr_rscale=[r19['sgr_rscale'][0]] * len(FRACS))


def setup():
    import v21_tune as vt
    d = vt.round_dir(K)
    if os.path.exists(os.path.join(d, 'round.json')):
        raise RuntimeError('round %d exists; move it to scrap first' % K)
    os.makedirs(d, exist_ok=True)
    info = _info()
    json.dump(info, open(os.path.join(d, 'round.json'), 'w'), indent=1)
    print('round %d: %d runs on the round 18 -> 19 line, f = %s, Sgr rescaled by %g'
          % (K, len(FRACS), FRACS, info['sgr_rscale'][0]))
    return info


def check():
    import v21_tune as vt
    d = json.load(open(os.path.join(vt.round_dir(K), 'round.json')))
    ref = _info()
    r19 = json.load(open(os.path.join(vt.round_dir(19), 'round.json')))
    print('round.json reproduces: %s; sgr_rscale: %s; f=1 would give round 19 theta: %s'
          % (all(d[key] == ref[key] for key in ref), set(d['sgr_rscale']),
             np.allclose(np.array(d['theta']) + np.array(d['offsets'][-1]) / FRACS[-1], r19['theta'])))


def plot(out='/home/ana/projects/glossary/plots/tune_r20_line.png'):
    """Every t=0 miss, Sgr arrival and Sgr M(<5 kpc) against f, with round 18 (f=0) and round 19 (f=1); locally."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import v21_tune as vt, sgr_timing as st
    plt.style.use('default')
    T = st.T
    d = json.load(open(os.path.join(T, 'round20', 'round.json')))
    ends = [np.array(json.load(open(os.path.join(T, r, 'round.json')))['endpoints'][0]) - vt.TARGET
            for r in ('round18', 'round19')]
    f = np.r_[0, d['fracs'], 1]
    E = np.vstack((ends[0], d['misses'], ends[1]))
    m5 = np.load(os.path.join(T, 'round20', 'sgr_mass_runs.npz'))['mass'][:, 1]
    fig, ax = plt.subplots(4, 4, figsize=(16, 13), sharex=True)
    units = ['kpc'] * 3 + ['km/s'] * 3
    for j in range(12):
        a = ax[j // 6 * 2 + j % 6 // 3, j % 3]
        a.axhspan(-vt.TOL[j], vt.TOL[j], color='0.85', lw=0)
        a.axhline(0, color='0.5', lw=0.8)
        a.plot(f, E[:, j], 'k.-')
        a.plot([0, 1], E[[0, -1], j], 'r:', lw=1)
        a.set_ylabel('%s miss [%s]' % (vt.NAMES[j], units[j % 6]))
    a = ax[0, 3]
    a.axhline(0, color='0.5', lw=0.8)
    a.plot(f, [st.arrival(e) for e in E], 'k.-')
    a.set_ylabel('Sgr arrival [Myr early]')
    a = ax[1, 3]
    a.axhspan(3e8, 4e8, color='0.85', lw=0)
    a.plot(d['fracs'], m5, 'k.-')
    a.set_ylabel(r'Sgr $M(<5\,{\rm kpc})$ [M$_\odot$]')
    for a in ax[2:, 3]:
        a.axis('off')
    for a in ax[-1, :3]:
        a.set_xlabel('f (0 = round 18 base, 1 = round 19 base)')
    ax[1, 3].set_xlabel('f')
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    return out


if __name__ == '__main__':
    setup() if len(sys.argv) > 1 and sys.argv[1] == 'setup' else check()
