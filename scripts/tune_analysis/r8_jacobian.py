"""J8: Sgr-block Jacobian around round-8 run 3 from the +/- nudge runs (7-12, 13-18); comparison with J7,
curvature, and damped (Levenberg-Marquardt) steps from run 3 with their linear forecasts; r9_offsets writes the round-9
Sgr start offsets (r9_offsets.json, passed to v21_tune.new_offset_round(9, ...))."""
import json, numpy as np
R8 = '/home/ana/projects/glossary/output/v21_sim/tune/round08/round.json'
R7 = '/home/ana/projects/glossary/output/v21_sim/tune/round07/round.json'
DELTA = np.array([0.05, 0.25, 0.05, 0.25, 0.25, 0.25])
TOL = np.array([0.1] * 3 + [1.] * 3)
NAMES = ['x0', 'y0', 'z0', 'vx0', 'vy0', 'vz0']
ROWS = ['x', 'y', 'z', 'vx', 'vy', 'vz']


def load():
    d = json.load(open(R8))
    F = np.array(d['endpoints'])[:, 6:]
    return d, F


def jacobians():
    d, F = load()
    f0, fp, fm = F[3], F[7:13], F[13:19]
    Jp = (fp - f0).T / DELTA
    Jm = (f0 - fm).T / DELTA
    Jc = (fp - fm).T / (2 * DELTA)
    curv = (fp + fm - 2 * f0).T            # second difference, endpoint units, per nudge
    J7 = np.array(json.load(open(R7))['jacobian_sgr'])
    return dict(plus=Jp, minus=Jm, central=Jc, J7=J7, curv=curv, f0=f0, target=np.array(d['target'])[6:])


def lm_steps(J, r, lams=(0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2), damping='marquardt'):
    """Steps in start space minimising |W(r + J s)|^2 + damping term, W = 1/TOL, in nudge units (s / DELTA).
    damping='marquardt': (AtA + lam diag(AtA)) u = -At b, as used for round-8 runs 0-6 and round 9;
    damping='trace': (AtA + lam trace(AtA)/6 I) u = -At b (equal damping; the first J8 step table)."""
    W, D = np.diag(1 / TOL), np.diag(1 / DELTA)
    A = W @ J @ np.linalg.inv(D)        # tolerance units per nudge unit
    b = W @ r
    M = A.T @ A
    damp = dict(marquardt=np.diag(np.diag(M)), trace=np.trace(M) / 6 * np.eye(6))[damping]
    out = []
    for lam in lams:
        u = -np.linalg.solve(M + lam * damp, A.T @ b)
        s = np.linalg.inv(D) @ u
        out.append((lam, s, r + J @ s))
    return out


def r9_offsets(out='/home/ana/projects/glossary/scripts/tune_analysis/r9_offsets.json',
               lams=(3e-4, 1e-3, 3e-3, 1e-2), base_lam=3e-3):
    """Round-9 Sgr start offsets relative to round-8 run 3: Marquardt-damped steps with the central J8 (runs 0-3),
    then the base_lam step + one forward nudge per start component (runs 4-9, for J9). Written to `out`, which was
    copied to OBS and passed to v21_tune.new_offset_round(9, base, offsets, labels, note)."""
    j = jacobians()
    steps = lm_steps(j['central'], j['f0'] - j['target'], lams)
    offs = [s for _, s, _ in steps]
    labs = ['J8 LM lambda=%g' % lam for lam in lams]
    base = offs[list(lams).index(base_lam)]
    bl = ('%.0e' % base_lam).replace('e-0', 'e-')     # 3e-3
    for i, n in enumerate(NAMES):
        o = base.copy(); o[i] += DELTA[i]
        offs.append(o); labs.append('lam%s +%s' % (bl, n))
    note = ('Marquardt-damped (lam*diag(AtA)) steps from round-8 run 3 with the central-difference J8 | '
            'runs %d-%d: run %d (lam=%s) + Sgr nudges (0.05,0.25,0.05 kpc; 0.25 km/s) for J9'
            % (len(lams), len(lams) + 5, list(lams).index(base_lam), bl))
    theta = np.array(json.load(open(R8))['runs'][3])
    d = dict(base=theta.tolist(), offsets=np.array(offs).tolist(), labels=labs, note=note)
    json.dump(d, open(out, 'w'), indent=1)
    return d


def report():
    j = jacobians()
    r = j['f0'] - j['target']
    np.set_printoptions(suppress=True, linewidth=150)
    print('run 3 Sgr miss: %s' % np.round(r, 3))
    for k in ('plus', 'minus', 'central', 'J7'):
        print('\n%s: response to one nudge (rows x,y,z kpc; vx,vy,vz km/s; columns %s)' % (k, ' '.join(NAMES)))
        print(np.round(j[k] * DELTA, 3))
    print('\n+/- disagreement (plus - minus) per nudge:')
    print(np.round((j['plus'] - j['minus']) * DELTA, 3))
    print('\nsingular values (tolerance units per nudge):')
    for k in ('plus', 'minus', 'central', 'J7'):
        A = np.diag(1 / TOL) @ j[k] @ np.diag(DELTA)
        print('  %-8s %s' % (k, np.round(np.linalg.svd(A, compute_uv=False), 3)))
    print('\nLM steps from run 3 with the central J8, and their linear forecasts:')
    for lam, s, f in lm_steps(j['central'], r):
        print('  lam %-7g start change %s | forecast %s' % (lam, np.round(s, 4), np.round(f, 3)))
    return j


if __name__ == '__main__':
    report()
