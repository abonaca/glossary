import sys, json, numpy as np
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import v21_tune as vt
np.set_printoptions(precision=2, suppress=True, linewidth=220)
O = '/home/ana/projects/glossary/output/v21_sim/tune/'; W = 1 / vt.TOL; L, S = slice(0, 6), slice(6, 12)
K = (5, 6, 7, 8)
def load(k, il, isg, im):
    c = np.load(O + 'round%02d/centre_check.npz' % k)
    assert c['r_vel_lmc'][il] == (5 if il == 3 else c['r_vel_lmc'][il])
    return np.hstack((c['lmc'][:, il] - c['mw'][:, im], c['sgr'][:, isg] - c['mw'][:, im]))
info = {k: json.load(open(O + 'round%02d/round.json' % k)) for k in K}
th = {k: np.array(info[k]['theta']) for k in K}; d = np.array(info[8]['delta'])
F = {'old': {k: load(k, 0, 2, 1) for k in K}, 'new': {k: load(k, 3, 2, 3) for k in K}}
for k in K: assert np.allclose(F['old'][k][0] - vt.TARGET, info[k]['residual'], atol=1e-3)
J = {v: {k: (F[v][k][1:] - F[v][k][0]).T / d for k in K} for v in F}
r = {v: {k: F[v][k][0] - vt.TARGET for k in K} for v in F}
print('BASE MISS, physical (LMC kpc,km/s | Sgr kpc,km/s), new centres [old]')
for k in K:
    print(' r%d new %s\n    old %s' % (k, r['new'][k], r['old'][k]))
print('\nBASE MISS /TOL, new centres; totals LMC, Sgr')
for k in K:
    q = r['new'][k] * W; print(' r%d %s  | LMC %.2f Sgr %.2f' % (k, q, np.linalg.norm(q[L]), np.linalg.norm(q[S])))
print('\nLMC-block entries (velocity rows, own velocity cols) across rounds, new centres  [old]')
for row in (3, 4, 5):
    for col in (3, 4, 5):
        print('  %s / %s:' % (vt.NAMES[row], vt.NAMES[col]), np.array([J['new'][k][row, col] for k in K]),
              ' [', np.array([J['old'][k][row, col] for k in K]), ']')
def spread(v, B, C):
    """relative scatter across rounds of the TOL-weighted, nudge-scaled block"""
    A = np.array([W[B, None] * J[v][k][B, C] * d[C] for k in K]); m = A.mean(0)
    return np.linalg.norm(A - m) / np.linalg.norm(m) / np.sqrt(len(K))
print('\nscatter of each block across J5-J8 (rms deviation / mean, TOL-weighted, nudge-scaled):')
for nm, B, C in (('LMC<-LMC', L, L), ('LMC<-Sgr', L, S), ('Sgr<-LMC', S, L), ('Sgr<-Sgr', S, S)):
    print('  %s  new %.2f   old %.2f' % (nm, spread('new', B, C), spread('old', B, C)))
print('\nPREDICTED vs ACTUAL change r(k)->r(k+1), /TOL, new centres')
for k in (5, 6, 7):
    dt = th[k + 1] - th[k]; act = (r['new'][k + 1] - r['new'][k]) * W
    print(' r%d->r%d actual   %s' % (k, k + 1, act))
    for jk in (k, k + 1):
        p = J['new'][jk] @ dt * W
        print('        J%d full  %s   LMC err %.2f Sgr err %.2f' % (jk, p, np.linalg.norm((p - act)[L]), np.linalg.norm((p - act)[S])))
        old = J['old'][jk] @ dt * W; acto = (r['old'][k + 1] - r['old'][k]) * W
        print('          (old centres: LMC err %.2f Sgr err %.2f)' % (np.linalg.norm((old - acto)[L]), np.linalg.norm((old - acto)[S])))
np.savez(O + 'round08/rounds5to8_newcentres.npz', **{'J%d' % k: J['new'][k] for k in K}, **{'r%d' % k: r['new'][k] for k in K})
