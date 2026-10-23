"""Round-7 start (the restart from round 6, round07/round.json; the earlier block-J6 round 7 from r7_theta.py /
theta7.json is round07_blockJ6): LMC fixed at round 6, Sgr = round 6 + a Marquardt-damped (lambda=3e-3) step on
the round-6 Sgr block, with endpoints re-measured from round06/centre_check.npz (MW velocity aperture 3 kpc, LMC
5 kpc, Sgr 5 kpc). The result was passed to v21_tune.new_sgr_round(7, theta7, sgr_delta, scan_dt, note)."""
import sys, json, numpy as np
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import v21_tune as vt
O = '/home/ana/projects/glossary/output/v21_sim/tune/'
SGR_DELTA = [0.05, 0.25, 0.05, 0.25, 0.25, 0.25]
SCAN_DT = [-1, -0.5, -0.25, 0.25, 0.5, 1]


def theta7(lam=3e-3):
    W, S = 1 / vt.TOL, slice(6, 12)
    c = np.load(O + 'round06/centre_check.npz')
    info = json.load(open(O + 'round06/round.json'))
    # aperture index 3: MW r_vel 3 kpc, LMC 5 kpc; Sgr index 2: 5 kpc
    F = np.hstack((c['lmc'][:, 3] - c['mw'][:, 3], c['sgr'][:, 2] - c['mw'][:, 3]))
    th6, d = np.array(info['theta']), np.array(info['delta'])
    r = (F[0, S] - vt.TARGET[S]) * W[S]
    A = W[S, None] * (F[7:, S] - F[0, S]).T          # tolerance units per round-6 nudge
    M = A.T @ A
    s = -np.linalg.solve(M + lam * np.diag(np.diag(M)), A.T @ r) * d[S]
    th7 = th6.copy()
    th7[S] += s
    return th7


def check():
    t = np.array(json.load(open(O + 'round07/round.json'))['theta'])
    print('max |theta7 - round07 theta| = %.2e' % np.abs(theta7() - t).max())


if __name__ == '__main__':
    check()
