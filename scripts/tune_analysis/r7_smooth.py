"""
Smooth lnL(t) for Sgr in round 7's base run: lnL at the snapshots (linear in between), lnL >= 0, fitted
to the per-interval misses (restart from the live state at each snapshot) with a curvature penalty;
the penalty strength from leave-one-interval-out cross-validation.

    import r7_smooth as sm
    lin = sm.linearise(m)                      # 3 integrations per interval
    L = sm.solve(lin, alpha)
"""
import numpy as np, r7_df as rd
from scipy.optimize import lsq_linear
np.set_printoptions(precision=3, suppress=True, linewidth=160)
W = rd.W


def linearise(m, L0=None):
    """Per interval j: miss(L_j, L_j+1) ~ c_j + A_j (L_j - L0_j) + B_j (L_j+1 - L0_j+1), weighted by 1/TOL."""
    t, live = m['times'], m['live']
    n = len(t)
    L0 = np.zeros(n) if L0 is None else np.asarray(L0, float)
    c, A, B = [], [], []
    for j in range(n - 1):
        def miss(a, b):
            f = L0.copy(); f[j] += a; f[j + 1] += b
            return (rd.integrate(m, f, t[j], t[j + 1], live[j])[1][-1] - live[j + 1]) * W
        c0 = miss(0, 0)
        c.append(c0); A.append(miss(1, 0) - c0); B.append(miss(0, 1) - c0)
    return dict(L0=L0, c=np.array(c), A=np.array(A), B=np.array(B))


def design(lin, keep=None):
    """Stacked linear system M (L - L0) = -c over the kept intervals."""
    n = len(lin['L0'])
    rows, rhs = [], []
    for j in (range(n - 1) if keep is None else keep):
        M = np.zeros((6, n)); M[:, j] = lin['A'][j]; M[:, j + 1] = lin['B'][j]
        rows.append(M); rhs.append(-lin['c'][j])
    return np.vstack(rows), np.concatenate(rhs)


def D2(n):
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i:i + 3] = (1, -2, 1)
    return D


def solve(lin, alpha, keep=None):
    M, y = design(lin, keep)
    L0 = lin['L0']
    D = D2(len(L0))
    MM = np.vstack((M, np.sqrt(alpha) * D))
    yy = np.concatenate((y, -np.sqrt(alpha) * D @ L0))
    x = lsq_linear(MM, yy, bounds=(-L0, np.inf)).x          # x = L - L0, L >= 0
    return L0 + x


def cv(lin, alphas):
    """Leave-one-interval-out prediction error (sum of squared weighted misses of the left-out interval)."""
    n = len(lin['L0'])
    out = []
    for a in alphas:
        e = 0.
        for k in range(n - 1):
            L = solve(lin, a, keep=[j for j in range(n - 1) if j != k])
            M, y = design(lin, [k])
            e += np.sum((M @ (L - lin['L0']) - y)**2)
        out.append(e)
    return np.array(out)


def misses(m, L):
    t, live = m['times'], m['live']
    return np.array([rd.integrate(m, L, t[j], t[j + 1], live[j])[1][-1] - live[j + 1] for j in range(len(t) - 1)])


if __name__ == '__main__':
    import json
    m = rd.load(cubic=True)
    t, live = m['times'], m['live']
    lin = linearise(m, np.full(len(t), 2.))
    alphas = np.logspace(-3, 4, 15)
    e = cv(lin, alphas)
    rough = []
    for a in alphas:
        L = solve(lin, a)
        M, y = design(lin)
        rough.append((np.sum((M @ (L - lin['L0']) - y)**2), np.sum((D2(len(L)) @ L)**2)))
    print('alpha     CV error   data chi2   roughness')
    for a, ei, (d, r) in zip(alphas, e, rough):
        print('%9.3g  %9.2f  %9.2f  %9.3g' % (a, ei, d, r))
    best = alphas[np.argmin(e)]
    L = solve(lin, best)
    lin2 = linearise(m, L)                                  # re-linearise at the solution
    L = solve(lin2, best)
    print('alpha (CV) = %.3g' % best)
    print('lnL at snapshots:', L)
    mi = misses(m, L)
    M, y = design(lin2)
    print('linear-model chi2 %.2f, actual chi2 %.2f' % (np.sum((M @ (L - lin2['L0']) - y)**2), np.sum((mi * W)**2)))
    for name, LL in (('constant 2.094', 2.094), ('smooth lnL(t)', L)):
        mm = misses(m, LL)
        fw = rd.integrate(m, LL, t[0], t[-1], live[0], t_eval=t)[1]
        bw = rd.integrate(m, LL, t[-1], t[0], live[-1], t_eval=t[::-1])[1][::-1]
        print('=== %s' % name)
        print('  interval misses rms: %.3f kpc %.2f km/s;  chi2 %.2f' % (np.sqrt(np.mean(np.sum(mm[:, :3]**2, 1))),
              np.sqrt(np.mean(np.sum(mm[:, 3:]**2, 1))), np.sum((mm * W)**2)))
        print('  forward from live t=-3, t=0 minus live:', fw[-1] - live[-1])
        print('  backward from live t=0, t=-3 minus round-7 start:', bw[0] - m['theta'][6:])
        print('     t[Myr]   |fw-live| kpc km/s   |bw-live| kpc km/s')
        for j in range(0, len(t), 3):
            print('    %6.0f   %6.2f %6.2f   %6.2f %6.2f' % (t[j] * rd.TU, np.linalg.norm(fw[j, :3] - live[j, :3]),
                  np.linalg.norm(fw[j, 3:] - live[j, 3:]), np.linalg.norm(bw[j, :3] - live[j, :3]), np.linalg.norm(bw[j, 3:] - live[j, 3:])))
    np.savez('r7_smooth.npz', L=L, alpha=best, alphas=alphas, cv=e, rough=np.array(rough), times=t)
