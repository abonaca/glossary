import sys, json, numpy as np
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
exec(open('sgr_timing_cross.py').read().split("lt = {k")[0].replace('(4, 5, 6, 7)', '(4, 5, 6, 7, 8)'))
lt = {k: lateness(k) for k in I}
print('Sgr lateness (Myr, + = late):', {k: round(float(v), 2) for k, v in lt.items()})
print('Sgr arrival shift per Sgr nudge unit (Myr, + = later), |cos| along timing')
for k in I:
    u = u_dir(k); C = W[S, None] * J[k][S, S]
    print(' J%d  tau %s  |cos| %s' % (k, -(u @ C) / (u @ u), np.abs(u @ C) / (np.linalg.norm(u) * np.linalg.norm(C, axis=0))))
for k in (4, 5, 6, 7):
    d = th[k + 1] - th[k]; u = u_dir(k)
    pl = -(u @ (W[S, None] * J[k][S, L] @ d[L])) / (u @ u); ps = -(u @ (W[S, None] * J[k][S, S] @ d[S])) / (u @ u)
    print(' r%d->r%d  J%d predicts: LMC %+.2f  Sgr own %+.2f  total %+.2f | actual change %+.2f Myr' % (k, k + 1, k, pl, ps, pl + ps, lt[k + 1] - lt[k]))
