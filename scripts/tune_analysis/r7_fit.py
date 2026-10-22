import numpy as np, r7_df as rd
m = rd.load(cubic=True)
out = {}
for kind in ('time', 'chandra'):
    print('=== %s-fixed' % kind if kind == 'time' else '=== Chandrasekhar-shaped', flush=True)
    res, dv = rd.fit_residual(m, kind=kind)
    np.save('r7_res_%s.npy' % kind, res)
    out[kind] = (res, dv)
    print('total dv over 3 time units: drag %.2f, in-plane %.2f, normal %.2f km/s' % tuple(dv.sum(0)), flush=True)
print(rd.plot_drag(m, {'time-fixed': (out['time'][0], 'time'), 'Chandrasekhar-shaped': (out['chandra'][0], 'chandra')},
                   '/home/ana/projects/glossary/scripts/tune_analysis/r7_drag.png'))
