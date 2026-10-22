import json, numpy as np, r7_df as rd
T = '/home/ana/projects/glossary/output/v21_sim/tune/round03'
m = rd.load('r3test', cubic=True)
res, dv = rd.fit_residual(m, verbose=True)
np.save('r3test/res_cubic.npy', res)
info = json.load(open(T + '/round.json'))
th0 = np.array(info['theta'])
runs, ths = [], []
for i in range(7, 13):
    runs.append(dict(np.load(T + '/run%02d_centres.npz' % i)))
    th = th0.copy(); th[i - 1] += info['delta'][i - 1]; ths.append(th)
rd.other_tracks(m, 0., res, runs, ths)
