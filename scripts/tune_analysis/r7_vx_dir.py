"""
Start shift for Sgr's target vx in round 7, from backward integration with several drag models:
start(live t=0 end with vx at TARGET) - start(live t=0 end), each integrated back to t=-3.
"""
import json, numpy as np, r7_df as rd, v21_tune as vt
np.set_printoptions(precision=4, suppress=True, linewidth=170)
mc = np.load('r7_mass_compare_5_10_20.npy', allow_pickle=True).item()
mb = np.load('r7_mass_compare_bound_held.npy', allow_pickle=True).item() if False else \
     np.load('r7_mass_compare_bound_bound_held.npy', allow_pickle=True).item()
models = [('constant lnL 2.09, M(<5)', None, 2.094), ('constant lnL 3 (V21), M(<5)', None, 3.),
          ('no drag', None, 0.),
          ('smooth lnL(t), M(<5)', None, mc['M(<5 kpc)']['L']), ('smooth lnL(t), M(<20)', 20., mc['M(<20 kpc)']['L']),
          ('smooth lnL(t), bound (t=0 held)', 'bound_held', mb['bound_held']['L'])]
info = json.load(open(rd.D + '/round.json'))
J = np.array(info['jacobian_sgr'])            # live d(end)/d(start), per kpc and km/s
out = {}
for name, mass, L in models:
    m = rd.load(cubic=True, mass=mass)
    t, live = m['times'], m['live']
    th = m['theta'][6:]
    dvx = vt.TARGET[9] - live[-1, 3]
    back = lambda e: rd.integrate(m, L, t[-1], t[0], e)[1][-1]
    b0 = back(live[-1])
    d = {}
    for f in (0.5, 1.):
        e = live[-1].copy(); e[3] += f * dvx
        d[f] = back(e) - b0
    D = d[1.]
    # timing direction at the start: d(state)/dt along the model orbit from the round-7 start
    h = 1. / rd.TU                               # 1 Myr
    s1 = rd.integrate(m, L, t[0], t[0] + h, th)[1][-1]
    tim = s1 - th                                # start change for +1 Myr along the orbit
    W6 = np.concatenate((np.full(3, 1 / 0.05), np.full(3, 1 / 0.25)))      # round-7 nudge sizes as scale
    cos_t = np.dot(D * W6, tim * W6) / np.linalg.norm(D * W6) / np.linalg.norm(tim * W6)
    dt_equiv = np.dot(D * W6, tim * W6) / np.dot(tim * W6, tim * W6)
    fw0 = rd.integrate(m, L, t[0], t[-1], th)[1][-1]
    fw1 = rd.integrate(m, L, t[0], t[-1], th + D)[1][-1]
    print('=== %s   (vx change at t=0: %+.3f km/s)' % (name, dvx))
    print('  start shift  dx %s kpc  dv %s km/s   |dx| %.4f kpc |dv| %.4f km/s' % (D[:3], D[3:], np.linalg.norm(D[:3]), np.linalg.norm(D[3:])))
    print('  linearity: 2 x half-step shift - full shift: %s' % (2 * d[0.5] - D))
    print('  timing part: cos(shift, +1 Myr direction) %.3f (in round-7 nudge units); equivalent %.2f Myr' % (cos_t, dt_equiv))
    print('  model forwards from round-7 start + shift, end change: %s' % (fw1 - fw0))
    print('  live Jacobian (round-7 nudges) applied to shift, end change: %s' % (J @ D))
    out[name] = dict(D=D, half=d[0.5], tim=tim, model_fw=fw1 - fw0, live_J=J @ D)
np.save('r7_vx_dir.npy', out, allow_pickle=True)
names = list(out)
print('\ncosine between start shifts (position+velocity, in round-7 nudge units):')
W6 = np.concatenate((np.full(3, 1 / 0.05), np.full(3, 1 / 0.25)))
for a in names:
    print('%-34s' % a[:34], ' '.join('%6.3f' % (np.dot(out[a]['D'] * W6, out[b]['D'] * W6) /
          np.linalg.norm(out[a]['D'] * W6) / np.linalg.norm(out[b]['D'] * W6)) for b in names))
