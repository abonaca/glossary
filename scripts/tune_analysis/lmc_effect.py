import sys, json, pickle, numpy as np
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import v21_surrogate as vs, v21_tune as vt
np.set_printoptions(precision=2, suppress=True, linewidth=200)
O = '/home/ana/projects/glossary/output/v21_sim/tune/'
vs.set_sigma_profile(tuple(np.load('/home/ana/projects/glossary/output/v21_sim/ics/mw_triax_full/halo_sigma.npy')))
c = pickle.load(open(O + 'round02/round3_candidates.pkl', 'rb')); L = c['lnL']
sm = np.load(O + 'round02/sgr_mass.npz')
def tracks(k):
    info = json.load(open(O + 'round%02d/round.json' % k))
    runs = [dict(np.load(O + 'round%02d/run%02d_centres.npz' % (k, i))) for i in range(13)]
    return dict(info=info, runs=runs, sgr_t=sm['times'], sgr_m=sm['mass'])   # round-2 Sgr mass loss for both
t2, t3 = tracks(2), tracks(3)
ev2 = vs.EvolvingField(O + 'round02/expansions', t2['runs'][0]); ev3 = vs.EvolvingField(O + 'round03/expansions', t3['runs'][0])
s2, s3 = np.array(t2['info']['theta'])[6:], np.array(t3['info']['theta'])[6:]
W = 1 / vt.TOL[6:]; T = vt.TARGET[6:]
m = lambda ev, tr, s: vs.sgr_evolving(ev, tr, tr['runs'][0], s, [0.0], lnL=L)[-1]
live2, live3 = vs._live_rel(t2['runs'][0])[-1, 6:], vs._live_rel(t3['runs'][0])[-1, 6:]
a22, a23, a33, a32 = m(ev2, t2, s2), m(ev2, t2, s3), m(ev3, t3, s3), m(ev3, t3, s2)
print('model - live, r2 start in r2 field :', (a22 - live2) * W)
print('model - live, r3 start in r3 field :', (a33 - live3) * W)
print('\nSgr r3 start: r3 field minus r2 field (effect of the LMC/MW change) /TOL:', (a33 - a23) * W)
print('Sgr r2 start: r3 field minus r2 field                                /TOL:', (a32 - a22) * W)
print('\nforecast used for r3 (r2 field, r3 Sgr start, r2 offset) resid/TOL:', (a23 - (a22 - live2) - T) * W)
print('same but in the r3 field                                   resid/TOL:', (a33 - (a22 - live2) - T) * W)
print('live r3                                                    resid/TOL:', (live3 - T) * W)
print('\nmodel change r2->r3 in the r3 field (Sgr start change only) /TOL:', (a33 - a32) * W)
print('live change r2->r3                                         /TOL:', (live3 - live2) * W)
