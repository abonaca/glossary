import sys, json, pickle, numpy as np
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import v21_surrogate as vs, v21_tune as vt
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicHermiteSpline
np.set_printoptions(precision=3, suppress=True, linewidth=200)
O = '/home/ana/projects/glossary/output/v21_sim/tune/'
vs.set_sigma_profile(tuple(np.load('/home/ana/projects/glossary/output/v21_sim/ics/mw_triax_full/halo_sigma.npy')))
tr = vs.load_tracks(2); info = tr['info']; base = tr['runs'][0]
c = pickle.load(open(O + 'round02/round3_candidates.pkl', 'rb')); L = c['lnL']; off = c['offset']
ev = vs.EvolvingField(O + 'round02/expansions', base)
th2 = np.array(info['theta'])[6:]; W = 1 / vt.TOL[6:]
T0, T1 = vt.T_START, vt.T_END

# same equations as vs.sgr_evolving, but integrable in either direction and returning absolute states
Rl = vs.sgr_reflex(tr, base)
R_live = CubicHermiteSpline(base['times'], Rl[:, :3], Rl[:, 3:]); dR_live = R_live.derivative()
vm_live = ev.xm.derivative()
m = lambda t: np.interp(t, tr['sgr_t'], tr['sgr_m'])
def rhs(t, w):
    x, v, R, VR = w[:3], w[3:6], w[6:9], w[9:]
    xm = ev.xm(t) - R_live(t) + R
    vm = vm_live(t) - dR_live(t) + VR
    d = x - xm
    a = ev.force(x, t, xm) + vs._friction_rho(ev.halo_density(d, t), v - vm, m(t), L, d)
    return np.hstack((v, a, VR, m(t) * d / (np.sum(d**2) + vs.SGR_PLUMMER**2)**1.5))
mw_live = lambda t: np.hstack((ev.xm(t), vm_live(t)))     # live MW track (reflex to live Sgr included)

def forward(th):
    """present-day Sgr relative to the model MW, from th (relative to the live MW at T0), as in sgr_evolving"""
    w0 = np.hstack((th + mw_live(T0), np.zeros(6)))
    w = solve_ivp(rhs, (T0, T1), w0, rtol=1e-10, atol=1e-10, method='DOP853').y[:, -1]
    return w[:6] - (mw_live(T1) - np.hstack((R_live(T1), dR_live(T1))) + w[6:])

def backward(y):
    """start (relative to the live MW at T0) whose backward orbit passes through y (relative to the MW) at T1"""
    w1 = np.hstack((y + mw_live(T1), R_live(T1), dR_live(T1)))
    w = solve_ivp(rhs, (T1, T0), w1, rtol=1e-10, atol=1e-10, method='DOP853').y[:, -1]
    return w[:6] - mw_live(T0), w[6:]

print('check: forward(round-2 start) - model at round-2 start (sgr_evolving):',
      forward(th2) - vs.sgr_evolving(ev, tr, base, th2, [T1], lnL=L)[-1], flush=True)
ystar = vt.TARGET[6:] + off       # model endpoint whose offset-corrected forecast is the target
y = ystar.copy()
for it in range(6):
    th, R0 = backward(y)
    F = forward(th)
    miss = (F - ystar) * W
    print('iter %d: MW reflex left at T0 %s\n  start %s  minus round 2 %s\n  forward miss/TOL %s' % (
        it, R0, th, th - th2, miss), flush=True)
    if np.all(np.abs(miss) < 0.1):
        break
    y = y - (F - ystar)
c['sgr_backward'] = th
pickle.dump(c, open(O + 'round02/round3_candidates.pkl', 'wb'))
print('\nexact-model Sgr start', th)
print('  minus round-2 start ', th - th2)
print('  minus round-3 cand. ', th - c['theta3'][6:])
