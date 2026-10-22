import sys, json, pickle, numpy as np
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import v21_surrogate as vs, v21_tune as vt
np.set_printoptions(precision=2, suppress=True, linewidth=200)
O = '/home/ana/projects/glossary/output/v21_sim/tune/'
def load(k):
    d = O + 'round%02d/' % k
    return json.load(open(d + 'round.json')), [dict(np.load(d + 'run%02d_centres.npz' % i)) for i in range(13)]
i2, R2 = load(2); i3, R3 = load(3)
c = pickle.load(open(O + 'round02/round3_candidates.pkl', 'rb'))
TOL = vt.TOL; T = vt.TARGET
F2 = vs._live_rel(R2[0])[-1]; F3 = vs._live_rel(R3[0])[-1]
th2, th3 = np.array(i2['theta']), np.array(i3['theta'])
print('theta3 - candidate', th3 - c['theta3'])
pred = np.concatenate((np.array([1.203, -1.906, -0.73, -6.86, -2.877, 3.817]), c['sgr_fun']))
print('r2 resid/TOL ', (F2 - T) / TOL)
print('forecast     ', pred)
print('r3 resid/TOL ', (F3 - T) / TOL)
print('r3 - forecast', (F3 - T) / TOL - pred)
# linear prediction of r3 from r2 live Jacobian
J2 = np.array(i2['jacobian'])
print('r2-Jacobian linear forecast', (F2 + J2 @ (th3 - th2) - T) / TOL)
# timing decomposition of the Sgr change: closest time on r3 base track to the target, and to r2 endpoint
for lab, R in (('r2', R2[0]), ('r3', R3[0])):
    tr = vs._live_rel(R); ts = R['times']
    tf = np.linspace(ts[-6], ts[-1], 2001)
    from scipy.interpolate import CubicHermiteSpline
    for b, sl in (('LMC', slice(0, 6)), ('Sgr', slice(6, 12))):
        x = CubicHermiteSpline(ts, tr[:, sl][:, :3], tr[:, sl][:, 3:])
        v = x.derivative()
        dx = np.linalg.norm(x(tf) - T[sl][:3], axis=1); j = np.argmin(dx)
        w = np.hstack((x(tf[j]), v(tf[j]) / 1.0227))
        print('%s %s closest approach %.2f kpc at t=%.1f Myr; resid/TOL there %s' % (lab, b, dx[j], tf[j] * 977.8, (w - T[sl]) / TOL[sl]))
# MW base offset check: MW velocity in base vs nudged runs
for lab, R in (('r2', R2), ('r3', R3)):
    mw = np.array([np.hstack((r['mw'][-1])) if 'mw' in r else None for r in R])
    print(lab, 'MW final (base, nudged mean, nudged std):', mw[0], mw[1:].mean(0), mw[1:].std(0))
# Jacobians: r3 columns and comparison with r2 per unit
J3 = np.array(i3['jacobian'])
W = 1 / TOL
print('column norms (in TOL per unit) r2:', np.linalg.norm(W[:, None] * J2, axis=0))
print('column norms (in TOL per unit) r3:', np.linalg.norm(W[:, None] * J3, axis=0))
print('rel diff |J3-J2|/|J3| per column:', np.linalg.norm(W[:, None] * (J3 - J2), axis=0) / np.linalg.norm(W[:, None] * J3, axis=0))
print('response per nudge (TOL) r3:', np.linalg.norm(W[:, None] * J3 * np.array(i3['delta']), axis=0))
# Sgr block SVD of r3 live Jacobian, residual in modes
Js = W[6:, None] * J3[6:, 6:] / W[None, 6:]  # TOL-out per TOL-in
U, S, Vt = np.linalg.svd(W[6:, None] * J3[6:, 6:])
print('Sgr block singular values (TOL per unit):', S)
print('r2 Sgr resid in modes:', U.T @ ((F2 - T)[6:] * W[6:]))
print('r3 Sgr resid in modes:', U.T @ ((F3 - T)[6:] * W[6:]))
print('U0 (mode 0 output direction):', U[:, 0], '\nV0:', Vt[0])
d = (F3 - F2)[6:]; print('Sgr r3-r2 change:', d, ' in modes (TOL):', U.T @ (d * W[6:]))
print('\n--- decomposition of the Sgr r2->r3 change (TOL), mode 0 first ---')
dth = th3 - th2
for lab, J in (('J2', J2), ('J3', J3), ('avg', 0.5 * (J2 + J3))):
    eL = J[6:, :6] @ dth[:6]; eS = J[6:, 6:] @ dth[6:]
    print(lab, 'from LMC step', eL * W[6:], 'modes', (U.T @ (eL * W[6:]))[:2], '| from Sgr step', eS * W[6:], 'modes', (U.T @ (eS * W[6:]))[:2])
print('actual', (F3 - F2)[6:] * W[6:])
print('per-column mode-0 contributions (avg J):', (U[:, 0] @ (W[6:, None] * 0.5 * (J2 + J3)[6:])) * dth)
# timing: shift of r3 Sgr track that best matches r2 endpoint... and target
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import minimize_scalar
ts = R3[0]['times']; tr = vs._live_rel(R3[0])[:, 6:]
x = CubicHermiteSpline(ts, tr[:, :3], tr[:, 3:]); v = x.derivative()
st = lambda t: np.hstack((x(t), v(t) / 1.0227))
for lab, y in (('target', T[6:]),):
    m = minimize_scalar(lambda t: np.sum(((st(t) - y) * W[6:])**2), bounds=(-0.02, 0.02), method='bounded')
    print('r3 Sgr best time shift to %s: %.1f Myr, resid/TOL there %s' % (lab, m.x * 977.8, (st(m.x) - y) * W[6:]))
print('\n--- timing-only Sgr correction ---')
r3 = (F3 - T)[6:] * W[6:]
step = -(U[:, 0] @ r3) / S[0] * Vt[0]            # in physical units (W applied to outputs only)
print('step (x0,y0,z0 kpc; v km/s):', step)
for lab, J in (('J2', J2), ('J3', J3)):
    print(lab, 'forecast resid/TOL:', r3 + W[6:] * (J[6:, 6:] @ step))
# alternative: least-squares x0,z0 only
for lab, J in (('J2', J2), ('J3', J3)):
    A = W[6:, None] * J[6:, 6:][:, [0, 2]]
    s2 = np.linalg.lstsq(A, -r3, rcond=None)[0]
    print(lab, 'x0,z0 lstsq step', s2, 'forecast', r3 + A @ s2)
print('\n--- is mode 0 timing? ---')
dst = np.hstack((v(0.0) / 1.0227, (v(0.0 + 1e-4) - v(0.0 - 1e-4)) / 2e-4 / 1.0227**2 / 1000))  # per Myr: kpc/Myr, km/s/Myr
dst = np.hstack((x(1e-4 / 0.9778 * 1) - x(-1e-4 / 0.9778), (v(1e-4 / 0.9778) - v(-1e-4 / 0.9778)))) / 2e-4  # per Gyr-ish units
tv = dst * W[6:]; tv /= np.linalg.norm(tv)
print('time-derivative direction (TOL-weighted):', tv)
print('U0                                       :', U[:, 0])
print('|cos| with U0..U5:', np.abs(U.T @ tv))
