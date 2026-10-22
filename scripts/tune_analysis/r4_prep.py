import sys, json, numpy as np, agama
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import v21_surrogate as vs, v21_tune as vt
from scipy.integrate import solve_ivp
np.set_printoptions(precision=3, suppress=True, linewidth=200)
O = '/home/ana/projects/glossary/output/v21_sim/tune/'
MYR = 1 / 977.8
i2 = json.load(open(O + 'round02/round.json')); i3 = json.load(open(O + 'round03/round.json'))
J2, J3 = np.array(i2['jacobian']), np.array(i3['jacobian'])
th3, r3 = np.array(i3['theta']), np.array(i3['residual']); TOL = vt.TOL; W = 1 / TOL
# --- Sgr: move the start along its own orbit by +5.6 Myr, in the round-3 field at t=-3
E = O + 'round03/expansions/'
mw = agama.Potential(agama.Potential(E + 'halo_00.coef'), agama.Potential(E + 'disk_00.coef'))
lmc = agama.Potential(E + 'lmc_00.coef')
xl = th3[:3]                                     # LMC start relative to the MW
def rhs(t, w):
    return np.hstack((w[3:], mw.force(w[:3]) + lmc.force(w[:3] - xl) - lmc.force(-xl)))
s0 = th3[6:]
s1 = solve_ivp(rhs, (0, 5.6 * MYR), s0, rtol=1e-11, atol=1e-11, method='DOP853').y[:, -1]
dS = s1 - s0
print('Sgr start r3', s0, '\n  advanced 5.6 Myr', s1, '\n  change', dS)
U, S, Vt = np.linalg.svd(W[6:, None] * J3[6:, 6:])
print('  cos(change, V0 of J3) = %.3f' % (abs(Vt[0] @ dS) / np.linalg.norm(dS)))
for lab, J in (('J2', J2), ('J3', J3)):
    e = W * (J[:, 6:] @ dS)
    print('  %s: effect of Sgr change on [LMC | Sgr] /TOL' % lab, e[:6], e[6:])
    print('      forecast Sgr resid/TOL', r3[6:] * W[6:] + e[6:])
# --- LMC: step from the LMC block given r3 residual + Sgr change's effect on the LMC
for lab, J in (('J3', J3), ('J2', J2), ('mean', 0.5 * (J2 + J3))):
    Jw = J[:6, :6] * W[:6, None]
    for (lam, nm) in ((0., 'full GN'), (0.3, 'damped 0.3')):
        rw = r3[:6] * W[:6] + W[:6] * (J[:6, 6:] @ dS)
        A = Jw.T @ Jw
        dl = -np.linalg.solve(A + lam * np.diag(np.diag(A)), Jw.T @ rw)
        print('%-4s %-10s LMC step %s  pred LMC /TOL %s  LMC-step effect on Sgr /TOL %s' % (
            lab, nm, dl, rw + Jw @ dl, W[6:] * (J[6:, :6] @ dl)))
# cross-check the LMC step with the other Jacobian
print('\n--- LMC step ignoring the Sgr->LMC cross-term ---')
for lab, J in (('J3', J3), ('J2', J2), ('mean', 0.5 * (J2 + J3))):
    Jw = J[:6, :6] * W[:6, None]; rw = r3[:6] * W[:6]; A = Jw.T @ Jw
    for (lam, nm) in ((0., 'full GN'), (0.3, 'damped 0.3')):
        dl = -np.linalg.solve(A + lam * np.diag(np.diag(A)), Jw.T @ rw)
        print('%-4s %-10s LMC step %s  pred LMC /TOL %s  effect on Sgr /TOL %s' % (lab, nm, dl, rw + Jw @ dl, W[6:] * (J[6:, :6] @ dl)))
# how well did each Jacobian predict the actual r2->r3 change of the LMC? (theta change known)
th2 = np.array(i2['theta']); F2 = np.array(i2['residual']); d = th3 - th2
for lab, J in (('J2', J2), ('J3', J3), ('mean', 0.5 * (J2 + J3))):
    print('%s predicted r2->r3 change /TOL' % lab, W * (J @ d), '\n   actual', W * (r3 - F2))
# time-shift check: what the 5.6 Myr forward integration says the Sgr change should be
print('expected Sgr change from +5.6 Myr (forward integration):', np.array([14.1, -1.7, 12.2, -12.7, -1.4, 6.5]))
