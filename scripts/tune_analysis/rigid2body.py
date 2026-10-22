import sys, numpy as np, agama
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import r0_noise as rn
agama.setUnits(length=1, velocity=1, mass=232500)
P = '/home/ana/projects/glossary/data/tango/potentials_triax/'
mw = agama.Potential(agama.Potential(P + 'mw00.pot'), *rn.v21_static())
lmc = agama.Potential(P + 'lmc00.pot')
L = np.loadtxt(P + 'trajlmc.txt')
d = np.load(sys.argv[1])
def rhs(t, w):
    r = w[:3]
    return np.hstack((w[3:], mw.force(r) - lmc.force(-r)))
from scipy.integrate import solve_ivp
w0 = L[L[:, 0] == -3][0, 1:]
ts = d['times']
sol = solve_ivp(rhs, (-3, 0), w0, t_eval=ts, rtol=1e-10, atol=1e-10)
def lead(xyz, t):
    j = np.argmin(np.linalg.norm(L[:, 1:4] - xyz, axis=1)); return L[j, 0] - t
print('    t   r_V21  r_rigid  r_live   lead_rigid  lead_live  (lead = V21 time at closest point - t)')
for j, t in enumerate(ts):
    ours = d['lmc'][j, :3] - d['mw'][j, :3]
    rv = np.linalg.norm([np.interp(t, L[:, 0], L[:, k]) for k in (1, 2, 3)])
    print('%6.3f %6.1f %7.1f %7.1f %9.3f %9.3f' % (t, rv, np.linalg.norm(sol.y[:3, j]), np.linalg.norm(ours),
          lead(sol.y[:3, j], t), lead(ours, t)))
