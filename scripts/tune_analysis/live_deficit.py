import sys, numpy as np, agama
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import r0_noise as rn
agama.setUnits(length=1, velocity=1, mass=232500)
P = '/home/ana/projects/glossary/data/tango/potentials_triax/'
mw = agama.Potential(agama.Potential(P + 'mw00.pot'), *rn.v21_static())
lmc = agama.Potential(P + 'lmc00.pot')
from scipy.integrate import solve_ivp
def excess(t, r, v):
    # velocity change not explained by the rigid two-body force, integrated along the track
    # (interpolate the track with a cubic spline in time so the force integral is resolved)
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(t, r); tf = np.linspace(t[0], t[-1], 3001)
    a2 = mw.force(cs(tf)) - lmc.force(-cs(tf))
    from scipy.integrate import cumulative_trapezoid as ct
    I = ct(a2, tf, axis=0, initial=0)
    return np.array([(v[j] - v[0]) - I[np.argmin(abs(tf - t[j]))] for j in range(len(t))])
d = np.load(sys.argv[1]); t = d['times']
rl = d['lmc'][:, :3] - d['mw'][:, :3]; vl = d['lmc'][:, 3:] - d['mw'][:, 3:]
L = np.loadtxt(P + 'trajlmc.txt'); L = L[L[:, 0] >= -3]
ev = excess(L[:, 0], L[:, 1:4], L[:, 4:7]); el = excess(t, rl, vl)
print('    t   r_live |dv_ex| live  radial | r_V21 |dv_ex| V21 radial   (km/s)')
for j, tt in enumerate(t):
    k = np.argmin(abs(L[:, 0] - tt))
    print('%6.3f %6.1f %7.2f %7.2f | %6.1f %7.2f %7.2f' % (tt, np.linalg.norm(rl[j]), np.linalg.norm(el[j]),
          el[j] @ rl[j] / np.linalg.norm(rl[j]), np.linalg.norm(L[k, 1:4]), np.linalg.norm(ev[k]),
          ev[k] @ L[k, 1:4] / np.linalg.norm(L[k, 1:4])))
