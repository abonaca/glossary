import sys, numpy as np, agama
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import r0_noise as rn
agama.setUnits(length=1, velocity=1, mass=232500)
P = '/home/ana/projects/glossary/data/tango/potentials_triax/'
mw = agama.Potential(agama.Potential(P + 'mw00.pot'), *rn.v21_static())
lmc = agama.Potential(P + 'lmc00.pot')
L = np.loadtxt(P + 'trajlmc.txt'); L = L[L[:, 0] >= -3]
t, r, v = L[:, 0], L[:, 1:4], L[:, 4:7]
a2 = mw.force(r) - lmc.force(-r)
from scipy.integrate import cumulative_trapezoid as ct
dv = (v - v[0]) - ct(a2, t, axis=0, initial=0)      # velocity change not explained by rigid two-body force
rh = r / np.linalg.norm(r, axis=1)[:, None]
print('    t      r   dv_excess (vector, km/s)        |dv|  radial')
for tt in (-2.95, -2.9, -2.8, -2.7, -2.6, -2.5, -2.25, -2, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0):
    j = np.argmin(abs(t - tt))
    print('%6.2f %6.1f  %s %6.2f %6.2f' % (t[j], np.linalg.norm(r[j]), np.array2string(dv[j], precision=2), np.linalg.norm(dv[j]), dv[j] @ rh[j]))
