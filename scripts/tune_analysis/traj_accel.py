import sys, numpy as np, agama
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import r0_noise as rn
agama.setUnits(length=1, velocity=1, mass=232500)
P = '/home/ana/projects/glossary/data/tango/potentials_triax/'
halo = agama.Potential(P + 'mw00.pot'); stat = agama.Potential(*rn.v21_static())
lmc = agama.Potential(P + 'lmc00.pot')
L = np.loadtxt(P + 'trajlmc.txt'); A = np.loadtxt(P + 'accel.txt')
t = L[:, 0]; r = L[:, 1:4]; v = L[:, 4:7]
a_traj = np.gradient(v, t, axis=0)
print('   t     r   |a_traj| |a_2body|  ratio  radial ratio | accel.txt |F_lmc(MW)| ratio | dx/dt-v')
for tt in (-4.5, -4, -3.5, -3, -2.5, -2, -1.75, -1.5, -1.25, -1.1, -1.0, -0.75, -0.5):
    j = np.argmin(abs(t - tt)); rr = r[j]
    a2 = halo.force(rr) + stat.force(rr) - lmc.force(-rr)
    rh = rr / np.linalg.norm(rr)
    k = np.argmin(abs(A[:, 0] - tt)); fl = lmc.force(-rr)
    dxdt = np.gradient(r, t, axis=0)[j] - v[j]
    print('%6.2f %6.1f %7.3f %7.3f %7.3f %7.3f | %7.3f %7.3f %7.3f | %.2f' % (t[j], np.linalg.norm(rr),
          np.linalg.norm(a_traj[j]), np.linalg.norm(a2), np.linalg.norm(a_traj[j]) / np.linalg.norm(a2),
          (a_traj[j] @ rh) / (a2 @ rh), np.linalg.norm(A[k, 1:]), np.linalg.norm(fl),
          np.linalg.norm(A[k, 1:]) / np.linalg.norm(fl), np.linalg.norm(dxdt)))
print('trajlmc time step', np.unique(np.round(np.diff(t), 6)))
