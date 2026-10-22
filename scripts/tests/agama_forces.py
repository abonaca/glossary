# reference forces from AGAMA for random points and times
import sys, numpy as np, agama
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import mw_lmc_agama as m
d = np.load(m.os.path.join(m.OUT_DIR, 'mw_lmc_1.5e+11.npz'))
rng = np.random.default_rng(1)
n = 300
r = 10**rng.uniform(-0.5, 2.3, n); ct = rng.uniform(-1, 1, n); ph = rng.uniform(0, 2*np.pi, n)
xyz = np.column_stack([r*np.sqrt(1-ct**2)*np.cos(ph), r*np.sqrt(1-ct**2)*np.sin(ph), r*ct])
xyz[:50,2] = rng.normal(0, 0.3, 50)   # points near the disk midplane
t_myr = rng.uniform(-3000, 0, n)
pot_mw = agama.Potential(m.PARAM_BULGE, m.PARAM_DISK, m.PARAM_HALO)
traj = np.column_stack([d['t']/m.TIME_UNIT_MYR, d['lmc_w'][:,:3]])
pot_lmc = agama.Potential(potential=agama.Potential(m.param_lmc(1.5e11)), center=traj)
acc = np.column_stack([d['t']/m.TIME_UNIT_MYR, -d['mw_acc']*m.TIME_UNIT_MYR/m.KMS_TO_KPCMYR])
pot_acc = agama.Potential(type='UniformAcceleration', file=acc)
conv = m.KMS_TO_KPCMYR / m.TIME_UNIT_MYR   # (km/s)/time unit -> kpc/Myr^2
f_mw = pot_mw.force(xyz) * conv
f_tot = np.array([agama.Potential(pot_mw, pot_lmc, pot_acc).force(xyz[i], t=t_myr[i]/m.TIME_UNIT_MYR) for i in range(n)]) * conv
np.savez(sys.argv[1], xyz=xyz, t=t_myr, f_mw=f_mw, f_tot=f_tot)
print('saved', sys.argv[1])
