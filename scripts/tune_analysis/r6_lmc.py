import json, numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
E = np.array([
[-0.65,-41.24,-27.01,-61.94,-210.29,202.93],
[-0.82,-40.2,-28.07,-67.24,-216.61,202.5],
[0.235,-38.4,-29.9,-65.5,-222,201],
[-0.51,-40.56,-27.43,-64.42,-213.57,203.43],
[0.68,-39.01,-30.22,-76.38,-217.23,197.39],
[2.34,-31.25,-36.83,-62.32,-238.48,180.04],
[-2.16,-42.8,-19.21,-68.86,-200.75,221.69]])
TL = np.array([.2,.2,.2,1,1,1.]); d = np.array([3,1,3,3,2,3.])
r = np.array([-0.05,0.06,0.09,1.96,3.51,-3.67])
J6 = (E[1:] - E[0]).T / d
print('J6 LMC block (rows x,y,z,vx,vy,vz; cols x0..vz0, per unit):\n', J6)
s = -np.linalg.solve(J6, r)
print('undamped GN step', s, ' predicted resid/TOL', (r + J6 @ s) / TL)
th6 = np.array(json.load(open('theta6.json'))) if True else None
print('theta6 LMC', np.array(th6)[:6], '-> theta7 LMC', np.array(th6)[:6] + s)
