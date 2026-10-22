import sys, json, numpy as np
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import v21_tune as vt
np.set_printoptions(precision=2, suppress=True, linewidth=220)
O = '/home/ana/projects/glossary/output/v21_sim/tune/round08/'
c = dict(np.load(O + 'centre_check.npz')); info = json.load(open(O + 'round.json'))
d = np.array(info['delta']); W = 1 / vt.TOL; L, S = slice(0, 6), slice(6, 12)
iL, iS, iM = 0, 2, 1          # r_vel_lmc = 1, r_sgr = 5, r_vel_mw = 1 (v21_tune)
def ends(il=iL, isg=iS, im=iM):
    return np.hstack((c['lmc'][:, il] - c['mw'][:, im], c['sgr'][:, isg] - c['mw'][:, im]))
print('reproduces v21_tune endpoints: max |diff|', np.abs(ends() - c['tune_endpoint']).max())
print('\nBOOTSTRAP error of one run (base, 100 resamples), std per component')
for nm, key, r in (('MW ', 'mw', c['r_vel_mw']), ('LMC', 'lmc', c['r_vel_lmc']), ('Sgr', 'sgr', c['r_sgr'])):
    s = c['boot_' + key].std(axis=0)
    for j, rr in enumerate(r):
        print('  %s aperture %3.1f kpc: pos %s kpc  vel %s km/s' % (nm, rr, s[j, :3], s[j, 3:]))
# endpoint error relative to MW (bootstrap the difference)
bl = c['boot_lmc'][:, iL] - c['boot_mw'][:, iM]; bs = c['boot_sgr'][:, iS] - c['boot_mw'][:, iM]
print('\n  endpoint (body - MW) bootstrap std, current apertures: LMC', bl.std(0), '| Sgr', bs.std(0))
print('  in TOL units:                                          LMC', bl.std(0) * W[L], '| Sgr', bs.std(0) * W[S])
print('  -> implied Jacobian error per column (sqrt2*std/nudge), TOL units per nudge unit:')
E = np.hstack((bl.std(0), bs.std(0))) * W * np.sqrt(2)
print('     rows LMC/Sgr x cols: rows', E, '\n     divide by nudges', d)
J = {}
for il, rl in enumerate(c['r_vel_lmc']):
    F = ends(il=il)
    J[rl] = (F[1:] - F[0]).T / d
print('\nLMC velocity rows of J8 (TOL units per nudge unit) for each LMC velocity aperture')
for row, nm in ((3, 'vx'), (4, 'vy'), (5, 'vz')):
    for rl in c['r_vel_lmc']:
        print('  LMC %s, r_vel %3.1f: %s' % (nm, rl, J[rl][row] * W[row]))
print('\nbase LMC endpoint miss /TOL per aperture:')
for il, rl in enumerate(c['r_vel_lmc']):
    print('  r_vel %3.1f: %s' % (rl, (ends(il=il)[0] - vt.TARGET)[L] * W[L]))
Js = {}
for isg, rs in enumerate(c['r_sgr']):
    F = ends(isg=isg); Js[rs] = (F[1:] - F[0]).T / d
print('\nSgr rows of J8 for each Sgr radius (TOL units per nudge unit), and base miss /TOL')
for rs in c['r_sgr']:
    print('  r %3.1f: miss %s' % (rs, ((Js[rs] * 0)[0] if False else (ends(isg=list(c['r_sgr']).index(rs))[0] - vt.TARGET)[S] * W[S])))
for row in range(6, 12):
    print('  %s:' % vt.NAMES[row], '  '.join('r%g %s' % (rs, Js[rs][row] * W[row]) for rs in (3., 5., 7.)))
