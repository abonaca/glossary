"""Tests of agama_jax against AGAMA itself (both installed in streamsculptor-env).

Run all:            pytest scripts/test_agama_jax.py -v -s
Or from ipython:    import test_agama_jax as ta; ta.report()

Errors are relative: |dPhi/Phi| and |dF|/|F|. AGAMA works in N-body units (kpc, km/s, G=1, time in kpc/(km/s)).
"""

import os
import re
import tempfile

import numpy as np
import jax
import jax.numpy as jnp
import diffrax
import agama

import agama_jax as aj

VARIANT = 'triax'
CONV = aj.KMS_TO_KPCMYR**2       # (km/s)^2 -> kpc^2/Myr^2
TMP_DIR = tempfile.mkdtemp(prefix='agama_jax_test_')
SNAP_TIMES = np.arange(-1.5, 0.25 + 1e-9, 0.0625)


# AGAMA reference potentials #

def ini_sections(variant=VARIANT, linear=False, keep=('halo', 'bulge', 'disk', 'lmc', 'acc')):
    """Text of the published potential_evolving.ini restricted to the given sections, with absolute file paths
    and optionally linearInterp=true added to the evolving components"""

    folder = aj.tango_dir(variant)
    with open(aj.tango_ini(variant)) as f:
        text = f.read()
    blocks = re.split(r'(?m)^(?=\[)', text)
    out = []
    for b in blocks:
        m = re.match(r'\[potential\s*(\w*)\]', b, re.I)
        if not m or m.group(1).lower() not in keep:
            continue
        b = re.sub(r'(?<=[=\s])([\w.]+\.(pot|txt))', lambda mm: os.path.join(folder, mm.group(1)), b)
        if linear:
            b = re.sub(r'(?im)^type=evolving$', 'type=evolving\nlinearInterp=true', b)
        out.append(b)
    return '\n'.join(out)

def write_ini(text, name):
    path = os.path.join(TMP_DIR, name)
    with open(path, 'w') as f:
        f.write(text)
    return path

def agama_ref(keep=('halo', 'bulge', 'disk', 'lmc', 'acc'), linear=False, variant=VARIANT):
    """AGAMA potential made of the given sections of the Tango ini"""

    return agama.Potential(write_ini(ini_sections(variant, linear, keep), f"ref_{'_'.join(keep)}_{linear}.ini"))

def agama_sgr(b=aj.SGR_SCALE_RADIUS):
    """AGAMA reference for the Sgr Plummer: an Evolving series of Plummer spheres (one per trajectory row) with
    linear interpolation, centered on the Hermite-interpolated trajectory"""

    d = np.loadtxt(os.path.join(aj.TANGO_DIR, 'Sgr_snapshot', 'trajectory.txt'))
    traj = os.path.join(TMP_DIR, 'sgr_traj.txt')
    np.savetxt(traj, d[:,:7])
    lines = ['[Potential sgr]', 'type=evolving', 'linearInterp=true', f'center={traj}', 'Timestamps']
    for i, row in enumerate(d):
        p = write_ini(f'[Potential]\ntype=Plummer\nmass={row[7]*aj.G_KMS:.15g}\nscaleRadius={b}\n', f'sgr_{i:03d}.ini')
        lines.append(f'{row[0]:.10f} {p}')
    return agama.Potential(write_ini('\n'.join(lines) + '\n', 'sgr.ini'))

def agama_reflex(plummer, t_grid=np.linspace(-3.2, 0.3, 3501)):
    """AGAMA UniformAcceleration tabulated densely from our PlummerReflex (tests sign and magnitude)"""

    acc = -np.array(jax.vmap(lambda t: aj.PlummerReflex(plummer).a_mw(t))(jnp.asarray(t_grid*aj.TIME_UNIT_MYR)))
    path = os.path.join(TMP_DIR, 'sgr_reflex.txt')
    np.savetxt(path, np.column_stack([t_grid, acc / CONV]))
    return agama.Potential(type='UniformAcceleration', file=path)


# Comparison helpers #

def random_points(n, rmin=1e-3, rmax=1e4, seed=1):
    """Points isotropic in direction and uniform in log radius [kpc]"""

    rng = np.random.default_rng(seed)
    r = 10**rng.uniform(np.log10(rmin), np.log10(rmax), n)
    u = rng.normal(size=(n, 3))
    return u / np.linalg.norm(u, axis=1)[:,None] * r[:,None]

def jax_eval(pot, x, t):
    """Potential and force of a streamsculptor potential in AGAMA units; x (n,3) kpc, t (n,) AGAMA time"""

    xs, ts = jnp.asarray(x), jnp.asarray(np.broadcast_to(t, len(x)) * aj.TIME_UNIT_MYR)
    phi = jax.jit(jax.vmap(pot.potential))(xs, ts)
    acc = -jax.jit(jax.vmap(pot.gradient))(xs, ts)
    return np.asarray(phi) / CONV, np.asarray(acc) / CONV

def agama_eval(pot, x, t):
    t = np.broadcast_to(t, len(x))
    return pot.potential(x, t=t), pot.force(x, t=t)

def errors(ref, new, phi_ref_scale=None):
    """Relative errors of potential and force; phi_ref_scale replaces |Phi_ref| (for potentials crossing zero)"""

    (p0, f0), (p1, f1) = ref, new
    ep = abs(p1 - p0) / (abs(p0) if phi_ref_scale is None else phi_ref_scale)
    ef = np.linalg.norm(f1 - f0, axis=1) / np.linalg.norm(f0, axis=1)
    return ep, ef

def summary(label, ep, ef):
    s = (f'{label:42s} n={len(ef):6d}  phi med {np.median(ep):.1e} max {ep.max():.1e}   '
         f'force med {np.median(ef):.1e} max {ef.max():.1e}')
    print(s)
    return dict(phi_med=np.median(ep), phi_max=ep.max(), force_med=np.median(ef), force_max=ef.max())


# Tests: single components #

def check_snapshots(files=None, n=3000):
    """Every Multipole snapshot file vs AGAMA, split into inner extrapolation, grid, and outer extrapolation"""

    folder = aj.tango_dir(VARIANT)
    if files is None:
        files = [f'mw{i:02d}.pot' for i in range(29)] + [f'lmc{i:02d}.pot' for i in range(1, 29)]
    x = random_points(n)
    r = np.linalg.norm(x, axis=1)
    res = {}
    for f in files:
        d = aj.read_pot_file(os.path.join(folder, f))
        p = aj.EvolvingMultipole([0.], [aj.prepare_multipole(d['radii'], d['Phi'], d['dPhi'])])
        ep, ef = errors(agama_eval(agama.Potential(os.path.join(folder, f)), x, 0.), jax_eval(p, x, 0.))
        for lab, m in [('inner', r < d['radii'][0]), ('grid', (r >= d['radii'][0]) & (r <= d['radii'][-1])),
                       ('outer', r > d['radii'][-1])]:
            res[(f, lab)] = dict(phi_med=np.median(ep[m]), phi_max=ep[m].max(), force_med=np.median(ef[m]),
                                 force_max=ef[m].max())
    for lab in ['inner', 'grid', 'outer']:
        vals = [v for (f, l), v in res.items() if l == lab]
        print(f'{len(vals)} snapshots, {lab:5s}: worst phi med {max(v["phi_med"] for v in vals):.1e} '
              f'max {max(v["phi_max"] for v in vals):.1e}   worst force med {max(v["force_med"] for v in vals):.1e} '
              f'max {max(v["force_max"] for v in vals):.1e}')
    return res

def test_snapshots():
    res = check_snapshots()
    for (f, lab), v in res.items():
        if lab == 'grid':
            assert v['force_med'] < 1e-5 and v['force_max'] < 1e-3 and v['phi_max'] < 1e-5, (f, v)
        else:
            assert v['force_max'] < 1e-12 and v['phi_max'] < 1e-12, (f, lab, v)

def test_bulge_disk(n=5000):
    """Bulge + disk table (same grid as the V21 table) vs AGAMA, within 0.2-400 kpc
    (inside ~0.05 kpc the force goes to zero and relative errors reach 1e-3)"""

    comps = aj.tango_components(VARIANT, static=True)
    x = random_points(n, 0.2, 400)
    e = errors(agama_eval(agama_ref(('bulge', 'disk')), x, 0.), jax_eval(comps['bulge_disk'], x, 0.))
    v = summary('bulge+disk table', *e)
    assert v['force_med'] < 1e-4 and v['phi_max'] < 1e-4

def test_lmc_initial(n=3000):
    """Initial analytic LMC (lmc00 spheroid) as a fine l=0 multipole vs AGAMA"""

    tab = aj.load_tango_tables(VARIANT)
    p = aj.EvolvingMultipole([0.], [aj.prepare_multipole(tab['lmc0_radii'], tab['lmc0_phi'][None],
                                                         tab['lmc0_dphi'][None])])
    d = aj.read_pot_file(os.path.join(aj.tango_dir(VARIANT), 'lmc00.pot'))
    x = random_points(n, 1e-2, 1e4)
    v = summary('lmc00 (initial LMC)', *errors(agama_eval(agama.Potential(**d['params']), x, 0.), jax_eval(p, x, 0.)))
    assert v['force_max'] < 1e-6 and v['phi_max'] < 1e-8


# Tests: time dependence #

def random_times(n, seed=2, ties=True):
    """Times [AGAMA units] uniform over -3.2..0.4, plus exactly at the snapshot times, just after their midpoints, and
    (if ties) exactly at the midpoints, where the nearest snapshot is a tie decided by the last bit of t"""

    rng = np.random.default_rng(seed)
    t = rng.uniform(-3.2, 0.4, n)
    special = np.concatenate([SNAP_TIMES, SNAP_TIMES[:-1] + 0.03125 + 1e-9] + ([SNAP_TIMES[:-1] + 0.03125] if ties else []))
    t[:len(special)] = special
    return t

def check_evolving(keep, linear, n=4000, rmax=300):
    comps = aj.tango_components(VARIANT, linear=linear)
    pots = [comps[k] for k in keep if k in comps] + ([comps['bulge_disk']] if 'bulge' in keep else [])
    from streamsculptor import potential
    pot = potential.Potential_Combine(pots)
    x, t = random_points(n, 0.5, rmax), random_times(n, ties=linear)
    ref = agama_eval(agama_ref(keep, linear), x, t)
    new = jax_eval(pot, x, t)
    # potentials containing the frame acceleration (Phi = -a.x) are compared relative to a typical |Phi|
    scale = np.abs(agama_eval(agama_ref(('halo',), linear), x, t)[0]) if 'acc' in keep else None
    return summary(f'{"+".join(keep)} ({"linear" if linear else "nearest"})', *errors(ref, new, scale))

def test_evolving_halo():
    for linear in (True, False):
        v = check_evolving(('halo',), linear)
        assert v['force_med'] < 1e-5 and v['force_max'] < 1e-3

def test_evolving_lmc():
    """Evolving LMC on its trajectory (tests snapshot blending, the analytic->multipole transition, and the center)"""

    for linear in (True, False):
        v = check_evolving(('lmc',), linear, rmax=600)
        assert v['force_med'] < 1e-5 and v['force_max'] < 1e-3

def test_acceleration(n=3000):
    comps = aj.tango_components(VARIANT)
    x, t = random_points(n, 0.5, 300), random_times(n)
    ref, new = agama_eval(agama_ref(('acc',)), x, t), jax_eval(comps['acc'], x, t)
    ef = np.linalg.norm(new[1] - ref[1], axis=1) / np.linalg.norm(ref[1], axis=1)
    print(f'{"frame acceleration":42s} n={n:6d}  force med {np.median(ef):.1e} max {ef.max():.1e}')
    assert ef.max() < 1e-10

def test_full_model():
    for linear in (True, False):
        v = check_evolving(('halo', 'bulge', 'disk', 'lmc', 'acc'), linear)
        assert v['force_med'] < 1e-4 and v['force_max'] < 1e-2

def test_sgr(n=4000):
    p = aj.sgr_plummer()
    x, t = random_points(n, 0.5, 300), random_times(n)
    v = summary('Sgr Plummer', *errors(agama_eval(agama_sgr(), x, t), jax_eval(p, x, t)))
    assert v['force_max'] < 1e-10 and v['phi_max'] < 1e-10
    t = np.clip(t, -3.2, 0.3)   # within the dense AGAMA table
    ref = agama_eval(agama_reflex(p), x, t)
    new = jax_eval(aj.PlummerReflex(p), x, t)
    ef = np.linalg.norm(new[1] - ref[1], axis=1) / np.linalg.norm(ref[1], axis=1)
    print(f'{"Sgr reflex (vs dense AGAMA table)":42s} n={n:6d}  force med {np.median(ef):.1e} max {ef.max():.1e}')
    assert np.median(ef) < 1e-5


# Tests: orbits #

def jax_orbit(pot, w0, t_end, n_out, jump_ts=None, rtol=1e-11, atol=1e-11):
    """Orbit [kpc, km/s] from w0 (kpc, km/s) at t=0 to t_end [AGAMA units], saved at n_out equal times"""

    ts = np.linspace(0, t_end, n_out) * aj.TIME_UNIT_MYR
    w = jnp.concatenate([jnp.asarray(w0[:3]), jnp.asarray(w0[3:]) * aj.KMS_TO_KPCMYR])
    term = diffrax.ODETerm(lambda t, y, args: jnp.concatenate([y[3:], -pot.gradient(y[:3], t)]))
    ctrl = diffrax.PIDController(rtol=rtol, atol=atol, jump_ts=jump_ts)
    sol = diffrax.diffeqsolve(term, diffrax.Dopri8(), t0=0., t1=ts[-1], dt0=-1., y0=w,
                              saveat=diffrax.SaveAt(ts=jnp.asarray(ts)), stepsize_controller=ctrl, max_steps=10**6)
    y = np.asarray(sol.ys)
    return np.column_stack([y[:,:3], y[:,3:] / aj.KMS_TO_KPCMYR])

def agama_orbit(pot, w0, t_end, n_out):
    t, o = agama.orbit(potential=pot, ic=w0, timestart=0., time=t_end, trajsize=n_out, accuracy=1e-12)
    return o

def gd1_ic():
    """GD-1 progenitor today [kpc, km/s] from the V21 static fit"""

    import gd1_test as gt
    w = gt.progenitor('v21')
    return np.hstack([w[:3], w[3:] / aj.KMS_TO_KPCMYR])

def check_orbits(linear=True, sgr=True, t_end=-2.6, n_out=261):
    """Orbits of the GD-1 progenitor and of halo test particles at 20-100 kpc, AGAMA vs jax.
    Returns max position difference [pc] for each orbit"""

    pot = aj.tango_potential(VARIANT, linear=linear, sgr=sgr, sgr_reflex=sgr)
    keep = ('halo', 'bulge', 'disk', 'lmc', 'acc')
    text = ini_sections(VARIANT, linear, keep)
    if sgr:
        p = aj.sgr_plummer()
        agama_sgr()
        agama_reflex(p)
        text += '\n[Potential sgr]\nfile=' + os.path.join(TMP_DIR, 'sgr.ini') + '\n'
        text += '\n[Potential sgr_reflex]\ntype=UniformAcceleration\nfile=' + os.path.join(TMP_DIR, 'sgr_reflex.txt') + '\n'
    ref = agama.Potential(write_ini(text, f'orbit_{linear}_{sgr}.ini'))

    ics = [('GD-1', gd1_ic())]
    rng = np.random.default_rng(3)
    for r in [20, 50, 100]:
        u = rng.normal(size=3); u /= np.linalg.norm(u)
        v = rng.normal(size=3); v -= u*np.dot(u, v); v /= np.linalg.norm(v)
        ics.append((f'halo r={r}', np.hstack([u*r, v*np.sqrt(-np.dot(u*r, ref.force(u*r)))])))
    jumps = None if linear else jnp.asarray((SNAP_TIMES[:-1] + 0.03125) * aj.TIME_UNIT_MYR)
    out = {}
    for name, w0 in ics:
        o_ref = agama_orbit(ref, w0, t_end, n_out)
        o_new = jax_orbit(pot, w0, t_end, n_out, jump_ts=jumps)
        dx = np.linalg.norm(o_new[:,:3] - o_ref[:,:3], axis=1) * 1e3
        out[name] = dx.max()
        print(f'orbit {name:12s} ({"linear" if linear else "nearest"}, sgr={sgr}): max |dx| = {dx.max():.2f} pc, '
              f'final {dx[-1]:.2f} pc, max r = {np.linalg.norm(o_ref[:,:3], axis=1).max():.0f} kpc')
    return out

def test_orbits():
    for linear in (True, False):
        out = check_orbits(linear=linear, sgr=True)
        assert out['GD-1'] < 10


def report():
    """Run all checks and print the summary"""

    check_snapshots()
    test_lmc_initial()
    test_bulge_disk()
    test_acceleration()
    for keep in [('halo',), ('lmc',), ('halo', 'bulge', 'disk', 'lmc', 'acc')]:
        for linear in (True, False):
            check_evolving(keep, linear, rmax=600 if keep == ('lmc',) else 300)
    test_sgr()
    for linear in (True, False):
        check_orbits(linear=linear, sgr=True)
