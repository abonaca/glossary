import sys
sys.path.insert(0, '.')
import agama_jax as aj, numpy as np, jax, jax.numpy as jnp
tab = aj.load_tango_tables('triax')
s = aj.prepare_multipole(tab['lmc0_radii'], tab['lmc0_phi'][None], tab['lmc0_dphi'][None])
p = aj.EvolvingMultipole([0.], [s])
st = p.snaps.st
which = sys.argv[1]
x = jnp.array([1., 2., 3.])
if which == 'quintic':
    f = jax.jit(lambda lr: aj.quintic_eval(lr, st['logr'][0], st['nr'][0], st['f'][0], st['df'][0], st['d2f'][0]))
    print(which, f(1.0))
elif which == 'radial':
    f = jax.jit(lambda lr: aj._radial_coefs(st, 0, lr)); print(which, f(1.0))
elif which == 'searchsorted':
    f = jax.jit(lambda lr: jnp.searchsorted(st['logr'][0], lr)); print(which, f(1.0))
elif which == 'sph':
    f = jax.jit(lambda q: aj.sph_harm_real(0, q, jnp.sqrt(jnp.sum(q**2)))); print(which, f(x))
elif which == 'snap':
    f = jax.jit(lambda q: p.snaps.snapshot_potential(q, 0)); print(which, f(x))
elif which == 'full':
    f = jax.jit(lambda q: p.potential(q, 0.)); print(which, f(x))
if which == 'pl':
    f = jax.jit(lambda d: aj._power_law_coefs(st['s_in'][0], st['u_in'][0], st['w_in'][0], p.snaps.l_of_c, d)); print(which, f(0.3))
if which == 'pl_out':
    f = jax.jit(lambda d: aj._power_law_coefs(st['s_out'][0], st['u_out'][0], st['w_out'][0], -p.snaps.l_of_c-1, d)); print(which, f(0.3))
if which == 'sum':
    f = jax.jit(lambda q: jnp.sum(aj._radial_coefs(st, 0, 1.0) * aj.sph_harm_real(0, q, jnp.sqrt(jnp.sum(q**2))))); print(which, f(x))
if which == 'where':
    def g(q):
        r = jnp.sqrt(jnp.sum(q**2)); logr = jnp.log1p(r - 1.)
        c_mid = aj._radial_coefs(st, 0, jnp.clip(logr, jnp.log1p(st['rmin'][0]), jnp.log1p(st['rmax'][0])) - 1. - 1.)
        return c_mid
    f = jax.jit(g); print(which, f(x))
import os
if which == 'snap4':
    d = aj.read_pot_file(os.path.join(aj.tango_dir('triax'), 'mw10.pot'))
    p4 = aj.EvolvingMultipole([0.], [aj.prepare_multipole(d['radii'], d['Phi'], d['dPhi'])])
    f = jax.jit(lambda q: p4.snaps.snapshot_potential(q, 0)); print(which, f(x))
def variant(q, mode):
    r = jnp.sqrt(jnp.sum(q**2)); logr = jnp.log1p(r - 1.)
    rmin, rmax = st['rmin'][0], st['rmax'][0]
    c_mid = aj._radial_coefs(st, 0, jnp.clip(logr, jnp.log1p(rmin), jnp.log1p(rmax)) - 1. - 1.)
    c_in = aj._power_law_coefs(st['s_in'][0], st['u_in'][0], st['w_in'][0], p.snaps.l_of_c, logr - jnp.log1p(rmin) - 1.)
    c_out = aj._power_law_coefs(st['s_out'][0], st['u_out'][0], st['w_out'][0], -p.snaps.l_of_c-1, logr - jnp.log1p(rmax) - 1.)
    if mode == 'a': return c_mid + c_in
    if mode == 'b': return c_in + c_out
    if mode == 'c': return jnp.where(r < rmin, c_in, c_mid)
    if mode == 'd': return jnp.where(r > rmax, c_out, c_mid)
if which.startswith('v_'):
    f = jax.jit(lambda q: variant(q, which[2:])); print(which, f(x))
