# Licensed under BSD-3-Clause License - see LICENSE
# Adapted from GC_formation_model/params_example.py
#
# Fixes applied vs. the original:
#  - p2_arr/p3_arr/kappa_arr/seed_list/test_mmr were referenced in the final
#    params dict but never defined anywhere in params_example.py (NameError
#    on load). Traced through the whole codebase: only test_mmr is ever read
#    downstream, and defensively (`if 'test_mmr' in params ...`), so the
#    other four are dead placeholders. Trivial single-element values here.
#  - base_tree/base_halo/resultspath/z_list now resolve relative to this
#    file's own location rather than the original's $-placeholder convention.

import os
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # .../configs
PROJECT_DIR = _THIS_DIR.rsplit(os.sep, 1)[0] if False else os.path.dirname(_THIS_DIR)  # project root
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

# disrupt mode: constant, tidal
disrupt_mode = 'tidal'

# Chen & Gnedin (2024a) values for TNG50
p2 = 18
p3 = 0.5
kappa = 1.5

d_tid = 0.3
disrupt_x = 0.67
disrupt_y = 1.33
pr = 0.5
log_mc = 7.0

seed = 473

allcat_base = 'allcat'
merit_name = 'merits.txt'

mpb_only = False

mmr_slope = 0.3
mmr_pivot = 9.0
mmr_evolution = 1.0
mmr0 = -0.5
max_feh = 0.3

tdep = 0.3

sigma_mg = 0.3
sigma_mc = 0
sigma_gas = 0.3

sm_scat = True
gaussian_process = True

gauss_l = 2
gauss_l_sm = 2

regen_feh = False
seed_feh = 0

log_Mhmin = 8.0
log_Mmin = 4.0

t_lag = 0.01

low_mass = False
log_Mmin_low_mass = 4.0
low_mass_attempt_N = 1

form_nuclear_cluster = True
no_random_at_formation = False

# --- paths, resolved relative to this file's location ---
base_tree = os.path.join(DATA_DIR, 'raw') + os.sep       # merger_tree_<id>.hdf5 lives here
base_halo = os.path.join(DATA_DIR, 'cutouts') + os.sep   # unused by form(), key must still exist
resultspath = os.path.join(PROJECT_DIR, 'output') + os.sep

redshift_snap = np.loadtxt(os.path.join(DATA_DIR, 'raw', 'z_list.txt'), dtype=float)

path_massloss = 'unused'  # confirmed dead: evolve.py:71 has it commented out

mu_sev = 0.55
rmax_form = 3.0

subs = [523889]  # TNG50-1, MW-like halo

color_metallicity = 'CGL18'
UVB_constraint = 'KM22'

exceed_stellar = False
fix_stellar = False

skip = None
verbose = True
full_snap_only = False
max_lag_ratio = 0.5

Om = 0.3089; Ob = 0.0486; h100 = 0.6774  # TNG50

full_snap = [2, 3, 4, 6, 8, 11, 13, 17, 21, 25, 33, 40, 50, 59, 67, 72, 78, 84, 91, 99]

analyse_snap = 99

# fixes for the undefined-name bug (see header note)
p2_arr = [p2]
p3_arr = [p3]
kappa_arr = [kappa]
seed_list = [seed]
test_mmr = False

params = {
    'disrupt_mode': disrupt_mode, 'h100': h100, 'Ob': Ob, 'Om': Om,
    'p2': p2, 'p3': p3, 'kappa': kappa,
    'p2_arr': p2_arr, 'p3_arr': p3_arr, 'kappa_arr': kappa_arr,
    'd_tid': d_tid, 'disrupt_x': disrupt_x, 'disrupt_y': disrupt_y,
    'log_mc': log_mc, 'seed': seed, 'seed_list': seed_list, 'seed_feh': seed_feh,
    'mpb_only': mpb_only, 'test_mmr': test_mmr,
    'mmr_slope': mmr_slope, 'mmr_pivot': mmr_pivot, 'mmr_evolution': mmr_evolution,
    'max_feh': max_feh, 'mmr0': mmr0, 'tdep': tdep,
    'sigma_mg': sigma_mg, 'sigma_mc': sigma_mc, 'sigma_gas': sigma_gas,
    'sm_scat': sm_scat, 'log_Mhmin': log_Mhmin, 'log_Mmin': log_Mmin, 'pr': pr,
    't_lag': t_lag, 'base_tree': base_tree, 'base_halo': base_halo,
    'redshift_snap': redshift_snap, 'path_massloss': path_massloss,
    'mu_sev': mu_sev, 'rmax_form': rmax_form, 'subs': subs, 'full_snap': full_snap,
    'color_metallicity': color_metallicity, 'resultspath': resultspath,
    'allcat_base': allcat_base, 'merit_name': merit_name,
    'low_mass': low_mass, 'log_Mmin_low_mass': log_Mmin_low_mass,
    'form_nuclear_cluster': form_nuclear_cluster, 'low_mass_attempt_N': low_mass_attempt_N,
    'no_random_at_formation': no_random_at_formation, 'skip': skip,
    'gaussian_process': gaussian_process, 'gauss_l': gauss_l, 'gauss_l_sm': gauss_l_sm,
    'regen_feh': regen_feh, 'UVB_constraint': UVB_constraint,
    'exceed_stellar': exceed_stellar, 'fix_stellar': fix_stellar, 'verbose': verbose,
    'analyse_snap': analyse_snap, 'full_snap_only': full_snap_only,
    'max_lag_ratio': max_lag_ratio,
}
