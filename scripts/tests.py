import numpy as np

from astropy.table import Table, hstack
import astropy.units as u

import h5py
from collections import Counter
import scipy.spatial as sp

import os
import requests
import importlib.util
import time

from GC_formation_model import astro_utils
from GC_formation_model.form import form
from GC_formation_model.offset import get_offset
from GC_formation_model.assign import assign
from GC_formation_model.get_tid import calc_eig
from GC_formation_model.evolve import evolve

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
CONFIG_DIR = os.path.join(PROJECT_DIR, "config")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")


# Load GC_formation_model output catalogs as astropy tables

ALLCAT_COLNAMES = ['subfindID_z0', 'logMh_z0', 'logMstar_z0', 'logMh_zform', 'logMstar_zform', 'logM_zform', 'zform', 'feh', 'isMPB', 'subfindID_zform', 'snapnum_zform']
GCID_COLNAMES = ['gcid', 'quality']

def _allcat_path(params):
    return os.path.join(OUTPUT_DIR, params['allcat_base'] + f"_s-{params['seed']:d}_p2-{params['p2']:g}_p3-{params['p3']:g}.txt")

def read_allcat(params):
    """Load an allcat_*.txt output as an astropy Table with named columns."""
    
    return Table.read(_allcat_path(params), format='ascii.no_header', names=ALLCAT_COLNAMES)



def fetch_redshift_snap(sim="TNG50-1", save=True):
    """Fetch the snapshot->redshift table for a given TNG run via the API"""
    api_key = os.environ["TNG_API_KEY"]
    headers = {"api-key": api_key}
    base = f"https://www.tng-project.org/api/{sim}"

    r = requests.get(f"{base}/snapshots/", headers=headers)
    r.raise_for_status()
    snaps = r.json()

    n = len(snaps)
    print("n_snapshots returned:", n)
    assert n == 100, f"expected 100 snapshots for {sim}, got {n}"

    redshift_snap = np.zeros(n)
    for s in snaps:
        redshift_snap[s["number"]] = s["redshift"]

    if save:
        out_path = os.path.join(DATA_DIR, "raw", "z_list.txt")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savetxt(out_path, redshift_snap)
        print("saved to", out_path)

    return redshift_snap


def load_params(run_name="params_run1.py"):
    """Load a run config file and return its params dictionary"""
    
    params_path = os.path.join(CONFIG_DIR, run_name)
    
    spec = importlib.util.spec_from_file_location("params_module", params_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    params = module.params
    
    params["allcat_name"] = params["allcat_base"] + f"_s-{params['seed']:d}_p2-{params['p2']:g}_p3-{params['p3']:g}.txt"
    params["cosmo"] = astro_utils.cosmo(h=params["h100"], omega_baryon=params["Ob"], omega_matter=params["Om"])
    
    return params


def run_formation(params=None, run_name="params_run1.py", print_params=False):
    """Run the GC formation stage using the Chen & Gnedin module"""

    if params is None:
        params = load_params(run_name)
        if print_params:
            print(params)

    os.makedirs(params["resultspath"], exist_ok=True)
    
    form(params)


def analyze_gcs(run_name="params_run1.py"):
    """Find how many of the clusters formed ex situ, and in how many distinct cluster formation events"""
    
    params = load_params(run_name)
    allcat = read_allcat(params)
    
    ex_situ = allcat[allcat['isMPB'] == 0]
    events = set(zip(ex_situ['subfindID_zform'], ex_situ['snapnum_zform']))

    in_situ = allcat[allcat['isMPB'] == 1]
    insitu_events = set(zip(in_situ['subfindID_zform'], in_situ['snapnum_zform']))

    print("total clusters:", len(allcat))
    print("ex-situ clusters:", len(ex_situ))
    print("distinct ex-situ formation events (snapshot, subfind_id):", len(events))
    print("distinct in-situ formation events (snapshot, subfind_id):", len(insitu_events))


def count_branches(run_name="params_run1.py"):
    """Map each ex-situ formation event to its branch (MainLeafProgenitorID) and report the number of distinct physical branches involved"""

    params = load_params(run_name)
    allcat = read_allcat(params)
    ex_situ = allcat[allcat['isMPB'] == 0]

    tree_path = f"{params['base_tree']}merger_tree_{params['subs'][0]}.hdf5"
    with h5py.File(tree_path, "r") as f:
        tree_snap    = f["SnapNum"][:]
        tree_subfind = f["SubfindID"][:]
        tree_leaf    = f["MainLeafProgenitorID"][:]

    idx_lookup = {(int(s), int(h)): i for i, s, h in zip(range(len(tree_snap)), tree_snap, tree_subfind)}

    branches = set()
    missing = 0
    for s, h in zip(ex_situ['snapnum_zform'], ex_situ['subfindID_zform']):
        idx = idx_lookup.get((int(s), int(h)))
        if idx is None:
            missing += 1
            continue
        branches.add(int(tree_leaf[idx]))

    print("ex-situ events:", len(ex_situ))
    print("events not found in tree:", missing)
    print("distinct physical branches:", len(branches))


def get_with_retry(url, headers, max_retries=8, wait=1, timeout=120):
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            # wait = backoff * (2 ** attempt)
            print(f"    retry {attempt+1}/{max_retries} after {wait}s ({e})")
            time.sleep(wait)


def download_cutouts(events, hid_root=523889, sim="TNG50-1", save_path=None):
    """
    events: iterable of (snap, subfind_id) pairs needing a cutout for assign()
    Writes/updates halo_<hid_root>.hdf5 in the on-disk layout loader.load_halo() expects.
    Skips any (snap, subfind_id) already present, so this is safe to re-run/resume.
    """
    
    api_key = os.environ["TNG_API_KEY"]
    headers = {"api-key": api_key}
    base = f"https://www.tng-project.org/api/{sim}"

    filename = os.path.join(save_path, f"halo_{hid_root}.hdf5")
    os.makedirs(save_path, exist_ok=True)

    # needs to be without spaces as it's used to construct an http request
    field_map = {
        "dm": "Coordinates,Velocities,ParticleIDs",
        "stars": "Coordinates,Velocities,ParticleIDs,GFM_StellarFormationTime,Masses",
    }

    with h5py.File(filename, "a") as f:
        for i, (snap, subfind_id) in enumerate(sorted(set(events))):
            grp_name = f"snap_{snap}_halo_{subfind_id}"
            if grp_name in f and f[grp_name].attrs.get("completed", False):
                continue

            grp = f.require_group(grp_name)
            for parttype, field_str in field_map.items():
                r = get_with_retry(
                    f"{base}/snapshots/{snap}/subhalos/{subfind_id}/cutout.hdf5?{parttype}={field_str}",
                    headers=headers)

                tmp_path = "/tmp/_cutout_tmp.hdf5"
                with open(tmp_path, "wb") as tf:
                    tf.write(r.content)

                pt_grp = grp.require_group(parttype)
                with h5py.File(tmp_path, "r") as cf:
                    key = {"dm": "PartType1", "stars": "PartType4"}[parttype]
                    if key not in cf:
                        pt_grp.attrs["count"] = 0
                        # add empty entries even for missing fields
                        for field in field_str.split(","):
                            shape = (0, 3) if field in ("Coordinates", "Velocities") else (0,)
                            if field not in pt_grp:
                                pt_grp.create_dataset(field, shape=shape, dtype='f8')
                        continue
                    pt_grp.attrs["count"] = cf[key]["Coordinates"].shape[0]
                    for field in cf[key].keys():
                        if field in pt_grp:
                            del pt_grp[field]
                        pt_grp.create_dataset(field, data=cf[key][field][:])
                os.remove(tmp_path)

            grp.attrs["completed"] = True
            if (i + 1) % 20 == 0:
                print(f"{i+1}/{len(events)} cutouts done")

        f.attrs["completed"] = True
    print("all cutouts done:", filename)


def test_download_cutouts(i=0, insitu=True, run_name="params_run1.py"):
    """Test cutout download for an individual GC
    i - index in the list
    insitu - bool, True for in-situ clusters"""
    
    params = load_params(run_name)
    allcat = read_allcat(params)
    sel = allcat[allcat['isMPB'] == int(insitu)]

    # grab one event
    snap, subfind_id = int(sel['snapnum_zform'][i]), int(sel['subfindID_zform'][i])
    print("testing on:", snap, subfind_id)
    
    t0 = time.time()
    download_cutouts(
        events=[(snap, subfind_id)],
        hid_root=523889,
        save_path=os.path.join(DATA_DIR, "cutouts"),
    )
    print("took", time.time() - t0, "s")

def run_download_cutouts(run_name="params_run1.py"):
    """Download all cutouts"""
    
    params = load_params(run_name)
    allcat = read_allcat(params)
    
    events = set(zip(allcat['snapnum_zform'], allcat['subfindID_zform']))
    print(len(events), "rows ->", len(set(events)), "unique events")

    download_cutouts(events, hid_root=523889, save_path=os.path.join(DATA_DIR, "cutouts"))

def count_empty():
    """Test how many cutouts have 0 dm vs star particles"""
    
    path = os.path.join(DATA_DIR, "cutouts", "halo_523889.hdf5")
    with h5py.File(path, "r") as f:
        dm_zero = sum(1 for g in f if g.startswith("snap_") and "dm" in f[g]
                      and f[g]["dm"].attrs.get("count") == 0)
        stars_zero = sum(1 for g in f if g.startswith("snap_") and "stars" in f[g]
                          and f[g]["stars"].attrs.get("count") == 0)
    print("groups with dm count=0:", dm_zero)
    print("groups with stars count=0:", stars_zero)

def repair_empty_groups(hid_root=523889):
    """Add empty placeholder datasets for any parttype group that came back
    with zero particles from the API (and so has no datasets at all),
    so loader.load_halo() can read it without KeyError."""

    path = os.path.join(DATA_DIR, "cutouts", f"halo_{hid_root}.hdf5")
    field_shapes = {
        "Coordinates": (0, 3), "Velocities": (0, 3),
        "ParticleIDs": (0,), "GFM_StellarFormationTime": (0,), "Masses": (0,),
    }
    field_map = {
        "dm":    ["Coordinates", "Velocities", "ParticleIDs"],
        "stars": ["Coordinates", "Velocities", "ParticleIDs", "GFM_StellarFormationTime", "Masses"],
    }

    n_fixed = 0
    with h5py.File(path, "a") as f:
        for grp_name in f.keys():
            if not grp_name.startswith("snap_"):
                continue
            grp = f[grp_name]
            for parttype, fields in field_map.items():
                if parttype not in grp:
                    continue
                pt_grp = grp[parttype]
                if pt_grp.attrs.get("count", None) == 0:
                    for field in fields:
                        if field not in pt_grp:
                            pt_grp.create_dataset(field, shape=field_shapes[field], dtype='f8')
                            n_fixed += 1
    print(f"added {n_fixed} placeholder datasets")



def run_offset(run_name="params_run1.py"):
    """Find unique GC formation events and store in _offset_root (main branch) and _offset (accreted branches) files"""
    params = load_params(run_name)
    get_offset(params)

def test_offset(run_name="params_run1.py"):
    """Test whether GC formation events have been identified properly"""
    
    params = load_params(run_name)
    fin_base = params['resultspath'] + params['allcat_name'][:-4]

    root = np.loadtxt(fin_base + '_offset_root.txt', ndmin=2, dtype=int)
    offset = np.loadtxt(fin_base + '_offset.txt', ndmin=2, dtype=int)

    print("offset_root shape:", root.shape)   # should be (1, 5) — one row, one root halo
    print(root)

    print("offset shape:", offset.shape)      # should be (n_events, 4)
    print("n_events:", len(offset))

    allcat = read_allcat(params)
    print("total clusters in allcat:", len(allcat))

    # the row-range covered by the single root entry should span the WHOLE allcat file
    _, idx_root_beg, idx_root_end, idx_beg_in_off, idx_end_in_off = root[0]
    print("root covers allcat rows:", idx_root_beg, "to", idx_root_end,
          "-- should equal 0 to", len(allcat))

    # and the event index range should span the whole offset table
    print("root covers offset rows:", idx_beg_in_off, "to", idx_end_in_off,
          "-- should equal 0 to", len(offset))




def run_assign(run_name="params_run1.py", ret=False):
    params = load_params(run_name)
    assign(params)
    if ret: return params

def read_gcid(params):
    """Read GC ID file as an Astropy Table"""
    path = _allcat_path(params)[:-4] + "_gcid.txt"
    
    return Table.read(path, format='ascii.no_header', names=GCID_COLNAMES)

def read_gc_catalog(run_name="params_run1.py"):
    """Join allcat (cluster properties) with _gcid.txt (tracer particle ID + quality) by row order"""
    
    params = load_params(run_name)
    allcat = read_allcat(params)
    gcid = read_gcid(params)
    
    assert len(allcat) == len(gcid), f"row count mismatch: allcat={len(allcat)}, gcid={len(gcid)}"
    
    return hstack([allcat, gcid])

def test_assign(run_name="params_run1.py", verbose=False):
    """Check properties of particles assigned to GCs"""
    
    t = read_gc_catalog(run_name)
    # print("quality distribution:", np.bincount(t['quality']))
    
    # reconstruct num_gc per event and attach it per-cluster
    event_sizes = Counter(zip(t['snapnum_zform'], t['subfindID_zform']))
    t['num_gc'] = [event_sizes[(s, h)] for s, h in zip(t['snapnum_zform'], t['subfindID_zform'])]

    for q in (0, 1, 2):
        print(f"quality={q}")
        z = t['zform'][t['quality'] == q]
        print(f"n={len(z)}, median zform={np.median(z):.2f}, zform range=[{z.min():.2f}, {z.max():.2f}]")
        
        n = t['num_gc'][t['quality'] == q]
        print(f"median event size={np.median(n):.0f}, mean={np.mean(n):.1f}\n")
        




def n_subhalo_per_snap(run_name="params_run1.py"):
    """Count the number of massive subhalos per snapshot, for which cutouts will be needed to calculate the tidal tensor"""
    
    params = load_params(run_name)
    tree_path = f"{params['base_tree']}merger_tree_{params['subs'][0]}.hdf5"
    with h5py.File(tree_path, "r") as f:
        snapnum = f["SnapNum"][:]
        mass = f["SubhaloMass"][:]

    threshold = 10**(params['log_Mhmin'] - 10) * params['h100']
    for snap in params['full_snap']:
        n = np.sum((snapnum == snap) & (mass > threshold))
        print(f"snap {snap}: {n} subhalos above threshold")


def select_cluster(flag_insitu=False, ind_select=0, quality=2, run_name="params_run1.py", verbose=False):
    """Return a gc catalog table row for a single globular cluster, with specified origin (insitu=1, exsitu=0), order, and quality (2 star particle, narrow time, 1 star particle, generous time, 0 dm particle)"""
    
    cat = read_gc_catalog(run_name)
    candidates = cat[(cat['isMPB'] == int(flag_insitu)) & (cat['quality'] == quality)]
    
    row = candidates[ind_select]
    if verbose: print(dict(zip(row.colnames, row)))
    
    return row

def walk_branch_full_snap_nodes(snap_form, subfind_id_form, params):
    """DescendantID walk from formation to z=0 (or until the branch drops
    out of the tree), keeping only nodes at a full_snap epoch."""
    
    tree_path = f"{params['base_tree']}merger_tree_{params['subs'][0]}.hdf5"
    with h5py.File(tree_path, "r") as f:
        subhalo_id = f["SubhaloID"][:]
        desc_id    = f["DescendantID"][:]
        snapnum    = f["SnapNum"][:]
        subfind_id = f["SubfindID"][:]

    id_to_idx = {sid: i for i, sid in enumerate(subhalo_id)}
    start = np.where((snapnum == snap_form) & (subfind_id == subfind_id_form))[0]
    if len(start) == 0:
        raise ValueError("formation node not found in tree")
    idx = start[0]

    full_snap_set = set(params['full_snap'])
    nodes, steps = [], 0
    while True:
        s, h = int(snapnum[idx]), int(subfind_id[idx])
        if s in full_snap_set:
            nodes.append((s, h))
        d = desc_id[idx]
        if d == -1 or d not in id_to_idx:
            break
        idx = id_to_idx[d]
        steps += 1
        if steps > 200:
            print("WARNING: exceeded 200 steps, stopping")
            break
    return sorted(set(nodes))

def accretion_landscape(run_name="params_run1.py"):
    """For every ex-situ cluster: when its host merged onto the MW branch,
    how many mergers along the way, and how long it spent outside the MW."""
    from GC_formation_model.evolve import _load_tree, _walk_hosts

    params = load_params(run_name)
    cat = read_gc_catalog(run_name)
    ex = cat[cat['isMPB'] == 0]
    tr = _load_tree(params)
    z_list = params['redshift_snap']
    t0 = params['cosmo'].cosmicTime(0, units='Gyr')

    cache = {}
    rows = []
    for r in ex:
        key = (int(r['snapnum_zform']), int(r['subfindID_zform']))
        if key not in cache:
            cache[key] = _walk_hosts(tr, *key)
        hosts, mergers = cache[key]
        mw = [m for m in mergers if m[2]]

        t_form = t0 - params['cosmo'].cosmicTime(z_list[key[0]], units='Gyr')
        t_acc = (t0 - params['cosmo'].cosmicTime(z_list[mw[0][0]], units='Gyr')) if mw else np.nan
        rows.append((int(r['gcid']), key[0], mw[0][0] if mw else -1,
                     len(mergers), t_form, t_acc, t_form - t_acc))

    t = Table(rows=rows, names=['gcid', 'snap_form', 'snap_mw_merge', 'n_mergers',
                                't_form_lb', 't_acc_lb', 'window_Gyr'])
    print('ex-situ clusters:', len(t))
    print('n_mergers distribution:', np.bincount(t['n_mergers']))
    w = t['window_Gyr']
    print('pre-accretion window (Gyr): median %.2f, 90th pct %.2f, max %.2f'
          % (np.median(w), np.percentile(w, 90), np.nanmax(w)))
    for thr in (0.5, 1, 2, 4):
        print(f'  fraction with window > {thr} Gyr: {np.mean(w > thr):.3f}')

    t.pprint()

def count_tid_nodes(run_name="params_run1.py", ex_situ_only=False):
    """Distinct (snap, subfind) nodes at full_snap epochs across all clusters'
    host branches -- i.e. the cutouts the scoped get_tid would actually need."""
    params = load_params(run_name)
    cat = read_gc_catalog(run_name)
    if ex_situ_only:
        cat = cat[cat['isMPB'] == 0]

    nodes, cache = set(), {}
    for key in set(zip(cat['snapnum_zform'], cat['subfindID_zform'])):
        key = (int(key[0]), int(key[1]))
        if key not in cache:
            cache[key] = walk_branch_full_snap_nodes(key[0], key[1], params)
        nodes.update(cache[key])
    print(f'distinct (snap, subfind) nodes: {len(nodes)}')
    print(f'-> {3*len(nodes)} cutout requests at ~27 s each '
          f'= {3*len(nodes)*27/3600:.1f} h')





###################
# Tidal evolution #
###################

def download_tid_cutouts(events, hid_root=523889, sim="TNG50-1", save_path=None):
    """dm+stars+gas, with Potential -- a separate file from assign()'s cutouts,
    since the field/parttype set is different."""
    
    api_key = os.environ["TNG_API_KEY"]
    headers = {"api-key": api_key}
    base = f"https://www.tng-project.org/api/{sim}"
    filename = os.path.join(save_path, f"halo_{hid_root}_tid.hdf5")
    os.makedirs(save_path, exist_ok=True)

    field_map = {
        "dm":    "Coordinates,Velocities,ParticleIDs,Potential",
        "stars": "Coordinates,Velocities,ParticleIDs,Potential",
        "gas":   "Coordinates,Velocities,ParticleIDs,Potential",
    }
    field_shapes = {"Coordinates": (0, 3), "Velocities": (0, 3),
                     "ParticleIDs": (0,), "Potential": (0,)}
    key_map = {"dm": "PartType1", "stars": "PartType4", "gas": "PartType0"}

    with h5py.File(filename, "a") as f:
        for i, (snap, subfind_id) in enumerate(sorted(set(events))):
            grp_name = f"snap_{snap}_halo_{subfind_id}"
            if grp_name in f and f[grp_name].attrs.get("completed", False):
                continue
            grp = f.require_group(grp_name)
            for parttype, field_str in field_map.items():
                r = get_with_retry(
                    f"{base}/snapshots/{snap}/subhalos/{subfind_id}/cutout.hdf5?{parttype}={field_str}",
                    headers=headers)
                tmp_path = "/tmp/_cutout_tid_tmp.hdf5"
                with open(tmp_path, "wb") as tf:
                    tf.write(r.content)
                pt_grp = grp.require_group(parttype)
                with h5py.File(tmp_path, "r") as cf:
                    key = key_map[parttype]
                    if key not in cf:
                        pt_grp.attrs["count"] = 0
                        for field in field_str.split(","):
                            if field not in pt_grp:
                                pt_grp.create_dataset(field, shape=field_shapes[field], dtype='f8')
                        continue
                    pt_grp.attrs["count"] = cf[key]["Coordinates"].shape[0]
                    for field in cf[key].keys():
                        if field in pt_grp:
                            del pt_grp[field]
                        pt_grp.create_dataset(field, data=cf[key][field][:])
                os.remove(tmp_path)
            grp.attrs["completed"] = True
            print(f"{i+1}/{len(events)} tid cutouts done")
        f.attrs["completed"] = True
    print("all tid cutouts done:", filename)


def compute_single_cluster_tid(row, branch_nodes, params, hid_root=523889):
    """Tag/eig arrays for ONE cluster, using the KNOWN host at each epoch
    (branch_nodes) instead of get_tid_unit's brute-force search."""

    d_tid = params['d_tid'] * params['h100']
    z_list = params['redshift_snap']
    full_snap = params['full_snap']
    gcid = int(row['gcid'])

    path = os.path.join(DATA_DIR, "cutouts", f"halo_{hid_root}_tid.hdf5")
    node_lookup = dict(branch_nodes)

    tag = np.zeros(len(full_snap), dtype=int)
    eig1 = np.zeros(len(full_snap))
    eig2 = np.zeros(len(full_snap))
    eig3 = np.zeros(len(full_snap))

    with h5py.File(path, "r") as f:
        for j, snap in enumerate(full_snap):
            if snap not in node_lookup:
                continue
            grp_name = f"snap_{snap}_halo_{node_lookup[snap]}"
            if grp_name not in f:
                print(f"WARNING: missing cutout for {grp_name}")
                continue

            scale_a = 1 / (1 + z_list[snap])
            grp = f[grp_name]
            pos_l, pid_l, pot_l = [], [], []
            for pt in ("dm", "stars", "gas"):
                if pt not in grp or grp[pt].attrs.get("count", 0) == 0:
                    continue
                pos_l.append(grp[pt]["Coordinates"][:] * scale_a)
                pid_l.append(grp[pt]["ParticleIDs"][:].astype(int))
                pot_l.append(grp[pt]["Potential"][:] / scale_a)
            if not pos_l:
                continue
            pos, pid, pot = np.concatenate(pos_l), np.concatenate(pid_l), np.concatenate(pot_l)

            idx_gc = np.where(pid == gcid)[0]
            if len(idx_gc) == 0:
                print(f"  gcid not found at snap {snap} (stripped away / lost?)")
                continue
            # else:
            #     idx_gc = idx_gc[0]
            # print(idx_gc)

            kdtree = sp.KDTree(pos)
            # print(kdtree, pos[idx_gc], pot[idx_gc], pos, pot, d_tid)
            # print(np.shape(pos[idx_gc]), np.shape(pot[idx_gc]))
            eig = calc_eig(kdtree, pos[idx_gc], pot[idx_gc], pos, pot, d_tid)
            eig_sorted = np.sort(eig, axis=1)
            tag[j] = 1
            eig1[j], eig2[j], eig3[j] = eig_sorted[0, 2], eig_sorted[0, 1], eig_sorted[0, 0]

    return tag, eig1, eig2, eig3

def write_single_row(row, colnames, path, int_cols=()):
    """Write a single astropy Row (or 1-row Table) as a whitespace-separated
    line, matching allcat's on-disk convention -- avoids astropy's ascii
    writer registry for what's really just one line of numbers."""
    fmt = " ".join("%d" if c in int_cols else "%.6f" for c in colnames)
    values = [row[c] for c in colnames]
    with open(path, "w") as f:
        f.write((fmt % tuple(values)) + "\n")

def read_mass_history(params, gcid_select=None):
    """Read the mass-loss history HDF5 file, with each cluster having its own group"""
    
    fin_name = params['resultspath'] + params['allcat_name']
    path = fin_name[:-4] + '_k-%g_mhistory_snap%d.hdf5' % (params['kappa'], params.get('analyse_snap', 99))

    if gcid_select is not None:
        return Table.read(path, path=str(gcid_select), format='hdf5')

    with h5py.File(path, 'r') as f:
        keys = [k for k in f.keys() if not k.endswith('__table_column_meta__')]
    
    return {int(k): Table.read(path, path=k, format='hdf5') for k in keys}
    # return Table.read('../output/mhistory.fits')

def evolve_single_cluster(flag_insitu=False, ind_select=0, quality=2, run_name="params_run1.py", hid_root=523889):
    
    params = load_params(run_name)
    row = select_cluster(flag_insitu=flag_insitu, ind_select=ind_select, quality=quality, run_name=run_name, verbose=True)

    branch_nodes = walk_branch_full_snap_nodes(int(row['snapnum_zform']), int(row['subfindID_zform']), params)
    print("full_snap nodes along this branch:", branch_nodes)

    download_tid_cutouts(branch_nodes, hid_root=hid_root, save_path=os.path.join(DATA_DIR, "cutouts"))

    tag, eig1, eig2, eig3 = compute_single_cluster_tid(row, branch_nodes, params, hid_root)
    print("tag:", tag, "\neig1:", eig1)

    orig_path = _allcat_path(params)
    new_base = orig_path[:-4] + "_single"
    
    write_single_row(row, ALLCAT_COLNAMES, new_base + ".txt", int_cols={'subfindID_z0', 'isMPB', 'subfindID_zform', 'snapnum_zform'})

    params_test = dict(params)
    params_test['allcat_name'] = os.path.basename(new_base + ".txt")
    get_offset(params_test)
    
    write_single_row(row, GCID_COLNAMES, new_base + "_gcid.txt", int_cols={'gcid', 'quality'})

    np.savetxt(new_base + "_tidtag.txt", tag.reshape(1, -1), fmt='%d')
    np.savetxt(new_base + "_tideig1.txt", eig1.reshape(1, -1), fmt='%.3e')
    np.savetxt(new_base + "_tideig2.txt", eig2.reshape(1, -1), fmt='%.3e')
    np.savetxt(new_base + "_tideig3.txt", eig3.reshape(1, -1), fmt='%.3e')

    m_out, t_disrupt, m_form, t_form, m_history, t_history = evolve(params_test, track_history=True, return_t_disrupt=True)

    t = read_mass_history(params_test, gcid_select=int(row['gcid']))
    t.pprint(max_lines=-1)
    print(t.meta)
    
    # print(row)
    # print(m_out, t_disrupt, m_form, t_form, m_history, t_history)



#####################
# Orbit integration #
#####################

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
from scipy.optimize import root

G   = 4.498502e-6      # kpc^3 / (Msun Gyr^2)
KMS = 1.0227121        # km/s -> kpc/Gyr
BOX = 35000.0          # ckpc/h


def _nfw_M(r, rs, vmax):
    """Enclosed NFW mass fixed by (rmax, Vmax); rs = rmax/2.1626."""
    x = np.clip(r / rs, 1e-8, None)
    return (rs * vmax**2 / (0.2162 * G)) * (np.log1p(x) - x / (1.0 + x))


def _hern_M(r, a, M):
    return M * r**2 / (r + a)**2


def host_splines(node_idx, params, tree):
    """Time splines of potential parameters and centre for one branch.
    node_idx: list of (snap, tree_index) along the branch."""
    h, z = params['h100'], params['redshift_snap']
    t, rs, vmax, astar, mstar, pos = [], [], [], [], [], []
    for snap, i in node_idx:
        a = 1.0 / (1.0 + z[snap])
        t.append(params['cosmo'].cosmicTime(z[snap], units='Gyr'))
        rs.append(max(tree['vmaxrad'][i] * a / h, 1e-3) / 2.1626)
        vmax.append(tree['vmax'][i] * KMS)
        astar.append(max(tree['hmr_star'][i] * a / h, 1e-3) / (1 + np.sqrt(2)))
        mstar.append(tree['mass_star'][i] * 1e10 / h)
        pos.append(tree['pos'][i])                      # ckpc/h, unwrapped later
    t = np.array(t)
    return dict(t0=t[0], t1=t[-1],
                rs=CubicSpline(t, rs), vmax=CubicSpline(t, vmax),
                astar=CubicSpline(t, astar), mstar=CubicSpline(t, mstar),
                pos_raw=np.array(pos), t_arr=t)


def accel(t, x, hosts):
    """Acceleration at MW-centric physical position x (kpc), t in Gyr."""
    acc = np.zeros(3)
    for H in hosts:
        if not (H['t0'] <= t <= H['t1']):
            continue
        dx = x - H['centre'](t)
        r = max(float(np.linalg.norm(dx)), 1e-6)
        M = _nfw_M(r, float(H['rs'](t)), float(H['vmax'](t))) \
            + _hern_M(r, float(H['astar'](t)), float(H['mstar'](t)))
        acc -= G * M / r**3 * dx
    return acc


def _rhs(t, y, hosts, c0, c1, ti):
    return np.concatenate([y[3:], accel(t, y[:3], hosts) + c0 + c1 * (t - ti)])


def integrate(y0, ti, tf, hosts, c0, c1, rtol=1e-10):
    s = solve_ivp(_rhs, (ti, tf), y0, args=(hosts, c0, c1, ti),
                  method='DOP853', rtol=rtol, atol=1e-12)
    return s.y[:, -1]


def shoot(y_i, y_f, ti, tf, hosts):
    """Solve for (c0, c1) so the integration lands exactly on y_f."""
    def resid(p):
        return integrate(y_i, ti, tf, hosts, p[:3], p[3:]) - y_f
    sol = root(resid, np.zeros(6), method='hybr', tol=1e-10)
    c0, c1 = sol.x[:3], sol.x[3:]
    tm = 0.5 * (ti + tf)
    ym = integrate(y_i, ti, tm, hosts, c0, c1)
    a_model = np.linalg.norm(accel(tm, ym[:3], hosts))
    a_corr = np.linalg.norm(c0 + c1 * (tm - ti))
    return c0, c1, sol.success, a_corr / max(a_model, 1e-30)


def orbit_radii(gcid, nodes, params, hid_root=523889, box=35000.0):
    """Tracer distance from its current host and from the MW main branch."""
    
    path = f"{params['base_tree']}merger_tree_{params['subs'][0]}.hdf5"
    with h5py.File(path, 'r') as f:
        spos, svel = f['SubhaloPos'][:], f['SubhaloVel'][:]
        sid, fprog = f['SubhaloID'][:], f['FirstProgenitorID'][:]
        snapnum = f['SnapNum'][:]
    idx_map = {int(s): i for i, s in enumerate(sid)}

    mw, i = {}, 0                                  # MW main branch by snapshot
    while True:
        mw[int(snapnum[i])] = i
        fp = int(fprog[i])
        if fp == -1 or fp not in idx_map:
            break
        i = idx_map[fp]

    h, z = params['h100'], params['redshift_snap']
    t0 = params['cosmo'].cosmicTime(0, units='Gyr')
    cut = os.path.join(DATA_DIR, "cutouts", f"halo_{hid_root}_orbit.hdf5")

    def sep(p, q, a):
        d = p - q
        d -= box * np.round(d / box)
        return d * a / h                            # physical kpc

    rows = []
    with h5py.File(cut, 'r') as f:
        for snap, subfind_id, tidx in nodes:
            g = f"snap_{snap}_halo_{subfind_id}"
            if g not in f or f[g].attrs.get('count', 0) == 0:
                continue
            j = np.where(f[g]['ParticleIDs'][:] == gcid)[0]
            if len(j) == 0:
                print(f'  tracer not in host at snap {snap}')
                continue
            a = 1.0 / (1.0 + z[snap])
            xp = f[g]['Coordinates'][j[0]]
            vp = f[g]['Velocities'][j[0]] * np.sqrt(a)

            d_host = sep(xp, spos[tidx], a)
            dv_host = vp - svel[tidx]
            if snap in mw:
                mi = mw[snap]
                r_mw = float(np.linalg.norm(sep(xp, spos[mi], a)))
                r_sat_mw = float(np.linalg.norm(sep(spos[tidx], spos[mi], a)))
            else:
                r_mw = r_sat_mw = np.nan

            rows.append((snap,
                         t0 - params['cosmo'].cosmicTime(z[snap], units='Gyr'),
                         subfind_id,
                         float(np.linalg.norm(d_host)), r_mw, r_sat_mw,
                         float(np.linalg.norm(dv_host))))

    t = Table(rows=rows, names=['snap', 't_lookback', 'host_subfind',
                                'r_host', 'r_MW', 'r_sat_MW', 'v_host'])
    t['r_host'].unit = t['r_MW'].unit = t['r_sat_MW'].unit = u.kpc
    t['v_host'].unit = u.km / u.s
    t['t_lookback'].unit = u.Gyr
    return t

def walk_branch_all_snaps(snap_form, subfind_form, params):
    """Every snapshot along the host branch, formation -> z=0, with the tree
    index so SubhaloPos/SubhaloVel can be looked up directly."""
    tree_path = f"{params['base_tree']}merger_tree_{params['subs'][0]}.hdf5"
    with h5py.File(tree_path, "r") as f:
        sid, desc = f["SubhaloID"][:], f["DescendantID"][:]
        snapnum, subfind = f["SnapNum"][:], f["SubfindID"][:]
    idx_map = {int(s): i for i, s in enumerate(sid)}

    start = np.where((snapnum == snap_form) & (subfind == subfind_form))[0]
    if len(start) == 0:
        raise ValueError("formation node not found in tree")
    i, out = start[0], []
    while True:
        out.append((int(snapnum[i]), int(subfind[i]), int(i)))
        d = int(desc[i])
        if d == -1 or d not in idx_map:
            break
        i = idx_map[d]
    return out

def download_orbit_cutouts(nodes, hid_root=523889, parttype="stars", sim="TNG50-1", save_path=None):
    """Position/velocity only, single parttype -- for tracer orbit extraction.
    nodes: list of (snap, subfind_id, tree_idx) from walk_branch_all_snaps."""
    headers = {"api-key": os.environ["TNG_API_KEY"]}
    base = f"https://www.tng-project.org/api/{sim}"
    filename = os.path.join(save_path, f"halo_{hid_root}_orbit.hdf5")
    os.makedirs(save_path, exist_ok=True)
    key = {"dm": "PartType1", "stars": "PartType4"}[parttype]
    fields = "Coordinates,Velocities,ParticleIDs"
    field_shapes = {"Coordinates": (0, 3), "Velocities": (0, 3), "ParticleIDs": (0,)}

    targets = sorted({(s, h) for s, h, _ in nodes})
    with h5py.File(filename, "a") as f:
        for n, (snap, subfind_id) in enumerate(targets):
            grp_name = f"snap_{snap}_halo_{subfind_id}"
            if grp_name in f and f[grp_name].attrs.get("completed", False):
                continue
            r = get_with_retry(
                f"{base}/snapshots/{snap}/subhalos/{subfind_id}/cutout.hdf5?{parttype}={fields}",
                headers=headers)
            tmp = "/tmp/_orbit_tmp.hdf5"
            with open(tmp, "wb") as tf:
                tf.write(r.content)
            grp = f.require_group(grp_name)
            with h5py.File(tmp, "r") as cf:
                if key not in cf:
                    grp.attrs["count"] = 0
                    for fld in fields.split(","):
                        if fld not in grp:
                            grp.create_dataset(fld, shape=field_shapes[fld], dtype='f8')
                else:
                    grp.attrs["count"] = cf[key]["Coordinates"].shape[0]
                    for fld in fields.split(","):
                        if fld in grp:
                            del grp[fld]
                        grp.create_dataset(fld, data=cf[key][fld][:])
            os.remove(tmp)
            grp.attrs["completed"] = True
            if (n + 1) % 10 == 0:
                print(f"{n+1}/{len(targets)} orbit cutouts done")
    print("orbit cutouts done:", filename)

import matplotlib.pyplot as plt

def cluster_radial_distance(flag_insitu=False, ind_select=0, quality=2, run_name="params_run1.py", hid_root=523889):
    """"""
    params = load_params(run_name=run_name)
    params['allcat_name'] = os.path.basename(_allcat_path(params)[:-4] + "_single.txt")
    
    row = select_cluster(flag_insitu=flag_insitu, ind_select=ind_select, quality=quality)
    
    nodes = walk_branch_all_snaps(int(row['snapnum_zform']), int(row['subfindID_zform']), params)
    download_orbit_cutouts(nodes, save_path=os.path.join(DATA_DIR, "cutouts"))
    
    t = orbit_radii(int(row['gcid']), nodes, params)
    # t.pprint(max_lines=-1)
    
    mt = read_mass_history(params, gcid_select=int(row['gcid']))
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(10,10), sharex=True)
    
    if mt.meta['disrupted']:
        print('dissolved at', mt['t_lookback'][-1], 'lookback')
        for i in range(2):
            plt.sca(ax[i])
            plt.axvline(mt['t_lookback'][-1], lw=0.5, ls=':', color='tab:blue')
    
    plt.sca(ax[0])
    plt.plot(t['t_lookback'], t['r_host'], 'ko')
    
    plt.gca().set_yscale('log')
    plt.ylabel('$r_{host}$ [kpc]')
    
    plt.sca(ax[1])
    plt.plot(t['t_lookback'], t['r_MW'], 'ko', label='GC')
    plt.plot(t['t_lookback'], t['r_sat_MW'], 'ro', label='host')
    
    plt.legend(loc=1, ncols=2, fontsize='small', handlelength=0.5)
    plt.gca().invert_xaxis()
    plt.gca().set_yscale('log')
    plt.xlabel('Lookback time [Gyr]')
    
    plt.ylabel('$r_{Milky Way}$ [kpc]')
    
    plt.tight_layout()
    plt.savefig('../plots/gc_distance_host_mw.png')
    
