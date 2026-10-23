"""Round-10 Sgr start offsets: a timing scan around round-9 run 2 (lambda=3e-3), Sgr's t=-3 state moved along its own
orbit by SCAN_DT Myr with v21_tune.sgr_orbit_shift (dt>0: further along) in round 3's t=-3 fields (the same fields
as round 7's timing scan; identical to OBS round03/run00/expansions). Run 4 is the exact -0.49 Myr rewind (run 2's
best in-tolerance moment was 0.49 Myr before t=0). Written to r10_offsets.json, copied to OBS and passed to
v21_tune.new_offset_round(10, base, offsets, labels, note)."""
import sys, json, numpy as np
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import v21_tune as vt
T = '/home/ana/projects/glossary/output/v21_sim/tune/'
E = T + 'round03/expansions'
OUT = '/home/ana/projects/glossary/scripts/tune_analysis/r10_offsets.json'
SCAN_DT = (-0.5, -0.25, 0.25, 0.5, -0.49)


def r10_offsets(out=OUT, scan_dt=SCAN_DT, base_run=2):
    theta = np.array(json.load(open(T + 'round09/round.json'))['runs'][base_run])
    offs = [vt.sgr_orbit_shift(theta, dt, E) - theta[6:] for dt in scan_dt]
    labs = ['r9 run%d orbit %+.2f Myr' % (base_run, dt) for dt in scan_dt]
    note = ('timing scan around round-9 run %d: Sgr t=-3 state moved along its orbit (sgr_orbit_shift, round-3 '
            'fields) by %s Myr; the last run is the -0.49 Myr rewind' % (base_run, list(scan_dt)))
    d = dict(base=theta.tolist(), offsets=np.array(offs).tolist(), labels=labs, note=note, scan_dt=list(scan_dt))
    if out:
        json.dump(d, open(out, 'w'), indent=1)
    return d


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True, linewidth=170)
    d = r10_offsets()
    for l, o in zip(d['labels'], d['offsets']):
        print('%-26s %s' % (l, np.array(o)))
