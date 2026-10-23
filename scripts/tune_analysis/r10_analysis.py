"""Round 10 (timing scan around round-9 run 2): Sgr/LMC misses at t=0, how far ahead along its orbit Sgr is
(+ = early), the best in-tolerance moment within +-30 Myr, and the change from round-9 run 2 (shift 0)."""
import json, numpy as np
from r9_analysis import timing_split
T = '/home/ana/projects/glossary/output/v21_sim/tune/'


def load(k):
    d = json.load(open(T + 'round%02d/round.json' % k))
    return d, np.array(d['misses'])


def report():
    np.set_printoptions(precision=3, suppress=True, linewidth=170)
    d9, M9 = load(9)
    d, M = load(10)
    dts = json.load(open('/home/ana/projects/glossary/scripts/tune_analysis/r10_offsets.json'))['scan_dt']
    rows = [(0.0, 'r9 run 2', M9[2], d9['lmc_ok'][2], d9['closest'][2])]
    rows += [(dt, 'r10 run %d' % i, M[i], d['lmc_ok'][i], d['closest'][i]) for i, dt in enumerate(dts)]
    tg = d['target'][9:]
    print('start shift | Sgr miss x,y,z kpc | vx,vy,vz km/s | ahead [Myr] | LMC ok | best moment')
    for dt, lab, m, ok, c in sorted(rows, key=lambda r: r[0]):
        ahead = timing_split(m[6:], tg)[0]
        best = ('t %+5.2f Myr %s' % (c['t_best'], np.array(c['miss_best'])) if c['t_best'] is not None
                else 'never within position tolerance')
        print('%+5.2f %-9s %s | %s | %+5.2f | %-5s | %s' % (dt, lab, m[6:9], m[9:], ahead, ok, best))
    print('\nLMC miss')
    for dt, lab, m, ok, c in sorted(rows, key=lambda r: r[0]):
        print('%+5.2f %-9s %s %s' % (dt, lab, m[:3], m[3:6]))


if __name__ == '__main__':
    report()
