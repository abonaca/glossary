"""Round-1 start: round-0 start with the LMC delayed by 0.148 (two-body) and Sgr solved in the restricted model
(v21_surrogate.restricted_start on the round-0 live tracks), rounded to 6 decimals and passed to
v21_tune.new_round(1, theta, note=...)."""
import sys, json, contextlib, io, numpy as np
sys.path.insert(0, '/home/ana/projects/glossary/scripts')
import v21_surrogate as vs
O = '/home/ana/projects/glossary/output/v21_sim/tune/'


def theta1():
    with contextlib.redirect_stdout(io.StringIO()):
        th = vs.restricted_start(vs.load_tracks(0))
    return np.array([float('%.6f' % x) for x in th])


def check():
    t = np.array(json.load(open(O + 'round01/round.json'))['theta'])
    print('max |theta1 - round01 theta| = %.2e' % np.abs(theta1() - t).max())


if __name__ == '__main__':
    check()
