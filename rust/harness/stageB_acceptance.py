"""Stage B acceptance scene (Derek): a full-canvas glaze at print size.

Prints resident memory after every phase and the peak, so the stage B work
can be measured against it. The target: the whole run under a 7 GB cgroup,
no swap (what OOM-killed Nives Cargnel's *Soglia* print).

    python rust/harness/stageB_acceptance.py [W H] [--watercolor]

Default 4500 x 3000, oil. --watercolor adds a full-canvas wet wash and lets
it settle (the watercolour state is allocated on first use).
"""
import os
import resource
import sys
import time

sys.path.insert(0, os.getcwd())
import numpy as np

import pinsel as ps
from pinsel import pigments as pg, path as P, tools as T

args = [a for a in sys.argv[1:] if not a.startswith('--')]
W, H = (int(args[0]), int(args[1])) if len(args) >= 2 else (4500, 3000)
WATERCOLOR = '--watercolor' in sys.argv
OUT = os.path.join(os.getcwd(), 'output')
os.makedirs(OUT, exist_ok=True)
t0 = time.time()


def rss_gb():
    with open('/proc/self/statm') as f:
        return int(f.read().split()[1]) * os.sysconf('SC_PAGE_SIZE') / 2**30


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def phase(name):
    print(f'{time.time() - t0:7.1f} s  {name:34s} rss {rss_gb():5.2f} GB   peak {peak_gb():5.2f} GB', flush=True)


phase('start')
cv = ps.Canvas(W, H, ground='canvas', seed=1)
phase('canvas')
cv.tone(pg.mix(pg.BURNT_SIENNA, 1, pg.YELLOW_OCHRE, 1).thinned(0.3), thickness=0.25)
phase('tone')
b = ps.Brush('flat', 60, seed=2)
rng = np.random.default_rng(0)
for i in range(40):
    b.load(pg.mix(pg.ULTRAMARINE, 1, pg.TITANIUM_WHITE, 2), 1.2)
    y = rng.uniform(0, H)
    cv.stroke(b, P.line((0, y), (W, y + rng.normal(0, 40))), pressure=0.8)
phase('40 strokes')
T.glaze(cv, pg.ALIZARIN_CRIMSON.thinned(0.6), thickness=0.12, region=None, seed=3)
phase('full-canvas oil glaze (wet)')
if WATERCOLOR:
    T.glaze(cv, pg.ULTRAMARINE.as_medium('watercolor'),
            thickness=0.1, region=None, seed=4)
    phase('full-canvas watercolour wash')
cv.dry()
phase('dry')
cv.save(os.path.join(OUT, f'stageB_acceptance_{W}x{H}.png'))
phase('save')
print(f'peak resident memory {peak_gb():.2f} GB')
