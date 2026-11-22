"""Minimal pinsel example: a few strokes, a glaze, and a render."""
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pinsel as ps
from pinsel import pigments as pg, path as P

rng = np.random.default_rng(0)
cv = ps.Canvas(900, 600, ground='canvas', seed=1)
cv.tone(pg.mix(pg.BURNT_SIENNA, 2, pg.YELLOW_OCHRE, 1).thinned(0.3), thickness=0.25)

# sky: overlapping strokes with streaky, partially mixed blues (lighter towards the horizon)
flat = ps.Brush('flat', 24, seed=2)
for y in np.arange(10, 430, 14):
    t = y / 430
    for x in np.arange(-100, 900, 260):
        flat.load([(pg.ULTRAMARINE, 1.2 - t), (pg.TITANIUM_WHITE, 1.5 + 4 * t), (pg.CERULEAN_BLUE, 0.8)],
                  mix=0.6)
        x0 = x + rng.uniform(-60, 60)
        cv.stroke(flat, P.wave((x0, y), (x0 + rng.uniform(250, 380), y + rng.normal(0, 4)), amplitude=5,
                               wavelength=400), pressure=P.taper(0.1, 0.2, 0.8))

# a sun, painted wet into the sky
sun = ps.Brush('round', 16, seed=4).load(pg.mix(pg.CADMIUM_YELLOW, 2, pg.TITANIUM_WHITE, 3), 1.6)
cv.stroke(sun, P.spiral((650, 140), 2, 26, turns=3), pressure=0.9)

# rolling hill in thick hog-bristle impasto, dabbed with varied greens
block = ps.Brush('flat', 30, seed=6)
for x in np.arange(-20, 920, 40):
    block.load(pg.mix(pg.SAP_GREEN, 2, pg.ULTRAMARINE, 0.6, pg.TITANIUM_WHITE, 0.8), 1.0)
    top = 370 + 40 * np.sin(x / 180) + 15 * np.sin(x / 57)
    cv.stroke(block, P.line((x, top + 8), (x + 8, 620)), pressure=0.85)
hog = ps.Brush('hog', 9, seed=3)
greens = [pg.mix(pg.SAP_GREEN, 2, pg.CADMIUM_YELLOW, 1, pg.TITANIUM_WHITE, 1),
          pg.mix(pg.SAP_GREEN, 2, pg.YELLOW_OCHRE, 1, pg.TITANIUM_WHITE, 0.6),
          pg.mix(pg.CADMIUM_YELLOW_LIGHT, 1, pg.SAP_GREEN, 1, pg.TITANIUM_WHITE, 1.5)]
for i in range(900):
    x = rng.uniform(-10, 910)
    top = 370 + 40 * np.sin(x / 180) + 15 * np.sin(x / 57)
    y = top + 4 + (620 - top) * rng.uniform() ** 1.6
    hog.load(greens[i % 3].impasto(1.5), amount=1.3)
    a = -np.pi / 2 + rng.normal(0.2, 0.35)
    L = 18 + 30 * (y - top) / 250
    cv.stroke(hog, P.quad((x, y), (x + 0.4 * L * np.cos(a) + 4, y + 0.5 * L * np.sin(a)),
                         (x + L * np.cos(a), y + L * np.sin(a))), pressure=P.flick(0.85, 0.25))

# let it dry, then glaze warm evening light over the hill
cv.dry()
glaze = ps.Brush('hake', 40, seed=5)
for y in range(380, 620, 50):
    glaze.load(pg.QUINACRIDONE_GOLD.thinned(0.85), 0.6)
    cv.stroke(glaze, P.line((-20, y), (920, y - 10)), pressure=0.7)

(ROOT / 'output').mkdir(exist_ok=True)
cv.save(ROOT / 'output' / 'quickstart.png')
print('saved output/quickstart.png')
