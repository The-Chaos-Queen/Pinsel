"""Pastel / charcoal / eraser exerciser for the Rust port (verify.sh and PINSEL_AB).

Exercises every code path of the pastel kernels: tip and side strokes, soft
pastel and vine/compressed charcoal, dust on and off, light and firm pressure,
a single dab, wobble, strokes laid over older strokes (wet film pushed into the
base layer), a masked region, and kneaded-eraser passes over wet paste paint
and over pastel (film and base layers).

Run from the root of a clone, or with an absolute path from verify.sh (which
runs scripts inside its fastmath-off copy of the package; the copy is the cwd):

    bash rust/harness/verify.sh $PWD/rust/harness/pastel_demo.py
"""
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np

import pinsel as ps
from pinsel import pigments as pg, path as P

OUT = os.path.join(os.getcwd(), 'output')
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(3)

cv = ps.Canvas(480, 320, ground='rough', seed=7)

# wet oil underpainting: the pastel lies over wet paste paint
flat = ps.Brush('flat', 26, seed=2)
for y in range(20, 300, 22):
    flat.load([(pg.ULTRAMARINE, 1.0), (pg.TITANIUM_WHITE, 2.0 + y / 100.0)], mix=0.6)
    cv.stroke(flat, P.line((10, y), (470, y + rng.normal(0, 6))), pressure=0.8)

# soft pastel, tip and side, light and firm pressure, different dust
sticks = [
    ps.SoftPastel(size=12, hardness=0.35, dust=0.35, side=0.0, seed=11),
    ps.SoftPastel(size=18, hardness=0.6, dust=0.0, side=1.0, seed=12),
    ps.SoftPastel(size=9, hardness=0.1, dust=1.0, side=0.5, seed=13),
]
colours = [pg.CADMIUM_YELLOW, pg.ALIZARIN_CRIMSON, pg.TITANIUM_WHITE]
for i, (st, col) in enumerate(zip(sticks, colours)):
    st.load(col, amount=1.0 + 0.2 * i)
    for k in range(6):
        y = 30 + 45 * k + 12 * i
        cv.stroke(st, P.wave((20, y), (300 + 30 * i, y + 25), amplitude=8, wavelength=120),
                  pressure=[0.25, 0.5, 0.9][i] if k % 2 else P.taper(0.1, 0.2, 0.8))
# stroke with wobble and an explicit seed, and a single dab
sticks[0].stroke(cv, P.line((40, 280), (400, 260)), pressure=0.7, wobble=0.6, seed=99)
sticks[1].stroke(cv, [(240, 150)], pressure=0.8)
sticks[2].stroke(cv, [(250, 150), (251, 150)], pressure=0.6, angle=35)

# charcoal: vine and compressed, crossing the pastel (older film goes to base)
vine = ps.Charcoal(size=8, seed=5)
comp = ps.Charcoal(size=14, compressed=True, side=0.7, seed=6)
for k in range(8):
    cv.stroke(vine, P.line((30 + 50 * k, 20), (60 + 50 * k, 300)), pressure=0.85)
cv.stroke(comp, P.line((20, 100), (460, 120)), pressure=0.9)
cv.stroke(comp, P.line((20, 200), (460, 180)), pressure=P.taper(0.2, 0.2, 0.8))

# a protected stripe: nothing may land (or lift) there
cv.mask[:, 200:230] = 1.0
cv.stroke(vine, P.line((10, 150), (470, 150)), pressure=1.0)
er = ps.KneadedEraser(size=16, strength=0.7)
er.stroke(cv, P.line((10, 60), (470, 70)), pressure=0.9)
cv.mask[:, 200:230] = 0.0

# eraser passes: over pastel, over the wet oil, a dab, a light pass, and a pass
# across a fresh stroke (film and base layers both lifted)
er.stroke(cv, P.line((60, 110), (420, 190)), pressure=1.0)
er.dab(cv, 300, 250, pressure=0.9)
er.stroke(cv, P.line((20, 240), (460, 240)), pressure=0.2)
ps.KneadedEraser(size=30, strength=1.0).stroke(cv, P.line((100, 20), (140, 300)))
cv.stroke(sticks[1], P.line((50, 30), (430, 30)), pressure=0.9)
er.stroke(cv, P.line((200, 20), (260, 50)), pressure=0.9)

cv.save(os.path.join(OUT, 'pastel_demo.png'))
print('film_vol sum', float(cv.film_vol.sum()), 'base_vol sum', float(cv.base_vol.sum()))
