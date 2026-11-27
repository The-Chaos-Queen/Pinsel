"""Airbrush / splatter / sponge / pen exerciser for the Rust port (verify.sh and PINSEL_AB).

Exercises every code path of the six kernels of the spray group:
  _spray_accumulate   airbrush with and without overspray tail, dots, fill, near the canvas edge
  _splat_discs        the coarse spatter of the airbrush (spatter > 0)
  _splat_drops        flick, tap, toothbrush and explicit drops on paste (dome 0.4/1.0) and fluid paint
  _sponge_dab         paste sponge over wet paint (pickup + deposit), fluid sponge damp (deposit)
                      and thirsty (blot, re-wet settled pigment), many small dabs
  _pen_raster         technical (round), broad nib, pointed nib, dots, hatching, a skipping dry pen
  _feather            fluid ink (feathering) and paste ink (none)

Run from the root of a clone, or with an absolute path from verify.sh (which
runs scripts inside its fastmath-off copy of the package; the copy is the cwd):

    bash rust/harness/verify.sh $PWD/rust/harness/spray_demo.py
"""
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np

import pinsel as ps
from pinsel import pigments as pg, path as P, tools as T

OUT = os.path.join(os.getcwd(), 'output')
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(5)


def oil(p):
    return p.as_medium('oil') if p.medium != 'oil' else p


def acr(p):
    return p.as_medium('acrylic')


def wc(p):
    return p.as_medium('watercolor')


# ==========================================================================
# Paste paints on canvas
# ==========================================================================
cv = ps.Canvas(480, 320, ground='canvas', seed=7)

# wet oil underpainting, so the sponge has something to pick up
flat = ps.Brush('flat', 26, seed=2)
for y in range(20, 300, 30):
    flat.load([(pg.ULTRAMARINE, 1.0), (pg.TITANIUM_WHITE, 2.0 + y / 100.0)], mix=0.6)
    cv.stroke(flat, P.line((10, y), (470, y + rng.normal(0, 6))), pressure=0.8)

# airbrush: with and without overspray tail, coarse spatter, dots, fill, edge
ab = T.Airbrush(seed=31, spatter=0.25, overspray=0.2).load(acr(pg.CADMIUM_RED))
for i in range(4):
    y = 40 + i * 20
    pts = P.line((-15, y), (150, y))
    ab.spray(cv, pts if i % 2 == 0 else pts[::-1], distance=40 + 20 * i, pressure=0.9 - 0.15 * i, wobble=2)
ab2 = T.Airbrush(seed=32, spatter=0.0, overspray=0.0, grain=2.5).load(acr(pg.PHTHALO_BLUE), dilution=0.3)
ab2.spray(cv, P.line((170, 30), (330, 80)), distance=70, pressure=0.7, dwell=0.5)
ab2.dot(cv, 400, 40, duration=0.5, distance=30)
ab2.dot(cv, 478, 5, duration=0.4, distance=20)           # at the canvas edge
ab2.fill(cv, 20, 150, 150, 220, distance=45, pressure=0.6)
with T.Stencil.ellipse(cv, 240, 190, 50, 40, invert=True, leak=0.0):
    ab.fill(cv, 190, 150, 290, 230, distance=45, pressure=0.6)

# splatter: drops of paste paint (dome 0.4 + frayed by tooth) in flicks, taps,
# toothbrush clouds and explicit drops with elongation / angle
sp = T.Splatter(seed=52)
sp.load(oil(pg.CADMIUM_YELLOW))
sp.flick(cv, (300, 250), direction=-20, count=60, distance=(15, 170), size=2.4)
sp.load(acr(pg.TITANIUM_WHITE).thinned(0.15))
sp.toothbrush(cv, (380, 200), radius=70, count=1500, size=0.45)
sp.toothbrush(cv, (360, 160), radius=30, count=200, size=0.9, direction=30)
sp.load(oil(pg.mix(pg.PHTHALO_GREEN, 1, pg.TITANIUM_WHITE, 1)))
sp.tap(cv, (420, 270), radius=45, count=14, size=3.8)
sp.drops(cv, [60, 100, 140, 470], [270, 285, 262, 300], [4.0, 2.2, 0.4, 3.0],
         elongation=[1.0, 2.5, 1.0, 3.0], angle=[0, 40, 0, 200])

# sponge on paste: wet blue underpainting is picked up (pickup > 0) and the
# reservoir deposits; a dirty sponge, then a second colour over it
sg = T.Sponge(size=22, porosity=0.55, seed=43)
sg.load(oil(pg.mix(pg.CADMIUM_ORANGE, 1, pg.TITANIUM_WHITE, 0.4)), 1.0)
for i in range(14):
    sg.dab(cv, 40 + i * 28 + rng.normal(0, 5), 120 + rng.normal(0, 12), pressure=float(rng.uniform(0.4, 0.9)))
sg.scatter(cv, (240, 170), 60, count=10, seed=3)
sg.drag(cv, P.line((30, 300), (450, 296)), pressure=0.5)
sg2 = T.Sponge(size=12, porosity=0.6, coarseness=0.8, seed=44)
sg2.load(acr(pg.SAP_GREEN), 0.8)
for i in range(60):
    sg2.dab(cv, rng.uniform(10, 470), rng.uniform(10, 310), pressure=float(rng.uniform(0.3, 0.8)))

# many small dabs: enough calls that verify.sh also compares the full-canvas
# checkpoint every 1000th call of the sponge kernel
sm = T.Sponge(size=4, porosity=0.4, seed=45)
sm.load(oil(pg.BURNT_UMBER), 1.0)
for i in range(300):
    x, y = rng.uniform(5, 475), rng.uniform(5, 315)
    sm.dab(cv, x, y, pressure=float(rng.uniform(0.3, 0.9)), rotation=float(rng.uniform(0, 360)))
    if sm.load_level < 0.3:
        sm.load(oil(pg.BURNT_UMBER), 1.0)

# paste ink pens (no feathering) and a dry (skipping) pen
ink_oil = pg.LAMP_BLACK.as_medium('oil')
pn = T.Pen('technical', width=2.0, seed=61).load(ink_oil)
pn.line(cv, P.line((20, 20), (200, 22)), wobble=0.5)
pn.dots(cv, [(30, 300), (45, 303), (60, 299)])
dry = T.Pen('technical', width=3.0, capacity=400, seed=62).load(ink_oil)
dry.line(cv, P.wave((20, 308), (460, 308), amplitude=5, wavelength=60))

cv.wait(20)
cv.dry()
cv.save(os.path.join(OUT, 'spray_demo.png'))

# ==========================================================================
# Fluid paints on paper: wet splatter, fluid airbrush, damp and thirsty
# sponge, pens with ink (feathering), nib and pointed pens
# ==========================================================================
pv = ps.Canvas(480, 320, ground='cold_press', seed=5)
g = np.random.default_rng(11)

T.glaze(pv, wc(pg.mix(pg.CERULEAN_BLUE, 1, pg.ULTRAMARINE, 0.6)), 0.16,
        region=(10, 10, 230, 120), feather=2, water=0.9, seed=73)
pv.wait(150)                                   # let the wash settle: the thirsty sponge lifts it
blot = T.Sponge(size=20, porosity=0.5, seed=74).wet(0.1)
for i in range(7):
    blot.dab(pv, 30 + i * 28 + g.normal(0, 6), 60 + g.normal(0, 14), pressure=float(g.uniform(0.5, 0.8)))
sgw = T.Sponge(size=18, porosity=0.55, seed=75)
for col, n, dy in [(pg.mix(pg.SAP_GREEN, 1, pg.INDIGO, 0.5), 8, 20), (pg.HANSA_YELLOW, 6, -8)]:
    sgw.load(wc(col), 1.0, water=0.9)
    for i in range(n):
        sgw.dab(pv, 20 + g.uniform(0, 200), 150 + dy + g.uniform(0, 40), pressure=float(g.uniform(0.35, 0.7)))
sgw.wet(0.05)
for i in range(6):                             # thirsty again: lifts what it just laid
    sgw.dab(pv, 40 + g.uniform(0, 160), 160 + g.uniform(0, 40), pressure=0.6)

aw = T.Airbrush(seed=71, flow=1.6).load(wc(pg.ULTRAMARINE), water=3.0)
for i in range(4):
    y = 20 + i * 14
    aw.spray(pv, P.line((250, y), (470, y)), distance=55, pressure=0.9 - 0.15 * i)
aw.load(wc(pg.QUINACRIDONE_GOLD), water=0.5)    # drier spray
aw.fill(pv, 270, 90, 430, 150, distance=40, pressure=0.6)

sw = T.Splatter(seed=72)
for col, org, dirn in [(pg.BURNT_SIENNA, (250, 250), -6), (pg.PHTHALO_BLUE, (250, 290), 4)]:
    sw.load(wc(col), water=1.0)
    sw.flick(pv, org, direction=dirn, count=30, distance=(20, 200), spread=22, size=2.4)
sw.load(wc(pg.QUINACRIDONE_GOLD))
sw.tap(pv, (380, 230), radius=40, count=14, size=3.0)
sw.toothbrush(pv, (320, 190), radius=40, count=600, size=0.5)

ink = pg.LAMP_BLACK.as_medium('ink')
sepia = pg.SEPIA.as_medium('ink')
tp = T.Pen('technical', width=1.3, seed=81).load(ink)
for k in range(5):
    tp.line(pv, P.line((20, 215 + k * 7), (200, 215 + k * 7 + g.normal(0, 3))), wobble=0.5)
tp.hatch(pv, 20, 255, 130, 310, angle=60, spacing=5.5, cross=True, seed=82)
tp2 = T.Pen('technical', width=1.6, feather=0.9, seed=84).load(ink)
tp2.dots(pv, [(150 + 6 * i, 280 + 5 * np.sin(i)) for i in range(20)])
nib = T.Pen('nib', nib=9, nib_angle=40, seed=86).load(sepia)
for k in range(3):
    cx = 270 + k * 50
    nib.line(pv, P.spline([(cx - 15, 200), (cx - 18, 175), (cx + 4, 165), (cx + 18, 185), (cx + 6, 205)]))
nib.dip(sepia)
nib.line(pv, P.wave((250, 215), (470, 215), amplitude=8, wavelength=40))
pt = T.Pen('pointed', flex=6.0, capacity=1800, seed=87).load(ink)
for k in range(6):
    cx = 270 + k * 30
    pt.line(pv, P.spline([(cx, 300), (cx + 5, 270), (cx + 14, 245)]), pressure=P.swell(0.05, 1.0, center=0.4, width=0.3))
pt.load(ink, 0.3)
pt.line(pv, P.spline([(250, 240), (330, 232), (400, 246), (470, 236)]), pressure=P.swell(0.2, 1.0, width=0.5), speed=1.8)

# many small dabs / dots (full-canvas checkpoint of verify.sh)
ss = T.Sponge(size=5, porosity=0.4, seed=90)
ss.load(wc(pg.SAP_GREEN), 1.0, water=0.9)
for i in range(400):
    ss.dab(pv, g.uniform(5, 475), g.uniform(5, 315), pressure=float(g.uniform(0.3, 0.8)))
    if ss.load_level < 0.3:
        ss.load(wc(pg.SAP_GREEN), 1.0, water=0.9)

pv.wait(20)
pv.dry()
pv.save(os.path.join(OUT, 'spray_demo_paper.png'))
print('film_vol sum', float(cv.film_vol.sum()), 'base_vol sum', float(cv.base_vol.sum()))
