"""
Watercolour demo: a sampler sheet of classic watercolour effects, all
produced by the fluid simulation in :mod:`pinsel.watercolor`.

    python examples/watercolor_demo.py

Writes output/watercolor_demo.png (1200 x 800, rough paper).
"""
import os
import time

import numpy as np
from PIL import Image, ImageDraw

import pinsel as ps
from pinsel import watercolor as wc
from pinsel import path as P

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output', 'watercolor_demo.png')

W, H = 1200, 800
cv = ps.Canvas(W, H, ground='rough', seed=11)
pg = lambda name: ps.pigments.get(name, 'watercolor')
ULTRA, SIENNA, PHTHALO = pg('ultramarine'), pg('burnt_sienna'), pg('phthalo_blue')
ROSE, YELLOW, CERULEAN = pg('permanent_rose'), pg('hansa_yellow'), pg('cerulean_blue')
SAP, INDIGO, OCHRE = pg('sap_green'), pg('indigo'), pg('yellow_ochre')
COBALT, ALIZ, GOLD = pg('cobalt_blue'), pg('alizarin_crimson'), pg('quinacridone_gold')

# 4 x 3 panels
PW, PH = 262, 214
def panel(c, r):
    x0 = 22 + c * 295
    y0 = 16 + r * 262
    return x0, y0, x0 + PW, y0 + PH

labels = {}
T0 = time.time()


def dry_all(max_s=600):
    """Wait (simulated) until every wash has dried (deposits stay re-wettable)."""
    t = 0
    while cv.wc is not None and cv.wc.bbox is not None and t < max_s:
        cv.wait(20)
        t += 20


# ---------------------------------------------------------------- row 1
# 1. flat and graded washes
x0, y0, x1, y1 = panel(0, 0)
xm = (x0 + x1) // 2
wc.wash(cv, (x0, y0, xm - 8, y1), ULTRA, concentration=0.55, water=0.8, seed=1)
wc.wash(cv, (xm + 8, y0, x1, y1), SIENNA, concentration=1.1, water=0.85,
        gradient=((0, y0), (0, y1), (1.0, 0.04), 1.3), seed=2)
labels[(0, 0)] = 'flat wash  |  graded wash'

# 4. granulation: ultramarine, burnt sienna (granulating) vs phthalo (smooth)
x0, y0, x1, y1 = panel(3, 0)
bw = (x1 - x0 - 16) // 3
for i, (p, c) in enumerate([(ULTRA, 0.7), (SIENNA, 0.9), (PHTHALO, 0.35)]):
    xa = x0 + i * (bw + 8)
    wc.wash(cv, (xa, y0, xa + bw, y1), p, concentration=c, water=0.85, seed=10 + i)
labels[(3, 0)] = 'granulation: ultramarine, sienna | phthalo'

# 2. wet-on-wet colour mixing
x0, y0, x1, y1 = panel(1, 0)
cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
wc.wet_paper(cv, wc.ellipse(cx, cy, PW * 0.47, PH * 0.45), water=0.55, soak=0.9, seed=3)
cv.wait(4)
mop = ps.Brush('mop', 16, seed=1)
mop.load(ROSE, amount=1.0, water=1.0)
cv.stroke(mop, P.spline([(x0 + 40, cy - 40), (cx, cy - 60), (x1 - 40, cy - 30)]), pressure=0.8)
cv.wait(2)
mop.load(YELLOW, amount=1.2, water=1.0)
cv.stroke(mop, P.spline([(x0 + 35, cy + 10), (cx, cy - 5), (x1 - 35, cy + 20)]), pressure=0.8)
cv.wait(2)
mop.load(PHTHALO, amount=0.6, water=1.0)
cv.stroke(mop, P.spline([(x0 + 45, cy + 55), (cx, cy + 45), (x1 - 45, cy + 60)]), pressure=0.8)
wc.drop(cv, cx - 50, cy + 25, ULTRA, radius=9, concentration=1.4, seed=4)
wc.drop(cv, cx + 55, cy - 10, ALIZ, radius=8, concentration=1.2, seed=5)
labels[(1, 0)] = 'wet-on-wet mixing'
cv.wait(3)

# 3. wet-on-dry strokes with edge darkening (first layer)
x0, y0, x1, y1 = panel(2, 0)
rnd = ps.Brush('round', 13, seed=2)
rnd.load(COBALT, amount=0.9, water=1.0)
cv.stroke(rnd, P.spline([(x0 + 20, y0 + 50), (x0 + 120, y0 + 25), (x1 - 20, y0 + 70)]), pressure=0.9)
rnd.load(ROSE, amount=0.8, water=1.0)
cv.stroke(rnd, P.spline([(x0 + 30, y1 - 40), (x0 + 130, y1 - 90), (x1 - 25, y1 - 50)]), pressure=0.95)
labels[(2, 0)] = 'wet-on-dry: hard edges, dark rims'

# ---------------------------------------------------------------- row 2 washes
# 5. backrun: wash now, clear water later into the half-dry wash
x0, y0, x1, y1 = panel(0, 1)
wc.wash(cv, (x0, y0, x1, y1), ULTRA, concentration=0.7, water=0.8, seed=20)
labels[(0, 1)] = 'backruns (blooms)'
# 6. salt: wash now, salt when the shine has gone
x0, y0, x1, y1 = panel(1, 1)
wc.variegated_wash(cv, (x0, y0, x1, y1), [(INDIGO, 0.0), (ULTRA, 0.5), (ALIZ, 1.0)],
                   ((x0, y0), (x1, y1)), concentration=0.9, water=0.85, seed=21)
labels[(1, 1)] = 'salt texture'
# 7. lifting: sky wash, clouds blotted with tissue while wet
x0, y0, x1, y1 = panel(2, 1)
wc.wash(cv, (x0, y0, x1, y1), CERULEAN, concentration=0.75, water=0.8,
        gradient=((0, y0), (0, y1), (1.0, 0.35)), seed=22)
wc.wash(cv, (x0, y0, x1, y0 + 90), ULTRA, concentration=1.2, water=0.12, soft=12, seed=23)
labels[(2, 1)] = 'lifting: tissue blots, damp brush'

cv.wait(24)
# clouds: blot the wet sky
for (bx, by, rx, ry) in [(x0 + 65, y0 + 55, 55, 26), (x0 + 185, y0 + 45, 60, 24),
                         (x0 + 125, y0 + 115, 70, 28)]:
    wc.blot(cv, wc.ellipse(bx, by, rx, ry), strength=1.0, scale=14.0, seed=int(bx))

# 8. dry brush on rough paper
x0, y0, x1, y1 = panel(3, 1)
hake = ps.Brush('hake', 22, seed=3)
for i, p in enumerate([SIENNA, ULTRA, SAP]):
    hake.load(p, amount=1.4, water=0.32)
    yy = y0 + 40 + i * 65
    cv.stroke(hake, P.wave((x0 + 10, yy), (x1 - 10, yy + 10), amplitude=6, wavelength=180),
              pressure=0.55, speed=2.0)
rd = ps.Brush('round', 6, seed=4)
rd.load(INDIGO, amount=1.5, water=0.3)
for k in range(5):
    xa = x0 + 25 + k * 48
    cv.stroke(rd, P.line((xa, y1 - 15), (xa + 25, y0 + 15)), pressure=0.6, speed=1.8)
labels[(3, 1)] = 'dry brush on rough paper'

# wait until the backrun / salt washes have lost their shine
cv.wait(64)
x0, y0, x1, y1 = panel(0, 1)
wc.drop(cv, x0 + 80, y0 + 75, None, radius=16, water=1.3, seed=30)
wc.drop(cv, x0 + 180, y0 + 140, None, radius=13, water=1.3, seed=31)
wb = ps.Brush('round', 9, seed=5)
wb.load(ULTRA, amount=0.0, water=1.0)        # clean, wet brush
cv.stroke(wb, P.line((x0 + 30, y1 - 30), (x0 + 120, y1 - 45)), pressure=0.9)
x0, y0, x1, y1 = panel(1, 1)
wc.salt(cv, wc.rect(x0 + 10, y0 + 10, x1 - 10, y1 - 10), density=0.0025, size=(0.7, 2.2), seed=32)
dry_all()

# second layer for panel 3 (glaze strokes crossing the dried first ones)
x0, y0, x1, y1 = panel(2, 0)
rnd.load(GOLD, amount=0.9, water=1.0)
cv.stroke(rnd, P.spline([(x0 + 70, y0 + 15), (x0 + 110, y0 + 110), (x0 + 90, y1 - 15)]), pressure=0.9)
# damp-brush lifting on the dried sky
x0, y0, x1, y1 = panel(2, 1)
wc.lift(cv, wc.ellipse(x0 + 205, y0 + 160, 30, 30), strength=0.8, seed=33)

# ---------------------------------------------------------------- row 3
# 9. layered glazes: each layer dries before the next
x0, y0, x1, y1 = panel(0, 2)
glz = [(ROSE, (x0 + 10, y0 + 10, x0 + 160, y0 + 140)),
       (YELLOW, (x0 + 95, y0 + 40, x1 - 10, y0 + 175)),
       (PHTHALO, (x0 + 45, y0 + 95, x0 + 205, y1 - 5))]
for p, box in glz:
    wc.wash(cv, box, p, concentration=0.45 if p is not PHTHALO else 0.22, water=0.75,
            seed=int(box[0]))
    dry_all()
labels[(0, 2)] = 'layered glazes'

# 10. variegated wash (sunset)
x0, y0, x1, y1 = panel(1, 2)
wc.variegated_wash(cv, (x0, y0, x1, y1), [(YELLOW, 0.05), (ROSE, 0.45), (ULTRA, 0.95)],
                   ((x0, y1), (x0, y0)), concentration=0.8, water=0.85, seed=40)
labels[(1, 2)] = 'variegated wash'

# 11. spritz into a damp wash + spatter
x0, y0, x1, y1 = panel(2, 2)
wc.wash(cv, (x0, y0, x1, y1), SAP, concentration=0.8, water=0.8, seed=41)
labels[(2, 2)] = 'spritz + spatter'

# 12. little landscape
x0, y0, x1, y1 = panel(3, 2)
hor = y0 + 120
wc.wet_paper(cv, (x0, y0, x1, hor + 8), water=0.5, soak=0.9, seed=42)
cv.wait(3)
wc.variegated_wash(cv, (x0, y0, x1, hor + 8), [(CERULEAN, 0.0), (ROSE, 0.85), (YELLOW, 1.0)],
                   ((x0, y0), (x0, hor)), concentration=0.55, water=0.2, seed=43)
labels[(3, 2)] = 'landscape: all of it together'

cv.wait(75)
x0s, y0s, x1s, y1s = panel(2, 2)
wc.spritz(cv, (x0s + 5, y0s + 5, x1s - 5, y1s - 5), density=0.0012, size=(1.0, 5.0), water=0.8, seed=44)
wc.spatter(cv, (x0s + 10, y0s + 120, x1s - 10, y1s - 10), SIENNA, density=0.0015, size=(0.6, 3.0),
           concentration=1.6, seed=45)
dry_all()

# landscape: distant mountains (wet-on-dry), near hills, trees, grass
mtn = [(x0, hor + 8)] + [(x0 + i * 26, hor - 25 - 30 * abs(np.sin(i * 1.3)) - 8 * (i % 2))
                         for i in range(11)] + [(x1, hor - 20), (x1, hor + 8)]
wc.wash(cv, mtn, ps.mix(ULTRA, 2, ALIZ, 1), concentration=0.55, water=0.8, seed=46)
dry_all()
hills = [(x0, y1)] + [(x0 + i * 22, hor + 30 - 22 * np.sin(i * 0.5 + 0.6)) for i in range(13)] + [(x1, y1)]
wc.wash(cv, hills, ps.mix(SAP, 2, OCHRE, 1), concentration=0.9, water=0.85, seed=47)
cv.wait(6)
wc.drop(cv, x0 + 70, hor + 60, INDIGO, radius=10, concentration=1.2, seed=48)
wc.drop(cv, x0 + 190, hor + 70, SIENNA, radius=9, concentration=1.2, seed=49)
dry_all()
tree = ps.Brush('round', 5, seed=6)
for tx in (x0 + 40, x0 + 58, x0 + 210):
    tree.load(ps.mix(SAP, 1, INDIGO, 1), amount=1.6, water=0.9)
    for k in range(7):
        yy = hor + 45 - k * 9
        wdt = 3 + (7 - k) * 2.2
        cv.stroke(tree, P.line((tx - wdt, yy), (tx + wdt, yy - 3)), pressure=0.8)
grass = ps.Brush('fan', 16, seed=7)
grass.load(ps.mix(SAP, 1, SIENNA, 1), amount=1.6, water=0.35)
for k in range(6):
    xa = x0 + 15 + k * 42
    cv.stroke(grass, P.line((xa, y1 - 5), (xa + 12, y1 - 40)), pressure=0.6, speed=1.6)
dry_all()

cv.dry()
img = cv.to_image()

# captions
d = ImageDraw.Draw(img)
for (c, r), text in labels.items():
    x0, y0, x1, y1 = panel(c, r)
    d.text((x0, y1 + 6), text, fill=(90, 84, 76))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
img.save(OUT)
print('wrote', os.path.abspath(OUT), 'in %.1f s (simulated %.0f s)' % (time.time() - T0, cv.time))
