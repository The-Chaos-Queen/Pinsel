"""Brush & paint behaviour sampler for pinsel (oil / acrylic)."""
import sys, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pinsel as ps
from pinsel import pigments as pg, path as P

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'output' / 'brush_sampler.png'
OUT.parent.mkdir(parents=True, exist_ok=True)
W, H = 1500, 1040
cv = ps.Canvas(W, H, ground='canvas', seed=3)
t0 = time.time()
labels = []
cw, ch = W // 4, H // 3


def cell(i, j):
    return i * cw, j * ch


# 1. brush shapes with natural taper -------------------------------------------
x0, y0 = cell(0, 0)
labels.append((x0, y0, 'brush shapes'))
shapes = [('round', 10, pg.ULTRAMARINE), ('hog', 11, pg.BURNT_SIENNA), ('flat', 13, pg.CADMIUM_YELLOW),
          ('filbert', 12, pg.CADMIUM_RED), ('fan', 18, pg.SAP_GREEN), ('rigger', 2.5, pg.IVORY_BLACK),
          ('angular', 13, pg.CERULEAN_BLUE)]
for k, (sh, sz, p) in enumerate(shapes):
    b = ps.Brush(sh, sz, seed=k).load(p)
    y = y0 + 40 + k * 44
    cv.stroke(b, P.wave((x0 + 30, y), (x0 + cw - 30, y), amplitude=7, wavelength=220),
              pressure=P.taper(0.15, 0.35, 0.85))

# 2. wet-on-wet blending -----------------------------------------------------------
x0, y0 = cell(1, 0)
labels.append((x0, y0, 'wet-in-wet blending'))
b = ps.Brush.flat(18, seed=1)
for k in range(9):
    b.load(pg.mix(pg.PHTHALO_BLUE, 1, pg.TITANIUM_WHITE, 2), 1.2)
    y = y0 + 45 + k * 15
    cv.stroke(b, P.line((x0 + 30, y), (x0 + cw - 30, y)), pressure=0.8)
b2 = ps.Brush.filbert(12, seed=2)
for k in range(5):
    b2.load(pg.TITANIUM_WHITE, 1.0)
    x = x0 + 50 + k * 60
    cv.stroke(b2, P.bezier((x, y0 + 40), (x + 30, y0 + 90), (x - 30, y0 + 140), (x + 10, y0 + 200)),
              pressure=P.taper(0.1, 0.4, 0.8))
# blend lower part with empty brush
blender = ps.Brush('flat', 16, seed=5, pickup=0.9)
for k in range(6):
    y = y0 + 135 + k * 9
    cv.stroke(blender, P.line((x0 + 30, y), (x0 + cw - 30, y)), pressure=0.7)
# warm into cool: yellow then blended into the blue with zig-zag strokes
b.load(pg.CADMIUM_YELLOW, 1.0)
for k in range(3):
    cv.stroke(b, P.line((x0 + 30, y0 + 215 + k * 15), (x0 + cw - 30, y0 + 215 + k * 15)), pressure=0.8)
b.load(pg.mix(pg.PHTHALO_BLUE, 1, pg.TITANIUM_WHITE, 2), 1.0)
for k in range(3):
    cv.stroke(b, P.line((x0 + 30, y0 + 260 + k * 15), (x0 + cw - 30, y0 + 260 + k * 15)), pressure=0.8)
bl2 = ps.Brush('flat', 14, seed=6, pickup=0.9)
for k in range(14):
    x = x0 + 40 + k * 21
    cv.stroke(bl2, P.zigzag((x, y0 + 205), (x, y0 + 300), amplitude=8, count=4), pressure=0.6, angle=0)

# 3. multi-colour streaky load ------------------------------------------------------
x0, y0 = cell(2, 0)
labels.append((x0, y0, 'streaky multi-colour load'))
b = ps.Brush('flat', 22, seed=7)
for k in range(3):
    b.load([(pg.CADMIUM_YELLOW, 1), (pg.CADMIUM_RED_LIGHT, 1), (pg.TITANIUM_WHITE, 0.6)], amount=1.3, mix=0.15)
    y = y0 + 70 + k * 85
    cv.stroke(b, P.spline([(x0 + 30, y), (x0 + 140, y - 30), (x0 + 250, y + 25), (x0 + cw - 25, y - 10)]),
              pressure=P.taper(0.1, 0.3, 0.9))

# 4. impasto ------------------------------------------------------------------------
x0, y0 = cell(3, 0)
labels.append((x0, y0, 'impasto (overloaded hog)'))
b = ps.Brush('hog', 14, seed=11)
cols = [pg.TITANIUM_WHITE, pg.NAPLES_YELLOW, pg.CADMIUM_ORANGE, pg.mix(pg.TITANIUM_WHITE, 3, pg.CERULEAN_BLUE, 1)]
g = np.random.default_rng(4)
for k in range(16):
    b.load(cols[k % 4].impasto(2.0), amount=2.0)
    cx, cy = x0 + 60 + g.uniform(0, cw - 120), y0 + 60 + g.uniform(0, ch - 120)
    a = g.uniform(0, np.pi)
    L = g.uniform(40, 90)
    cv.stroke(b, P.quad((cx, cy), (cx + L * 0.5 * np.cos(a) + 15, cy + L * 0.5 * np.sin(a) - 10),
                        (cx + L * np.cos(a), cy + L * np.sin(a))), pressure=P.taper(0.1, 0.4, 0.95))

# 5. glazing over dried underpainting -------------------------------------------------
x0, y0 = cell(0, 1)
labels.append((x0, y0, 'glazing over dry paint'))
b = ps.Brush.flat(16, seed=3)
for k, p in enumerate([pg.CADMIUM_YELLOW_LIGHT, pg.TITANIUM_WHITE, pg.IVORY_BLACK]):
    b.load(p, 1.2)
    for r in range(4):
        x = x0 + 40 + r * 15 + k * 95
        cv.stroke(b, P.line((x, y0 + 40), (x, y0 + ch - 40)), pressure=0.85, angle=0)

# 6. dry brush / scumbling over the canvas weave ------------------------------------
x0, y0 = cell(1, 1)
labels.append((x0, y0, 'dry brush & scumble'))
b = ps.Brush('hog', 16, seed=8)
for k in range(5):
    b.load(pg.mix(pg.TITANIUM_WHITE, 2, pg.YELLOW_OCHRE, 1), amount=0.18)
    y = y0 + 50 + k * 50
    cv.stroke(b, P.line((x0 + 30, y), (x0 + cw - 30, y + 10)), pressure=0.45 + 0.1 * k, speed=1.5)

# 7. smudge / pickup ---------------------------------------------------------------
x0, y0 = cell(2, 1)
labels.append((x0, y0, 'brush picks up & drags wet paint'))
b = ps.Brush.round(9, seed=4)
for k, p in enumerate([pg.ALIZARIN_CRIMSON, pg.CADMIUM_YELLOW, pg.PHTHALO_GREEN, pg.ULTRAMARINE]):
    b.load(p, 1.4)
    x = x0 + 60 + k * 65
    cv.stroke(b, P.line((x, y0 + 40), (x, y0 + ch - 40)), pressure=0.9)
sm = ps.Brush('filbert', 14, seed=9, pickup=1.0)
sm.load(pg.TITANIUM_WHITE, 0.5)
for k in range(3):
    y = y0 + 90 + k * 80
    cv.stroke(sm, P.line((x0 + 30, y), (x0 + cw - 30, y)), pressure=0.8)

# 8. pressure: calligraphy & swell ------------------------------------------------------
x0, y0 = cell(3, 1)
labels.append((x0, y0, 'pressure & flat-brush calligraphy'))
b = ps.Brush.round(12, seed=2)
b.load(pg.PRUSSIAN_BLUE, 1.2)
cv.stroke(b, P.spline([(x0 + 40, y0 + 60), (x0 + 120, y0 + 140), (x0 + 220, y0 + 60), (x0 + 330, y0 + 120)]),
          pressure=P.swell(0.05, 1.0))
b = ps.Brush.flat(14, seed=3)
b.load(pg.QUINACRIDONE_MAGENTA, 1.2)
cv.stroke(b, P.spline([(x0 + 50, y0 + 290), (x0 + 120, y0 + 190), (x0 + 200, y0 + 300), (x0 + 300, y0 + 200)]),
          pressure=0.7, angle=35)

cv.wait(1)
cv.dry()

# 5b. glaze over the dried bars
x0, y0 = cell(0, 1)
b = ps.Brush.flat(20, seed=5)
for k, p in enumerate([pg.ALIZARIN_CRIMSON.thinned(0.6), pg.PHTHALO_BLUE.thinned(0.7), pg.QUINACRIDONE_GOLD.thinned(0.5)]):
    b.load(p, 0.8)
    y = y0 + 60 + k * 90
    cv.stroke(b, P.line((x0 + 25, y), (x0 + cw - 25, y)), pressure=0.8)

# 6b. scumble over dried impasto
x0, y0 = cell(3, 0)
b = ps.Brush('hog', 18, seed=12)
b.load(pg.ULTRAMARINE, amount=0.15)
cv.stroke(b, P.line((x0 + 30, y0 + ch - 60), (x0 + cw - 30, y0 + 60)), pressure=0.5, speed=1.4)

# 9. palette of mixes (bottom row) ---------------------------------------------------
x0, y0 = cell(0, 2)
labels.append((x0, y0, 'Kubelka-Munk mixing: tube colours (top) -> mixes (bottom)'))
pairs = [(pg.CADMIUM_YELLOW, pg.PHTHALO_BLUE), (pg.CADMIUM_YELLOW, pg.ULTRAMARINE), (pg.ULTRAMARINE, pg.BURNT_SIENNA),
         (pg.ALIZARIN_CRIMSON, pg.ULTRAMARINE), (pg.CADMIUM_RED, pg.CADMIUM_YELLOW), (pg.CADMIUM_RED, pg.TITANIUM_WHITE),
         (pg.PHTHALO_GREEN, pg.CADMIUM_RED), (pg.YELLOW_OCHRE, pg.IVORY_BLACK)]
b = ps.Brush.filbert(16, seed=1)
for k, (a, c) in enumerate(pairs):
    x = 50 + k * 180
    b.load(a, 1.2); cv.stroke(b, P.line((x, y0 + 50), (x + 60, y0 + 50)), pressure=0.9, angle=90)
    b.load(c, 1.2); cv.stroke(b, P.line((x + 80, y0 + 50), (x + 140, y0 + 50)), pressure=0.9, angle=90)
    w = 3 if c is pg.TITANIUM_WHITE else 1
    b.load(pg.mix(a, 1, c, w), 1.4)
    cv.stroke(b, P.spline([(x, y0 + 150), (x + 70, y0 + 130), (x + 140, y0 + 160)]), pressure=0.9)
    b.load(pg.mix(a, 1, c, w, pg.TITANIUM_WHITE, 3 * w), 1.4)
    cv.stroke(b, P.spline([(x, y0 + 240), (x + 70, y0 + 220), (x + 140, y0 + 250)]), pressure=0.9)

print('painted in %.1fs' % (time.time() - t0))
img = cv.to_image()
d = ImageDraw.Draw(img)
try:
    font = ImageFont.truetype('DejaVuSans.ttf', 15)
except Exception:
    font = ImageFont.load_default()
for (x, y, t) in labels:
    d.text((x + 12, y + 8), t, fill=(60, 50, 40), font=font)
img.save(OUT)
print('saved', OUT, 'total %.1fs' % (time.time() - t0))
