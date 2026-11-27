"""Little Tern: a tern over a cold sea at dawn. Watercolour on cold-press paper.

Painted by Tern (Claude Opus 5.5) on 2026-11-27, the day pinsel got its Rust
kernels. The bird is reserved with masking, the washes go over it, then the
mask comes off and the cap, beak and wing tips go in. About 15 s, 0.7 GB.
"""
import sys
from pathlib import Path

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pinsel as ps
from pinsel import pigments as pg, path as P, watercolor as wc, tools as T

W, H = 1200, 800
HORIZON = 520
cv = ps.Canvas(W, H, ground='cold_press', ground_color='#f3ecdf', seed=27)
rng = np.random.default_rng(9)
w = lambda p: p.as_medium('watercolor')

# --- the bird, as a silhouette (flying left, far wing up, near wing down)
CX, CY, S = 790, 255, 1.35
FAR = [(-18, -14), (-8, -40), (5, -62), (35, -95), (75, -128), (62, -112), (32, -84), (18, -60), (12, -30), (8, -12)]
NEAR = [(5, 13), (-2, 36), (-14, 58), (-12, 80), (6, 112), (15, 128), (-12, 95), (-30, 58), (-26, 38), (-14, 10)]
BIRD = ([(-96, 1), (-80, -4), (-70, -10), (-55, -11), (-35, -7)] + FAR +
        [(35, -6), (60, -6), (112, -22), (74, 0), (110, 20), (58, 7), (25, 11)] + NEAR + [(-50, 8), (-80, 4)])
pt = lambda pts: [(CX + S * x, CY + S * y) for x, y in pts]
mask = T.Stencil.polygon(cv, pt(BIRD))
mask.apply()

# --- sky: cool grey-blue above, warming to a pale gold band at the horizon
sky_stops = [(w(pg.mix(pg.ULTRAMARINE, 1, pg.BURNT_SIENNA, 0.7)), 0.0, 0.9),
             (w(pg.mix(pg.COBALT_BLUE, 1, pg.BURNT_SIENNA, 0.25)), 0.35, 0.45),
             (w(pg.mix(pg.PERMANENT_ROSE, 1, pg.COBALT_BLUE, 1)), 0.70, 0.35),
             (w(pg.QUINACRIDONE_GOLD), 0.90, 0.45),
             (w(pg.NAPLES_YELLOW), 1.0, 0.35)]
wc.variegated_wash(cv, wc.rect(-30, -30, W + 30, HORIZON + 4), sky_stops, direction=((0, 0), (0, HORIZON)),
                   concentration=0.75, water=0.75, variation=0.18, rough=0.3, seed=1)
cv.wait(4)
# a few soft cloud lifts while the sky is still damp
for cx, cy in [(240, 150), (470, 110), (980, 170)]:
    wc.lift(cv, wc.ellipse(cx, cy, 130, 20), strength=0.35, seed=cx)
cv.wait(60)

# --- sea: pale at the horizon, deep and cold at the bottom
sea_stops = [(w(pg.mix(pg.COBALT_BLUE, 1, pg.QUINACRIDONE_GOLD, 0.3)), 0.0, 0.5),
             (w(pg.mix(pg.ULTRAMARINE, 1, pg.BURNT_SIENNA, 0.5)), 0.45, 1.1),
             (w(pg.INDIGO), 1.0, 1.6)]
wc.variegated_wash(cv, wc.rect(-30, HORIZON, W + 30, H + 30), sea_stops, direction=((0, HORIZON), (0, H)),
                   concentration=1.0, water=0.7, variation=0.15, rough=0.4, seed=2)
cv.wait(90)
# light on the water under the bright band
wc.lift(cv, wc.ellipse(560, HORIZON + 40, 380, 28), strength=0.45, seed=3)
wc.settle(cv)          # dry before the waves, so they keep their edges

# --- waves: broken horizontal strokes, longer and darker towards the front
brush = ps.Brush('flat', 10, seed=4)
for i in range(160):
    t = (i / 160) ** 1.5
    y = HORIZON + 8 + t * (H - HORIZON - 10) + rng.normal(0, 3)
    x0 = rng.uniform(-60, W)
    L = 25 + 200 * t
    brush.size = 1.5 + 5 * t
    brush.load(w(pg.mix(pg.INDIGO, 1, pg.ULTRAMARINE, 1)), amount=0.25 + 0.5 * t, water=0.35)
    cv.stroke(brush, P.wave((x0, y), (x0 + L, y + rng.normal(0, 2)), amplitude=1.5 + 2 * t,
                            wavelength=L / 1.5), pressure=P.taper(0.2, 0.3, 0.5 + 0.3 * t))
cv.wait(120)
mask.remove()

# --- the bird: soft grey under the wings, black cap, red beak, dark tips
wc.wash(cv, wc.polygon(pt(NEAR)),
        w(pg.mix(pg.PAYNES_GREY, 1, pg.COBALT_BLUE, 0.5)), concentration=0.25, water=0.5, seed=5)
wc.wash(cv, wc.polygon(pt(FAR)),
        w(pg.mix(pg.PAYNES_GREY, 1, pg.COBALT_BLUE, 0.5)), concentration=0.12, water=0.5, seed=6)
wc.wash(cv, wc.polygon(pt([(25, 11), (5, 13), (-14, 10), (-50, 8), (-30, 4), (20, 4)])),
        w(pg.PAYNES_GREY), concentration=0.18, water=0.5, seed=7)
cv.wait(30)
wc.wash(cv, wc.polygon(pt([(-81, -4), (-70, -10), (-55, -11), (-38, -8), (-48, -2), (-66, -1)])),
        w(pg.mix(pg.IVORY_BLACK, 1, pg.INDIGO, 0.3)), concentration=3.0, water=0.4, seed=8)
wc.wash(cv, wc.polygon(pt([(-97, 1), (-80, -4), (-79, 4)])),
        w(pg.mix(pg.CADMIUM_RED, 1, pg.CADMIUM_ORANGE, 0.4)), concentration=2.5, water=0.4, seed=9)
for tip in ([(75, -128), (62, -112), (50, -106), (66, -123)], [(15, 128), (6, 112), (-1, 106), (9, 123)]):
    wc.wash(cv, wc.polygon(pt(tip)), w(pg.PAYNES_GREY), concentration=1.2, water=0.4, seed=10)
cv.wait(200)
cv.dry()
(ROOT / 'output').mkdir(exist_ok=True)
cv.save(ROOT / 'output' / 'little_tern.png')
print('saved output/little_tern.png')
