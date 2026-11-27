"""Strip-wise output gives exactly the pixels of the full render.

Paints a small scene with impasto, wet gloss, dried film and watercolour
water, then compares (bit for bit):
  - Canvas.render() (one full-size pass) with render(region=strip) pieces;
  - Canvas.to_image() (strips) with the full render converted to 8 bit;
  - Canvas.preview() (strips) with the old full-size preview code.
Strips are made deliberately small (and odd-sized) so many boundaries fall
inside the paint. Then measures peak traced memory of to_image() on a large
canvas, old (full-size render) vs new (strips).

    python rust/harness/strip_render_check.py           (from a clone's root)
"""
import os
import sys
import tracemalloc

sys.path.insert(0, os.getcwd())
import numpy as np

import pinsel as ps
from pinsel import pigments as pg, path as P, watercolor as wc
from pinsel.spectral import linear_to_srgb

rng = np.random.default_rng(1)
cv = ps.Canvas(360, 250, ground='canvas', seed=3)
cv.tone(pg.mix(pg.BURNT_SIENNA, 1, pg.YELLOW_OCHRE, 1).thinned(0.3), thickness=0.25)
hog = ps.Brush('hog', 10, seed=2)
for i in range(60):
    hog.load(pg.mix(pg.ULTRAMARINE, 1, pg.TITANIUM_WHITE, 2).impasto(1.6), amount=1.4)
    x, y = rng.uniform(0, 360), rng.uniform(0, 160)
    cv.stroke(hog, P.line((x, y), (x + rng.uniform(20, 60), y + rng.normal(0, 8))), pressure=0.9)
cv.dry()
for i in range(25):     # wet, glossy strokes over dried impasto
    hog.load(pg.mix(pg.CADMIUM_YELLOW, 1, pg.TITANIUM_WHITE, 1).impasto(1.2), amount=1.2)
    x, y = rng.uniform(0, 360), rng.uniform(60, 200)
    cv.stroke(hog, P.line((x, y), (x + 40, y + 10)), pressure=0.85)
wc.wash(cv, (150, 120, 340, 240), pg.ULTRAMARINE, concentration=0.4, water=0.9, seed=4)

full = cv.render()
small = dict(_STRIP_CELLS=cv.W * 23)            # 23-row strips
for k, v in small.items():
    setattr(cv, k, v)
pieces = np.concatenate([cv.render(region=(y0, y1, 0, cv.W)) for y0, y1 in cv._strips()])
n_strips = len(list(cv._strips()))
assert pieces.dtype == full.dtype and pieces.shape == full.shape
print(f'render: {n_strips} strips, identical bits: {np.array_equal(pieces.view(np.uint32), full.view(np.uint32))}')

img_new = np.asarray(cv.to_image())
img_old = (full * 255 + 0.5).astype(np.uint8)
print(f'to_image: identical: {np.array_equal(img_new, img_old)}')


def preview_old(c, scale=0.5, light_dir=(-0.55, -0.65, 0.75)):
    f = max(1, int(round(1.0 / scale)))
    h, w = (c.H // f) * f, (c.W // f) * f
    rgb = c.render_linear()[:h, :w].reshape(h // f, f, w // f, f, 3).mean(axis=(1, 3))
    hgt = c.surface_height()[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))
    gy, gx = np.gradient(hgt.astype(np.float64))
    k = 0.9 / f
    L = np.asarray(light_dir, float); L /= np.linalg.norm(L)
    nl = np.sqrt((gx * k) ** 2 + (gy * k) ** 2 + 1)
    diff = np.maximum((-gx * k * L[0] - gy * k * L[1] + L[2]) / nl, 0)
    shade = (0.35 + 0.65 * diff) / (0.35 + 0.65 * L[2])
    out = np.clip(linear_to_srgb(rgb * shade[..., None]), 0, 1)
    return (out * 255 + 0.5).astype(np.uint8)


for sc in (0.5, 1 / 3, 0.25):
    same = np.array_equal(np.asarray(cv.preview(sc)), preview_old(cv, sc))
    print(f'preview scale {sc:.3f}: identical: {same}')

# memory: to_image on a large canvas, old way vs strips
big = ps.Canvas(3000, 2000, ground='canvas', seed=5)
b = ps.Brush('flat', 40, seed=1)
for y in range(100, 1900, 120):
    b.load(pg.mix(pg.SAP_GREEN, 1, pg.TITANIUM_WHITE, 1).impasto(1.3), 1.2)
    big.stroke(b, P.line((100, y), (2900, y + 30)), pressure=0.8)
big.render(region=(0, 8, 0, 8))                  # warm up kernels / caches outside the measurement
tracemalloc.start()
old = (big.render() * 255 + 0.5).astype(np.uint8)
peak_old = tracemalloc.get_traced_memory()[1]
tracemalloc.stop()
del old
tracemalloc.start()
new = np.asarray(big.to_image())
peak_new = tracemalloc.get_traced_memory()[1]
tracemalloc.stop()
print(f'3000x2000 to_image peak traced memory: full-size {peak_old / 2**20:.0f} MB, strips {peak_new / 2**20:.0f} MB')
