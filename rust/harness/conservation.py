"""Paint conservation for one stroke: total optical depth K per spectral band.

  PINSEL_BACKEND=numba|rust venv/bin/python rust/harness/conservation.py

Two strokes on a fixed 256x256 canvas, one loaded brush each:
  oil        flat brush, paste paint (ground 'canvas')
  watercolor round brush, watercolour medium (ground 'cold_press')

Total K per band = brush (res_K + tip_K) + canvas (film_K + base_K), and for
watercolour also canvas.wc.K + canvas.wc.dK. It is summed before and after one
stroke, and the relative change per band is printed. Target: about 1e-5.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pinsel as ps  # noqa: E402
from pinsel import _backend, path as P, pigments as pg  # noqa: E402
from pinsel.spectral import NB  # noqa: E402

TARGET = 1e-5
STROKE = P.line((60, 128), (196, 128))


def totals(cv, brush):
    """Return (brush K, canvas K) as float64 arrays of shape (NB,)."""
    brush_K = brush.res_K.sum(axis=0, dtype=np.float64) + brush.tip_K.sum(axis=0, dtype=np.float64)
    canvas_K = (cv.film_K.sum(axis=(0, 1), dtype=np.float64)
                + cv.base_K.sum(axis=(0, 1), dtype=np.float64))
    if cv.wc is not None:
        canvas_K += cv.wc.K.sum(axis=(0, 1), dtype=np.float64) + cv.wc.dK.sum(axis=(0, 1), dtype=np.float64)
    return brush_K, canvas_K


def run(label, ground, brush, watercolor):
    cv = ps.Canvas(256, 256, ground=ground, seed=5)
    if watercolor:
        cv.ensure_watercolor()  # so wc arrays exist (all zero) before the stroke
    brush.load(pg.ULTRAMARINE.as_medium('watercolor') if watercolor else pg.ULTRAMARINE)
    b0, c0 = totals(cv, brush)
    cv.stroke(brush, STROKE)
    b1, c1 = totals(cv, brush)
    before = b0 + c0
    after = b1 + c1
    print(f"\n{label}: ground={ground}")
    print(f"{'band':>4s} {'before':>14s} {'brush after':>14s} {'canvas after':>14s} {'after':>14s} {'rel change':>12s}")
    rel = np.full(NB, np.nan)
    for i in range(NB):
        if before[i] != 0:
            rel[i] = (after[i] - before[i]) / before[i]
        print(f"{i:4d} {before[i]:14.6f} {b1[i]:14.6f} {c1[i]:14.6f} {after[i]:14.6f} {rel[i]:12.3e}")
    worst = np.nanmax(np.abs(rel))
    print(f"max |rel change| = {worst:.3e}  ->  {'OK' if worst <= TARGET else 'ABOVE 1e-5'}")
    return worst


def main():
    print(f"backend: {_backend.NAME}")
    flat = ps.Brush('flat', 14, seed=11)
    round_wc = ps.Brush('round', 14, seed=11)
    w1 = run('oil flat stroke (paste)', 'canvas', flat, watercolor=False)
    w2 = run('watercolour round stroke', 'cold_press', round_wc, watercolor=True)
    print(f"\nworst over both strokes: {max(w1, w2):.3e}")


if __name__ == '__main__':
    main()
