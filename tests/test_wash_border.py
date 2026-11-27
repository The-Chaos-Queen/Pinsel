"""A rough-edged wash over the whole canvas reaches the canvas border.

region_mask used to smooth the rough edge with morphology that treated the
outside of the array as unpainted, which ate the outermost row and column and
left white dashes along the border of every rough wash (found 2026-11-27).
"""
import numpy as np

import pinsel as ps
from pinsel import pigments as pg, watercolor as wc


def test_rough_wash_reaches_the_border():
    cv = ps.Canvas(160, 110, ground='cold_press', seed=1)
    wc.wash(cv, None, pg.INDIGO.as_medium('watercolor'), concentration=1.5, water=0.7, rough=0.5, seed=1)
    cv.wait(30)
    cv.dry()
    a = np.asarray(cv.to_image()).astype(int).sum(axis=2)
    light = a > 400                                   # unpainted paper is far above, indigo far below
    assert not light.any(), f'{int(light.sum())} unpainted pixels, rows {sorted(set(np.nonzero(light)[0]))[:6]}'


def test_interior_edge_stays_rough():
    cv = ps.Canvas(160, 110, ground='cold_press', seed=1)
    wc.wash(cv, wc.rect(30, 30, 130, 80), pg.INDIGO.as_medium('watercolor'), concentration=1.5, water=0.7, rough=0.5, seed=1)
    cv.wait(30)
    cv.dry()
    a = np.asarray(cv.to_image()).astype(int).sum(axis=2)
    painted_per_row = [(a[y, 30:130] < 400).sum() for y in range(25, 35)]
    partial = [n for n in painted_per_row if 0 < n < 100]
    assert partial, f'the wash edge should wander along the paper grain, rows: {painted_per_row}'
