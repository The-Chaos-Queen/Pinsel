"""
Painting grounds: the surface you paint on.

Each ground provides
  * a height field (surface relief, in pinsel height units; 1 unit is
    roughly the thickness of one normal oil stroke),
  * a base spectral reflectance (colour of the gesso / paper / linen),
  * a few physical parameters (absorbency for watercolour).

Available kinds:
  'canvas'        primed cotton duck, plain weave
  'linen'         fine, irregular linen weave with slubs
  'coarse_canvas' heavy, open weave
  'cold_press'    watercolour paper, medium tooth (alias 'paper')
  'rough'         watercolour paper, strong tooth
  'hot_press'     smooth watercolour paper
  'panel'         gessoed wooden panel with faint brush marks
  'smooth'        perfectly flat (for tests)
"""
from __future__ import annotations

import numpy as np

from . import noise as nz
from .spectral import rgb_to_spectrum, NB


class Ground:
    def __init__(self, height, reflectance, kind, relief, absorbency=0.5, sizing=0.5):
        self.height = height.astype(np.float32)          # in height units
        self.reflectance = np.asarray(reflectance, dtype=np.float32)  # (NB,)
        self.kind = kind
        self.relief = relief
        self.absorbency = absorbency    # how fast paper soaks up water (watercolour)
        self.sizing = sizing            # resistance to pigment staining / lifting ease


def _weave(shape, period, g, irregular=0.15, slub=0.25, gap=0.08, twill=False):
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    # threads wander a little
    wx = nz.fbm(shape, scale=period * 9, octaves=3, seed=g) * irregular * period
    wy = nz.fbm(shape, scale=period * 9, octaves=3, seed=g) * irregular * period
    u = (xx + wx) / period
    v = (yy + wy) / period
    iu, iv = np.floor(u), np.floor(v)
    fu, fv = u - iu, v - iv
    # per-thread thickness variation (slubs): noise along each thread
    tw = 1.0 + slub * nz.fbm(shape, scale=period * 6, octaves=2, seed=g)
    th = 1.0 + slub * nz.fbm(shape, scale=period * 6, octaves=2, seed=g)
    half = 0.5 - gap
    pw = np.clip(1.0 - ((fu - 0.5) / np.maximum(half * tw, 1e-6)) ** 2, 0, 1) ** 0.6   # warp (vertical threads)
    pf = np.clip(1.0 - ((fv - 0.5) / np.maximum(half * th, 1e-6)) ** 2, 0, 1) ** 0.6   # weft (horizontal threads)
    shift = (iu + iv) if not twill else (iu + 2 * iv)
    lw = 0.5 + 0.5 * np.cos(np.pi * (v + iu + 0.5 * twill * iv))
    lf = 0.5 - 0.5 * np.cos(np.pi * (u + iv))
    hw = pw * (0.45 + 0.55 * lw)
    hf = pf * (0.45 + 0.55 * lf)
    height = np.maximum(hw, hf)
    del shift
    return height


def make_ground(kind, shape, color=None, relief=None, scale=1.0, seed=0):
    g = np.random.default_rng(seed)
    h, w = shape
    kind = {'paper': 'cold_press'}.get(kind, kind)
    absorb, sizing = 0.2, 0.5
    if kind in ('canvas', 'coarse_canvas', 'linen'):
        if kind == 'canvas':
            period, irr, slub, rel, col = 5.0, 0.22, 0.35, 0.4, '#efeadf'
        elif kind == 'coarse_canvas':
            period, irr, slub, rel, col = 8.5, 0.18, 0.3, 0.9, '#ece6d9'
        else:
            period, irr, slub, rel, col = 3.6, 0.25, 0.45, 0.45, '#ede7da'
        H = _weave(shape, period * scale, g, irregular=irr, slub=slub)
        # gesso partly fills the weave
        H = 0.75 * H + 0.25 * nz.blur(H, 1.2 * scale)
        H += 0.08 * nz.spectral_noise(shape, beta=1.2, seed=g)
        H = nz.blur(H, 0.5)
    elif kind in ('cold_press', 'rough', 'hot_press'):
        cell = {'cold_press': 9.0, 'rough': 16.0, 'hot_press': 5.0}[kind] * scale
        rel = {'cold_press': 0.55, 'rough': 1.0, 'hot_press': 0.12}[kind]
        wor = nz.worley(shape, cell=cell, seed=g)
        bumps = 1.0 - wor / (wor.max() + 1e-9)
        bumps = nz.blur(bumps, cell * 0.15)
        fine = nz.spectral_noise(shape, beta=1.6, seed=g, highcut=0.25)
        H = nz.normalize(bumps) * 0.8 + 0.35 * fine
        H += 0.4 * nz.fibers(shape, count=int(h * w / 500), seed=g)
        H += 0.25 * nz.fbm(shape, scale=cell * 6, octaves=3, seed=g)
        H = (H - H.min()) / (H.max() - H.min())
        col = '#f6f2e9'
        absorb = {'cold_press': 0.5, 'rough': 0.6, 'hot_press': 0.35}[kind]
    elif kind == 'panel':
        marks = nz.spectral_noise(shape, beta=2.2, seed=g, aniso=(1.0, 12.0))
        H = 0.3 * marks + 0.15 * nz.spectral_noise(shape, beta=1.0, seed=g)
        rel, col = 0.12, '#f1ede4'
    elif kind == 'smooth':
        H = np.zeros(shape)
        rel, col = 0.0, '#f2efe8'
    else:
        raise ValueError(f'unknown ground kind {kind!r}')
    H = H - H.min()
    if H.max() > 0:
        H = H / H.max()
    rel = rel if relief is None else relief
    R = rgb_to_spectrum(color if color is not None else col)
    return Ground(H * rel, R, kind, rel, absorbency=absorb, sizing=sizing)
