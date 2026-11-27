"""Procedural noise helpers (numpy; filters via pinsel._filters)."""
from __future__ import annotations

import numpy as np
from . import _filters
from . import _filters as ndimage
from ._filters import KDTree as cKDTree


def rng(seed=None):
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def normalize(a):
    a = a - a.mean()
    s = a.std()
    return a / s if s > 0 else a


def spectral_noise(shape, beta=2.0, lowcut=0.0, highcut=None, seed=None, aniso=(1.0, 1.0)):
    """Gaussian noise with power spectrum ~ 1/f^beta (FFT synthesis).

    ``aniso`` = (sy, sx) frequency weights: a value > 1 suppresses variation
    along that axis, i.e. features become *elongated* along it
    (``aniso=(1, 8)`` -> long horizontal streaks).  Returns zero-mean,
    unit-std float32.
    """
    g = rng(seed)
    h, w = shape
    fy = _filters.fftfreq(h)[:, None] * aniso[0]
    fx = _filters.rfftfreq(w)[None, :] * aniso[1]
    f = np.sqrt(fx * fx + fy * fy)
    f[0, 0] = 1.0
    amp = f ** (-beta / 2.0)
    if lowcut > 0:
        amp *= 1.0 - np.exp(-(f / lowcut) ** 2)
    if highcut is not None:
        amp *= np.exp(-(f / highcut) ** 2)
    amp[0, 0] = 0.0
    spec = (g.standard_normal(amp.shape) + 1j * g.standard_normal(amp.shape)) * amp
    out = _filters.irfft2(spec, s=shape)
    return normalize(out).astype(np.float32)


def fbm(shape, scale=64.0, octaves=5, persistence=0.5, seed=None):
    """Fractal value noise (sum of smoothly upsampled random grids)."""
    g = rng(seed)
    h, w = shape
    out = np.zeros(shape, dtype=np.float64)
    amp, s, total = 1.0, float(scale), 0.0
    for _ in range(octaves):
        gh, gw = int(np.ceil(h / s)) + 3, int(np.ceil(w / s)) + 3
        grid = g.standard_normal((gh, gw))
        z = ndimage.zoom(grid, s, order=3, mode='grid-wrap')
        oy, ox = g.integers(0, max(1, int(s))), g.integers(0, max(1, int(s)))
        out += amp * z[oy:oy + h, ox:ox + w]
        total += amp
        amp *= persistence
        s = max(1.0, s / 2.0)
    return normalize(out).astype(np.float32)


def worley(shape, cell=12.0, seed=None, jitter=1.0):
    """Distance to the nearest feature point (cellular noise), normalised 0..1."""
    g = rng(seed)
    h, w = shape
    n = max(4, int(h * w / (cell * cell)))
    pts = np.column_stack([g.uniform(-cell, h + cell, n), g.uniform(-cell, w + cell, n)])
    tree = cKDTree(pts)
    yy, xx = np.mgrid[0:h, 0:w]
    d, _ = tree.query(np.column_stack([yy.ravel(), xx.ravel()]), k=1, workers=-1)
    d = d.reshape(h, w)
    return (d / d.max()).astype(np.float32)


def noise1d(n, scale=20.0, seed=None, octaves=3):
    """Smooth 1-D noise, zero mean, roughly unit amplitude."""
    g = rng(seed)
    out = np.zeros(n)
    amp, s = 1.0, float(scale)
    x = np.arange(n)
    for _ in range(octaves):
        m = int(n / s) + 4
        pts = g.standard_normal(m)
        out += amp * np.interp(x / s, np.arange(m), pts)
        amp *= 0.5
        s = max(1.0, s / 2)
    sd = out.std()
    return out / sd if sd > 0 else out


def blur(a, sigma):
    return ndimage.gaussian_filter(a, sigma, mode='reflect')


def fibers(shape, count=4000, length=(6, 30), seed=None, width=0.6):
    """Random short curved fibres (for paper), returns 0..1 float32."""
    g = rng(seed)
    h, w = shape
    img = np.zeros(shape, dtype=np.float32)
    for _ in range(count):
        L = g.uniform(*length)
        x, y = g.uniform(0, w), g.uniform(0, h)
        a = g.uniform(0, np.pi)
        curv = g.normal(0, 0.05)
        steps = int(L)
        for _s in range(steps):
            xi, yi = int(x), int(y)
            if 0 <= xi < w and 0 <= yi < h:
                img[yi, xi] += 1.0
            x += np.cos(a); y += np.sin(a); a += curv
    img = blur(img, width)
    return img / (img.max() + 1e-9)
