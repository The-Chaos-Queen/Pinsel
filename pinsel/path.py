"""
Stroke paths and pressure profiles.

A path is an (N, 2) array of (x, y) points, or (N, 3) with a pressure column.
All builders return plain numpy arrays so you can manipulate them freely.

    >>> from pinsel import path as P
    >>> pts = P.spline([(100, 400), (300, 250), (520, 380), (700, 200)])
    >>> pts = P.wobble(pts, amount=2.0, seed=3)
    >>> canvas.stroke(brush, pts, pressure=P.taper(0.1, 0.3))
"""
from __future__ import annotations

import numpy as np

from .noise import noise1d


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------
def _n_for(length, density=1.0):
    return max(2, int(length * density) + 2)


def line(p0, p1, n=None):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    n = n or _n_for(np.linalg.norm(p1 - p0))
    t = np.linspace(0, 1, n)[:, None]
    return p0 + (p1 - p0) * t


def polyline(points):
    return np.asarray(points, dtype=float)


def bezier(p0, c0, c1, p1, n=None):
    """Cubic Bezier curve."""
    p0, c0, c1, p1 = (np.asarray(p, float) for p in (p0, c0, c1, p1))
    approx = np.linalg.norm(c0 - p0) + np.linalg.norm(c1 - c0) + np.linalg.norm(p1 - c1)
    n = n or _n_for(approx)
    t = np.linspace(0, 1, n)[:, None]
    mt = 1 - t
    return mt ** 3 * p0 + 3 * mt * mt * t * c0 + 3 * mt * t * t * c1 + t ** 3 * p1


def quad(p0, c, p1, n=None):
    """Quadratic Bezier curve."""
    p0, c, p1 = (np.asarray(p, float) for p in (p0, c, p1))
    n = n or _n_for(np.linalg.norm(c - p0) + np.linalg.norm(p1 - c))
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * c + t * t * p1


def spline(points, closed=False, samples_per_px=1.0, alpha=0.5):
    """Centripetal Catmull-Rom spline through ``points`` (keeps extra columns,
    e.g. pressure, interpolated along)."""
    P = np.asarray(points, dtype=float)
    if len(P) < 2:
        return P
    if closed:
        P = np.vstack([P[-1], P, P[0], P[1]])
    else:
        P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]

        def tj(ti, a, b):
            return ti + max(np.linalg.norm(b[:2] - a[:2]), 1e-6) ** alpha
        t0 = 0.0
        t1 = tj(t0, p0, p1)
        t2 = tj(t1, p1, p2)
        t3 = tj(t2, p2, p3)
        seg = np.linalg.norm(p2[:2] - p1[:2])
        n = max(2, int(seg * samples_per_px) + 1)
        t = np.linspace(t1, t2, n, endpoint=(i == len(P) - 3))[:, None]
        A1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        A2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        A3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
        B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
        out.append((t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2)
    return np.vstack(out)


def arc(center, radius, a0, a1, n=None, ry=None):
    """Circular (or elliptical with ``ry``) arc, angles in degrees."""
    cx, cy = center
    ry = radius if ry is None else ry
    a0, a1 = np.radians(a0), np.radians(a1)
    n = n or _n_for(abs(a1 - a0) * max(radius, ry))
    a = np.linspace(a0, a1, n)
    return np.column_stack([cx + radius * np.cos(a), cy + ry * np.sin(a)])


def circle(center, radius, start=0.0, n=None):
    return arc(center, radius, start, start + 360.0, n)


def spiral(center, r0, r1, turns=3.0, start=0.0):
    cx, cy = center
    length = np.pi * (r0 + r1) * turns
    n = _n_for(length)
    t = np.linspace(0, 1, n)
    a = np.radians(start) + 2 * np.pi * turns * t
    r = r0 + (r1 - r0) * t
    return np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])


def wave(p0, p1, amplitude=10.0, wavelength=60.0, phase=0.0):
    """Sine wave travelling from p0 to p1."""
    base = line(p0, p1)
    d = np.asarray(p1, float) - np.asarray(p0, float)
    L = np.linalg.norm(d)
    nrm = np.array([-d[1], d[0]]) / (L + 1e-9)
    s = np.linspace(0, L, len(base))
    return base + nrm * (amplitude * np.sin(2 * np.pi * s / wavelength + phase))[:, None]


def zigzag(p0, p1, amplitude=10.0, count=6):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    nrm = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
    pts = [p0 + d * (i / (2 * count)) + nrm * amplitude * (1 if i % 2 else -1) * (0 < i < 2 * count)
           for i in range(2 * count + 1)]
    return spline(pts, alpha=0.0) if count > 0 else line(p0, p1)


def scribble(center, radius, length=400.0, seed=None, smooth=25.0):
    """A wandering, hand-drawn looking path inside a circle."""
    g = np.random.default_rng(seed)
    n = int(length)
    ang = np.cumsum(noise1d(n, scale=smooth, seed=g) * 0.08)
    x, y = np.zeros(n), np.zeros(n)
    px, py = 0.0, 0.0
    for i in range(n):
        px += np.cos(ang[i]); py += np.sin(ang[i])
        r = np.hypot(px, py)
        if r > radius:
            ang[i:] += np.pi * 0.7
            px *= radius / r; py *= radius / r
        x[i], y[i] = px, py
    return np.column_stack([x + center[0], y + center[1]])


def concat(*paths):
    return np.vstack([np.asarray(p, float) for p in paths])


# --------------------------------------------------------------------------
# Modifiers
# --------------------------------------------------------------------------
def arclength(pts):
    pts = np.asarray(pts, float)
    d = np.sqrt((np.diff(pts[:, :2], axis=0) ** 2).sum(axis=1))
    return np.concatenate([[0.0], np.cumsum(d)])


def resample(pts, spacing):
    """Resample to (approximately) equal arc-length ``spacing`` (all columns)."""
    pts = np.asarray(pts, float)
    s = arclength(pts)
    L = s[-1]
    if L < 1e-9:
        return pts[:1].repeat(2, axis=0)
    n = max(2, int(np.ceil(L / spacing)) + 1)
    si = np.linspace(0, L, n)
    keep = np.concatenate([[True], np.diff(s) > 1e-12])
    s, pts = s[keep], pts[keep]
    return np.column_stack([np.interp(si, s, pts[:, k]) for k in range(pts.shape[1])])


def wobble(pts, amount=1.5, scale=40.0, seed=None):
    """Add smooth hand tremor perpendicular to the path."""
    pts = np.asarray(pts, float).copy()
    if len(pts) < 3 or amount <= 0:
        return pts
    s = arclength(pts)
    L = max(s[-1], 1.0)
    n = int(L) + 2
    nz = noise1d(n, scale=scale, seed=seed)
    off = np.interp(s, np.arange(n), nz) * amount
    d = np.gradient(pts[:, :2], axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    pts[:, 0] += -d[:, 1] * off
    pts[:, 1] += d[:, 0] * off
    return pts


def transform(pts, translate=(0, 0), rotate=0.0, scale=1.0, origin=(0, 0)):
    pts = np.asarray(pts, float).copy()
    o = np.asarray(origin, float)
    a = np.radians(rotate)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    pts[:, :2] = (pts[:, :2] - o) @ R.T * scale + o + np.asarray(translate, float)
    return pts


def reverse(pts):
    return np.asarray(pts)[::-1].copy()


# --------------------------------------------------------------------------
# Pressure profiles: functions t (0..1 along the stroke) -> pressure (0..1)
# --------------------------------------------------------------------------
def _smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def constant(p=0.7):
    return lambda t: np.full_like(np.asarray(t, float), p)


def taper(attack=0.12, release=0.25, peak=0.8, start=0.05, end=0.0):
    """Press in, hold, lift off -- the most natural stroke."""
    def f(t):
        t = np.asarray(t, float)
        a = start + (peak - start) * _smooth(t / max(attack, 1e-6))
        r = end + (peak - end) * _smooth((1 - t) / max(release, 1e-6))
        return np.minimum(a, r)
    return f


def ramp(p0=0.2, p1=0.9):
    return lambda t: p0 + (p1 - p0) * np.asarray(t, float)


def swell(lo=0.25, hi=0.95, center=0.5, width=0.35):
    """Thin - thick - thin (calligraphic swell)."""
    return lambda t: lo + (hi - lo) * np.exp(-((np.asarray(t, float) - center) / width) ** 2)


def flick(peak=0.9, hold=0.15):
    """Heavy touch down, then a fast lift (grass blades, hair, sparkles)."""
    def f(t):
        t = np.asarray(t, float)
        return peak * np.where(t < hold, 0.6 + 0.4 * _smooth(t / max(hold, 1e-6)),
                               (1 - _smooth((t - hold) / (1 - hold))) ** 1.3)
    return f


def dab(peak=0.9):
    """Press-and-lift on a short path (stippling, leaves)."""
    return lambda t: peak * np.sin(np.pi * np.clip(np.asarray(t, float), 0, 1)) ** 0.6


def jittered(profile, amount=0.08, scale=0.08, seed=None):
    """Add slow random variation (an unsteady hand) to a profile."""
    g = np.random.default_rng(seed)
    knots = g.normal(0, 1, 64)

    def f(t):
        t = np.asarray(t, float)
        v = np.interp(t / scale, np.arange(64), knots, period=64)
        return np.clip(profile(t) * (1 + amount * v), 0, 1)
    return f
