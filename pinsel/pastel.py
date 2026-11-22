"""Dry powdery drawing media: soft pastel, charcoal, and kneaded eraser.

Powdery media deposit matte pigment that catches the paper tooth at light
pressure, fills the grain at firm pressure, leaves a subtle dust halo, and
remains smudgeable / liftable (with :class:`pinsel.tools.Smudge` or
:class:`KneadedEraser`) until locked down with :meth:`pinsel.Canvas.fix`.
"""
from __future__ import annotations

import math
import numpy as np
from numba import njit

from . import pigments as pg
from .path import arclength, resample, taper, wobble as _wobble
from .pigments import Paint
from .spectral import NB


@njit(cache=True, fastmath=True)
def _clamp01(v):
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


@njit(cache=True, fastmath=True)
def _pastel_stroke(px, py, pp, pa, rx_base, ry_base, hardness, dust, amount,
                   Kpv, Spv, film_K, film_S, film_vol, wet, gloss, body, drate,
                   tooth, mask, base_K, base_S, base_vol, top_id, sid, seed):
    H, W = film_vol.shape
    nb = film_K.shape[2]
    M = px.shape[0]
    bx0, by0, bx1, by1 = W, H, -1, -1
    # simple deterministic LCG for powdery grain & dust micro-specks
    state = np.uint64(seed * 6364136223846793005 + 1442695040888963407)

    for j in range(M):
        cx = px[j]
        cy = py[j]
        p = pp[j]
        if p <= 1e-4:
            continue
        a = pa[j]
        ca = math.cos(a)
        sa = math.sin(a)
        scale = 0.45 + 0.55 * p
        rx = rx_base * scale
        ry = ry_base * scale
        dust_pad = 1.0 + 0.55 * dust
        rmax = max(rx, ry) * dust_pad + 1.0
        x0 = max(0, int(math.floor(cx - rmax)))
        x1 = min(W - 1, int(math.ceil(cx + rmax)))
        y0 = max(0, int(math.floor(cy - rmax)))
        y1 = min(H - 1, int(math.ceil(cy + rmax)))
        if x1 < x0 or y1 < y0:
            continue
        if x0 < bx0:
            bx0 = x0
        if y0 < by0:
            by0 = y0
        if x1 > bx1:
            bx1 = x1
        if y1 > by1:
            by1 = y1

        # Light pressure catches only high tooth peaks; firm pressure reaches valleys
        level = 0.95 - 1.95 * p - 0.35 * (1.0 - hardness)
        soft = 0.35 + 0.35 * (1.0 - hardness)
        inv_rx2 = 1.0 / (rx * rx + 1e-9)
        inv_ry2 = 1.0 / (ry * ry + 1e-9)
        inv_drx2 = 1.0 / ((rx * dust_pad) ** 2 + 1e-9)
        inv_dry2 = 1.0 / ((ry * dust_pad) ** 2 + 1e-9)

        for Y in range(y0, y1 + 1):
            dy = Y - cy
            for X in range(x0, x1 + 1):
                m = 1.0 - mask[Y, X]
                if m <= 0.0:
                    continue
                dx = X - cx
                lx = dx * ca + dy * sa
                ly = -dx * sa + dy * ca
                d2 = lx * lx * inv_rx2 + ly * ly * inv_ry2
                if d2 < 1.0:
                    fall = (1.0 - d2) ** 1.3
                    t = tooth[Y, X]
                    contact = _clamp01((t - level) / soft + 0.5)
                    # crumbly grain variation
                    state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
                    rnd = float((state >> np.uint64(33)) & np.uint64(0x7FFFFFFF)) / 2147483647.0
                    grain = 0.65 + 0.7 * rnd
                    dv = 0.14 * amount * (0.25 + 0.75 * p) * fall * contact * grain * m
                elif dust > 0.0:
                    dd2 = lx * lx * inv_drx2 + ly * ly * inv_dry2
                    if dd2 >= 1.0:
                        continue
                    state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
                    rnd = float((state >> np.uint64(33)) & np.uint64(0x7FFFFFFF)) / 2147483647.0
                    if rnd > 0.28 * dust * (1.0 - dd2):
                        continue
                    dv = 0.035 * amount * dust * (1.0 - dd2) * p * m
                else:
                    continue

                if dv <= 1e-6:
                    continue
                if top_id[Y, X] != sid:
                    if film_vol[Y, X] > 0.0 or film_K[Y, X, 0] > 0.0:
                        for b in range(nb):
                            base_K[Y, X, b] += film_K[Y, X, b]
                            base_S[Y, X, b] += film_S[Y, X, b]
                            film_K[Y, X, b] = 0.0
                            film_S[Y, X, b] = 0.0
                        base_vol[Y, X] += film_vol[Y, X]
                        film_vol[Y, X] = 0.0
                    top_id[Y, X] = sid

                fv = film_vol[Y, X]
                nv = fv + dv
                for b in range(nb):
                    film_K[Y, X, b] += Kpv[b] * dv
                    film_S[Y, X, b] += Spv[b] * dv
                # Matte, unfixed powdery deposit (wet=1 keeps it smudgeable until fix())
                wet[Y, X] = (wet[Y, X] * fv + 1.0 * dv) / nv
                gloss[Y, X] = (gloss[Y, X] * fv + 0.0 * dv) / nv
                body[Y, X] = (body[Y, X] * fv + 0.04 * dv) / nv
                drate[Y, X] = (drate[Y, X] * fv + 0.0 * dv) / nv
                film_vol[Y, X] = nv
    return bx0, by0, bx1, by1


@njit(cache=True, fastmath=True)
def _eraser_stroke(px, py, pp, r_base, strength, film_K, film_S, film_vol,
                   base_K, base_S, base_vol, mask):
    H, W = film_vol.shape
    nb = film_K.shape[2]
    M = px.shape[0]
    bx0, by0, bx1, by1 = W, H, -1, -1
    for j in range(M):
        cx = px[j]
        cy = py[j]
        p = pp[j]
        if p <= 1e-4:
            continue
        r = r_base * (0.5 + 0.5 * p)
        r2 = r * r
        x0 = max(0, int(math.floor(cx - r)))
        x1 = min(W - 1, int(math.ceil(cx + r)))
        y0 = max(0, int(math.floor(cy - r)))
        y1 = min(H - 1, int(math.ceil(cy + r)))
        if x1 < x0 or y1 < y0:
            continue
        if x0 < bx0:
            bx0 = x0
        if y0 < by0:
            by0 = y0
        if x1 > bx1:
            bx1 = x1
        if y1 > by1:
            by1 = y1
        for Y in range(y0, y1 + 1):
            dy = Y - cy
            for X in range(x0, x1 + 1):
                dx = X - cx
                d2 = dx * dx + dy * dy
                if d2 >= r2:
                    continue
                m = 1.0 - mask[Y, X]
                if m <= 0.0:
                    continue
                fall = 1.0 - d2 / r2
                lift = _clamp01(strength * p * fall * fall * m * 0.45)
                if lift <= 1e-6:
                    continue
                keep = 1.0 - lift
                if film_vol[Y, X] > 0.0:
                    for b in range(nb):
                        film_K[Y, X, b] *= keep
                        film_S[Y, X, b] *= keep
                    film_vol[Y, X] *= keep
                if base_vol[Y, X] > 0.0:
                    for b in range(nb):
                        base_K[Y, X, b] *= keep
                        base_S[Y, X, b] *= keep
                    base_vol[Y, X] *= keep
    return bx0, by0, bx1, by1


class SoftPastel:
    """Chalky soft pastel stick.

    Draws on the tip (``side=0``) or broad side (``side=1``), catching the
    paper tooth on light passes and filling the grain on firm passes. Stays
    smudgeable and liftable until :meth:`pinsel.Canvas.fix` is called.
    """

    def __init__(self, size=10.0, hardness=0.35, dust=0.35, side=0.0, seed=None):
        self.size = float(size)
        self.hardness = float(np.clip(hardness, 0.0, 1.0))
        self.dust = float(np.clip(dust, 0.0, 1.0))
        self.side = float(np.clip(side, 0.0, 1.0))
        self.rng = np.random.default_rng(seed)
        self.paint: Paint | None = None
        self.amount = 1.0

    def load(self, paint, amount=1.0):
        """Set the pigment carried by the stick (converted to matte gouache-like scatter)."""
        p = paint if isinstance(paint, Paint) else pg.paint(paint)
        self.paint = p.as_medium('gouache') if p.medium != 'gouache' else p.copy()
        self.amount = float(amount)
        return self

    def stroke(self, canvas, path, pressure=None, angle=None, wobble=0.0, seed=None):
        """Drag the pastel stick along ``path``."""
        if self.paint is None:
            raise ValueError('pastel stick has no colour loaded; call .load(paint) first')
        pts = np.asarray(path, dtype=float)
        if pts.ndim == 1:
            pts = pts[None, :]
        if pts.ndim != 2 or len(pts) < 1:
            return None
        spacing = max(0.8, self.size * 0.18)
        if len(pts) == 1 or arclength(pts)[-1] < 0.5:
            p0 = pts[0, :2]
            xy = np.repeat(p0[None, :], 4, axis=0)
            p_val = float(pts[0, 2]) if pts.shape[1] >= 3 else (
                0.7 if pressure is None else (float(pressure(0.5)) if callable(pressure) else float(pressure)))
            pp = np.full(4, np.clip(p_val, 0.0, 1.0), dtype=np.float64)
            pa = np.full(4, math.radians(0.0 if angle is None else float(angle)), dtype=np.float64)
        else:
            if wobble > 0:
                pts = _wobble(pts, amount=wobble, scale=max(18.0, 2.5 * self.size), seed=seed)
            rs = resample(pts, spacing)
            s = arclength(rs)
            t = s / max(s[-1], 1e-9)
            if pts.shape[1] >= 3 and pressure is None:
                pp = rs[:, 2]
            elif pressure is None:
                pp = taper(0.1, 0.2, 0.8)(t)
            elif callable(pressure):
                pp = pressure(t)
            elif np.ndim(pressure) == 1:
                pp = np.interp(t, np.linspace(0, 1, len(pressure)), np.asarray(pressure, float))
            else:
                pp = np.full(len(t), float(pressure))
            pp = np.clip(np.asarray(pp, dtype=np.float64), 0.0, 1.0)
            xy = rs[:, :2]
            if angle is None:
                d = np.gradient(xy, axis=0)
                pa = np.arctan2(d[:, 1], d[:, 0]) + math.pi / 2
            elif callable(angle):
                pa = np.radians(np.asarray(angle(t), dtype=np.float64))
            else:
                pa = np.full(len(t), math.radians(float(angle)), dtype=np.float64)

        if not (pp > 1e-4).any():
            return None
        rx = 0.5 * self.size * (1.0 + 2.2 * self.side)
        ry = 0.5 * self.size * (1.0 - 0.35 * self.side)
        st_seed = int(self.rng.integers(1, 2 ** 31 - 1)) if seed is None else int(seed)
        bbox = _pastel_stroke(
            np.ascontiguousarray(xy[:, 0], dtype=np.float64),
            np.ascontiguousarray(xy[:, 1], dtype=np.float64),
            np.ascontiguousarray(pp, dtype=np.float64),
            np.ascontiguousarray(pa, dtype=np.float64),
            rx, ry, self.hardness, self.dust, self.amount,
            np.ascontiguousarray(self.paint.K, dtype=np.float64),
            np.ascontiguousarray(self.paint.S, dtype=np.float64),
            canvas.film_K, canvas.film_S, canvas.film_vol, canvas.wet,
            canvas.gloss, canvas.body, canvas.drate, canvas.tooth, canvas.mask,
            canvas.base_K, canvas.base_S, canvas.base_vol, canvas.top_id,
            int(canvas.stroke_id), st_seed,
        )
        canvas._after_stroke()
        return bbox


class Charcoal(SoftPastel):
    """Vine or compressed charcoal stick, pre-loaded with deep matte carbon."""

    def __init__(self, size=8.0, hardness=0.3, dust=0.45, side=0.0, compressed=False, seed=None):
        super().__init__(size=size, hardness=hardness, dust=dust, side=side, seed=seed)
        base = pg.mix(pg.MARS_BLACK, 2.0, pg.IVORY_BLACK, 1.0) if compressed else pg.mix(
            pg.IVORY_BLACK, 2.0, pg.RAW_UMBER, 0.4)
        self.load(base, amount=1.25 if compressed else 0.85)


class KneadedEraser:
    """A pliable eraser that lifts unfixed charcoal, pastel, or wet paint."""

    def __init__(self, size=14.0, strength=0.65, seed=None):
        self.size = float(size)
        self.strength = float(np.clip(strength, 0.0, 1.0))
        self.rng = np.random.default_rng(seed)

    def stroke(self, canvas, path, pressure=None):
        """Lift unfixed pigment along ``path`` (respects ``canvas.mask``)."""
        pts = np.asarray(path, dtype=float)
        if pts.ndim == 1:
            pts = pts[None, :]
        if pts.ndim != 2 or len(pts) < 1:
            return None
        spacing = max(1.0, self.size * 0.25)
        if len(pts) == 1 or arclength(pts)[-1] < 0.5:
            xy = np.repeat(pts[0:1, :2], 3, axis=0)
            p_val = 0.8 if pressure is None else float(pressure)
            pp = np.full(3, np.clip(p_val, 0.0, 1.0), dtype=np.float64)
        else:
            rs = resample(pts, spacing)
            s = arclength(rs)
            t = s / max(s[-1], 1e-9)
            if pressure is None:
                pp = taper(0.15, 0.2, 0.85)(t)
            elif callable(pressure):
                pp = pressure(t)
            else:
                pp = np.full(len(t), float(pressure))
            pp = np.clip(np.asarray(pp, dtype=np.float64), 0.0, 1.0)
            xy = rs[:, :2]
        bbox = _eraser_stroke(
            np.ascontiguousarray(xy[:, 0], dtype=np.float64),
            np.ascontiguousarray(xy[:, 1], dtype=np.float64),
            np.ascontiguousarray(pp, dtype=np.float64),
            0.5 * self.size, self.strength,
            canvas.film_K, canvas.film_S, canvas.film_vol,
            canvas.base_K, canvas.base_S, canvas.base_vol, canvas.mask,
        )
        canvas._after_stroke()
        return bbox

    def dab(self, canvas, x, y, pressure=0.8):
        """Blot/lift a highlight at ``(x, y)``."""
        return self.stroke(canvas, [(x, y)], pressure=pressure)
