"""
Non-brush painting tools.

Every tool here works on the same physical canvas state as the bristle
brushes (see :mod:`pinsel.canvas`): paste paints (oil, acrylic, gouache) are
exchanged with the wet film (``film_K/film_S/film_vol`` + properties), fluid
paints (watercolour, ink) with the watercolour layer (``canvas.wc``).  All
tools respect ``canvas.mask`` and call ``canvas._after_stroke()`` once per
user-level mark.

Tools
-----
:class:`PaletteKnife`
    rigid steel blade.  ``spread`` lays flat slabs of thick paint with crisp
    edges, squeeze-out ridges and broken "skips" where the blade rides over
    relief; ``scrape`` levels wet impasto and collects the paint on the
    blade; ``sgraffito`` scratches lines through wet paint.
:class:`Airbrush`
    gaussian spray cone, Poisson droplet statistics (speckle at the fringe),
    coarse spatter, overspray halo; pressure -> flow, distance -> spread.
:class:`Splatter`
    flicked / tapped / toothbrush spatter: ballistic drops with a size
    distribution, oblique-impact elongation, satellites and streaks.
:class:`Sponge`
    natural sea sponge: porous multi-scale cell structure, pressure dependent
    contact, rotation per dab, load depletion, pick-up / blotting.
:class:`OilCrayon`
    a solid pigment stick used on its tip or side. Light pressure catches
    the grain; heavier passes lay down a denser, burnished film.
:class:`Pen`
    technical pen (constant width) and dip pens (broad-edge calligraphy nib,
    pointed flexible nib with railroading), ink pooling, depletion, tooth
    skipping and fibre feathering.
:class:`Smudge` / :class:`Blender`
    finger smudging that drags and softens wet paint without adding any;
    a dry-brush blender wrapper around :class:`pinsel.brush.Brush`.
:func:`glaze`, :class:`Stencil`, :func:`stipple`
    uniform transparent layers with soft edges, masking tape / stencils /
    friskets (context managers on ``canvas.mask``), stippling with blue-noise
    dot placement.

Quick example::

    import pinsel as ps
    from pinsel import tools as T, pigments as pg
    cv = ps.Canvas(800, 600)
    knife = T.PaletteKnife(length=50).load(pg.CADMIUM_YELLOW, 1.2)
    knife.spread(cv, ps.path.line((100, 300), (400, 320)), pressure=0.5)
    T.Airbrush().load(pg.ULTRAMARINE).spray(cv, ps.path.line((50, 100), (700, 100)))
"""
from __future__ import annotations

import math
import numpy as np
from ._jit import njit
from . import _filters as ndimage

from .path import resample, arclength, wobble as _wobble
from .pigments import Paint
from . import _backend as _B
from .spectral import NB

__all__ = ['PaletteKnife', 'Airbrush', 'Splatter', 'Sponge', 'OilCrayon', 'Pen', 'Smudge', 'Blender',
           'Stencil', 'glaze', 'stipple', 'blue_noise_points']


# ==========================================================================
# Shared helpers
# ==========================================================================
def _rng(seed):
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def _paint_props(paint):
    """(fluid, gloss, body, drate, granulation, staining) of a paint."""
    return dict(fluid=bool(paint.fluid), gloss=float(paint.gloss), body=float(paint.body),
                drate=1.0 / max(float(paint.dry_time), 1e-3),
                gran=float(paint.granulation), stain=float(paint.staining))


def _mix_paints(paint):
    """Accept a Paint or [(paint, weight), ...]; return (main, K, S) of the mix."""
    paints = paint if isinstance(paint, (list, tuple)) else [(paint, 1.0)]
    paints = [(p, 1.0) if isinstance(p, Paint) else p for p in paints]
    w = np.array([float(x[1]) for x in paints])
    w = w / w.sum()
    K = sum(p.K * wi for (p, _), wi in zip(paints, w))
    S = sum(p.S * wi for (p, _), wi in zip(paints, w))
    main = paints[int(np.argmax(w))][0]
    return main, np.asarray(K, float), np.asarray(S, float), paints, w


def _pressure_array(pressure, t, default):
    if pressure is None:
        pressure = default
    if callable(pressure):
        pp = np.asarray(pressure(t), float)
    elif np.ndim(pressure) == 0:
        pp = np.full(len(t), float(pressure))
    else:
        pr = np.asarray(pressure, float)
        pp = np.interp(t, np.linspace(0, 1, len(pr)), pr)
    return np.clip(np.broadcast_to(pp, t.shape).astype(float), 0.0, 1.0)


def _prep_path(path, spacing, pressure=None, default_pressure=0.7, wobble=0.0,
               wobble_scale=40.0, smooth=4.0, seed=None):
    """Resample a user path like :meth:`Brush.stroke` does.

    Returns ``xy (M,2), s (M,) arclength, t (M,), pp (M,) pressure,
    phi (M,) smoothed travel direction (radians)`` or None.
    A third path column is used as pressure when ``pressure`` is None.
    """
    pts = np.asarray(path, dtype=float)
    if pts.ndim != 2 or len(pts) < 1:
        return None
    if len(pts) == 1:
        extra = [0.0] * (pts.shape[1] - 2)
        pts = np.vstack([pts, pts + np.array([0.01, 0.0] + extra)])
    has_p = pts.shape[1] >= 3
    if wobble > 0:
        pts = _wobble(pts, amount=wobble, scale=wobble_scale, seed=seed)
    rs = resample(pts, spacing)
    s = arclength(rs)
    L = max(s[-1], 1e-9)
    t = s / L
    if pressure is None and has_p:
        pp = np.clip(rs[:, 2], 0, 1)
    else:
        pp = _pressure_array(pressure, t, default_pressure)
    xy = np.ascontiguousarray(rs[:, :2])
    d = np.gradient(xy, axis=0) if len(xy) > 1 else np.array([[1.0, 0.0]])
    phi = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    k = max(1, int(smooth / spacing))
    if len(phi) > 2 * k + 1:
        ker = np.ones(2 * k + 1) / (2 * k + 1)
        phi = np.convolve(np.pad(phi, k, mode='edge'), ker, mode='valid')
    return xy, s, t, np.ascontiguousarray(pp), np.ascontiguousarray(phi)


def _angle_array(angle, t, phi, rel_deg):
    """Tool orientation: None -> relative to motion by ``rel_deg``; number ->
    fixed absolute degrees; callable(t) -> absolute degrees."""
    if angle is None:
        return phi + math.radians(rel_deg)
    if callable(angle):
        return np.radians(np.asarray(angle(t), float)) * np.ones_like(t)
    return np.full(len(t), math.radians(float(angle)))


def _finish(canvas, bbox, fluid, pad=6):
    """Grow the watercolour bbox for fluid deposits, count the mark."""
    if fluid and bbox is not None and canvas.wc is not None:
        x0, y0, x1, y1 = (int(v) for v in bbox)
        if x1 >= x0 and y1 >= y0:
            canvas.wc.grow_bbox(x0, y0, x1, y1, canvas.H, canvas.W, pad=pad)
    canvas._after_stroke()
    return bbox


def _aniso_noise(shape, sigma, g):
    """Smoothed gaussian noise, zero mean, unit std (float64)."""
    n = ndimage.gaussian_filter(g.standard_normal(shape), sigma, mode='wrap')
    sd = n.std()
    return n / sd if sd > 0 else n


@njit(cache=True)
def _nb_seed(s):
    np.random.seed(s)


@njit(cache=True, fastmath=True, inline='always')
def _cl01(v):
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


@njit(cache=True, fastmath=True, inline='always')
def _sstep(v):
    v = _cl01(v)
    return v * v * (3.0 - 2.0 * v)


from .canvas import LEVELING as _LEV  # keep in sync with Canvas.surface_height


def _film(canvas):
    """The wet paste film as one tuple passed to kernels:
    (film_K, film_S, film_vol, base_K, base_S, base_vol, top_id, wet, gloss, body, drate).
    ``film_*`` is the top sub-layer (most recent mark), ``base_*`` older wet
    paint underneath; properties are shared and weighted by total volume."""
    return (canvas.film_K, canvas.film_S, canvas.film_vol, canvas.base_K, canvas.base_S,
            canvas.base_vol, canvas.top_id, canvas.wet, canvas.gloss, canvas.body, canvas.drate)


@njit(cache=True, fastmath=True, inline='always')
def _surface_top(hg, hd, fv, bd):
    """Visible surface height (same formula as Canvas.surface_height);
    ``fv`` = total wet volume (top + base)."""
    ph = hd + fv * bd
    return hg * math.exp(-_LEV * ph) + ph


@njit(cache=True, fastmath=True, inline='always')
def _solid_top(hg, hd):
    """Height of ground + dried paint (what a rigid tool rides on)."""
    return hg * math.exp(-_LEV * hd) + hd


@njit(cache=True, fastmath=True, inline='always')
def _solve_paint_height(hg, B, lo):
    """Paint height ph >= lo with hg*exp(-_LEV*ph) + ph == B.

    f(ph) = ph + hg*exp(-_LEV*ph) is convex; on its increasing branch
    (ph >= ph_min) Newton iteration started right of the root converges
    monotonically.  If even ph = max(lo, ph_min) gives a surface above B
    (blade below the ground's texture), all wet paint goes: returns lo.
    """
    pm = lo
    khg = _LEV * hg
    if khg > 1.0:
        pmin = math.log(khg) / _LEV
        if pmin > pm:
            pm = pmin
    if pm + hg * math.exp(-_LEV * pm) >= B:
        return lo
    ph = B if B > pm else pm
    for _ in range(10):
        e = hg * math.exp(-_LEV * ph)
        fp = 1.0 - _LEV * e
        if fp < 1e-6:
            break
        ph -= (ph + e - B) / fp
        if ph < pm:
            ph = pm
            break
    return ph if ph > lo else lo


@njit(cache=True, fastmath=True, inline='always')
def _push(F, sid, y, x):
    """A new mark lies on top of older wet paint: push the pixel's top layer
    down into the base layer (same rule as the core's paste_stroke)."""
    tid = F[6]
    if tid[y, x] != sid:
        fK = F[0]
        fS = F[1]
        fV = F[2]
        if fV[y, x] > 0.0 or fK[y, x, 0] > 0.0:
            bK = F[3]
            bS = F[4]
            for b in range(fK.shape[2]):
                bK[y, x, b] += fK[y, x, b]
                bS[y, x, b] += fS[y, x, b]
                fK[y, x, b] = 0.0
                fS[y, x, b] = 0.0
            F[5][y, x] += fV[y, x]
            fV[y, x] = 0.0
        tid[y, x] = sid


@njit(cache=True, fastmath=True, inline='always')
def _dep(F, sid, y, x, dv, Kpv, Spv, p_gloss, p_body, p_drate):
    """Deposit ``dv`` volume of fresh paint (per-volume K/S) on top of the pixel."""
    _dep_w(F, sid, y, x, dv, Kpv, Spv, 1.0, p_gloss, p_body, p_drate)


@njit(cache=True, fastmath=True, inline='always')
def _dep_w(F, sid, y, x, dv, Kpv, Spv, p_wet, p_gloss, p_body, p_drate):
    """Deposit ``dv`` volume of paint with wetness ``p_wet`` on top of the pixel."""
    if dv <= 0.0:
        return
    _push(F, sid, y, x)
    fK = F[0]
    fS = F[1]
    fV = F[2]
    fv = fV[y, x] + F[5][y, x]
    nv = fv + dv
    for b in range(fK.shape[2]):
        fK[y, x, b] += Kpv[b] * dv
        fS[y, x, b] += Spv[b] * dv
    wet = F[7]
    gl = F[8]
    bd = F[9]
    dr = F[10]
    wet[y, x] = (wet[y, x] * fv + p_wet * dv) / nv
    gl[y, x] = (gl[y, x] * fv + p_gloss * dv) / nv
    bd[y, x] = (bd[y, x] * fv + p_body * dv) / nv
    dr[y, x] = (dr[y, x] * fv + p_drate * dv) / nv
    fV[y, x] += dv


@njit(cache=True, fastmath=True, inline='always')
def _take(F, y, x, dv, rK, rS, ri):
    """Remove volume ``dv`` of wet paint (top layer first, then the older base
    layer) and add its K/S to row ``ri`` of ``rK``/``rS``.  Returns the volume
    actually removed."""
    if dv <= 0.0:
        return 0.0
    fK = F[0]
    fS = F[1]
    fV = F[2]
    bK = F[3]
    bS = F[4]
    bV = F[5]
    ft = fV[y, x]
    fb = bV[y, x]
    dt = dv if dv < ft else ft
    db = dv - dt
    if db > fb:
        db = fb
    frt = dt / ft if ft > 1e-12 else 0.0
    frb = db / fb if fb > 1e-12 else 0.0
    for b in range(fK.shape[2]):
        k1 = fK[y, x, b] * frt
        s1 = fS[y, x, b] * frt
        k2 = bK[y, x, b] * frb
        s2 = bS[y, x, b] * frb
        fK[y, x, b] -= k1
        fS[y, x, b] -= s1
        bK[y, x, b] -= k2
        bS[y, x, b] -= s2
        rK[ri, b] += k1 + k2
        rS[ri, b] += s1 + s2
    fV[y, x] = ft - dt
    bV[y, x] = fb - db
    return dt + db


@njit(cache=True, fastmath=True)
def _gauss_dep_bin(F, sid, mask, ex, ey, sig, dvt, rK, rS, rV, ri, p_gloss, p_body, p_drate, bb):
    """Deposit ``dvt`` volume from reservoir row ``ri`` as a gaussian blob
    (ridge segment) centred at (ex, ey).  Updates bbox ``bb``; returns used."""
    vb = rV[ri]
    if vb <= 1e-9 or dvt <= 1e-10:
        return 0.0
    H, W = mask.shape
    nb = rK.shape[1]
    if dvt > vb:
        dvt = vb
    rr = int(math.ceil(2.5 * sig))
    ix = int(round(ex))
    iy = int(round(ey))
    i2 = 0.5 / (sig * sig)
    wsum = 0.0
    for Y in range(iy - rr, iy + rr + 1):
        for X in range(ix - rr, ix + rr + 1):
            wsum += math.exp(-((X - ex) ** 2 + (Y - ey) ** 2) * i2)
    if wsum <= 1e-12:
        return 0.0
    kK = np.empty(nb)
    kS = np.empty(nb)
    for b in range(nb):
        kK[b] = rK[ri, b] / vb
        kS[b] = rS[ri, b] / vb
    used = 0.0
    for Y in range(iy - rr, iy + rr + 1):
        for X in range(ix - rr, ix + rr + 1):
            if X < 0 or Y < 0 or X >= W or Y >= H:
                continue
            dv = dvt * math.exp(-((X - ex) ** 2 + (Y - ey) ** 2) * i2) / wsum * (1.0 - mask[Y, X])
            if dv <= 1e-10:
                continue
            used += dv
            _dep(F, sid, Y, X, dv, kK, kS, p_gloss, p_body, p_drate)
            if X < bb[0]:
                bb[0] = X
            if Y < bb[1]:
                bb[1] = Y
            if X > bb[2]:
                bb[2] = X
            if Y > bb[3]:
                bb[3] = Y
    fu = used / vb
    for b in range(nb):
        rK[ri, b] *= (1.0 - fu)
        rS[ri, b] *= (1.0 - fu)
    rV[ri] = vb - used
    return used


@njit(cache=True, fastmath=True, inline='always')
def _wc_add(wK, wS, wwat, wm, wg, wst, y, x, water, m, Kpv, Spv, gran, stain):
    """Add ``water`` and pigment mass ``m`` (suspended) to the watercolour layer."""
    if m > 0.0:
        for b in range(wK.shape[2]):
            wK[y, x, b] += Kpv[b] * m
            wS[y, x, b] += Spv[b] * m
        wm[y, x] += m
        wg[y, x] += m * gran
        wst[y, x] += m * stain
    if water > 0.0:
        wwat[y, x] += water


@njit(cache=True, fastmath=True, inline='always')
def _wc_settle_add(dK, dS, dm, dg, dst, y, x, m, Kpv, Spv, gran, stain):
    """Add pigment mass ``m`` directly to the deposited (settled) layer."""
    if m <= 0.0:
        return
    for b in range(dK.shape[2]):
        dK[y, x, b] += Kpv[b] * m
        dS[y, x, b] += Spv[b] * m
    dm[y, x] += m
    dg[y, x] += m * gran
    dst[y, x] += m * stain


@njit(cache=True, fastmath=True)
def _apply_volume_map(vol, ox, oy, Kpv, Spv, p_gloss, p_body, p_drate, F, sid, mask):
    """Deposit a bbox-local paint volume map on top of the wet film (respects mask)."""
    h, w = vol.shape
    H, W = mask.shape
    for i in range(h):
        Y = oy + i
        if Y < 0 or Y >= H:
            continue
        for j in range(w):
            X = ox + j
            if X < 0 or X >= W:
                continue
            dv = vol[i, j] * (1.0 - mask[Y, X])
            if dv > 1e-9:
                _dep(F, sid, Y, X, dv, Kpv, Spv, p_gloss, p_body, p_drate)


@njit(cache=True, fastmath=True)
def _apply_fluid_map(vol, ox, oy, water_ratio, settle_frac, Kpv, Spv, gran, stain,
                     wK, wS, wwat, wm, wg, wst, dK, dS, dm, dg, dst, mask):
    """Deposit a bbox-local pigment-mass map into the watercolour layer:
    ``settle_frac`` of it lands as dried-on-contact pigment, the rest is
    suspended in ``water_ratio * mass`` of water."""
    h, w = vol.shape
    H, W = wwat.shape
    mx, my = max(0, ox), max(0, oy)
    for i in range(h):
        Y = oy + i
        if Y < 0 or Y >= H:
            continue
        for j in range(w):
            X = ox + j
            if X < 0 or X >= W:
                continue
            m = vol[i, j] * (1.0 - mask[Y - my, X - mx])
            if m <= 1e-10:
                continue
            ms = m * settle_frac
            _wc_settle_add(dK, dS, dm, dg, dst, Y, X, ms, Kpv, Spv, gran, stain)
            _wc_add(wK, wS, wwat, wm, wg, wst, Y, X, m * water_ratio, m - ms, Kpv, Spv, gran, stain)


_apply_volume_map = _B.port('tools_apply_volume_map', _apply_volume_map, mutable={8: 'F'})
_apply_fluid_map = _B.port('tools_apply_fluid_map', _apply_fluid_map, mutable={
    9: 'wK', 10: 'wS', 11: 'wwat', 12: 'wm', 13: 'wg', 14: 'wst', 15: 'dK', 16: 'dS', 17: 'dm',
    18: 'dg', 19: 'dst'})


def _deposit_map(canvas, vol, ox, oy, paint_K, paint_S, props, body_scale=1.0,
                 water_ratio=3.0, settle_frac=0.0, gloss_scale=1.0):
    """Deposit a bbox-local volume map either on top of the film or into the
    watercolour layer (fluid paints: ``vol`` is pigment mass)."""
    vol = np.ascontiguousarray(vol, dtype=np.float64)
    Kp = np.ascontiguousarray(paint_K, dtype=np.float64)
    Sp = np.ascontiguousarray(paint_S, dtype=np.float64)
    if props['fluid']:
        wc = canvas.ensure_watercolor()
        h, w = vol.shape
        region = np.s_[max(0, oy):max(0, oy + h), max(0, ox):max(0, ox + w)]
        _apply_fluid_map(vol, int(ox), int(oy), float(water_ratio), float(settle_frac), Kp, Sp,
                         props['gran'], props['stain'], wc.K, wc.S, wc.water, wc.m, wc.g, wc.st,
                         wc.dK, wc.dS, wc.dm, wc.dg, wc.dst, canvas.watercolor_mask(region))
    else:
        _apply_volume_map(vol, int(ox), int(oy), Kp, Sp, props['gloss'] * gloss_scale,
                          props['body'] * body_scale, props['drate'], _film(canvas),
                          int(canvas.stroke_id), canvas.mask)


class _Reservoir:
    """A well-mixed paint load (used by sponge, pen, airbrush, splatter)."""

    def __init__(self):
        self.K = np.zeros(NB)
        self.S = np.zeros(NB)
        self.vol = 0.0
        self.cap = 1.0
        self.water = 0.0
        self.paint = None
        self.props = dict(fluid=False, gloss=0.8, body=1.0, drate=1e-5, gran=0.0, stain=0.3)

    def load(self, paint, amount, cap, water=1.0, clean=True):
        main, Km, Sm, _, _ = _mix_paints(paint)
        if clean:
            self.K[:] = 0
            self.S[:] = 0
            self.vol = 0.0
            self.water = 0.0
        self.paint = main
        self.props = _paint_props(main)
        self.cap = float(cap)
        v = float(cap) * float(amount)
        self.K += Km * v
        self.S += Sm * v
        self.vol += v
        self.water += float(cap) * float(water)
        return self

    @property
    def Kpv(self):
        return self.K / self.vol if self.vol > 1e-12 else np.zeros(NB)

    @property
    def Spv(self):
        return self.S / self.vol if self.vol > 1e-12 else np.zeros(NB)

    @property
    def level(self):
        return self.vol / self.cap if self.cap > 0 else 0.0

    def take(self, v):
        """Remove volume ``v`` (returns the fraction actually taken)."""
        if self.vol <= 1e-12:
            return 0.0
        f = min(1.0, v / self.vol)
        self.K *= (1 - f)
        self.S *= (1 - f)
        self.vol *= (1 - f)
        return f

    def clean(self):
        self.K[:] = 0
        self.S[:] = 0
        self.vol = 0.0
        self.water = 0.0


# ==========================================================================
# Palette knife
# ==========================================================================
@njit(cache=True, fastmath=True, inline='always')
def _bins_shift(res_K, res_S, res_vol, src, dst, fr):
    """Move fraction ``fr`` of knife bin ``src`` into bin ``dst``."""
    if fr <= 0.0:
        return
    for b in range(res_K.shape[1]):
        dk = res_K[src, b] * fr
        ds = res_S[src, b] * fr
        res_K[src, b] -= dk
        res_S[src, b] -= ds
        res_K[dst, b] += dk
        res_S[dst, b] += ds
    dv = res_vol[src] * fr
    res_vol[src] -= dv
    res_vol[dst] += dv


@njit(cache=True, fastmath=True)
def _knife_kernel(px, py, pp, pa, ps, mode, half, nk, res_K, res_S, res_vol, cap, prm,
                  lump_tex, stri, chat, flex_n, p_gloss, p_body, p_drate,
                  F, sid, h_ground, h_dry, mask, smax, sox, soy):
    """Sweep a rigid blade along the path.

    mode 0 = spread (lay paint), 1 = scrape.  The blade is a segment of half
    length ``half`` at angle ``pa`` (direction along the blade).  A pixel is
    processed once, when the blade's trailing edge passes over it.  The blade
    level follows the highest solid relief under it (flexibility ``flex_n``
    in blade samples); paint is filled up to the blade level where the
    knife's (lumpy) paint load reaches the surface, wet paint above the blade
    is scraped onto the knife.  The knife's load lives in bins along the
    blade, migrates along an angled blade (snow-plough) and squeezes out at
    the blade ends as ridges.
    """
    thickness = prm[0]
    float_k = prm[1]
    reach_base = prm[2]
    reach_press = prm[3]
    lumpiness = prm[4]
    stri_amp = prm[5]
    ride = prm[6]
    edge_w = prm[7]
    transfer = prm[8]
    pickup = prm[9]
    squeeze = prm[10]
    plow = prm[11]
    ridge_w = prm[12]
    leave = prm[13]
    scrape_eff = prm[14]
    lift_frac = prm[15]
    touch_frac = prm[16]
    lift_noise = prm[17]
    spill_cap = prm[18]
    diffuse = prm[19]
    bite = prm[20]
    fV = F[2]
    bV = F[5]
    wet = F[7]
    body = F[9]
    H, W = fV.shape
    nb = res_K.shape[1]
    M = px.shape[0]
    nbin = res_vol.shape[0]
    L = 2.0 * half
    NU = lump_tex.shape[0]
    NS = lump_tex.shape[1]
    bb = np.array([W, H, -1, -1])
    lev = np.empty(nk)
    wlev = np.zeros(nk)
    C = np.empty(nk)
    tmp = np.empty(nk)
    kK = np.empty(nb)
    kS = np.empty(nb)
    pnx, pny, pcx, pcy = 0.0, 0.0, 0.0, 0.0
    soft = 0.12 * thickness + 0.03
    for j in range(M):
        cx = px[j]
        cy = py[j]
        p = pp[j]
        a = pa[j]
        bxd = math.cos(a)
        byd = math.sin(a)
        j0 = j - 1 if j > 0 else 0
        j1 = j if j > 0 else (1 if M > 1 else 0)
        mx = px[j1] - px[j0]
        my = py[j1] - py[j0]
        step = math.sqrt(mx * mx + my * my)
        if step > 1e-9:
            mx /= step
            my /= step
        else:
            mx, my = -byd, bxd
        nx = -byd
        ny = bxd
        if nx * mx + ny * my < 0.0:
            nx = -nx
            ny = -ny
        bm = bxd * mx + byd * my          # cos(angle between blade and motion)
        # rigid blade: one float height from the blade-averaged load
        lsum = 0.0
        csum = 0.0
        for ib in range(nbin):
            lsum += res_vol[ib]
            csum += cap[ib]
        lmean = min(lsum / (csum + 1e-12), 1.5)
        tkm = thickness * lmean ** 0.7 if lmean > 0.0 else 0.0
        if j == 0:
            pnx, pny, pcx, pcy = nx, ny, cx, cy
            # touch down: a little paint squeezed out behind the blade
            if mode == 0 and touch_frac > 0.0:
                for k in range(nk):
                    u = -half + L * (k + 0.5) / nk
                    ib = min(max(int((u + half) / L * nbin), 0), nbin - 1)
                    vb = res_vol[ib]
                    if vb <= 1e-9:
                        continue
                    ldt = min(vb / (cap[ib] + 1e-12), 1.5)
                    X = int(round(cx + bxd * u - nx * 0.7))
                    Y = int(round(cy + byd * u - ny * 0.7))
                    if X < 0 or Y < 0 or X >= W or Y >= H:
                        continue
                    dv = touch_frac * thickness * ldt ** 0.7 * (0.5 + 0.5 * p) * (1.0 - mask[Y, X])
                    if dv <= 0.0:
                        continue
                    fd = min(dv / vb, 0.5)
                    dv = vb * fd
                    for b in range(nb):
                        kK[b] = res_K[ib, b] / vb
                        kS[b] = res_S[ib, b] / vb
                        res_K[ib, b] -= res_K[ib, b] * fd
                        res_S[ib, b] -= res_S[ib, b] * fd
                    res_vol[ib] = vb - dv
                    _dep(F, sid, Y, X, dv, kK, kS, p_gloss, p_body, p_drate)
            continue
        if pnx * nx + pny * ny < 0.0:
            pnx = -pnx
            pny = -pny
        # scraping: the sharp, pressed edge bites a little below the ridges
        f = tkm * float_k * (1.0 - p) + 0.02 if mode == 0 else leave - bite * p
        # ---- blade level along the blade: rides on the highest solid relief;
        #      a lightly held blade also rides on soft wet paint (it lays its
        #      load on top), a firmly pressed / scraping one cuts through
        rf = min(ride * math.sqrt(max(1.0 - p, 0.0)), 1.0) if mode == 0 else 0.0
        for k in range(nk):
            u = -half + L * k / (nk - 1)
            # the blade face rests on what lies just ahead of its trailing edge
            lv = -1e9
            wlev[k] = 0.0
            for ahead in (1.5, 4.0):
                X = int(round(cx + bxd * u + nx * ahead))
                Y = int(round(cy + byd * u + ny * ahead))
                if X < 0 or Y < 0 or X >= W or Y >= H:
                    continue
                hd = h_dry[Y, X]
                hg = h_ground[Y, X]
                sol = _solid_top(hg, hd)
                top = _surface_top(hg, hd, fV[Y, X] + bV[Y, X], body[Y, X])
                if sol > lv:
                    lv = sol
                wv = (top - sol) * rf
                if wv > wlev[k]:
                    wlev[k] = wv
            lev[k] = lv
        # rigid-ish blade rests on the highest solid points within its flex
        # window; soft wet paint only supports it locally
        for k in range(nk):
            mxv = -1e9
            for q in range(max(0, k - flex_n), min(nk, k + flex_n + 1)):
                if lev[q] > mxv:
                    mxv = lev[q]
            tmp[k] = mxv
        for k in range(nk):
            sm = 0.0
            cnt = 0
            for q in range(max(0, k - flex_n), min(nk, k + flex_n + 1)):
                if tmp[q] > -1e8:
                    sm += tmp[q]
                    cnt += 1
            sw = 0.0
            cw = 0
            for q in range(max(0, k - 3), min(nk, k + 4)):
                sw += wlev[q]
                cw += 1
            C[k] = (sm / cnt if cnt > 0 else 0.0) + (sw / cw if cw > 0 else 0.0)
        # ---- bbox of the swept quad
        xa = min(pcx, cx) - abs(bxd) * half - 2
        xb = max(pcx, cx) + abs(bxd) * half + 2
        ya = min(pcy, cy) - abs(byd) * half - 2
        yb = max(pcy, cy) + abs(byd) * half + 2
        x0 = max(int(math.floor(xa)), 0)
        x1 = min(int(math.ceil(xb)), W - 1)
        y0 = max(int(math.floor(ya)), 0)
        y1 = min(int(math.ceil(yb)), H - 1)
        isj = min(int(ps[j]), NS - 1)
        chv = chat[min(isj, chat.shape[0] - 1)]
        for Y in range(y0, y1 + 1):
            for X in range(x0, x1 + 1):
                dc = (X - cx) * nx + (Y - cy) * ny
                if dc > 0.0:
                    continue
                dp = (X - pcx) * pnx + (Y - pcy) * pny
                if dp <= 0.0 or dc < -(step + 2.5):
                    continue
                u = (X - cx) * bxd + (Y - cy) * byd
                au = abs(u)
                if au > half + edge_w:
                    continue
                cov = _cl01((half - au) / edge_w + 0.5)
                m = (1.0 - mask[Y, X]) * cov
                if m <= 0.0:
                    continue
                if X < bb[0]:
                    bb[0] = X
                if Y < bb[1]:
                    bb[1] = Y
                if X > bb[2]:
                    bb[2] = X
                if Y > bb[3]:
                    bb[3] = Y
                kf = min(max((u + half) / L * (nk - 1), 0.0), nk - 1.0)
                k0 = int(kf)
                k1 = min(k0 + 1, nk - 1)
                fk = kf - k0
                Cu = C[k0] * (1.0 - fk) + C[k1] * fk
                ib = min(max(int((u + half) / L * nbin), 0), nbin - 1)
                load = min(res_vol[ib] / (cap[ib] + 1e-12), 1.5)
                tk = thickness * load ** 0.7 if load > 0.0 else 0.0
                iu = min(max(int(u + half), 0), NU - 1)
                B = Cu + f + stri_amp * stri[iu] + chv
                hd = h_dry[Y, X]
                hg = h_ground[Y, X]
                if mode == 0:
                    # fresh knife paint lies on top of older wet paint
                    _push(F, sid, Y, X)
                fv = fV[Y, X] + bV[Y, X]
                bd = body[Y, X]
                top = _surface_top(hg, hd, fv, bd)
                gap = B - top
                if gap < 0.0:
                    # -- wet paint above the blade is scraped onto the knife
                    wh = fv * bd
                    if fv > 1e-7 and wh > 1e-9:
                        pht = _solve_paint_height(hg, B, hd)
                        nwh = max(pht - hd, 0.0)
                        fr = (wh - nwh) / wh * scrape_eff * m * _cl01(1.6 * wet[Y, X])
                        if fr > 0.98:
                            fr = 0.98
                        if fr > 0.0:
                            res_vol[ib] += _take(F, Y, X, fv * fr, res_K, res_S, ib)
                    continue
                if mode != 0:
                    continue
                # -- spread: knife paint bridges the gap where its lumpy load reaches
                lf = 1.6 - min(load, 1.0)
                tl = tk * (1.0 + lumpiness * lf * lump_tex[iu, isj])
                if tl < 0.0:
                    tl = 0.0
                wetc = wet[Y, X] if fv > 0.05 else 0.0
                R = f + tl * (reach_base + reach_press * p * p) * (1.0 + 0.4 * wetc)
                # viscous paint bridges the small valleys of the weave: contact
                # is decided against the local maximum of the solid relief
                topc = top
                iy2 = Y - soy
                ix2 = X - sox
                if 0 <= iy2 < smax.shape[0] and 0 <= ix2 < smax.shape[1]:
                    if smax[iy2, ix2] > topc:
                        topc = smax[iy2, ix2]
                c = _cl01((R - (B - topc)) / soft + 0.5)
                if c <= 0.0:
                    continue
                # pick-up of underlying wet paint (smearing / dirty knife)
                if pickup > 0.0 and fv > 1e-6:
                    pf = min(pickup * c * m * wet[Y, X], 0.5)
                    res_vol[ib] += _take(F, Y, X, fv * pf, res_K, res_S, ib)
                    fv = fV[Y, X] + bV[Y, X]
                    top = _surface_top(hg, hd, fv, bd)
                    gap = B - top
                    if gap <= 0.0:
                        continue
                vb = res_vol[ib]
                if vb <= 1e-9:
                    continue
                dv = min(gap / max(p_body, 0.05) * transfer * c * m, 0.35 * vb)
                if dv <= 1e-9:
                    continue
                fd = dv / vb
                for b in range(nb):
                    kK[b] = res_K[ib, b] / vb
                    kS[b] = res_S[ib, b] / vb
                    res_K[ib, b] -= res_K[ib, b] * fd
                    res_S[ib, b] -= res_S[ib, b] * fd
                res_vol[ib] = vb - dv
                _dep(F, sid, Y, X, dv, kK, kS, p_gloss, p_body, p_drate)
        # ---- pressing spreads the load along the blade a little
        if nbin > 1 and diffuse > 0.0:
            dq = min(diffuse * step * (0.3 + p), 0.24)
            for ib in range(nbin - 1):
                for b in range(nb):
                    dk = (res_K[ib, b] - res_K[ib + 1, b]) * dq
                    ds = (res_S[ib, b] - res_S[ib + 1, b]) * dq
                    res_K[ib, b] -= dk
                    res_K[ib + 1, b] += dk
                    res_S[ib, b] -= ds
                    res_S[ib + 1, b] += ds
                dv = (res_vol[ib] - res_vol[ib + 1]) * dq
                res_vol[ib] -= dv
                res_vol[ib + 1] += dv
        # ---- scraping: the pressed bead squeezes outwards along the blade
        if mode == 1 and nbin > 2:
            qo = min(0.06 * squeeze * step * p, 0.3)
            hb = nbin // 2
            for ib in range(1, hb):
                _bins_shift(res_K, res_S, res_vol, ib, ib - 1, qo)
            for ib in range(nbin - 2, nbin - 1 - hb, -1):
                _bins_shift(res_K, res_S, res_vol, ib, ib + 1, qo)
        # ---- paint migrates along an angled blade towards its trailing end
        if plow > 0.0 and nbin > 1:
            q = min(plow * abs(bm) * 0.16 * step / (L / nbin), 0.45)
            if bm > 0.0:      # end at u<0 trails: move towards lower bins
                for ib in range(1, nbin):
                    _bins_shift(res_K, res_S, res_vol, ib, ib - 1, q)
            else:
                for ib in range(nbin - 2, -1, -1):
                    _bins_shift(res_K, res_S, res_vol, ib, ib + 1, q)
        # ---- squeeze-out ridges at both blade ends
        for e in (-1, 1):
            ib = 0 if e < 0 else nbin - 1
            vb = res_vol[ib]
            if vb <= 1e-9:
                continue
            trail = max(-e * bm, 0.0)
            ld = vb / (cap[ib] + 1e-12)
            if mode == 0:
                tke = min(thickness * min(ld, 1.5) ** 0.7, 1.5 * tkm)
                dvt = squeeze * (0.3 + p) * step * tke * 0.5 * (1.0 + 3.0 * trail)
                if ld > spill_cap:
                    dvt += 0.02 * (ld - spill_cap) * cap[ib] * step
            else:
                # the scraped bead piles up at the blade ends
                dvt = vb * min(0.5, 0.05 * squeeze * step * (0.3 + p) * (1.0 + 2.0 * trail))
            dvt = min(dvt, 0.5 * vb)
            ex = cx + bxd * e * (half + 0.6 + ridge_w) - nx * 0.4
            ey = cy + byd * e * (half + 0.6 + ridge_w) - ny * 0.4
            _gauss_dep_bin(F, sid, mask, ex, ey, ridge_w, dvt, res_K, res_S, res_vol, ib,
                           p_gloss, p_body, p_drate, bb)
        pnx, pny, pcx, pcy = nx, ny, cx, cy
    # ---- lift-off: the paint bead under the blade stays behind as a ridge
    if lift_frac > 0.0 and M > 1:
        cx = px[M - 1]
        cy = py[M - 1]
        a = pa[M - 1]
        bxd = math.cos(a)
        byd = math.sin(a)
        ns = max(int(L * 2), 4)
        rw = ridge_w * 1.2
        for k in range(ns):
            u = -half + L * (k + 0.5) / ns
            ib = min(max(int((u + half) / L * nbin), 0), nbin - 1)
            vb = res_vol[ib]
            if vb <= 1e-9:
                continue
            iu = min(max(int(u + half), 0), NU - 1)
            nzv = max(1.0 + lift_noise * lump_tex[iu, NS - 1], 0.1)
            ldl = min(vb / (cap[ib] + 1e-12), 3.0)
            dvt = min(lift_frac * thickness * ldl ** 0.7 * 5.0 * (L / ns) * nzv, 0.6 * vb)
            _gauss_dep_bin(F, sid, mask, cx + bxd * u + pnx * 0.9, cy + byd * u + pny * 0.9,
                           rw, dvt, res_K, res_S, res_vol, ib, p_gloss, p_body, p_drate, bb)
    return bb[0], bb[1], bb[2], bb[3]


_knife_kernel = _B.port('tools_knife_kernel', _knife_kernel, mutable={
    8: 'res_K', 9: 'res_S', 10: 'res_vol', 20: 'F'})


@njit(cache=True, fastmath=True)
def _sgraffito_kernel(px, py, pp, r0, r1, valley, burr, carry, ragged, noise, nox, noy,
                      F, sid, h_ground, h_dry, mask):
    """Scratch a line through the wet film (both wet sub-layers) with a tip.

    Removed paint is pushed to both sides of the groove (burr ridges); a
    little is carried along and dropped where the tip lifts.  In the canvas
    weave's valleys below the tip some paint survives.  Half-dry paint tears
    out raggedly (``ragged`` with bbox-local ``noise``).
    """
    fV = F[2]
    bV = F[5]
    wet = F[7]
    gloss = F[8]
    body = F[9]
    drate = F[10]
    H, W = fV.shape
    nb = F[0].shape[2]
    M = px.shape[0]
    bb = np.array([W, H, -1, -1])
    crK = np.zeros((1, nb))     # carried crumb on the tip
    crS = np.zeros((1, nb))
    rem = np.zeros((1, nb))     # scratch row for one pixel's removal
    remS = np.zeros((1, nb))
    kK = np.empty(nb)
    kS = np.empty(nb)
    cv = 0.0
    cg = 0.0
    cb = 0.0
    cd = 0.0
    for j in range(M):
        cx = px[j]
        cy = py[j]
        p = pp[j]
        if p <= 0.0:
            continue
        r = r0 + (r1 - r0) * p
        j0 = j - 1 if j > 0 else 0
        j1 = j + 1 if j < M - 1 else M - 1
        mx = px[j1] - px[j0]
        my = py[j1] - py[j0]
        ml = math.sqrt(mx * mx + my * my)
        if ml > 1e-9:
            mx /= ml
            my /= ml
        else:
            mx, my = 1.0, 0.0
        nx = -my
        ny = mx
        x0 = max(int(math.floor(cx - r - 1)), 0)
        x1 = min(int(math.ceil(cx + r + 1)), W - 1)
        y0 = max(int(math.floor(cy - r - 1)), 0)
        y1 = min(int(math.ceil(cy + r + 1)), H - 1)
        if x1 < x0 or y1 < y0:
            continue
        # contact level of the tip: highest solid point under it
        ctop = -1e9
        for Y in range(y0, y1 + 1):
            for X in range(x0, x1 + 1):
                if (X - cx) ** 2 + (Y - cy) ** 2 <= r * r:
                    s = _solid_top(h_ground[Y, X], h_dry[Y, X])
                    if s > ctop:
                        ctop = s
        for Y in range(y0, y1 + 1):
            for X in range(x0, x1 + 1):
                dx = X - cx
                dy = Y - cy
                d = math.sqrt(dx * dx + dy * dy)
                if d > r + 0.7:
                    continue
                m = 1.0 - mask[Y, X]
                fv = fV[Y, X] + bV[Y, X]
                if m <= 0.0 or fv <= 1e-7:
                    continue
                cov = _cl01(r - d + 0.5)
                deep = ctop - _solid_top(h_ground[Y, X], h_dry[Y, X])
                keep = _cl01(deep / (valley + 1e-6) - 0.3) * 0.85
                wv = wet[Y, X]
                nzv = 0.0
                iy = Y - noy
                ix = X - nox
                if 0 <= iy < noise.shape[0] and 0 <= ix < noise.shape[1]:
                    nzv = noise[iy, ix]
                # drier paint tears raggedly instead of parting cleanly
                tear = _cl01((1.0 - wv) * ragged * 2.0)
                thr = _cl01(1.0 - tear * (0.5 + 0.5 * nzv))
                fr = min(cov * m * (1.0 - keep) * thr * (0.4 + 0.6 * p), 0.995)
                if fr <= 0.0:
                    continue
                pg = gloss[Y, X]
                pb = body[Y, X]
                pd = drate[Y, X]
                for b in range(nb):
                    rem[0, b] = 0.0
                    remS[0, b] = 0.0
                dv = _take(F, Y, X, fv * fr, rem, remS, 0)
                if dv <= 0.0:
                    continue
                # burr on the side of the groove, the rest is carried
                lat = dx * nx + dy * ny
                side = 1.0 if lat >= 0.0 else -1.0
                X2 = int(round(cx + nx * side * (r + 1.0)))
                Y2 = int(round(cy + ny * side * (r + 1.0)))
                fb = burr
                if X2 < 0 or Y2 < 0 or X2 >= W or Y2 >= H or mask[Y2, X2] >= 1.0:
                    fb = 0.0
                if fb > 0.0:
                    for b in range(nb):
                        kK[b] = rem[0, b] / dv
                        kS[b] = remS[0, b] / dv
                    _dep(F, sid, Y2, X2, dv * fb, kK, kS, pg, pb, pd)
                    if wet[Y2, X2] < wv:
                        wet[Y2, X2] = wv
                    if X2 < bb[0]:
                        bb[0] = X2
                    if Y2 < bb[1]:
                        bb[1] = Y2
                    if X2 > bb[2]:
                        bb[2] = X2
                    if Y2 > bb[3]:
                        bb[3] = Y2
                cvn = dv * (1.0 - fb)
                if cvn > 0.0:
                    for b in range(nb):
                        crK[0, b] += rem[0, b] * (1.0 - fb)
                        crS[0, b] += remS[0, b] * (1.0 - fb)
                    tw = cv + cvn
                    cg = (cg * cv + pg * cvn) / tw
                    cb = (cb * cv + pb * cvn) / tw
                    cd = (cd * cv + pd * cvn) / tw
                    cv = tw
                if X < bb[0]:
                    bb[0] = X
                if Y < bb[1]:
                    bb[1] = Y
                if X > bb[2]:
                    bb[2] = X
                if Y > bb[3]:
                    bb[3] = Y
        # the carried crumb leaks back a little behind the tip, at both sides
        if cv > 1e-7 and carry < 1.0:
            for e in (-1.0, 1.0):
                X = int(round(cx - mx * (r + 1.0) + e * nx * (r + 0.8)))
                Y = int(round(cy - my * (r + 1.0) + e * ny * (r + 0.8)))
                if X < 0 or Y < 0 or X >= W or Y >= H or mask[Y, X] >= 1.0:
                    continue
                dv = cv * (1.0 - carry) * 0.5
                for b in range(nb):
                    kK[b] = crK[0, b] / cv
                    kS[b] = crS[0, b] / cv
                _dep(F, sid, Y, X, dv, kK, kS, cg, cb, cd)
                for b in range(nb):
                    crK[0, b] -= kK[b] * dv
                    crS[0, b] -= kS[b] * dv
                cv -= dv
    # the crumb is dropped where the tip lifts
    if cv > 1e-7 and M > 0:
        cx = px[M - 1]
        cy = py[M - 1]
        rr = 2
        wsum = 0.0
        for Y in range(int(cy) - rr, int(cy) + rr + 2):
            for X in range(int(cx) - rr, int(cx) + rr + 2):
                wsum += math.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / 2.0)
        for b in range(nb):
            kK[b] = crK[0, b] / cv
            kS[b] = crS[0, b] / cv
        for Y in range(int(cy) - rr, int(cy) + rr + 2):
            for X in range(int(cx) - rr, int(cx) + rr + 2):
                if X < 0 or Y < 0 or X >= W or Y >= H:
                    continue
                dv = cv * math.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / 2.0) / wsum * (1.0 - mask[Y, X])
                _dep(F, sid, Y, X, dv, kK, kS, cg, cb, cd)
    return bb[0], bb[1], bb[2], bb[3]


_sgraffito_kernel = _B.port('tools_sgraffito_kernel', _sgraffito_kernel, mutable={12: 'F'})


class PaletteKnife:
    """A flexible steel painting knife.

    Parameters
    ----------
    length : float       blade length in px (= width of the mark)
    angle : float        default blade angle relative to the direction of
                         motion in degrees (90 = blade perpendicular to the
                         stroke, smaller = angled, pushes paint to one side)
    sharpness : float    0..1, crispness of the mark's side edges
    flex : float         0 (rigid trowel, rides on the highest bump) ..
                         1 (very flexible, follows the relief)
    thickness : float    paint layer the loaded blade carries (height units)
    reach : float        px a full load lasts at full width
    lumpiness : float    unevenness of the paint on the blade -> skips
    pickup : float       how strongly wet canvas paint is taken onto the blade
    squeeze : float      paint squeezed out at the blade ends (edge ridges)
    plow : float         paint migration along an angled blade
    seed : int
    """

    def __init__(self, length=40.0, angle=80.0, sharpness=0.85, flex=0.4, thickness=3.0,
                 reach=320.0, lumpiness=0.55, pickup=0.08, squeeze=0.45, plow=0.6,
                 seed=None):
        self.length = float(length)
        self.angle = float(angle)
        self.sharpness = float(sharpness)
        self.flex = float(flex)
        self.thickness = float(thickness)
        self.reach = float(reach)
        self.lumpiness = float(lumpiness)
        self.pickup = float(pickup)
        self.squeeze = float(squeeze)
        self.plow = float(plow)
        self.rng = np.random.default_rng(seed)
        self.nbin = int(np.clip(self.length / 2.0, 6, 64))
        self.res_K = np.zeros((self.nbin, NB))
        self.res_S = np.zeros((self.nbin, NB))
        self.res_vol = np.zeros(self.nbin)
        self.paint = None
        self.props = dict(fluid=False, gloss=0.8, body=1.0, drate=1e-5, gran=0.0, stain=0.3)
        self._set_cap(0.5)
        # fine scratches / nicks in the blade -> striations along the stroke
        nu = int(self.length) + 4
        g = self.rng
        stri = 0.35 * _aniso_noise((nu,), 1.5, g)
        uu = np.arange(nu)
        for _ in range(max(1, int(self.length / 14))):      # a few nicks
            c = g.uniform(0, nu)
            stri -= g.uniform(0.6, 1.6) * np.exp(-0.5 * ((uu - c) / g.uniform(0.4, 0.9)) ** 2)
        self._stri = np.ascontiguousarray(stri)

    def _set_cap(self, body):
        # volume one bin holds at a full load: enough for ``reach`` px of a
        # layer about half the knife's thickness (height = volume * body)
        bw = self.length / self.nbin
        self.cap = np.full(self.nbin, self.reach * bw * self.thickness * 0.5 / max(body, 0.1))

    # ------------------------------------------------------------- paint
    @property
    def load_level(self):
        return float(self.res_vol.sum() / self.cap.sum())

    def clean(self):
        """Wipe the blade completely."""
        self.res_K[:] = 0
        self.res_S[:] = 0
        self.res_vol[:] = 0
        return self

    def wipe(self, fraction=0.7):
        f = 1.0 - fraction
        self.res_K *= f
        self.res_S *= f
        self.res_vol *= f
        return self

    def load(self, paint, amount=1.0, mix=1.0, clean=True, marble=12.0):
        """Pick up paint with the blade.

        paint  : a Paint or ``[(paint, weight), ...]`` -- several colours picked
                 up together lie side by side on the blade (marbled strokes)
        amount : 1 = normal load, 2 = heavy impasto load, 0.3 = scant
        mix    : 1 = colours mixed on the palette, 0 = unmixed streaks
        marble : typical width (px along the blade) of unmixed colour streaks
        """
        if clean:
            self.clean()
        main, Km, Sm, paints, w = _mix_paints(paint)
        if main.fluid:
            raise ValueError('a palette knife lays paste paints (oil, acrylic, gouache); '
                             'use paint.as_medium(...) for fluid paints')
        if clean or self.paint is None:
            self._set_cap(main.body)
        self.paint = main
        self.props = _paint_props(main)
        g = self.rng
        nbin = self.nbin
        if len(paints) > 1:
            nzc = _aniso_noise((nbin,), max(0.5, marble / (self.length / nbin) * 0.5), g)
            # rank-based assignment keeps the colour proportions of the load
            q = (np.argsort(np.argsort(nzc)) + 0.5) / nbin
            cat = np.clip(np.searchsorted(np.cumsum(w), q), 0, len(paints) - 1)
            onehot = np.zeros((nbin, len(paints)))
            onehot[np.arange(nbin), cat] = 1
            comp = mix * w[None, :] + (1 - mix) * onehot
            Kp = comp @ np.array([p.K for p, _ in paints])
            Sp = comp @ np.array([p.S for p, _ in paints])
        else:
            Kp = np.tile(Km, (nbin, 1))
            Sp = np.tile(Sm, (nbin, 1))
        lv = np.clip(1.0 + 0.15 * g.normal(0, 1, nbin), 0.5, 1.5)
        # a knife holds a bit less paint at its tip end
        lv *= np.linspace(1.05, 0.9, nbin)
        vol = self.cap * float(amount) * lv
        self.res_vol += vol
        self.res_K += vol[:, None] * Kp
        self.res_S += vol[:, None] * Sp
        return self

    # ------------------------------------------------------------- marks
    def _blade_angles(self, t, phi, angle, rel):
        rel = self.angle if rel is None else rel
        pa = _angle_array(angle, t, phi, rel)
        # a blade dragged along its own length sweeps nothing: keep >= 12 deg
        d = (pa - phi + math.pi / 2) % math.pi - math.pi / 2   # in (-90, 90]
        lim = math.radians(12.0)
        bad = np.abs(d) < lim
        if bad.any():
            pa = np.where(bad, phi + np.where(d >= 0, lim, -lim), pa)
        return pa

    def _run(self, canvas, path, mode, pressure, angle, rel_angle, speed, wobble, seed,
             leave=0.05, default_p=0.55, bite=0.0):
        g = _rng(seed) if seed is not None else self.rng
        prep = _prep_path(path, 0.5, pressure, default_p, wobble=wobble, seed=g, smooth=6.0)
        if prep is None:
            return None
        xy, s, t, pp, phi = prep
        pa = self._blade_angles(t, phi, angle, rel_angle)
        half = self.length / 2.0
        nk = max(4, int(self.length) + 1)
        NU = int(self.length) + 4
        NS = int(s[-1]) + 4
        # lumpy paint on the blade: patches elongated along the stroke with
        # jagged borders (two octaves)
        sp = max(speed, 0.2)
        lump = 0.85 * _aniso_noise((NU, NS), (2.8, 12.0 * sp), g) + \
            0.45 * _aniso_noise((NU, NS), (0.9, 3.0 * sp), g)
        lump = np.ascontiguousarray(lump / lump.std())
        # local maximum of the solid relief (paint bridges small valleys)
        m = int(half) + 6
        sx0 = int(max(0, xy[:, 0].min() - m)); sx1 = int(min(canvas.W, xy[:, 0].max() + m + 1))
        sy0 = int(max(0, xy[:, 1].min() - m)); sy1 = int(min(canvas.H, xy[:, 1].max() + m + 1))
        if sx1 > sx0 and sy1 > sy0:
            hd = canvas.h_dry[sy0:sy1, sx0:sx1]
            sol = canvas.h_ground[sy0:sy1, sx0:sx1] * np.exp(-_LEV * hd) + hd
            smax = ndimage.maximum_filter(sol, size=5).astype(np.float32)
        else:
            smax = np.zeros((1, 1), np.float32)
        chat = np.ascontiguousarray((0.05 if mode == 0 else 0.015) * self.thickness *
                                    _aniso_noise((NS,), 6.0, g))
        flex_n = int(max(1, (1.0 - self.flex) * self.length * 0.5 + 1))
        if mode == 1:
            # pressed hard, the thin blade bends into the relief
            flex_n = max(1, flex_n // 4)
        edge_w = 0.35 + 2.5 * (1.0 - self.sharpness)
        pr = self.props
        spd = max(float(speed), 0.05)
        thickness = self.thickness * spd ** -0.25
        prm = np.array([thickness, 0.9, 0.55, 1.4, self.lumpiness * spd ** 0.3,
                        0.025 * self.thickness, 1.1, edge_w, 0.95,
                        self.pickup if mode == 0 else 0.0, self.squeeze, self.plow, 1.1,
                        float(leave), 0.97, 0.3 if mode == 0 else 0.7, 0.15, 0.9, 1.6,
                        0.012, float(bite)], dtype=np.float64)
        bbox = _knife_kernel(xy[:, 0].copy(), xy[:, 1].copy(), pp, pa, s, mode, half, nk,
                             self.res_K, self.res_S, self.res_vol, self.cap, prm,
                             lump, self._stri, chat, flex_n,
                             pr['gloss'], pr['body'], pr['drate'],
                             _film(canvas), int(canvas.stroke_id), canvas.h_ground,
                             canvas.h_dry, canvas.mask, smax, sx0, sy0)
        return _finish(canvas, bbox, False)

    def spread(self, canvas, path, pressure=None, angle=None, rel_angle=None, speed=1.0,
               wobble=0.0, seed=None):
        """Lay paint with the flat of the blade (also ``lay``).

        pressure : 0..1 (number, array or fn(t)); light = thick slab that
                   breaks up over relief, heavy = thin layer pressed into
                   the weave
        angle    : None -> blade at ``rel_angle`` (default ``self.angle``) to
                   the motion; number/fn(t) -> fixed blade orientation (deg)
        speed    : fast strokes skip more and lay thinner
        """
        return self._run(canvas, path, 0, pressure, angle, rel_angle, speed, wobble, seed,
                         default_p=0.45)

    lay = spread

    def scrape(self, canvas, path, pressure=None, angle=None, rel_angle=None, leave=0.0,
               bite=0.35, speed=1.0, wobble=0.0, seed=None):
        """Drag the edge firmly over wet paint: everything above the blade is
        collected on the knife (levelling impasto, revealing the layer below
        and the weave); the knife's load smears into the next mark.

        leave : thickness of paint left above the relief the blade rides on
        bite  : how far the pressed, flexing edge digs below the relief tops
                (x pressure); 0 = only level the paint
        """
        return self._run(canvas, path, 1, pressure, angle, rel_angle, speed, wobble, seed,
                         leave=leave, default_p=0.9, bite=bite)

    def sgraffito(self, canvas, path, width=2.5, pressure=None, wobble=0.0, seed=None,
                  burr=0.55):
        """Scratch a thin line through wet paint with the knife tip, revealing
        the dried layer (or ground) below.  ``width`` in px at full pressure."""
        g = _rng(seed) if seed is not None else self.rng
        prep = _prep_path(path, 0.35, pressure, 0.85, wobble=wobble, seed=g)
        if prep is None:
            return None
        xy, s, t, pp, phi = prep
        r1 = max(0.5, width / 2.0)
        r0 = max(0.45, r1 * 0.45)
        x0 = int(max(0, xy[:, 0].min() - width - 4)); x1 = int(min(canvas.W, xy[:, 0].max() + width + 5))
        y0 = int(max(0, xy[:, 1].min() - width - 4)); y1 = int(min(canvas.H, xy[:, 1].max() + width + 5))
        if x1 > x0 and y1 > y0:
            noise = np.tanh(_aniso_noise((y1 - y0, x1 - x0), 0.8, g))
        else:
            noise = np.zeros((1, 1))
        bbox = _sgraffito_kernel(xy[:, 0].copy(), xy[:, 1].copy(), pp, r0, r1, 0.35, burr, 0.85,
                                 0.8, noise, x0, y0, _film(canvas), int(canvas.stroke_id),
                                 canvas.h_ground, canvas.h_dry, canvas.mask)
        return _finish(canvas, bbox, False)

    def __repr__(self):
        return f'<PaletteKnife length={self.length:g} load={self.load_level:.2f}>'


# ==========================================================================
# Airbrush
# ==========================================================================
@njit(cache=True, fastmath=True)
def _spray_accumulate(px, py, amt, sig, tail_frac, tail_mult, ox, oy, acc):
    """Add gaussian spray cones (integral ``amt[j]``) into the bbox map ``acc``."""
    h, w = acc.shape
    for j in range(px.shape[0]):
        a = amt[j]
        if a <= 0.0:
            continue
        s = sig[j]
        st = s * tail_mult
        r = 3.2 * (st if tail_frac > 0.0 else s)
        cx = px[j] - ox
        cy = py[j] - oy
        x0 = max(int(math.floor(cx - r)), 0)
        x1 = min(int(math.ceil(cx + r)), w - 1)
        y0 = max(int(math.floor(cy - r)), 0)
        y1 = min(int(math.ceil(cy + r)), h - 1)
        if x1 < x0 or y1 < y0:
            continue
        nx = x1 - x0 + 1
        gxs = np.empty(nx)
        gxt = np.empty(nx)
        i2s = 0.5 / (s * s)
        i2t = 0.5 / (st * st)
        for i in range(nx):
            dx = x0 + i - cx
            gxs[i] = math.exp(-dx * dx * i2s)
            gxt[i] = math.exp(-dx * dx * i2t)
        ns = a * (1.0 - tail_frac) / (2.0 * math.pi * s * s)
        nt = a * tail_frac / (2.0 * math.pi * st * st)
        for Y in range(y0, y1 + 1):
            dy = Y - cy
            gys = math.exp(-dy * dy * i2s) * ns
            gyt = math.exp(-dy * dy * i2t) * nt
            for i in range(nx):
                acc[Y, x0 + i] += gys * gxs[i] + gyt * gxt[i]


_spray_accumulate = _B.port('tools_spray_accumulate', _spray_accumulate, mutable={8: 'acc'})


@njit(cache=True, fastmath=True)
def _splat_discs(xs, ys, rs, vs, ox, oy, acc, tooth, rough):
    """Add droplets (dome-shaped discs of volume ``vs``) into the map ``acc``.
    ``rough`` lets the paper tooth fray the droplet outline."""
    h, w = acc.shape
    H, W = tooth.shape
    for k in range(xs.shape[0]):
        r = rs[k]
        cx = xs[k] - ox
        cy = ys[k] - oy
        R = r + 1.5
        x0 = max(int(math.floor(cx - R)), 0)
        x1 = min(int(math.ceil(cx + R)), w - 1)
        y0 = max(int(math.floor(cy - R)), 0)
        y1 = min(int(math.ceil(cy + R)), h - 1)
        if x1 < x0 or y1 < y0:
            continue
        # sub-pixel droplets: bilinear splat keeps the volume
        if r < 0.6:
            fx = cx - math.floor(cx)
            fy = cy - math.floor(cy)
            ix = int(math.floor(cx))
            iy = int(math.floor(cy))
            for dy in range(2):
                for dx in range(2):
                    X = ix + dx
                    Y = iy + dy
                    if 0 <= X < w and 0 <= Y < h:
                        wx = fx if dx == 1 else 1.0 - fx
                        wy = fy if dy == 1 else 1.0 - fy
                        acc[Y, X] += vs[k] * wx * wy
            continue
        wsum = 0.0
        for Y in range(y0, y1 + 1):
            for X in range(x0, x1 + 1):
                d = math.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
                tY = Y + oy
                tX = X + ox
                tf = 0.0
                if 0 <= tY < H and 0 <= tX < W:
                    tf = tooth[tY, tX]
                re = r * (1.0 + rough * 0.12 * tf)
                q = d / re
                if q < 1.0 + 0.5 / re:
                    cov = _cl01((re - d) + 0.5)
                    wsum += cov * math.sqrt(max(1.0 - q * q, 0.05))
        if wsum <= 0.0:
            continue
        for Y in range(y0, y1 + 1):
            for X in range(x0, x1 + 1):
                d = math.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
                tY = Y + oy
                tX = X + ox
                tf = 0.0
                if 0 <= tY < H and 0 <= tX < W:
                    tf = tooth[tY, tX]
                re = r * (1.0 + rough * 0.12 * tf)
                q = d / re
                if q < 1.0 + 0.5 / re:
                    cov = _cl01((re - d) + 0.5)
                    acc[Y, X] += vs[k] * cov * math.sqrt(max(1.0 - q * q, 0.05)) / wsum


_splat_discs = _B.port('tools_splat_discs', _splat_discs, mutable={6: 'acc'})


class Airbrush:
    """A double-action airbrush.

    Parameters
    ----------
    nozzle : float     spray width right at the nozzle (px, gaussian sigma)
    cone : float       full spray cone angle in degrees (distance -> spread)
    flow : float       paint volume sprayed per px of travel at full trigger
    grain : float      droplet size of the fine mist (speckle; 0 = perfectly
                       smooth, 1 = typical, 3 = coarse / low air pressure)
    spatter : float    share of paint thrown as coarse droplets (dots)
    overspray : float  share of paint drifting into a wide faint halo
    seed : int
    """

    def __init__(self, nozzle=1.2, cone=20.0, flow=2.0, grain=1.0, spatter=0.07,
                 overspray=0.12, seed=None):
        self.nozzle = float(nozzle)
        self.cone = float(cone)
        self.flow = float(flow)
        self.grain = float(grain)
        self.spatter = float(spatter)
        self.overspray = float(overspray)
        self.rng = np.random.default_rng(seed)
        self.K = np.zeros(NB)
        self.S = np.zeros(NB)
        self.paint = None
        self.props = None
        self.water = 4.0

    def load(self, paint, dilution=0.0, water=4.0):
        """Fill the colour cup.  ``dilution`` 0..1 thins the paint with medium
        / water (lighter, more transparent tone per pass).  ``water``: fluid
        paints only, water carried per unit of pigment (wetness of the spray)."""
        main, Km, Sm, _, _ = _mix_paints(paint)
        f = 1.0 - float(np.clip(dilution, 0.0, 0.99))
        self.K = Km * f
        self.S = Sm * f
        self.paint = main
        self.props = _paint_props(main)
        self.water = float(water)
        return self

    def sigma(self, distance):
        """Spray gaussian sigma (px) at ``distance`` px from the surface."""
        return self.nozzle + np.asarray(distance, float) * math.tan(math.radians(self.cone) / 2) * 0.55

    def spray(self, canvas, path, pressure=None, distance=60.0, speed=1.0, wobble=0.0,
              seed=None, dwell=None):
        """Spray along ``path``.

        pressure : trigger 0..1 (number, array or fn(t)) -> paint flow
        distance : px from the surface (number, array or fn(t)); further away =
                   wider, softer, lighter, more speckled
        speed    : hand speed (>1 lighter, <1 heavier)
        dwell    : optional extra seconds-equivalent of spraying at the end
                   points (dark spots where the hand pauses)
        """
        if self.props is None:
            raise RuntimeError('load the airbrush first')
        g = _rng(seed) if seed is not None else self.rng
        prep = _prep_path(path, 1.0, pressure, 0.8, wobble=wobble, seed=g)
        if prep is None:
            return None
        xy, s, t, pp, phi = prep
        if callable(distance):
            dist = np.asarray(distance(t), float) * np.ones_like(t)
        elif np.ndim(distance) == 0:
            dist = np.full(len(t), float(distance))
        else:
            dist = np.interp(t, np.linspace(0, 1, len(distance)), np.asarray(distance, float))
        dist = np.maximum(dist, 0.0)
        sig = self.sigma(dist)
        # sub-sample the path to ~sigma/4 (the cone is smooth)
        step = max(0.5, float(sig.min()) * 0.25)
        if len(s) > 2 and step > 1.0:
            si = np.arange(0.0, s[-1] + 1e-9, step)
            xs = np.interp(si, s, xy[:, 0]); ys = np.interp(si, s, xy[:, 1])
            pp = np.interp(si, s, pp); sig = np.interp(si, s, sig); dist = np.interp(si, s, dist)
            ds = np.full(len(si), step)
        else:
            xs, ys = xy[:, 0], xy[:, 1]
            ds = np.gradient(s) if len(s) > 1 else np.ones(1)
        spd = max(float(speed), 0.05)
        trig = np.clip(pp, 0, 1) ** 1.3
        amt = self.flow * trig * ds / spd
        if dwell:
            amt[0] += self.flow * trig[0] * float(dwell) * 60.0
            amt[-1] += self.flow * trig[-1] * float(dwell) * 60.0
        return self._emit(canvas, np.ascontiguousarray(xs, float), np.ascontiguousarray(ys, float),
                          np.ascontiguousarray(amt, float), np.ascontiguousarray(sig, float),
                          dist, g)

    def dot(self, canvas, x, y, duration=0.3, distance=30.0, pressure=0.8, seed=None):
        """Hold the airbrush still: a soft round spot."""
        if self.props is None:
            raise RuntimeError('load the airbrush first')
        g = _rng(seed) if seed is not None else self.rng
        sig = float(self.sigma(distance))
        amt = self.flow * float(pressure) ** 1.3 * float(duration) * 60.0
        return self._emit(canvas, np.array([float(x)]), np.array([float(y)]), np.array([amt]),
                          np.array([sig]), np.array([float(distance)]), g)

    def fill(self, canvas, x0, y0, x1, y1, distance=60.0, pressure=0.7, passes=None,
             spacing=None, seed=None, wobble=1.5):
        """Even tone over a rectangle with overlapping back-and-forth passes
        (combine with a :class:`Stencil` for hard-edged shapes)."""
        g = _rng(seed) if seed is not None else self.rng
        sig = float(self.sigma(distance))
        sp = spacing or 1.6 * sig
        ys = np.arange(y0 - sig, y1 + sig + 1e-6, sp)
        pts = []
        for i, yy in enumerate(ys):
            a, b = (x0 - 2 * sig, x1 + 2 * sig) if i % 2 == 0 else (x1 + 2 * sig, x0 - 2 * sig)
            pts.append((a, yy))
            pts.append((b, yy))
        from .path import polyline
        res = None
        for _ in range(int(passes or 1)):
            res = self.spray(canvas, polyline(pts), pressure=pressure, distance=distance,
                             seed=g, wobble=wobble)
        return res

    # -- emission -------------------------------------------------------------
    def _emit(self, canvas, xs, ys, amt, sig, dist, g):
        tail_mult = 2.6
        r = float(sig.max()) * (3.2 * tail_mult if self.overspray > 0 else 3.2) + 4
        x0 = int(max(0, math.floor(xs.min() - r))); x1 = int(min(canvas.W, math.ceil(xs.max() + r + 1)))
        y0 = int(max(0, math.floor(ys.min() - r))); y1 = int(min(canvas.H, math.ceil(ys.max() + r + 1)))
        if x1 <= x0 or y1 <= y0:
            canvas._after_stroke()
            return None
        h, w = y1 - y0, x1 - x0
        acc = np.zeros((h, w))
        # fine mist share (the rest is thrown as coarse spatter droplets)
        sp = float(np.clip(self.spatter, 0, 0.9))
        _spray_accumulate(xs, ys, amt * (1.0 - sp), sig, float(self.overspray), tail_mult,
                          float(x0), float(y0), acc)
        # droplet statistics: Poisson count of mist droplets per pixel.  Far
        # from the surface droplets coalesce a little and land sparser.
        dmean = float(np.mean(dist))
        vdrop = 0.0007 * self.grain * (1.0 + dmean / 120.0)
        if vdrop > 0:
            lam = acc / vdrop
            cnt = g.poisson(np.minimum(lam, 1e7))
            acc = cnt * vdrop
        # the larger droplets of the spray (visible at the fringe as dots);
        # the tail of the size distribution forms little round spots
        if sp > 0:
            tot = float(amt.sum()) * sp
            rmed = 0.2 * (0.5 + 0.5 * self.grain) * (1.0 + dmean / 200.0)
            slog = 0.4
            vk = 0.5
            vmean = vk * rmed ** 3 * math.exp(4.5 * slog ** 2)
            n = int(min(g.poisson(tot / max(vmean, 1e-9)), 600000))
            if n > 0:
                pr = amt / amt.sum()
                idx = g.choice(len(xs), size=n, p=pr)
                ssp = sig[idx] * 1.15
                dx = g.normal(0, 1, n) * ssp
                dy = g.normal(0, 1, n) * ssp
                rr = rmed * np.exp(g.normal(0, slog, n))
                vv = vk * rr ** 3
                _splat_discs(xs[idx] + dx, ys[idx] + dy, rr, vv, int(x0), int(y0), acc,
                             canvas.tooth, 0.0)
        # a touch of blur: droplets spread a little on contact
        acc = ndimage.gaussian_filter(acc, 0.35)
        pr = self.props
        Kp = self.K
        Sp = self.S
        if pr['fluid']:
            # fine mist dries almost on contact (more so from further away);
            # a wet spray (lots of water) stays suspended and can bleed
            settle = float(np.clip(0.75 + dmean / 400.0 - self.water / 20.0, 0.1, 0.9))
            _deposit_map(canvas, acc, x0, y0, Kp, Sp, pr, water_ratio=self.water,
                         settle_frac=settle)
        else:
            _deposit_map(canvas, acc, x0, y0, Kp, Sp, pr, body_scale=0.04, gloss_scale=0.8)
        nzr = acc > 1e-6
        if nzr.any():
            yy, xx = np.nonzero(nzr)
            bbox = (x0 + xx.min(), y0 + yy.min(), x0 + xx.max(), y0 + yy.max())
        else:
            bbox = None
        return _finish(canvas, bbox, pr['fluid'])

    def __repr__(self):
        return f'<Airbrush nozzle={self.nozzle:g} cone={self.cone:g} flow={self.flow:g}>'


# ==========================================================================
# Splatter
# ==========================================================================
@njit(cache=True, fastmath=True)
def _splat_drops(xs, ys, rs, es, th, vs, ph, ox, oy, acc, tooth, rough, dome):
    """Rasterise impact drops into the bbox map ``acc``.

    Each drop is an ellipse (radius ``rs``, elongation ``es`` along ``th``)
    with a wobbly outline (harmonics with phases ``ph``), frayed by the paper
    tooth, plus a tapering tail pointing in the direction of flight for
    oblique impacts.  ``dome`` 1 = rounded blob profile (paste), 0 = flat.
    The drop volume ``vs`` is distributed over its footprint.
    """
    h, w = acc.shape
    H, W = tooth.shape
    for k in range(xs.shape[0]):
        r = rs[k]
        e = es[k]
        a = r * math.sqrt(e)
        b = r / math.sqrt(e)
        ct = math.cos(th[k])
        st = math.sin(th[k])
        lt = 0.0
        if e > 1.5:
            lt = (e - 1.0) * r * 1.3
        cx = xs[k] - ox
        cy = ys[k] - oy
        R = a + lt + 2.0
        x0 = max(int(math.floor(cx - R)), 0)
        x1 = min(int(math.ceil(cx + R)), w - 1)
        y0 = max(int(math.floor(cy - R)), 0)
        y1 = min(int(math.ceil(cy + R)), h - 1)
        if x1 < x0 or y1 < y0:
            continue
        p0 = ph[k]
        if r < 0.55:
            # sub-pixel droplet: bilinear splat
            fx = cx - math.floor(cx)
            fy = cy - math.floor(cy)
            ix = int(math.floor(cx))
            iy = int(math.floor(cy))
            for dy in range(2):
                for dx in range(2):
                    X = ix + dx
                    Y = iy + dy
                    if 0 <= X < w and 0 <= Y < h:
                        acc[Y, X] += vs[k] * (fx if dx == 1 else 1.0 - fx) * (fy if dy == 1 else 1.0 - fy)
            continue
        wsum = 0.0
        for pas in range(2):
            for Y in range(y0, y1 + 1):
                for X in range(x0, x1 + 1):
                    dx = X - cx
                    dy = Y - cy
                    u = dx * ct + dy * st
                    v = -dx * st + dy * ct
                    phi = math.atan2(v / b, u / a)
                    wob = 1.0 + 0.07 * math.sin(3.0 * phi + p0) + 0.05 * math.sin(5.0 * phi + 2.3 * p0) \
                        + 0.03 * math.sin(8.0 * phi + 4.1 * p0)
                    tY = Y + oy
                    tX = X + ox
                    tf = 0.0
                    if 0 <= tY < H and 0 <= tX < W:
                        tf = tooth[tY, tX]
                    wob *= 1.0 + rough * 0.10 * tf
                    q = math.sqrt((u / a) ** 2 + (v / b) ** 2) / wob
                    val = 0.0
                    edge = (1.0 - q) * min(a, b) + 0.5
                    if edge > 0.0:
                        prof = math.sqrt(max(1.0 - q * q, 0.0)) if dome > 0.5 else 1.0
                        val = _cl01(edge) * (0.15 + 0.85 * prof)
                    if lt > 0.0 and u > 0.3 * a:
                        tu = (u - 0.3 * a) / (lt + 0.7 * a)
                        if tu < 1.0:
                            hw = b * 0.6 * (1.0 - tu) ** 0.8 * wob
                            te = hw - abs(v) + 0.5
                            if te > 0.0:
                                tv = _cl01(te) * (0.6 if dome > 0.5 else 1.0) * (1.0 - 0.5 * tu)
                                if tv > val:
                                    val = tv
                    if val <= 0.0:
                        continue
                    if pas == 0:
                        wsum += val
                    else:
                        acc[Y, X] += vs[k] * val / wsum
            if wsum <= 1e-9:
                break


_splat_drops = _B.port('tools_splat_drops', _splat_drops, mutable={9: 'acc'})


class Splatter:
    """Spattered drops: flicked from a loaded brush, tapped off it, or sprayed
    from a toothbrush.

    Parameters
    ----------
    size : float        typical (median) drop radius in px
    satellites : float  how many satellite droplets big drops throw off
    streaks : float     share of drops that land as elongated streaks
    seed : int
    """

    def __init__(self, size=2.5, satellites=0.6, streaks=0.12, seed=None):
        self.size = float(size)
        self.satellites = float(satellites)
        self.streaks = float(streaks)
        self.rng = np.random.default_rng(seed)
        self.res = _Reservoir()

    def load(self, paint, amount=1.0, water=1.0):
        """Load the brush.  Paste paints should be thinned to splatter (they
        land as little raised blobs); fluid paints land as wet drops."""
        self.res.load(paint, amount, 1.0, water=water)
        return self

    @property
    def paint(self):
        return self.res.paint

    # -- drop generation -------------------------------------------------------
    def _satellites(self, g, x, y, r, th, e, k_sat, crown=False):
        """Satellite droplets thrown ahead of (or, for vertical impacts, all
        around) the big drops."""
        n = g.poisson(k_sat * np.clip(r - 0.8, 0, None) * 1.6)
        tot = int(n.sum())
        if tot == 0:
            return [np.zeros(0)] * 6
        idx = np.repeat(np.arange(len(r)), n)
        rr = r[idx]
        if crown:
            ang = g.uniform(0, 2 * np.pi, tot)
            dist = rr * g.uniform(1.4, 2.6, tot)
        else:
            ang = th[idx] + g.normal(0, 0.45, tot)
            dist = rr * np.sqrt(e[idx]) * g.uniform(1.3, 4.0, tot)
        sx = x[idx] + np.cos(ang) * dist
        sy = y[idx] + np.sin(ang) * dist
        sr = rr * np.exp(g.normal(np.log(0.2), 0.4, tot))
        se = 1.0 + np.abs(g.normal(0, 0.3, tot))
        return [sx, sy, sr, se, ang, np.full(tot, 1.0)]

    def _render(self, canvas, xs, ys, rs, es, th, g, dome=None, vol_scale=1.0):
        if len(xs) == 0:
            canvas._after_stroke()
            return None
        pr = self.res.props
        fluid = pr['fluid']
        if dome is None:
            dome = 0.0 if fluid else 1.0
        # drop volume ~ r^3 (a sphere flattening on impact)
        vs = 0.55 * rs ** 3 * vol_scale * max(self.res.level, 0.0)
        pad = float((rs * (np.sqrt(es) + np.maximum(es - 1, 0) * 1.3)).max()) + 4
        x0 = int(max(0, math.floor(xs.min() - pad))); x1 = int(min(canvas.W, math.ceil(xs.max() + pad + 1)))
        y0 = int(max(0, math.floor(ys.min() - pad))); y1 = int(min(canvas.H, math.ceil(ys.max() + pad + 1)))
        if x1 <= x0 or y1 <= y0:
            canvas._after_stroke()
            return None
        acc = np.zeros((y1 - y0, x1 - x0))
        ph = g.uniform(0, 2 * np.pi, len(xs))
        _splat_drops(np.ascontiguousarray(xs, float), np.ascontiguousarray(ys, float),
                     np.ascontiguousarray(rs, float), np.ascontiguousarray(es, float),
                     np.ascontiguousarray(th, float), np.ascontiguousarray(vs, float), ph,
                     int(x0), int(y0), acc, canvas.tooth, 1.0 if fluid else 0.4, float(dome))
        res = self.res
        if fluid:
            # pigment mass and water: drops are juicy
            _deposit_map(canvas, acc * 0.22, x0, y0, res.Kpv, res.Spv, pr,
                         water_ratio=4.5 * max(res.water, 0.05), settle_frac=0.0)
        else:
            _deposit_map(canvas, acc, x0, y0, res.Kpv, res.Spv, pr, body_scale=0.8)
        # the brush slowly empties
        res.take(float(vs.sum()) * 0.002)
        nz = acc > 1e-6
        bbox = None
        if nz.any():
            yy, xx = np.nonzero(nz)
            bbox = (x0 + xx.min(), y0 + yy.min(), x0 + xx.max(), y0 + yy.max())
        return _finish(canvas, bbox, fluid, pad=4)

    def flick(self, canvas, origin, direction=0.0, spread=30.0, count=60, distance=(10, 220),
              size=None, energy=1.0, seed=None):
        """Flick the loaded brush from ``origin`` towards ``direction`` (deg).

        Drops fan out by ``spread`` degrees and land between ``distance`` px;
        far drops arrive at a shallow angle and are elongated with tails,
        big drops throw satellite droplets ahead of them.
        """
        g = _rng(seed) if seed is not None else self.rng
        size = self.size if size is None else float(size)
        n = int(count)
        th0 = math.radians(direction)
        d0, d1 = distance
        u = g.uniform(0, 1, n)
        dist = d0 + (d1 - d0) * u ** 0.8
        ang = th0 + np.radians(spread) * 0.5 * g.normal(0, 0.6, n) * (0.4 + u)
        r = size * np.exp(g.normal(0, 0.55, n))
        # big drops stay nearer the main axis and do not fly quite as far
        ang = th0 + (ang - th0) / (1.0 + 0.25 * r / size)
        x = origin[0] + np.cos(ang) * dist
        y = origin[1] + np.sin(ang) * dist
        elong = 1.0 + 1.8 * energy * u ** 1.5 * g.uniform(0.4, 1.4, n)
        st = g.uniform(0, 1, n) < self.streaks
        elong[st] = g.uniform(3.0, 7.0, st.sum()) * (0.6 + 0.4 * energy)
        r[st] *= 0.75
        th = ang + g.normal(0, 0.06, n)
        sat = self._satellites(g, x, y, r, th, elong, self.satellites * energy)
        # a fine mist of tiny droplets along the flick
        nm = int(n * 1.5)
        um = g.uniform(0, 1, nm)
        am = th0 + np.radians(spread) * 0.6 * g.normal(0, 0.6, nm)
        dm = d0 + (d1 - d0) * um ** 0.7
        mx = origin[0] + np.cos(am) * dm
        my = origin[1] + np.sin(am) * dm
        mr = size * 0.18 * np.exp(g.normal(0, 0.5, nm))
        xs = np.concatenate([x, sat[0], mx]); ys = np.concatenate([y, sat[1], my])
        rs = np.concatenate([r, sat[2], mr]); es = np.concatenate([elong, sat[3], np.ones(nm)])
        ths = np.concatenate([th, sat[4], am])
        return self._render(canvas, xs, ys, rs, es, ths, g)

    def tap(self, canvas, center, radius=40.0, count=40, size=None, seed=None):
        """Tap the loaded brush against a finger or stick above the surface:
        drops fall nearly vertically -> round, with crown satellites."""
        g = _rng(seed) if seed is not None else self.rng
        size = self.size if size is None else float(size)
        n = int(count)
        x = center[0] + g.normal(0, radius * 0.5, n)
        y = center[1] + g.normal(0, radius * 0.5, n)
        r = size * np.exp(g.normal(0, 0.6, n))
        e = 1.0 + np.abs(g.normal(0, 0.12, n))
        th = g.uniform(0, 2 * np.pi, n)
        sat = self._satellites(g, x, y, r, th, e, self.satellites * 0.8, crown=True)
        xs = np.concatenate([x, sat[0]]); ys = np.concatenate([y, sat[1]])
        rs = np.concatenate([r, sat[2]]); es = np.concatenate([e, sat[3]])
        ths = np.concatenate([th, sat[4]])
        return self._render(canvas, xs, ys, rs, es, ths, g)

    def toothbrush(self, canvas, center, radius=60.0, count=3000, size=0.5, direction=None,
                   falloff=1.0, seed=None):
        """Run a thumb over a loaded toothbrush: a cloud of fine dots
        (stars, sand, granite, weathering)."""
        g = _rng(seed) if seed is not None else self.rng
        n = int(count)
        # dense centre thinning out towards the edge of the cloud
        rr = radius * 0.5 * np.sqrt(-2.0 * np.log(g.uniform(1e-6, 1, n))) ** falloff
        a = g.uniform(0, 2 * np.pi, n)
        x = center[0] + rr * np.cos(a)
        y = center[1] + rr * np.sin(a)
        r = float(size) * np.exp(g.normal(0, 0.5, n))
        if direction is None:
            th = g.uniform(0, 2 * np.pi, n)
            e = 1.0 + np.abs(g.normal(0, 0.15, n))
        else:
            th = math.radians(direction) + g.normal(0, 0.2, n)
            e = 1.0 + np.abs(g.normal(0.4, 0.3, n))
        return self._render(canvas, x, y, r, e, th, g)

    def drops(self, canvas, xs, ys, radii, elongation=None, angle=None, seed=None):
        """Place drops explicitly (radii in px, angles in degrees)."""
        g = _rng(seed) if seed is not None else self.rng
        xs = np.atleast_1d(np.asarray(xs, float)); ys = np.atleast_1d(np.asarray(ys, float))
        rs = np.broadcast_to(np.asarray(radii, float), xs.shape).copy()
        es = np.ones_like(xs) if elongation is None else np.broadcast_to(np.asarray(elongation, float), xs.shape).copy()
        th = g.uniform(0, 2 * np.pi, len(xs)) if angle is None else \
            np.radians(np.broadcast_to(np.asarray(angle, float), xs.shape))
        return self._render(canvas, xs, ys, rs, es, th, g)

    def __repr__(self):
        return f'<Splatter size={self.size:g} load={self.res.level:.2f}>'


# ==========================================================================
# Sponge
# ==========================================================================
def _sponge_texture(size, porosity, coarseness, g):
    """Height map of a natural sea sponge surface, (T, T) float64.

    Material height ~1 with a large-scale undulation (only the bumps touch
    under light pressure), holes of a wide (power-law) size distribution
    (-1), and an irregular rounded outline.
    """
    R = float(size)
    T = int(2 * math.ceil(R * 1.3) + 1)
    c = (T - 1) / 2.0
    yy, xx = np.mgrid[0:T, 0:T].astype(float)
    # large-scale undulation and fine fibrous detail
    und = _aniso_noise((T, T), max(1.5, R / 4.0), g)
    fine = _aniso_noise((T, T), 0.8, g)
    h = 1.0 - 0.22 * und - 0.06 * fine
    # pores: union of irregular discs with power-law radii (continuous field
    # so that pore walls are smooth after interpolation)
    area = T * T
    rmin, rmax = 0.8 * coarseness, max(2.0, 0.2 * R * coarseness)
    pert = _aniso_noise((T, T), 1.2, g)
    n = int(porosity * area / (math.pi * (1.1 * coarseness) ** 2) * 0.62)
    hf = np.full((T, T), -1.0)
    if n > 0:
        u = g.uniform(0, 1, n)
        a = 2.3       # power-law exponent of pore radii
        rr = (rmin ** (1 - a) + u * (rmax ** (1 - a) - rmin ** (1 - a))) ** (1 / (1 - a))
        pxs = g.uniform(0, T, n)
        pys = g.uniform(0, T, n)
        for x, y, r in zip(pxs, pys, rr):
            x0, x1 = int(max(0, x - r - 2)), int(min(T, x + r + 3))
            y0, y1 = int(max(0, y - r - 2)), int(min(T, y + r + 3))
            if x1 <= x0 or y1 <= y0:
                continue
            d = np.hypot(xx[y0:y1, x0:x1] - x, yy[y0:y1, x0:x1] - y)
            v = (r - d) / max(r, 1.0) + 0.12 * pert[y0:y1, x0:x1]
            hf[y0:y1, x0:x1] = np.maximum(hf[y0:y1, x0:x1], v)
    hole = np.clip(hf * 4.0 + 0.5, 0, 1)
    h = h * (1 - hole) - 1.0 * hole
    # irregular rounded outline
    ang = np.arctan2(yy - c, xx - c)
    rad = np.hypot(yy - c, xx - c)
    k = np.arange(1, 7)
    amp = g.normal(0, 1, 6) * 0.12 / k ** 0.7
    phs = g.uniform(0, 2 * np.pi, 6)
    Rphi = R * (1.0 + (amp[:, None, None] * np.cos(k[:, None, None] * ang[None] + phs[:, None, None])).sum(0))
    edge = rad / Rphi
    h = h - np.clip((edge - 0.65) / 0.35, 0, None) ** 2 * 0.9
    h = np.where(edge > 1.0, -1.0, h)
    return np.ascontiguousarray(h)


@njit(cache=True, fastmath=True)
def _sponge_dab(cx, cy, rot, scale, tex, depth, soft, micro, alpha, pickup,
                rK, rS, rV, cap, p_gloss, p_body, p_drate, fluid, water_lvl, wat_ratio, gran, stain,
                F, sid, mask, x0, y0, tooth, wK, wS, wwat, wm, wg, wst, dK, dS, dm, dg, dst, wdamp):
    """Press the sponge texture once at (cx, cy) rotated by ``rot``."""
    T = tex.shape[0]
    c = (T - 1) / 2.0
    H, W = tooth.shape
    nb = rK.shape[1]
    x1, y1 = x0 + mask.shape[1] - 1, y0 + mask.shape[0] - 1
    bb = np.array([W, H, -1, -1])
    if x1 < x0 or y1 < y0:
        return bb[0], bb[1], bb[2], bb[3]
    cr = math.cos(rot)
    sr = math.sin(rot)
    level = 1.0 - depth
    kK = np.empty(nb)
    kS = np.empty(nb)
    fV = F[2]
    bV = F[5]
    wet = F[7]
    v0 = rV[0]
    for Y in range(y0, y1 + 1):
        for X in range(x0, x1 + 1):
            m = 1.0 - mask[Y - y0, X - x0]
            if m <= 0.0:
                continue
            dx = (X - cx) / scale
            dy = (Y - cy) / scale
            u = cr * dx + sr * dy + c
            v = -sr * dx + cr * dy + c
            if u < 0.0 or v < 0.0 or u >= T - 1.0 or v >= T - 1.0:
                continue
            iu = int(u)
            iv = int(v)
            fu = u - iu
            fv_ = v - iv
            hgt = (tex[iv, iu] * (1 - fu) * (1 - fv_) + tex[iv, iu + 1] * fu * (1 - fv_) +
                   tex[iv + 1, iu] * (1 - fu) * fv_ + tex[iv + 1, iu + 1] * fu * fv_)
            # a soft sponge conforms to the tooth only a little
            hgt += 0.06 * tooth[Y, X]
            cnt = _cl01((hgt - level) / soft + 0.5) * m
            if cnt <= 0.0:
                continue
            # pressed deeper -> more paint transferred
            pen = 0.35 + 0.65 * _cl01((hgt - level) / 0.3)
            if X < bb[0]:
                bb[0] = X
            if Y < bb[1]:
                bb[1] = Y
            if X > bb[2]:
                bb[2] = X
            if Y > bb[3]:
                bb[3] = Y
            mic = 1.0 + micro * (tex[iv, iu] - tex[iv + 1, iu + 1]) * 4.0
            if mic < 0.2:
                mic = 0.2
            if fluid:
                # water + pigment from the sponge; a damp sponge blots
                wsat = wwat[Y, X]
                if water_lvl > 0.25 * (1.0 + wsat):
                    dw = alpha * cnt * pen * water_lvl * mic * 0.9
                    vb = rV[0]
                    dmass = dw * wat_ratio
                    if dmass > 0.02 * vb:
                        dmass = 0.02 * vb
                    if vb > 1e-12:
                        for b in range(nb):
                            kK[b] = rK[0, b] / vb
                            kS[b] = rS[0, b] / vb
                        fr = dmass / vb
                        for b in range(nb):
                            rK[0, b] -= rK[0, b] * fr
                            rS[0, b] -= rS[0, b] * fr
                        rV[0] = vb - dmass
                        _wc_add(wK, wS, wwat, wm, wg, wst, Y, X, dw, dmass, kK, kS, gran, stain)
                    else:
                        wwat[Y, X] += dw
                elif pickup > 0.0:
                    thirst = 1.0 - water_lvl
                    # a thirsty sponge also draws moisture out of the paper
                    wdamp[Y, X] *= 1.0 - min(0.6 * cnt * thirst, 0.9)
                    # it re-wets settled pigment and lifts it (staining
                    # pigments resist)
                    dmm = dm[Y, X]
                    if dmm > 1e-6:
                        stn = dst[Y, X] / dmm
                        fl = min(pickup * cnt * pen * 2.6 * thirst * (1.0 - min(stn, 1.0)), 0.85)
                        for b in range(nb):
                            dk = dK[Y, X, b] * fl
                            dsg = dS[Y, X, b] * fl
                            dK[Y, X, b] -= dk
                            dS[Y, X, b] -= dsg
                            rK[0, b] += dk
                            rS[0, b] += dsg
                        dm[Y, X] = dmm * (1.0 - fl)
                        dg[Y, X] *= (1.0 - fl)
                        dst[Y, X] *= (1.0 - fl)
                        rV[0] += dmm * fl
                    if wsat <= 1e-4:
                        continue
                    fr = min(pickup * cnt * pen * 2.6 * thirst, 0.92)
                    mm = wm[Y, X] * fr
                    for b in range(nb):
                        dk = wK[Y, X, b] * fr
                        dsg = wS[Y, X, b] * fr
                        wK[Y, X, b] -= dk
                        wS[Y, X, b] -= dsg
                        rK[0, b] += dk
                        rS[0, b] += dsg
                    wm[Y, X] -= mm
                    wg[Y, X] *= (1.0 - fr)
                    wst[Y, X] *= (1.0 - fr)
                    wwat[Y, X] *= (1.0 - fr)
                    rV[0] += mm
                continue
            # paste: pick up wet paint (sponge gets dirty / lifts texture) ...
            fvt = fV[Y, X] + bV[Y, X]
            if pickup > 0.0 and fvt > 1e-6:
                room = 1.0 - rV[0] / (1.5 * cap + 1e-12)
                if room > 0.0:
                    pf = min(pickup * cnt * wet[Y, X] * room, 0.6)
                    rV[0] += _take(F, Y, X, fvt * pf, rK, rS, 0)
            # ... and deposit from its load, textured by the sponge surface
            vb = rV[0]
            if vb <= 1e-9:
                continue
            ld = vb / (cap + 1e-12)
            dv = alpha * cnt * pen * mic * min(ld, 1.5) ** 1.3
            if dv > 0.05 * vb:
                dv = 0.05 * vb
            fr = dv / vb
            for b in range(nb):
                kK[b] = rK[0, b] / vb
                kS[b] = rS[0, b] / vb
                rK[0, b] -= rK[0, b] * fr
                rS[0, b] -= rS[0, b] * fr
            rV[0] = vb - dv
            _dep(F, sid, Y, X, dv, kK, kS, p_gloss, p_body, p_drate)
    return bb[0], bb[1], bb[2], bb[3]


_sponge_dab = _B.port('tools_sponge_dab', _sponge_dab, mutable={
    10: 'rK', 11: 'rS', 12: 'rV', 22: 'F', 28: 'wK', 29: 'wS', 30: 'wwat', 31: 'wm', 32: 'wg', 33: 'wst',
    34: 'dK', 35: 'dS', 36: 'dm', 37: 'dg', 38: 'dst', 39: 'wdamp'})


class Sponge:
    """A natural sea sponge.

    Parameters
    ----------
    size : float        radius of the contact patch in px
    porosity : float    0..1, share of holes in the surface
    coarseness : float  scale of the pores (1 = medium, 2 = coarse)
    softness : float    0..1, how much the sponge flattens under pressure
    seed : int          every sponge is different
    """

    def __init__(self, size=30.0, porosity=0.5, coarseness=1.0, softness=0.5, seed=None):
        self.size = float(size)
        self.porosity = float(porosity)
        self.coarseness = float(coarseness)
        self.softness = float(softness)
        self.rng = np.random.default_rng(seed)
        self.tex = _sponge_texture(self.size, self.porosity, self.coarseness, self.rng)
        self.res = _Reservoir()
        self.res_K = np.zeros((1, NB))
        self.res_S = np.zeros((1, NB))
        self.res_V = np.zeros(1)
        self.cap = math.pi * self.size ** 2 * 6.0
        self.water = 0.0

    def load(self, paint, amount=1.0, water=1.0, clean=True):
        """Press the sponge into paint (``amount`` 1 = normal, 0.3 = nearly
        dry); fluid paints: ``water`` = how wet the sponge is."""
        main, Km, Sm, _, _ = _mix_paints(paint)
        if clean:
            self.clean()
        self.res.paint = main
        self.res.props = _paint_props(main)
        v = self.cap * float(amount) * (0.35 if main.fluid else 1.0)
        self.res_K[0] += Km * v
        self.res_S[0] += Sm * v
        self.res_V[0] += v
        if main.fluid:
            self.water = float(water)
        return self

    def wet(self, water=1.0):
        """Dampen a clean sponge (for lifting / blotting watercolour)."""
        self.water = float(water)
        self.res.props = dict(self.res.props, fluid=True)
        return self

    def clean(self):
        self.res_K[:] = 0
        self.res_S[:] = 0
        self.res_V[:] = 0
        self.water = 0.0
        return self

    @property
    def load_level(self):
        return float(self.res_V[0] / self.cap)

    def dab(self, canvas, x, y, pressure=0.6, rotation=None, seed=None, _count=True):
        """Press the sponge once at (x, y).  ``rotation`` in degrees (random
        by default: painters keep turning the sponge)."""
        g = _rng(seed) if seed is not None else self.rng
        rot = g.uniform(0, 2 * np.pi) if rotation is None else math.radians(rotation)
        p = float(np.clip(pressure, 0, 1))
        depth = 0.12 + 0.55 * p * (0.6 + 0.8 * self.softness)
        scale = (1.0 + 0.12 * p * self.softness) * float(np.exp(g.normal(0, 0.05)))
        pr = self.res.props
        fluid = bool(pr['fluid'])
        wc = canvas.ensure_watercolor() if fluid else None
        if wc is None:
            d2 = np.zeros((1, 1), np.float32)
            d3 = np.zeros((1, 1, NB), np.float32)
            wK = wS = dK = dS = d3
            wwat = wm = wg = wst = dm = dg = dst = wdamp = d2
        else:
            wK, wS, wwat, wm, wg, wst = wc.K, wc.S, wc.water, wc.m, wc.g, wc.st
            dK, dS, dm, dg, dst, wdamp = wc.dK, wc.dS, wc.dm, wc.dg, wc.dst, wc.damp
        radius = (self.tex.shape[0] - 1) * 0.5 * scale + 1.0
        x0, y0 = max(0, math.floor(x - radius)), max(0, math.floor(y - radius))
        x1, y1 = max(0, math.ceil(x + radius) + 1), max(0, math.ceil(y + radius) + 1)
        region = np.s_[y0:y1, x0:x1]
        mask = canvas.watercolor_mask(region) if fluid else canvas.mask[region]
        bbox = _sponge_dab(float(x), float(y), rot, scale, self.tex, depth, 0.08, 0.35,
                           0.9 if not fluid else 0.6, 0.35, self.res_K, self.res_S, self.res_V,
                           self.cap, pr['gloss'], pr['body'], pr['drate'], fluid,
                           float(self.water), 0.6, pr['gran'], pr['stain'],
                           _film(canvas), int(canvas.stroke_id), mask, x0, y0, canvas.tooth,
                           wK, wS, wwat, wm, wg, wst, dK, dS, dm, dg, dst, wdamp)
        if fluid:
            self.water *= 0.985
        if _count:
            return _finish(canvas, bbox, fluid)
        return bbox

    def dabs(self, canvas, points, pressure=0.6, jitter=0.0, reload=None, seed=None):
        """Dab at many points (one mark each).  ``pressure`` may be a number
        or a sequence; ``jitter`` adds random offsets (px)."""
        g = _rng(seed) if seed is not None else self.rng
        pts = np.asarray(points, float)
        pr = np.broadcast_to(np.asarray(pressure, float), (len(pts),))
        out = []
        for i, (x, y) in enumerate(pts[:, :2]):
            if jitter > 0:
                x += g.normal(0, jitter)
                y += g.normal(0, jitter)
            out.append(self.dab(canvas, x, y, pressure=pr[i], seed=g))
            if reload is not None and self.load_level < reload[1]:
                self.load(reload[0], reload[2] if len(reload) > 2 else 1.0)
        return out

    def scatter(self, canvas, center, radius, count=12, pressure=(0.4, 0.8), seed=None):
        """Dab ``count`` times at random inside a circle (foliage masses)."""
        g = _rng(seed) if seed is not None else self.rng
        r = radius * np.sqrt(g.uniform(0, 1, count))
        a = g.uniform(0, 2 * np.pi, count)
        pts = np.column_stack([center[0] + r * np.cos(a), center[1] + r * np.sin(a)])
        ps = g.uniform(pressure[0], pressure[1], count) if np.ndim(pressure) else pressure
        return self.dabs(canvas, pts, ps, seed=g)

    def drag(self, canvas, path, pressure=0.4, spacing=None, seed=None):
        """Wipe the sponge along a path: overlapping prints smeared in the
        direction of motion (one mark)."""
        g = _rng(seed) if seed is not None else self.rng
        sp = spacing or max(1.0, self.size * 0.08)
        prep = _prep_path(path, sp, pressure, 0.4)
        if prep is None:
            return None
        xy, s, t, pp, phi = prep
        rot0 = g.uniform(0, 2 * np.pi)
        boxes = []
        for (x, y), p, ph in zip(xy, pp, phi):
            boxes.append(self.dab(canvas, x, y, pressure=p, rotation=math.degrees(rot0 + 0.15 * (ph - phi[0])),
                                  seed=g, _count=False))
        b = np.array([bx for bx in boxes if bx is not None and bx[2] >= bx[0]])
        bbox = (b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()) if len(b) else None
        return _finish(canvas, bbox, bool(self.res.props['fluid']))

    def __repr__(self):
        return f'<Sponge size={self.size:g} load={self.load_level:.2f}>'


# ==========================================================================
# Pen
# ==========================================================================
@njit(cache=True, fastmath=True)
def _pen_raster(px, py, rad, ink, skip, gap, fill, kind, nca, nsa, nib_hw, nib_ht,
                ox, oy, cov, amt, tooth):
    """Rasterise a pen line into bbox maps: ``cov`` = max coverage, ``amt`` =
    max ink density.  kind 0 = round tip, 1 = broad-edge nib (rectangle of
    half sizes nib_hw x nib_ht along (nca, nsa)), 2 = pointed flexible nib
    (two tines ``gap`` apart, inside filled by ``fill``)."""
    h, w = cov.shape
    H, W = tooth.shape
    M = px.shape[0]
    for j in range(M):
        cx = px[j] - ox
        cy = py[j] - oy
        r = rad[j]
        if kind == 1:
            R = nib_hw + nib_ht + 1.5
        elif kind == 2:
            R = 0.5 * gap[j] + r + 1.5
        else:
            R = r + 1.5
        x0 = max(int(math.floor(cx - R)), 0)
        x1 = min(int(math.ceil(cx + R)), w - 1)
        y0 = max(int(math.floor(cy - R)), 0)
        y1 = min(int(math.ceil(cy + R)), h - 1)
        if x1 < x0 or y1 < y0:
            continue
        j0 = j - 1 if j > 0 else 0
        j1 = j + 1 if j < M - 1 else M - 1
        mx = px[j1] - px[j0]
        my = py[j1] - py[j0]
        ml = math.sqrt(mx * mx + my * my)
        if ml > 1e-9:
            mx /= ml
            my /= ml
        else:
            mx, my = 1.0, 0.0
        for Y in range(y0, y1 + 1):
            for X in range(x0, x1 + 1):
                dx = X - cx
                dy = Y - cy
                if kind == 1:
                    u = dx * nca + dy * nsa
                    v = -dx * nsa + dy * nca
                    c = _cl01(nib_hw - abs(u) + 0.5) * _cl01(nib_ht - abs(v) + 0.5)
                elif kind == 2:
                    lat = -dx * my + dy * mx
                    alo = dx * mx + dy * my
                    hg = 0.5 * gap[j]
                    # distance to the nearer tine, and to the inner band
                    dt = math.sqrt(alo * alo + (abs(lat) - hg) ** 2)
                    c = _cl01(r - dt + 0.5)
                    if abs(lat) < hg:
                        inner = _cl01(r - abs(alo) + 0.5) * fill[j]
                        if inner > c:
                            c = inner
                else:
                    d = math.sqrt(dx * dx + dy * dy)
                    c = _cl01(r - d + 0.5)
                if c <= 0.0:
                    continue
                if skip[j] > -2.5:
                    tY = Y + oy
                    tX = X + ox
                    if 0 <= tY < H and 0 <= tX < W:
                        c *= _cl01((tooth[tY, tX] - skip[j]) / 0.45 + 0.5)
                if c > cov[Y, X]:
                    cov[Y, X] = c
                a = c * ink[j]
                if a > amt[Y, X]:
                    amt[Y, X] = a


_pen_raster = _B.port('tools_pen_raster', _pen_raster, mutable={14: 'cov', 15: 'amt'})


@njit(cache=True, fastmath=True)
def _feather(cov, amt, perm, iters, decay):
    """Ink wicking into paper fibres: coverage creeps outward along
    high-permeability paths (hairy, feathered line edges)."""
    h, w = cov.shape
    c2 = cov.copy()
    a2 = amt.copy()
    for _ in range(iters):
        for Y in range(1, h - 1):
            for X in range(1, w - 1):
                pm = perm[Y, X]
                if pm <= 0.0:
                    continue
                best = 0.0
                ba = 0.0
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if dx == 0 and dy == 0:
                            continue
                        v = cov[Y + dy, X + dx]
                        if v > best:
                            best = v
                            ba = amt[Y + dy, X + dx]
                nv = best * pm * decay
                if nv > c2[Y, X]:
                    c2[Y, X] = nv
                    a2[Y, X] = max(a2[Y, X], ba * pm * decay * 0.8)
        for Y in range(h):
            for X in range(w):
                cov[Y, X] = c2[Y, X]
                amt[Y, X] = a2[Y, X]


_feather = _B.port('tools_feather', _feather, mutable={0: 'cov', 1: 'amt'})


class Pen:
    """Ink pens.

    kind : 'technical'  constant line width, steady ink flow (cartridge)
           'nib'        broad-edge dip pen: thick/thin depends on the stroke
                        direction relative to ``nib_angle`` (calligraphy)
           'pointed'    flexible pointed dip pen: pressure spreads the tines
                        (``flex`` px extra width), railroading when the ink
                        cannot keep up
    width : float       line width of the technical pen / hairline of dip pens
    nib : float         broad nib edge width in px
    nib_angle : float   angle of the broad nib edge (degrees)
    flex : float        extra width of a pointed nib at full pressure
    capacity : float    px of line a dip lasts (None = cartridge, unlimited)
    feather : float     0..1, ink wicking into paper fibres
    pooling : float     ink pooling where the pen starts, stops and turns
    water : float       fluid inks: water per unit of pigment left on the
                        paper (more = lines that can bleed in the simulation)
    """

    def __init__(self, kind='technical', width=1.4, nib=8.0, nib_angle=35.0, flex=4.0,
                 capacity=None, feather=0.35, pooling=0.6, water=0.45, seed=None):
        if kind not in ('technical', 'nib', 'pointed'):
            raise ValueError("kind must be 'technical', 'nib' or 'pointed'")
        self.kind = kind
        self.width = float(width)
        self.nib = float(nib)
        self.nib_angle = float(nib_angle)
        self.flex = float(flex)
        if capacity is None and kind != 'technical':
            capacity = 2200.0 if kind == 'pointed' else 1600.0
        self.capacity = capacity
        self.feather = float(feather)
        self.pooling = float(pooling)
        self.water = float(water)
        self.rng = np.random.default_rng(seed)
        self.level = 0.0
        self.K = np.zeros(NB)
        self.S = np.zeros(NB)
        self.props = None
        self.paint = None

    def load(self, paint, amount=1.0):
        """Dip the pen (or fill the cartridge) with ink / paint."""
        main, Km, Sm, _, _ = _mix_paints(paint)
        self.K, self.S = Km, Sm
        self.paint = main
        self.props = _paint_props(main)
        if main.medium == 'ink':
            # inks are fine dispersions / dyes: they do not granulate
            self.props['gran'] *= 0.15
        self.level = float(amount)
        return self

    dip = load

    def _perm(self, canvas, x0, y0, x1, y1):
        t = canvas.tooth[y0:y1, x0:x1].astype(np.float64)
        hp = t - ndimage.uniform_filter(t, 3)
        wet = 0.0
        if canvas.wc is not None:
            wet = np.clip(canvas.wc.damp[y0:y1, x0:x1] + canvas.wc.water[y0:y1, x0:x1], 0, 1)
        return np.clip(0.55 + 0.9 * hp - 0.15 * t + 0.4 * wet, 0, 1.0)

    def line(self, canvas, path, pressure=None, speed=1.0, wobble=0.0, seed=None):
        """Draw a line along ``path`` (pressure: number, array, fn(t))."""
        if self.props is None:
            raise RuntimeError('load the pen first')
        g = _rng(seed) if seed is not None else self.rng
        prep = _prep_path(path, 0.3, pressure, 0.6, wobble=wobble, seed=g, smooth=2.0)
        if prep is None:
            return None
        xy, s, t, pp, phi = prep
        L = s[-1]
        spd = max(float(speed), 0.05)
        kind = {'technical': 0, 'nib': 1, 'pointed': 2}[self.kind]
        n = len(s)
        gap = np.zeros(n)
        fill = np.ones(n)
        if kind == 0:
            rad = np.full(n, self.width / 2.0)
            wdt = np.full(n, self.width)
        elif kind == 1:
            a = math.radians(self.nib_angle)
            wdt = np.abs(np.sin(phi - a)) * self.nib + self.width
            rad = np.full(n, self.width / 2.0)
        else:
            spread = self.flex * np.clip(pp, 0, 1) ** 1.6
            rad = np.full(n, self.width / 2.0)
            gap = spread
            wdt = spread + self.width
        # ink supply along the line
        if self.capacity is None:
            lvl = np.full(n, max(self.level, 0.0))
        else:
            use = np.gradient(s) * (wdt / max(self.width + 0.5 * self.nib * (kind == 1), 1.0)) \
                / float(self.capacity) * (0.6 + 0.4 * spd)
            lvl = self.level - np.cumsum(use)
            self.level = float(max(lvl[-1], 0.0))
        lvl = np.clip(lvl, 0.0, None)
        ink = np.clip(lvl / 0.35, 0, 1) ** 0.5
        # pooling at start / end and where the pen slows in sharp turns
        curv = np.abs(np.gradient(np.unwrap(phi))) / 0.3
        pool = self.pooling * (np.exp(-s / 3.0) + 0.6 * np.exp(-(L - s) / 3.0)
                               + 0.5 * np.clip(curv, 0, 1)) / spd ** 0.5
        ink = ink * (1.0 + pool) * (1.0 + 0.08 * (pp - 0.5))
        blob = 1.0 + 0.45 * self.pooling * np.exp(-s / 1.5) * (kind != 0)
        rad = rad * blob
        # a dry or fast pen skips over the valleys of the tooth
        skip = np.where(lvl > 0.25, -3.0, -1.6 + 2.2 * (1 - lvl / 0.25))
        if spd > 1.5:
            skip = np.maximum(skip, -1.6 + 0.4 * (spd - 1.5))
        if kind == 2:
            # railroading: wide tine spread with not enough ink flow
            demand = (gap / max(self.flex, 1e-6)) * spd
            fill = np.clip(1.3 * np.clip(lvl, 0, 1) / (demand + 0.25), 0, 1) ** 2
        r = float(max(rad.max(), 0.5 * gap.max() + rad.max(), self.nib)) + 6
        x0 = int(max(0, math.floor(xy[:, 0].min() - r))); x1 = int(min(canvas.W, math.ceil(xy[:, 0].max() + r + 1)))
        y0 = int(max(0, math.floor(xy[:, 1].min() - r))); y1 = int(min(canvas.H, math.ceil(xy[:, 1].max() + r + 1)))
        if x1 <= x0 or y1 <= y0:
            canvas._after_stroke()
            return None
        cov = np.zeros((y1 - y0, x1 - x0))
        amt = np.zeros_like(cov)
        a = math.radians(self.nib_angle)
        _pen_raster(xy[:, 0].copy(), xy[:, 1].copy(), np.ascontiguousarray(rad),
                    np.ascontiguousarray(ink), np.ascontiguousarray(skip, float),
                    np.ascontiguousarray(gap), np.ascontiguousarray(fill), kind,
                    math.cos(a), math.sin(a), self.nib / 2.0, max(self.width / 2.0, 0.45),
                    int(x0), int(y0), cov, amt, canvas.tooth)
        return self._deposit(canvas, cov, amt, x0, y0, x1, y1)

    def _deposit(self, canvas, cov, amt, x0, y0, x1, y1):
        pr = self.props
        if pr['fluid'] and self.feather > 0:
            iters = int(round(self.feather * 4))
            if iters > 0:
                perm = self._perm(canvas, x0, y0, x1, y1)
                if canvas.wax is not None:
                    wax = canvas.wax[y0:y1, x0:x1]
                    # Ink that failed to touch paper cannot feed a feathered edge.
                    cov *= 1.0 - wax
                    amt[wax >= 1.0] = 0.0
                    perm *= 1.0 - wax
                _feather(cov, amt, perm, iters, 0.82)
        if pr['fluid']:
            # ink: concentrated pigment in little water; part soaks in at once
            m = amt * 0.45
            _deposit_map(canvas, m, x0, y0, self.K, self.S, pr, water_ratio=self.water,
                         settle_frac=0.7)
        else:
            _deposit_map(canvas, amt * 0.45, x0, y0, self.K, self.S, pr, body_scale=0.5)
        nz = amt > 1e-4
        bbox = None
        if nz.any():
            yy, xx = np.nonzero(nz)
            bbox = (x0 + xx.min(), y0 + yy.min(), x0 + xx.max(), y0 + yy.max())
        return _finish(canvas, bbox, pr['fluid'], pad=3)

    def dot(self, canvas, x, y, size=None, seed=None):
        """A single dot (the pen touches down and lifts)."""
        return self.dots(canvas, [(x, y)], None if size is None else [size], seed=seed)

    def dots(self, canvas, points, sizes=None, seed=None):
        """Many dots as one mark (stippling); ``sizes`` = dot diameters."""
        if self.props is None:
            raise RuntimeError('load the pen first')
        g = _rng(seed) if seed is not None else self.rng
        pts = np.asarray(points, float).reshape(-1, 2)
        if len(pts) == 0:
            return None
        base = self.width if self.kind != 'nib' else 0.5 * (self.width + self.nib * 0.5)
        d = np.full(len(pts), base) if sizes is None else np.broadcast_to(np.asarray(sizes, float), (len(pts),))
        d = d * np.exp(g.normal(0, 0.12, len(pts)))
        rad = d / 2.0 * (1.0 + 0.1 * self.pooling)
        ink = np.full(len(pts), 1.0 + 0.5 * self.pooling) * min(1.0, max(self.level, 0.0) / 0.35) ** 0.5
        r = float(rad.max()) + 6
        x0 = int(max(0, math.floor(pts[:, 0].min() - r))); x1 = int(min(canvas.W, math.ceil(pts[:, 0].max() + r + 1)))
        y0 = int(max(0, math.floor(pts[:, 1].min() - r))); y1 = int(min(canvas.H, math.ceil(pts[:, 1].max() + r + 1)))
        if x1 <= x0 or y1 <= y0:
            canvas._after_stroke()
            return None
        cov = np.zeros((y1 - y0, x1 - x0))
        amt = np.zeros_like(cov)
        # each dot: a tiny pen touch (round, slightly irregular)
        jit = g.normal(0, 0.15, (len(pts), 2))
        zeros = np.zeros(len(pts))
        for k in range(2):
            _pen_raster(pts[:, 0] + jit[:, 0] * k, pts[:, 1] + jit[:, 1] * k, rad * (1 - 0.15 * k), ink,
                        np.full(len(pts), -3.0), zeros, np.ones(len(pts)), 0, 1.0, 0.0, 0.0, 0.5,
                        int(x0), int(y0), cov, amt, canvas.tooth)
        if self.capacity is not None:
            self.level = max(0.0, self.level - float((d ** 2).sum()) / float(self.capacity) * 0.15)
        return self._deposit(canvas, cov, amt, x0, y0, x1, y1)

    def hatch(self, canvas, x0, y0, x1, y1, angle=45.0, spacing=5.0, cross=False,
              wobble=0.6, jitter=0.6, region=None, seed=None, reload=True):
        """Parallel hatching (and optional cross-hatching) over a rectangle,
        optionally clipped by a boolean ``region`` map (H, W)."""
        g = _rng(seed) if seed is not None else self.rng
        out = []
        for ang in ([angle, angle + 90.0] if cross else [angle]):
            a = math.radians(ang)
            d = np.array([math.cos(a), math.sin(a)])
            nrm = np.array([-d[1], d[0]])
            c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
            half = math.hypot(x1 - x0, y1 - y0) / 2
            for o in np.arange(-half, half + 1e-6, spacing):
                o2 = o + g.normal(0, jitter)
                p0 = c + nrm * o2 - d * half
                p1 = c + nrm * o2 + d * half
                tt = np.linspace(0, 1, int(2 * half) + 2)
                pts = p0[None] + (p1 - p0)[None] * tt[:, None]
                ok = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
                if region is not None:
                    xi = np.clip(pts[:, 0].astype(int), 0, canvas.W - 1)
                    yi = np.clip(pts[:, 1].astype(int), 0, canvas.H - 1)
                    ok &= np.asarray(region)[yi, xi] > 0.5
                # split into runs inside the region
                idx = np.flatnonzero(ok)
                if len(idx) < 3:
                    continue
                runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
                for rn in runs:
                    if len(rn) < 4:
                        continue
                    if reload and self.capacity is not None and self.level < 0.3:
                        self.level = 1.0
                    out.append(self.line(canvas, pts[rn], wobble=wobble, seed=g,
                                         pressure=0.5 + 0.1 * g.normal()))
        return out

    def __repr__(self):
        return f'<Pen {self.kind} width={self.width:g} ink={self.level:.2f}>'


# ==========================================================================
# Smudge / blender
# ==========================================================================
@njit(cache=True, fastmath=True)
def _smudge_kernel(px, py, pp, R, rate, spread, F, sid, mask, lK, lS, lV, lP):
    """Finger smudge along the path.

    Paint separates from the finger at its trailing contact line.  Across
    the finger that line is sampled in lanes (lateral offsets -R..R); every
    lane carries a little wet paint.  At each step a lane takes a fraction
    of the paint under it and lays down the same fraction of what it carries
    -> each colour is smeared forward with an exponential tail of length
    ~1/rate px, edges soften, impasto is flattened and where the finger
    runs off the paint a fading trail is left.  Lanes exchange a little
    paint sideways (``spread``).  Volume is conserved; only wet paint moves.
    """
    fV = F[2]
    bV = F[5]
    wet = F[7]
    gloss = F[8]
    body = F[9]
    drate = F[10]
    H, W = mask.shape
    nb = lK.shape[1]
    nl = 2 * R + 1
    prof = np.zeros(nl)
    for i in range(nl):
        d2 = ((i - R) / (R + 0.5)) ** 2
        prof[i] = (1.0 - d2) ** 0.6 if d2 < 1.0 else 0.0
    tK = np.zeros((1, nb))
    tS = np.zeros((1, nb))
    kK = np.empty(nb)
    kS = np.empty(nb)
    bb = np.array([W, H, -1, -1])
    M = px.shape[0]
    lastX = np.full(nl, -1)
    lastY = np.full(nl, -1)
    for j in range(M):
        p = pp[j]
        if p <= 0.0:
            continue
        j0 = j - 1 if j > 0 else 0
        j1 = j + 1 if j < M - 1 else M - 1
        mx = px[j1] - px[j0]
        my = py[j1] - py[j0]
        ml = math.sqrt(mx * mx + my * my)
        if ml < 1e-9:
            continue
        mx /= ml
        my /= ml
        nx = -my
        ny = mx
        for i in range(nl):
            w = prof[i] * p
            if w <= 0.0:
                continue
            l = i - R
            X = int(round(px[j] + nx * l))
            Y = int(round(py[j] + ny * l))
            if X < 0 or Y < 0 or X >= W or Y >= H:
                continue
            if X == lastX[i] and Y == lastY[i]:
                continue          # the lane has not moved onto a new pixel yet
            lastX[i] = X
            lastY[i] = Y
            m = 1.0 - mask[Y, X]
            if m <= 0.0:
                continue
            f = min(rate * w * m, 0.6)
            # lay down part of the carried paint
            v = lV[i]
            out = v * f
            # take part of the paint under the lane
            fv = fV[Y, X] + bV[Y, X]
            mob = _cl01(wet[Y, X] * 1.4)
            dvt = 0.0
            pw = wet[Y, X]
            pg = gloss[Y, X]
            pb = body[Y, X]
            pd = drate[Y, X]
            if fv > 1e-7 and mob > 0.0:
                for c in range(nb):
                    tK[0, c] = 0.0
                    tS[0, c] = 0.0
                dvt = _take(F, Y, X, fv * f * mob, tK, tS, 0)
            if out > 1e-9:
                for c in range(nb):
                    kK[c] = lK[i, c] / v
                    kS[c] = lS[i, c] / v
                _dep_w(F, sid, Y, X, out, kK, kS, lP[i, 0] / v, lP[i, 1] / v,
                       lP[i, 2] / v, lP[i, 3] / v)
                for c in range(nb):
                    lK[i, c] -= kK[c] * out
                    lS[i, c] -= kS[c] * out
                for c in range(4):
                    lP[i, c] *= (1.0 - f)
                lV[i] = v - out
            if dvt > 0.0:
                for c in range(nb):
                    lK[i, c] += tK[0, c]
                    lS[i, c] += tS[0, c]
                lV[i] += dvt
                lP[i, 0] += pw * dvt
                lP[i, 1] += pg * dvt
                lP[i, 2] += pb * dvt
                lP[i, 3] += pd * dvt
            if X < bb[0]:
                bb[0] = X
            if Y < bb[1]:
                bb[1] = Y
            if X > bb[2]:
                bb[2] = X
            if Y > bb[3]:
                bb[3] = Y
        # neighbouring lanes share a little paint (soft, smeared edges)
        if spread > 0.0:
            for i in range(nl - 1):
                q = spread
                dv = (lV[i] - lV[i + 1]) * q
                for c in range(nb):
                    dk = (lK[i, c] - lK[i + 1, c]) * q
                    ds = (lS[i, c] - lS[i + 1, c]) * q
                    lK[i, c] -= dk
                    lK[i + 1, c] += dk
                    lS[i, c] -= ds
                    lS[i + 1, c] += ds
                for c in range(4):
                    dq = (lP[i, c] - lP[i + 1, c]) * q
                    lP[i, c] -= dq
                    lP[i + 1, c] += dq
                lV[i] -= dv
                lV[i + 1] += dv
    return bb[0], bb[1], bb[2], bb[3]


_smudge_kernel = _B.port('tools_smudge_kernel', _smudge_kernel, mutable={
    6: 'F', 9: 'lK', 10: 'lS', 11: 'lV', 12: 'lP'})


class Smudge:
    """Finger / rag / tortillon smudging of wet paste paint (no paint added).

    size : float      radius of the finger pad in px
    strength : float  0..1; how far colours are dragged (smear length
                      ~ 4..25 px)
    softness : float  0..1, sideways smearing (softer swath edges)
    """

    def __init__(self, size=10.0, strength=0.6, softness=0.5):
        self.size = float(size)
        self.strength = float(strength)
        self.softness = float(softness)
        self._alloc()

    def _alloc(self):
        R = max(1, int(round(self.size)))
        self.R = R
        nl = 2 * R + 1
        self.lK = np.zeros((nl, NB))
        self.lS = np.zeros((nl, NB))
        self.lV = np.zeros(nl)
        self.lP = np.zeros((nl, 4))

    def clean(self):
        """Wipe the finger on a rag."""
        self._alloc()
        return self

    def stroke(self, canvas, path, pressure=None, clean=True, wobble=0.0, seed=None):
        """Smudge along ``path``.  ``clean``: start with a clean finger
        (False keeps the paint picked up by the previous smudge)."""
        if clean:
            self.clean()
        prep = _prep_path(path, 0.7, pressure, 0.7, wobble=wobble,
                          seed=seed if seed is not None else canvas.rng)
        if prep is None:
            return None
        xy, s, t, pp, phi = prep
        rate = 0.25 - 0.21 * float(np.clip(self.strength, 0, 1))
        bbox = _smudge_kernel(xy[:, 0].copy(), xy[:, 1].copy(), pp, self.R, rate,
                              0.15 * self.softness, _film(canvas), int(canvas.stroke_id),
                              canvas.mask, self.lK, self.lS, self.lV, self.lP)
        return _finish(canvas, bbox, False)

    def __repr__(self):
        return f'<Smudge size={self.size:g}>'


@njit(cache=True)
def _crayon_stroke(film, sid, tooth, relief, mask, wax, xy, pressures, angles,
                   size, side, K, S):
    height, width = mask.shape
    left, top, right, bottom = width, height, -1, -1
    for i in range(len(xy)):
        p = pressures[i]
        if p <= 0:
            continue
        cx, cy = xy[i]
        radius = 0.5 * size * (0.48 + 0.52 * math.sqrt(p))
        major = radius * (1.0 + 1.8 * side)
        minor = radius * (1.0 - 0.45 * side)
        c, s = math.cos(angles[i]), math.sin(angles[i])
        dx = math.hypot(major * c, minor * s) + 1.0
        dy = math.hypot(major * s, minor * c) + 1.0
        for y in range(max(0, math.floor(cy - dy)), min(height, math.ceil(cy + dy) + 1)):
            for x in range(max(0, math.floor(cx - dx)), min(width, math.ceil(cx + dx) + 1)):
                if mask[y, x] >= 1.0:
                    continue
                u = ((x - cx) * c + (y - cy) * s) / major
                v = ((y - cy) * c - (x - cx) * s) / minor
                edge = _cl01(0.5 + (1.0 - math.sqrt(u*u + v*v)) * minor)
                if edge == 0.0:
                    continue
                # The paper pattern stays put; the crumbling stick leaves a
                # slightly different scatter of pigment each time it passes.
                grain = ((x * 73856093) ^ (y * 19349663) ^ (sid * 83492791)) % 1021 / 1021.0
                this_mark = film[6][y, x] == sid
                under = film[5][y, x] + (0.0 if this_mark else film[2][y, x])
                bite = relief * tooth[y, x] / (1.0 + under)
                contact = _sstep(1.65*p + bite + 0.65*(grain - 0.5) - 0.12)
                target = edge * contact * p * (0.9 + p) * (1.0 - mask[y, x])
                # A single sweep coats each pixel once even as adjacent
                # footprints overlap. A fresh stroke can build a new layer.
                dv = target - (film[2][y, x] if this_mark else 0.0)
                if dv > 1e-6:
                    _dep_w(film, sid, y, x, dv, K, S, 0.12, 0.26, 0.16, 0.0)
                    # Count only newly transferred wax. Overlapping dabs in
                    # this pass fill the same footprint; later passes build it up.
                    wax[y, x] = min(1.0, wax[y, x] + dv)
                    left, top = min(left, x), min(top, y)
                    right, bottom = max(right, x), max(bottom, y)
    return left, top, right, bottom


_crayon_stroke = _B.port('tools_crayon_stroke', _crayon_stroke, mutable={0: 'film', 5: 'wax'})


class OilCrayon:
    """Oil pastel stick. Size is the tip diameter in pixels at full pressure.

    Roll from the round tip (side=0) onto the broad side (side=1). The long
    axis lies across the path by default; angle fixes it in absolute degrees
    from horizontal, or can be a function of fractional path length.
    """

    def __init__(self, size=10.0, side=0.0):
        self.size = float(size)
        self.side = float(side)
        if not np.isfinite(self.size) or self.size <= 0:
            raise ValueError("crayon size must be finite and positive")
        if not 0 <= self.side <= 1:
            raise ValueError("crayon side must be between 0 and 1")
        self.paint = None

    def load(self, paint):
        """Choose the pigment or premixed colour of the stick."""
        if not isinstance(paint, Paint):
            raise TypeError("load expects a Paint")
        self.paint = paint
        return self

    def stroke(self, canvas, path, pressure=None, angle=None):
        if self.paint is None:
            raise ValueError("load the crayon with a pigment before drawing")
        spacing = max(0.35, min(3.0, self.size * 0.15))
        prepared = _prep_path(path, spacing, pressure, default_pressure=0.65)
        if prepared is None:
            return None
        xy, _, t, pp, phi = prepared
        angles = _angle_array(angle, t, phi, 90.0)
        relief = min(1.65, math.sqrt(max(0.0, canvas.ground.relief)))
        if canvas.wax is None:
            canvas.wax = np.zeros(canvas.shape, dtype=np.float32)
        box = _crayon_stroke(_film(canvas), canvas.stroke_id, canvas.tooth, relief,
                             canvas.mask, canvas.wax, xy, pp, angles, self.size, self.side,
                             self.paint.K, self.paint.S)
        return _finish(canvas, box if box[2] >= box[0] else None, False)


class Blender:
    """A clean, dry blending brush (soft filbert / fan / mop): it picks up
    wet paint and redistributes it, softening transitions without adding
    colour.  Wraps :class:`pinsel.brush.Brush` with an empty reservoir."""

    def __init__(self, size=12.0, shape='filbert', pickup=0.85, seed=None, **brush_kw):
        from .brush import Brush
        brush_kw.setdefault('plough', 0.0)
        self.brush = Brush(shape, size, pickup=pickup, seed=seed, **brush_kw)

    def stroke(self, canvas, path, pressure=0.8, wipe=True, **kw):
        """One blending stroke; ``wipe`` cleans the brush on a rag first."""
        if wipe:
            self.brush.clean()
        return self.brush.stroke(canvas, path, pressure=pressure, **kw)

    def __repr__(self):
        return f'<Blender {self.brush.shape} size={self.brush.size:g}>'


# ==========================================================================
# Glaze, stencils / tape, stippling
# ==========================================================================
def _region_map(canvas, region):
    """Region spec -> float (H, W) map in 0..1.

    None = whole canvas; (H, W) array; (x0, y0, x1, y1) rectangle;
    a list of (x, y) points = polygon; ('ellipse', cx, cy, rx, ry).
    """
    H, W = canvas.shape
    if region is None:
        return np.ones((H, W), np.float32)
    if isinstance(region, np.ndarray) and region.shape == (H, W):
        return np.clip(region.astype(np.float32), 0, 1)
    if isinstance(region, tuple) and len(region) == 5 and region[0] == 'ellipse':
        _, cx, cy, rx, ry = region
        yy, xx = np.mgrid[0:H, 0:W]
        return ((((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0).astype(np.float32)
    arr = np.asarray(region, float)
    if arr.ndim == 1 and arr.size == 4:
        x0, y0, x1, y1 = arr
        m = np.zeros((H, W), np.float32)
        m[int(max(0, y0)):int(min(H, y1)), int(max(0, x0)):int(min(W, x1))] = 1.0
        return m
    from PIL import Image, ImageDraw
    img = Image.new('F', (W, H), 0.0)
    ImageDraw.Draw(img).polygon([tuple(map(float, p[:2])) for p in arr], fill=1.0)
    return np.asarray(img, dtype=np.float32).copy()


def glaze(canvas, paint, thickness=0.12, region=None, feather=6.0, variation=0.12,
          wet=True, water=0.9, seed=None):
    """Apply a uniform transparent layer over ``region`` with soft edges.

    paint     : usually a transparent paint (e.g. ``pg.ALIZARIN_CRIMSON``,
                ``paint.thinned(0.6)``); fluid paints give a flat wash
    thickness : paint volume (paste) / pigment mass (fluid) per pixel
    region    : None, (H, W) map, (x0, y0, x1, y1), polygon points or
                ('ellipse', cx, cy, rx, ry)
    feather   : softness of the region edge (px, gaussian)
    variation : gentle unevenness of a hand-laid glaze
    wet       : fluid paints only -- True: a wet wash the watercolour
                simulation will move; False: perfectly flat dried wash
    water     : fluid paints only -- water depth of the wash (1 = puddle-wet)
    """
    g = _rng(seed) if seed is not None else canvas.rng
    m = _region_map(canvas, region).astype(np.float64)
    if feather > 0:
        m = ndimage.gaussian_filter(m, feather)
    if variation > 0:
        from .noise import fbm
        m = m * np.clip(1.0 + variation * fbm(canvas.shape, scale=min(canvas.shape) / 5.0,
                                               octaves=5, seed=g), 0, None)
    ys, xs = np.nonzero(m > 1e-4)
    if len(ys) == 0:
        canvas._after_stroke()
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    vol = m[y0:y1, x0:x1] * float(thickness)
    main, Km, Sm, _, _ = _mix_paints(paint)
    props = _paint_props(main)
    if props['fluid']:
        _deposit_map(canvas, vol, x0, y0, Km, Sm, props,
                     water_ratio=float(water) / max(float(thickness), 1e-6) if wet else 0.0,
                     settle_frac=0.0 if wet else 1.0)
    else:
        # glazing medium: little relief, keeps the paint's gloss
        _deposit_map(canvas, vol, x0, y0, Km, Sm, props, body_scale=0.3)
    return _finish(canvas, (x0, y0, x1 - 1, y1 - 1), props['fluid'] and wet)


class Stencil:
    """Masking tape, stencils, friskets and masking fluid on ``canvas.mask``.

    Use as a context manager (or ``apply()`` / ``remove()``)::

        with T.Stencil.rect(cv, 100, 100, 300, 200):
            airbrush.fill(cv, 80, 80, 320, 220)      # paints around the rect
        with T.Stencil.tape(cv, (50, 400), (750, 380), width=24):
            knife.spread(cv, ...)                   # crisp tape edge

    leak : 0..1, paint creeping under the edge in the valleys of a rough
           surface (tape never seals the weave completely)
    lift : px the stencil is held above the surface (soft, overspray edge)
    invert : protect everything *outside* the shape (a window stencil)
    """

    def __init__(self, canvas, shape_map, leak=0.25, lift=0.0, invert=False):
        self.canvas = canvas
        m = np.clip(np.asarray(shape_map, np.float32), 0, 1)
        if invert:
            m = 1.0 - m
        if lift > 0:
            m = ndimage.gaussian_filter(m, lift).astype(np.float32)
        if leak > 0:
            # under the edge band the seal fails where the surface is low
            inner = ndimage.gaussian_filter(m, 1.2)
            band = np.clip((1.0 - inner) * 3.0, 0, 1) * (m > 0.5)
            valley = np.clip(-canvas.tooth * 0.8, 0, 1)
            m = m * (1.0 - leak * band * valley)
        self.map = m.astype(np.float32)
        self._saved = None

    # -- constructors --------------------------------------------------------
    @classmethod
    def from_array(cls, canvas, arr, **kw):
        return cls(canvas, arr, **kw)

    @classmethod
    def rect(cls, canvas, x0, y0, x1, y1, **kw):
        return cls(canvas, _region_map(canvas, (x0, y0, x1, y1)), **kw)

    @classmethod
    def polygon(cls, canvas, points, **kw):
        return cls(canvas, _region_map(canvas, points), **kw)

    @classmethod
    def ellipse(cls, canvas, cx, cy, rx, ry, **kw):
        return cls(canvas, _region_map(canvas, ('ellipse', cx, cy, rx, ry)), **kw)

    @classmethod
    def tape(cls, canvas, p0, p1, width=20.0, torn=True, seed=None, **kw):
        """A strip of masking tape from p0 to p1 with (optionally) torn ends."""
        g = _rng(seed) if seed is not None else canvas.rng
        p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
        d = p1 - p0
        L = np.linalg.norm(d) + 1e-9
        d /= L
        nrm = np.array([-d[1], d[0]])
        hw = width / 2.0
        pts = [p0 + nrm * hw, p1 + nrm * hw]
        k = max(4, int(width / 2))
        end = np.linspace(hw, -hw, k)
        for o in end:                       # torn end at p1
            pts.append(p1 + nrm * o + d * (g.normal(0, 1.5) if torn else 0))
        pts += [p1 - nrm * hw, p0 - nrm * hw]
        for o in end[::-1]:                 # torn end at p0
            pts.append(p0 + nrm * o - d * (g.normal(0, 1.5) if torn else 0))
        return cls(canvas, _region_map(canvas, np.array(pts)), **kw)

    @classmethod
    def masking_fluid(cls, canvas, path, width=6.0, seed=None, **kw):
        """Masking fluid painted along a path (blobby, slightly irregular)."""
        g = _rng(seed) if seed is not None else canvas.rng
        pts = resample(np.asarray(path, float)[:, :2], 1.0)
        H, W = canvas.shape
        m = np.zeros((H, W), np.float32)
        wv = width / 2.0 * (1.0 + 0.25 * _aniso_noise((len(pts),), 6.0, g))
        yy, xx = np.mgrid[0:H, 0:W]
        x0 = int(max(0, pts[:, 0].min() - width * 2)); x1 = int(min(W, pts[:, 0].max() + width * 2 + 1))
        y0 = int(max(0, pts[:, 1].min() - width * 2)); y1 = int(min(H, pts[:, 1].max() + width * 2 + 1))
        sub = np.zeros((y1 - y0, x1 - x0), np.float32)
        sy, sx = yy[y0:y1, x0:x1], xx[y0:y1, x0:x1]
        for (x, y), r in zip(pts, wv):
            sel = (sx - x) ** 2 + (sy - y) ** 2 <= r * r
            sub[sel] = 1.0
        m[y0:y1, x0:x1] = ndimage.gaussian_filter(sub, 0.6) > 0.5
        kw.setdefault('leak', 0.0)
        return cls(canvas, m, **kw)

    # -- use -------------------------------------------------------------------
    def apply(self):
        if self._saved is None:
            self._saved = self.canvas.mask.copy()
            self.canvas.mask = np.maximum(self.canvas.mask, self.map).astype(np.float32)
        return self

    def remove(self):
        """Peel the tape / lift the stencil (restores the previous mask)."""
        if self._saved is not None:
            self.canvas.mask = self._saved
            self._saved = None
        return self

    def __enter__(self):
        return self.apply()

    def __exit__(self, *exc):
        self.remove()
        return False


def blue_noise_points(shape, density, min_dist=2.0, max_points=200000, seed=None, k=12):
    """Well-spaced random points (variable-radius Poisson-disc dart throwing).

    density : number (points per 100 px^2) or an (H, W) map of points per
              100 px^2 -- darker tones get more, closer dots (stippling)
    Returns an (N, 2) array of (x, y).
    """
    g = _rng(seed)
    H, W = shape
    dens = np.broadcast_to(np.asarray(density, float), (H, W))
    # target spacing from density: one point per (100/dens) px^2
    with np.errstate(divide='ignore'):
        spacing = np.where(dens > 1e-6, np.sqrt(100.0 / np.maximum(dens, 1e-6)) * 0.85, np.inf)
    spacing = np.maximum(spacing, min_dist)
    expected = float((dens / 100.0).sum())
    if expected <= 0:
        return np.zeros((0, 2))
    n_try = int(min(max_points * 6, max(1000, expected * k)))
    # candidates drawn proportionally to the density (importance sampling)
    flat = dens.ravel()
    idx = g.choice(flat.size, size=n_try, p=flat / flat.sum())
    ys = idx // W + g.uniform(0, 1, n_try)
    xs = idx % W + g.uniform(0, 1, n_try)
    xs = np.minimum(xs, W - 1e-3)
    ys = np.minimum(ys, H - 1e-3)
    cell = float(max(min_dist, 1.0))
    gw, gh = int(W / cell) + 1, int(H / cell) + 1
    grid = {}
    out = []
    rmax = float(np.nanmax(np.where(np.isfinite(spacing), spacing, 0))) if np.isfinite(spacing).any() else min_dist
    reach = int(math.ceil(rmax / cell))
    for x, y in zip(xs, ys):
        r = spacing[int(y), int(x)]
        if not np.isfinite(r):
            continue
        cx, cy = int(x / cell), int(y / cell)
        rr = int(math.ceil(r / cell))
        ok = True
        for gy in range(max(0, cy - rr), min(gh, cy + rr + 1)):
            for gx in range(max(0, cx - rr), min(gw, cx + rr + 1)):
                for (qx, qy) in grid.get((gx, gy), ()):
                    if (qx - x) ** 2 + (qy - y) ** 2 < r * r:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            grid.setdefault((cx, cy), []).append((x, y))
            out.append((x, y))
            if len(out) >= max_points:
                break
    del reach
    return np.array(out, float).reshape(-1, 2)


def stipple(canvas, pen, tone, region=None, max_density=None, dot_size=None, seed=None):
    """Stippling: dots whose density follows ``tone``.

    tone        : number 0..1 or an (H, W) map (1 = darkest)
    region      : optional region spec limiting the dots (see :func:`glaze`)
    max_density : dots per 100 px^2 at tone 1
    pen         : a loaded :class:`Pen` (or :class:`Splatter` for blobs)
    """
    H, W = canvas.shape
    t = np.broadcast_to(np.asarray(tone, float), (H, W))
    if region is not None:
        t = t * _region_map(canvas, region)
    dsz = float(dot_size or getattr(pen, 'width', 1.5))
    min_d = 1.15 * dsz
    if max_density is None:
        # darkest tone: dots nearly touching
        max_density = 100.0 / (1.5 * dsz) ** 2
    pts = blue_noise_points((H, W), np.clip(t, 0, 1) * max_density, min_dist=max(min_d, 1.0),
                            seed=seed if seed is not None else canvas.rng)
    if len(pts) == 0:
        canvas._after_stroke()
        return None
    if isinstance(pen, Splatter):
        return pen.drops(canvas, pts[:, 0], pts[:, 1], (dot_size or 1.0) / 2.0, seed=seed)
    return pen.dots(canvas, pts, None if dot_size is None else dot_size, seed=seed)
