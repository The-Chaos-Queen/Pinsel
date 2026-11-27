"""
Bristle brushes.

A :class:`Brush` is a bundle of individually simulated bristles.  Every
bristle has its own root position in the ferrule, length (the pressure at
which it touches the canvas), stiffness (how much its tip lags behind in
curves), contact radius and its *own paint reservoir*.  During a stroke each
bristle

* deposits paint proportional to its load, pressure and contact with the
  canvas tooth (a nearly empty brush only touches the peaks of the weave:
  dry-brush / scumbling happens by itself),
* picks up wet paint from the canvas (colours smear and the brush gets
  dirty, just like a real one),
* ploughs wet paint aside (stiff hog bristles leave ridges),
* runs out of paint at its own pace, so strokes break up into streaks.

Watercolour / ink brushes exchange water and suspended pigment with the
watercolour layer instead (and a damp brush lifts paint back up).

Shapes: 'round', 'flat', 'bright', 'filbert', 'angular', 'fan', 'rigger',
'mop', 'hake', 'hog' (coarse stiff round), 'sable' (soft fine round).
"""
from __future__ import annotations

import math
import numpy as np

from . import _kernels as K
from . import _backend as B
from .path import resample, arclength, wobble as _wobble, taper
from .pigments import Paint
from .spectral import NB
from .noise import noise1d

SHAPES = {
    #           layout  aspect  stiff  splay  sx0   sx1  sy0  sy1  gamma  tip       density coarse plough pickup reach
    'round':   ('disc', 1.00, 0.50, 0.25, 0.35, 1.0, 0.35, 1.0, 0.70, 'pointed', 5.0, 1.00, 0.10, 0.35, 1.0),
    'sable':   ('disc', 1.00, 0.35, 0.15, 0.30, 1.0, 0.30, 1.0, 0.70, 'pointed', 7.0, 0.80, 0.00, 0.3, 1.1),
    'hog':     ('disc', 1.00, 0.85, 0.45, 0.55, 1.0, 0.55, 1.0, 0.80, 'blunt', 3.2, 1.35, 0.45, 0.45, 0.8),
    'flat':    ('rect', 0.20, 0.70, 0.30, 0.90, 1.0, 0.40, 1.0, 0.80, 'square', 4.5, 1.10, 0.30, 0.4, 1.0),
    'bright':  ('rect', 0.28, 0.85, 0.35, 0.90, 1.0, 0.45, 1.0, 0.80, 'square', 4.0, 1.20, 0.50, 0.45, 0.7),
    'filbert': ('rect', 0.24, 0.60, 0.25, 0.75, 1.0, 0.40, 1.0, 0.75, 'oval', 5.0, 1.00, 0.20, 0.35, 1.0),
    'angular': ('rect', 0.20, 0.65, 0.25, 0.90, 1.0, 0.40, 1.0, 0.80, 'angled', 4.5, 1.00, 0.20, 0.35, 1.0),
    'fan':     ('fan',  0.10, 0.30, 0.20, 0.95, 1.0, 0.60, 1.0, 0.80, 'square', 1.6, 0.85, 0.00, 0.3, 0.8),
    'rigger':  ('disc', 1.00, 0.12, 0.10, 0.40, 1.0, 0.40, 1.0, 0.60, 'pointed', 6.0, 0.80, 0.00, 0.2, 3.0),
    'mop':     ('disc', 1.00, 0.20, 0.30, 0.30, 1.0, 0.30, 1.0, 0.60, 'pointed', 6.0, 1.00, 0.00, 0.4, 1.6),
    'hake':    ('rect', 0.12, 0.25, 0.25, 0.95, 1.0, 0.50, 1.0, 0.70, 'square', 5.0, 1.00, 0.00, 0.3, 1.4),
}


class Brush:
    """A simulated bristle brush.

    Parameters
    ----------
    shape : str           one of :data:`SHAPES`
    size : float          brush radius in pixels (half the widest mark)
    bristles : int        number of bristles (default: derived from size)
    stiffness : float     0 (floppy sable) .. 1 (stiff hog); tips lag in curves
    splay : float         how much bristles fan out under heavy pressure
    bristle_radius : float  contact radius of one bristle in px
    reach : float         distance (px) a full load lasts before thinning out
    pickup : float        0..1, how strongly wet canvas paint is picked up
    plough : float        0..1, how much wet paint is pushed into ridges
    seed : int            randomness of the bristle bundle
    """

    def __init__(self, shape='round', size=10.0, bristles=None, stiffness=None,
                 splay=None, bristle_radius=None, reach=None, pickup=None,
                 plough=None, aspect=None, seed=None, contact_softness=0.35):
        if shape not in SHAPES:
            raise ValueError(f'unknown brush shape {shape!r}; choose from {list(SHAPES)}')
        (layout, asp, stiff, spl, sx0, sx1, sy0, sy1, gam, tip, dens, coarse,
         plo, pick, reach_f) = SHAPES[shape]
        self.shape = shape
        self.size = float(size)
        self.aspect = asp if aspect is None else aspect
        self.stiffness = stiff if stiffness is None else stiffness
        self.splay = spl if splay is None else splay
        self.sx0, self.sx1, self.sy0, self.sy1, self.sgamma = sx0, sx1, sy0, sy1, gam
        self.pickup = pick if pickup is None else pickup
        self.plough = plo if plough is None else plough
        self.contact_softness = contact_softness
        self.rng = np.random.default_rng(seed)
        g = self.rng
        rb = bristle_radius or float(np.clip((0.55 + 0.045 * self.size) * coarse, 0.55, 3.0))
        self.bristle_radius = rb
        if bristles is None:
            span = 2 * self.size if layout != 'rect' else 2 * self.size
            bristles = int(np.clip(dens * span / rb, 6, 1400))
        N = int(bristles)
        self.N = N
        self.reach = float(reach or (60.0 * self.size ** 0.6 + 100.0) * reach_f)
        self.spacing = float(np.clip(0.35 * rb, 0.3, 1.0))

        # ---- bristle layout in unit coordinates (x = across width, y = thickness)
        if layout == 'disc':
            i = np.arange(N) + 0.5
            r = np.sqrt(i / N)
            th = i * math.pi * (3 - math.sqrt(5))
            off = np.column_stack([r * np.cos(th), r * np.sin(th)])
            off += g.normal(0, 0.6 / math.sqrt(N), off.shape)
            rr = np.linalg.norm(off, axis=1)
            off[rr > 1] /= rr[rr > 1, None]
            off[:, 1] *= self.aspect
            rnorm = np.minimum(np.linalg.norm(off, axis=1), 1.0)
        elif layout == 'rect':
            off = np.column_stack([g.uniform(-1, 1, N), g.uniform(-1, 1, N) * self.aspect])
            rnorm = np.abs(off[:, 0])
        else:  # fan: thin arc made of clumps
            nclump = max(5, int(N / 7))
            centers = np.sort(g.uniform(-1, 1, nclump))
            cid = g.integers(0, nclump, N)
            x = np.clip(centers[cid] + g.normal(0, 0.6 / nclump, N), -1, 1)
            y = (np.cos(x * 0.9) - np.cos(0.45)) * 0.35 + g.normal(0, 0.03, N)
            off = np.column_stack([x, y])
            rnorm = np.abs(x)
        self.offsets = off.astype(np.float64)

        # ---- bristle lengths -> pressure threshold for contact
        nz = g.normal(0, 1, N)
        if tip == 'pointed':
            thr = 0.82 * rnorm ** 1.6 + 0.04 * nz
        elif tip == 'blunt':
            thr = 0.35 * rnorm ** 2 + 0.06 * nz
        elif tip == 'oval':
            thr = 0.6 * rnorm ** 2.2 + 0.04 * nz
        elif tip == 'angled':
            thr = 0.5 * (off[:, 0] + 1) / 2 + 0.04 * nz
        else:
            thr = 0.04 + 0.05 * np.abs(nz)
        self.thresholds = np.clip(thr, 0.0, 0.95)

        # ---- per-bristle stiffness, radius, splay direction
        st = np.clip(self.stiffness + g.normal(0, 0.08, N), 0.02, 1.0)
        self._follow_base = (0.18 + 0.82 * st ** 0.8)
        self.radii = rb * np.clip(g.normal(1.0, 0.18, N), 0.5, 1.6)
        self.bsplay = np.clip(g.normal(0.5, 0.35, N), 0, 1.5)
        self.cap = np.clip(g.normal(1.0, 0.2, N), 0.4, 1.6)
        self.thickness = 1.5
        self.cap *= (2 * self.size * self.reach * self.thickness) / self.cap.sum()
        # some bristles release paint more readily than others -> streaks
        self.bdep = np.exp(g.normal(0, 0.35, N))

        # ---- reservoirs
        self.res_K = np.zeros((N, NB))
        self.res_S = np.zeros((N, NB))
        self.res_vol = np.zeros(N)
        self.res_water = np.zeros(N)
        # paint picked up from the canvas sits loosely on the bristle tips
        self.tip_K = np.zeros((N, NB))
        self.tip_S = np.zeros((N, NB))
        self.tip_vol = np.zeros(N)
        self.tip_reach = 1.2 * self.size + 8.0
        self.paint = None
        self._props = dict(fluid=False, gloss=0.8, body=1.0, drate=1e-5, gran=0.0, stain=0.3)
        self._tips = np.full((N, 2), np.nan)

    # ------------------------------------------------------------- presets
    @classmethod
    def round(cls, size=8, **kw):
        return cls('round', size, **kw)

    @classmethod
    def flat(cls, size=14, **kw):
        return cls('flat', size, **kw)

    @classmethod
    def filbert(cls, size=12, **kw):
        return cls('filbert', size, **kw)

    @classmethod
    def fan(cls, size=20, **kw):
        return cls('fan', size, **kw)

    @classmethod
    def rigger(cls, size=2.5, **kw):
        return cls('rigger', size, **kw)

    @classmethod
    def mop(cls, size=30, **kw):
        return cls('mop', size, **kw)

    @classmethod
    def hog(cls, size=12, **kw):
        return cls('hog', size, **kw)

    # ------------------------------------------------------------- paint handling
    @property
    def load_level(self):
        """Remaining paint, 1 = fully loaded."""
        if self._props['fluid']:
            return float(self.res_water.sum() / self.cap.sum())
        return float(self.res_vol.sum() / self.cap.sum())

    def _set_props(self, paint):
        self.paint = paint
        self._props = dict(fluid=bool(paint.fluid), gloss=float(paint.gloss), body=float(paint.body),
                           drate=1.0 / max(float(paint.dry_time), 1e-3),
                           gran=float(paint.granulation), stain=float(paint.staining))

    def clean(self):
        """Rinse the brush completely."""
        self.res_K[:] = 0; self.res_S[:] = 0; self.res_vol[:] = 0; self.res_water[:] = 0
        self.tip_K[:] = 0; self.tip_S[:] = 0; self.tip_vol[:] = 0
        return self

    def wipe(self, fraction=0.6):
        """Wipe on a rag: removes a fraction of paint (and water)."""
        f = 1 - fraction
        self.res_K *= f; self.res_S *= f; self.res_vol *= f; self.res_water *= f
        self.tip_K *= f * f; self.tip_S *= f * f; self.tip_vol *= f * f
        return self

    def load(self, paint, amount=1.0, mix=1.0, jitter=0.25, water=1.0, clean=True):
        """Load the brush.

        paint   : a :class:`Paint`, or a list ``[(paint, weight), ...]`` to load
                  several colours at once (streaky, partially mixed loading)
        amount  : 1 = full load, 2 = overloaded (thick impasto), 0.2 = nearly dry
        mix     : 1 = colours fully mixed, 0 = each bristle carries one colour
        jitter  : random variation of load between bristles (streakiness)
        water   : watercolour only: water content (1 = juicy, 0.2 = damp)
        clean   : rinse first (False = dip a dirty brush into new paint)
        """
        if clean:
            self.clean()
        paints = paint if isinstance(paint, (list, tuple)) else [(paint, 1.0)]
        paints = [(p, 1.0) if isinstance(p, Paint) else p for p in paints]
        g = self.rng
        N = self.N
        w = np.array([float(x[1]) for x in paints])
        w /= w.sum()
        if len(paints) > 1:
            lat = self.offsets[:, 0]
            nzc = noise1d(256, scale=float(g.uniform(12, 30)), seed=g)
            idx = np.clip(((lat + 1) / 2 * 255).astype(int), 0, 255)
            q = 0.5 + 0.5 * np.tanh(nzc[idx] * 1.2) * 0.999
            q = np.clip(q + g.normal(0, 0.08, N), 0, 0.9999)
            cat = np.searchsorted(np.cumsum(w), q)
            onehot = np.zeros((N, len(paints)))
            onehot[np.arange(N), np.clip(cat, 0, len(paints) - 1)] = 1
            comp = mix * w[None, :] + (1 - mix) * onehot
        else:
            comp = np.ones((N, 1))
        Kp = np.array([p.K for p, _ in paints])
        Sp = np.array([p.S for p, _ in paints])
        lv = np.clip(1.0 + jitter * g.normal(0, 1, N), 0.05, 2.0)
        main = paints[int(np.argmax(w))][0]
        self._set_props(main)
        if self._props['fluid']:
            # watercolour: water carries pigment; amount = pigment concentration
            wat = self.cap * float(water) * np.clip(lv, 0.5, 1.5)
            pig = wat * float(amount) * 0.35
            self.res_water += wat
            self.res_vol += pig
            self.res_K += pig[:, None] * (comp @ Kp)
            self.res_S += pig[:, None] * (comp @ Sp)
        else:
            vol = self.cap * float(amount) * lv
            self.res_vol += vol
            self.res_K += vol[:, None] * (comp @ Kp)
            self.res_S += vol[:, None] * (comp @ Sp)
        return self

    def side_load(self, paint_a, paint_b, amount=1.0, softness=0.3, **kw):
        """Load colour A on one side of the brush and B on the other."""
        self.load(paint_a, amount, **kw)
        x = self.offsets[:, 0]
        fb = np.clip(0.5 + x / (2 * max(softness, 1e-3)), 0, 1)
        vol = self.res_vol.copy()
        self.res_K = vol[:, None] * ((1 - fb)[:, None] * paint_a.K + fb[:, None] * paint_b.K) \
            * (1 if not self._props['fluid'] else 1)
        self.res_S = vol[:, None] * ((1 - fb)[:, None] * paint_a.S + fb[:, None] * paint_b.S)
        return self

    def dip(self, paint, amount=0.3, water=0.6):
        """Dip the (possibly dirty) brush into more paint without rinsing."""
        return self.load(paint, amount, clean=False, water=water)

    def add_water(self, amount=0.5):
        """Dip in water (watercolour): dilutes what is on the brush."""
        self.res_water += self.cap * amount
        return self

    # ------------------------------------------------------------- stroke
    def stroke(self, canvas, path, pressure=None, angle=None, angle_offset=0.0,
               speed=1.0, wobble=0.0, pressure_jitter=0.0, seed=None):
        """Drag the brush along ``path``.

        path      : (N, 2) points, or (N, 3) with per-point pressure
        pressure  : None (natural taper), a number, an array per input point,
                    or a function of t in 0..1 (see :mod:`pinsel.path`)
        angle     : None -> brush width stays perpendicular to the motion
                    (broad strokes with flats); a number = fixed orientation
                    of the brush's wide axis in degrees (calligraphic);
                    or a function of t returning degrees
        angle_offset : degrees added to the orientation (e.g. 90 to drag a
                    flat sideways for thin edge strokes)
        speed     : >1 faster (less paint, more skipping), <1 slow and wet
        wobble    : hand tremor amplitude in px
        """
        pts = np.asarray(path, dtype=float)
        if pts.ndim == 1:
            pts = pts[None, :]
        if pts.ndim != 2 or len(pts) < 1:
            return None
        if arclength(pts)[-1] < 0.5:
            return self._dab(canvas, pts[0], pressure, angle, angle_offset, speed)
        has_p = pts.shape[1] >= 3
        s0 = arclength(pts)
        L0 = max(s0[-1], 1e-9)
        if pressure is not None and not callable(pressure) and np.ndim(pressure) == 1:
            pcol = np.interp(s0, s0, np.asarray(pressure, float)) if len(pressure) == len(pts) \
                else np.interp(s0 / L0, np.linspace(0, 1, len(pressure)), np.asarray(pressure, float))
            pts = np.column_stack([pts[:, :2], pcol])
            has_p = True
        # seed=None: the brush's own seeded stream (fresh OS entropy before 2026-11-27)
        g_hand = np.random.default_rng(seed) if seed is not None else self.rng
        if wobble > 0:
            pts = _wobble(pts, amount=wobble, scale=max(20.0, 3 * self.size), seed=g_hand)
        rs = resample(pts, self.spacing)
        s = arclength(rs)
        L = max(s[-1], 1e-9)
        t = s / L
        if has_p and pressure is None or (has_p and not callable(pressure) and np.ndim(pressure) == 1):
            pp = rs[:, 2]
        elif pressure is None:
            pp = taper()(t)
        elif callable(pressure):
            pp = pressure(t)
        else:
            pp = np.full(len(t), float(pressure))
        pp = np.clip(np.asarray(pp, float), 0, 1)
        if pressure_jitter > 0:
            g = g_hand
            n = int(L / max(4.0, self.size)) + 4
            pp = np.clip(pp * (1 + pressure_jitter * np.interp(t * (n - 1), np.arange(n), g.normal(0, 1, n))), 0, 1)
        xy = rs[:, :2]
        # smoothed travel direction
        d = np.gradient(xy, axis=0)
        phi = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        k = max(1, int(self.size * 0.5 / self.spacing))
        if len(phi) > 2 * k + 1:
            ker = np.ones(2 * k + 1) / (2 * k + 1)
            phi = np.convolve(np.pad(phi, k, mode='edge'), ker, mode='valid')
        if angle is None:
            pa = phi + math.pi / 2 + math.radians(angle_offset)
        elif callable(angle):
            pa = np.radians(np.asarray(angle(t), float) + angle_offset)
        else:
            pa = np.full(len(t), math.radians(float(angle) + angle_offset))
        return self._run_kernel(canvas, xy[:, 0], xy[:, 1], pp, pa, speed)

    def _run_kernel(self, canvas, px, py, pp, pa, speed):
        """Run the bristle simulation on an already resampled path."""
        follow = 1.0 - (1.0 - self._follow_base) ** (self.spacing / 0.5)
        self._tips[:] = np.nan
        alpha = 1.0 / self.reach
        pr = self._props
        px = np.ascontiguousarray(px, dtype=np.float64); py = np.ascontiguousarray(py, dtype=np.float64)
        pp = np.ascontiguousarray(pp, dtype=np.float64); pa = np.ascontiguousarray(pa, dtype=np.float64)
        if pr['fluid']:
            wc = canvas.ensure_watercolor()
            bbox = K.fluid_stroke(px, py, pp, pa, self.spacing, float(speed),
                                  self.offsets, self.thresholds, follow, self.radii, self.bsplay, self.bdep,
                                  self._tips, self.res_K, self.res_S, self.res_vol, self.res_water,
                                  self.cap, self.size, self.sx0, self.sx1, self.sy0, self.sy1,
                                  self.sgamma, self.splay, alpha * 1.4, self.pickup * 1.5,
                                  self.contact_softness, pr['gran'], pr['stain'],
                                  wc.water, wc.K, wc.S, wc.m, wc.g, wc.st,
                                  canvas.tooth, canvas.watercolor_mask(), 1.0)
            wc.grow_bbox(*bbox, canvas.H, canvas.W, pad=int(self.size) + 10)
        else:
            bbox = B.paste_stroke(px, py, pp, pa, self.spacing, float(speed),
                                  self.offsets, self.thresholds, follow, self.radii, self.bsplay, self.bdep,
                                  self._tips, self.res_K, self.res_S, self.res_vol, self.cap,
                                  self.tip_K, self.tip_S, self.tip_vol, 1.0 / self.tip_reach,
                                  self.size, self.sx0, self.sx1, self.sy0, self.sy1, self.sgamma,
                                  self.splay, alpha, self.pickup, self.plough,
                                  self.contact_softness, pr['gloss'], pr['body'], pr['drate'],
                                  canvas.film_K, canvas.film_S, canvas.film_vol, canvas.wet,
                                  canvas.gloss, canvas.body, canvas.drate, canvas.tooth,
                                  canvas.mask, 1.2, canvas.base_K, canvas.base_S,
                                  canvas.base_vol, canvas.top_id, int(canvas.stroke_id))
        canvas._after_stroke()
        return bbox

    def dab(self, canvas, x, y, pressure=None, angle=0.0, speed=1.0):
        """Press the brush straight down and lift it again (stippling, leaves,
        flower petals).  ``pressure`` = peak pressure, or a profile f(t)."""
        return self._dab(canvas, (x, y), pressure, angle, 0.0, speed)

    def _dab(self, canvas, p0, pressure, angle, angle_offset, speed):
        n = 24
        t = np.linspace(0, 1, n)
        if pressure is None:
            pp = 0.9 * np.sin(np.pi * t) ** 0.6
        elif callable(pressure):
            pp = pressure(t)
        else:
            pp = float(pressure) * np.sin(np.pi * t) ** 0.6
        # an almost stationary path: the hand never stands perfectly still
        jit = self.rng.normal(0, 0.03, (n, 2)).cumsum(axis=0)
        a = 0.0 if (angle is None or callable(angle)) else float(angle)
        pa = np.full(n, math.radians(a + angle_offset))
        return self._run_kernel(canvas, p0[0] + jit[:, 0], p0[1] + jit[:, 1],
                                np.clip(pp, 0, 1), pa, speed)

    def __repr__(self):
        return (f'<Brush {self.shape} size={self.size:g} bristles={self.N} '
                f'load={self.load_level:.2f}>')
