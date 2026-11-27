"""
The Canvas: physical state of a painting and its rendering.

State layout (all numpy float32, shape (H, W[, NB]))
----------------------------------------------------
ground:
  h_ground      surface relief of the bare ground (height units)
  tooth         normalised local relief (~ -1.5..1.5) used for dry-brush contact
dried paint:
  below_R       spectral reflectance of everything that has dried (NB bands)
  h_dry         accumulated height of dried paint
  gloss_dry     gloss of dried paint
wet paint film (oil / acrylic / gouache), two sub-layers:
  film_K, film_S  optical depths K*x and S*x of the *top* wet layer (the
                most recent stroke at that pixel; NB bands; additive on mixing)
  film_vol      physical volume of the top wet layer
  base_K, base_S, base_vol   older wet paint underneath; a new stroke pushes
                the previous top layer down (``top_id`` = stroke id), so fresh
                paint lies on top and colours mix only via the brush
  wet           wetness 0..1 (1 = fresh); paint dries -> baked into below_R
  gloss, body, drate   volume-weighted properties of the wet paint
                (gloss, height per volume, drying rate in 1/seconds)
masking:
  mask          0..1, 1 = fully protected (masking fluid / tape)
watercolour (lazily allocated, see :class:`WatercolorState`)

Height of the paint surface = ground texture (levelled by paint) + h_dry
+ film_vol * body.
"""
from __future__ import annotations

import math
import time as _time
import numpy as np

from . import _kernels as K
from .grounds import make_ground
from .spectral import NB, SPEC_TO_RGB, linear_to_srgb, rgb_to_spectrum, km_composite_np
from .noise import blur


# how strongly paint levels the ground texture in the surface height
# (h = h_ground * exp(-LEVELING * paint_h) + paint_h); tools mirror this
LEVELING = 2.5


class WatercolorState:
    """Arrays of the watercolour layer (allocated on first use).

    water       surface water depth (height units; ~1 = puddle-wet)
    damp        water absorbed into the paper (0..1 saturation)
    K, S        suspended pigment optical depths (NB)
    m, g, st    suspended pigment mass, granulation-weighted mass,
                staining-weighted mass
    dK, dS      deposited (settled, still re-wettable) pigment (NB)
    dm, dg, dst deposited mass channels
    u, v        optional velocity fields (owned by the fluid simulation)
    bbox        [y0, y1, x0, x1] of the region that may contain water, or None
    """

    def __init__(self, H, W):
        z2 = lambda: np.zeros((H, W), dtype=np.float32)
        z3 = lambda: np.zeros((H, W, NB), dtype=np.float32)
        self.water, self.damp = z2(), z2()
        self.K, self.S = z3(), z3()
        self.m, self.g, self.st = z2(), z2(), z2()
        self.dK, self.dS = z3(), z3()
        self.dm, self.dg, self.dst = z2(), z2(), z2()
        self.u = np.zeros((H, W + 1), dtype=np.float32)
        self.v = np.zeros((H + 1, W), dtype=np.float32)
        self.bbox = None

    def grow_bbox(self, x0, y0, x1, y1, H, W, pad=8):
        if x1 < x0 or y1 < y0:
            return
        b = [max(0, y0 - pad), min(H, y1 + pad + 1), max(0, x0 - pad), min(W, x1 + pad + 1)]
        if self.bbox is None:
            self.bbox = b
        else:
            self.bbox = [min(self.bbox[0], b[0]), max(self.bbox[1], b[1]),
                         min(self.bbox[2], b[2]), max(self.bbox[3], b[3])]


class Canvas:
    """A virtual painting surface.

    Parameters
    ----------
    width, height : int        size in pixels
    ground : str               'canvas', 'linen', 'coarse_canvas', 'cold_press'
                               (or 'paper'), 'rough', 'hot_press', 'panel', 'smooth'
    ground_color : colour      colour of the gesso / paper (default depends on ground)
    texture_scale : float      scale of the ground texture
    relief : float             override height of the ground texture
    seed : int                 seed for all procedural randomness
    """

    def __init__(self, width, height, ground='canvas', ground_color=None,
                 texture_scale=1.0, relief=None, seed=0):
        self.W, self.H = int(width), int(height)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        H, W = self.H, self.W
        self.ground = make_ground(ground, (H, W), color=ground_color, relief=relief,
                                  scale=texture_scale, seed=seed)
        self.h_ground = self.ground.height
        hn = self.h_ground / (self.ground.relief + 1e-9)
        ao = (1.0 - 0.07 * (1.0 - hn))[..., None]
        self.below_R = (self.ground.reflectance[None, None, :] * ao).astype(np.float32)
        z2 = lambda: np.zeros((H, W), dtype=np.float32)
        self.film_K = np.zeros((H, W, NB), dtype=np.float32)
        self.film_S = np.zeros((H, W, NB), dtype=np.float32)
        # older wet paint underneath the most recent stroke
        self.base_K = np.zeros((H, W, NB), dtype=np.float32)
        self.base_S = np.zeros((H, W, NB), dtype=np.float32)
        self.base_vol = np.zeros((H, W), dtype=np.float32)
        self.top_id = np.zeros((H, W), dtype=np.int32)
        self.film_vol, self.wet, self.gloss, self.body, self.drate = z2(), z2(), z2(), z2(), z2()
        self.h_dry, self.gloss_dry = z2(), z2()
        self.mask = z2()
        self.wax = None  # crayon coverage, allocated on the first crayon stroke
        self.wc = None
        self.time = 0.0
        self.strokes = 0
        self.stroke_id = 1
        self.frames = []
        self.record_every = 0
        self.record_scale = 0.5
        self._checkpoints = []
        self._update_tooth()

    # ------------------------------------------------------------------ basics
    @property
    def shape(self):
        return (self.H, self.W)

    def ensure_watercolor(self):
        if self.wc is None:
            self.wc = WatercolorState(self.H, self.W)
        return self.wc

    def _update_tooth(self):
        """Normalised local relief used for dry-brush contact."""
        # dried paint fills the weave; its own (levelled) relief adds tooth
        h = self.h_ground * np.exp(-self.h_dry * 1.5) + self.h_dry
        d = h - blur(h, 3.0)
        if not hasattr(self, '_tooth_noise'):
            from .noise import spectral_noise
            self._tooth_noise = spectral_noise(self.shape, beta=1.4, seed=self.seed + 7, highcut=0.3)
            g0 = self.h_ground - blur(self.h_ground, 3.0)
            # absolute scale fixed by the bare ground, so smooth paint gives less tooth
            self._tooth_scale = float(max(g0.std(), 0.02))
        d = 0.8 * d / self._tooth_scale + 0.55 * self._tooth_noise
        self.tooth = np.clip(d / 2.0, -1.6, 1.6).astype(np.float32)

    # ------------------------------------------------------------------ painting
    def stroke(self, brush, path, **kw):
        """Paint one stroke with ``brush`` along ``path``.  See
        :meth:`pinsel.brush.Brush.stroke` for all options."""
        return brush.stroke(self, path, **kw)

    def _after_stroke(self):
        self.strokes += 1
        self.stroke_id = self.strokes + 1
        if self.record_every and self.strokes % self.record_every == 0:
            self.snapshot()

    # ------------------------------------------------------------------ masks
    def watercolor_mask(self, region=None):
        """Fraction of paper unavailable to water, including tape and crayon wax.

        A (yslice, xslice) region keeps local drops local. With no crayon on
        the canvas, return the original mask or a view into it.
        """
        mask = self.mask if region is None else self.mask[region]
        if self.wax is None:
            return mask
        wax = self.wax if region is None else self.wax[region]
        return mask + (1.0 - mask) * wax

    def mask_polygon(self, pts, value=1.0, feather=0.0):
        from PIL import Image, ImageDraw
        img = Image.new('F', (self.W, self.H), 0.0)
        ImageDraw.Draw(img).polygon([tuple(map(float, p)) for p in pts], fill=1.0)
        m = np.asarray(img, dtype=np.float32)
        if feather > 0:
            m = blur(m, feather)
        self.mask = np.maximum(self.mask, m * value).astype(np.float32)

    def mask_ellipse(self, cx, cy, rx, ry, value=1.0, feather=0.0):
        yy, xx = np.mgrid[0:self.H, 0:self.W]
        m = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0).astype(np.float32)
        if feather > 0:
            m = blur(m, feather)
        self.mask = np.maximum(self.mask, m * value).astype(np.float32)

    def mask_rect(self, x0, y0, x1, y1, value=1.0):
        self.mask[int(y0):int(y1), int(x0):int(x1)] = np.maximum(
            self.mask[int(y0):int(y1), int(x0):int(x1)], value)

    def set_mask(self, m):
        self.mask = np.clip(np.asarray(m, dtype=np.float32), 0, 1)

    def clear_mask(self):
        self.mask[:] = 0

    # ------------------------------------------------------------------ time
    def wait(self, seconds=60.0, watercolor_dt=None):
        """Let time pass: watercolour flows and dries, paint films dry by
        their medium's drying time (oil takes days!)."""
        seconds = float(seconds)
        if self.wc is not None and self.wc.bbox is not None:
            from . import watercolor
            watercolor.simulate(self, seconds)
        self.time += seconds
        live = (self.film_vol + self.base_vol) > 0
        if live.any():
            self.wet[live] = np.maximum(0.0, self.wet[live] - self.drate[live] * seconds)
            sel = live & (self.wet <= 1e-4)
            if sel.any():
                self._bake(sel)

    def dry(self):
        """Dry everything completely (as if the painting rested for weeks)."""
        if self.wc is not None:
            from . import watercolor
            watercolor.settle(self)
            sel = np.ones((self.H, self.W), dtype=np.bool_)
            K.bake_deposit(sel, self.below_R, self.wc.dK, self.wc.dS)
            self.wc.dm[:] = 0; self.wc.dg[:] = 0; self.wc.dst[:] = 0
        sel = (self.film_vol > 0) | (self.base_vol > 0) | (self.film_K.sum(axis=2) > 0) \
            | (self.base_K.sum(axis=2) > 0)
        self._bake(sel)

    def _bake(self, sel):
        h_before = self.h_dry.copy()
        K.bake_film(sel.astype(np.bool_), self.below_R, self.film_K, self.film_S,
                    self.film_vol, self.wet, self.gloss, self.body, self.drate,
                    self.h_dry, self.gloss_dry, 0.7, self.base_K, self.base_S, self.base_vol)
        # paint levels a little while it dries (viscous flow softens bristle ridges)
        # oil paint flows into the weave valleys and softens fine ridges as it dries
        added = self.h_dry - h_before
        self.h_dry = (h_before + 0.3 * added + 0.7 * blur(added, 1.6)).astype(np.float32)
        self._update_tooth()

    # ------------------------------------------------------------------ direct layers
    def tone(self, paint, thickness=0.3, variation=0.25, seed=None):
        """Lay a thin, even toned ground (imprimatura) over the whole canvas
        and let it dry -- a classic first step in oil painting."""
        # seed=None: use the canvas's own seeded stream, so a seeded canvas
        # paints the same picture every run (fresh OS entropy before 2026-11-27)
        g = np.random.default_rng(seed) if seed is not None else self.rng
        from .noise import fbm
        var = 1.0 + variation * fbm(self.shape, scale=self.W / 6, octaves=5, seed=g)
        t = (np.clip(thickness * var, 0, None) * (1.0 - self.mask))[..., None]
        self.film_K += (paint.K[None, None, :] * t).astype(np.float32)
        self.film_S += (paint.S[None, None, :] * t).astype(np.float32)
        self.film_vol += (t[..., 0] * 0.3).astype(np.float32)
        self.body[:] = 0.3
        self.gloss[:] = paint.gloss
        self.wet[:] = 1.0
        self._bake(np.ones(self.shape, dtype=np.bool_))

    def varnish(self, gloss=0.75, depth=0.12):
        """Apply a final varnish (e.g. dammar/satin) over the dried painting.

        Unifies ``gloss_dry`` toward ``gloss`` and subtly deepens dark passages
        (saturation boost from reduced surface scatter), respecting ``mask``.
        """
        m = (1.0 - self.mask).astype(np.float32)
        g_target = float(np.clip(gloss, 0.0, 1.0))
        self.gloss_dry = (self.gloss_dry * (1.0 - m) + g_target * m).astype(np.float32)
        d = float(np.clip(depth, 0.0, 0.6))
        if d > 0.0:
            R = np.clip(self.below_R, 1e-5, 1.0)
            deepened = np.power(R, 1.0 + d * m[..., None])
            self.below_R = np.clip(deepened, 0.0, 1.0).astype(np.float32)
        return self

    def fix(self, strength=1.0):
        """Spray fixative over unfixed dry/wet media (charcoal, pastel, etc.).

        Bakes ``strength`` (0..1) of the current film into ``below_R`` while
        keeping matte gloss, locking it against further smudging or lifting.
        """
        s = float(np.clip(strength, 0.0, 1.0))
        if s <= 0.0:
            return self
        sel = ((self.film_vol > 0) | (self.base_vol > 0)) & (self.mask < 0.999)
        if not sel.any():
            return self
        if s >= 0.999:
            self._bake(sel)
            return self
        # Partial fixative: bake a fraction into below_R and leave the rest loose
        f_keep = 1.0 - s
        m = (1.0 - self.mask) * s
        # Compute K/S fraction to bake into below_R
        bk = self.base_K * m[..., None]
        bs = self.base_S * m[..., None]
        fk = self.film_K * m[..., None]
        fs = self.film_S * m[..., None]
        self.below_R = km_composite_np(bk, bs, self.below_R).astype(np.float32)
        self.below_R = km_composite_np(fk, fs, self.below_R).astype(np.float32)
        keep_arr = (1.0 - m).astype(np.float32)
        self.base_K *= keep_arr[..., None]; self.base_S *= keep_arr[..., None]; self.base_vol *= keep_arr
        self.film_K *= keep_arr[..., None]; self.film_S *= keep_arr[..., None]; self.film_vol *= keep_arr
        self._update_tooth()
        return self

    # ------------------------------------------------------------------ rendering
    def _parse_region(self, region):
        if region is None:
            return 0, self.H, 0, self.W
        if len(region) == 2 and isinstance(region[0], slice) and isinstance(region[1], slice):
            y0, y1, _ = region[0].indices(self.H)
            x0, x1, _ = region[1].indices(self.W)
            return y0, y1, x0, x1
        y0, y1, x0, x1 = (int(v) for v in region)
        return max(0, y0), min(self.H, y1), max(0, x0), min(self.W, x1)

    def surface_height(self, region=None):
        y0, y1, x0, x1 = self._parse_region(region)
        sl = (slice(y0, y1), slice(x0, x1))
        paint_h = self.h_dry[sl] + (self.film_vol[sl] + self.base_vol[sl]) * self.body[sl]
        level = np.exp(-paint_h * LEVELING)
        return self.h_ground[sl] * level + paint_h

    def render_linear(self, region=None):
        """Linear sRGB colour (without lighting), float32 (h, w, 3).

        Pass ``region=(y0, y1, x0, x1)`` (or ``(yslice, xslice)``) to render
        only a sub-rectangle of the canvas.
        """
        y0, y1, x0, x1 = self._parse_region(region)
        sl = (slice(y0, y1), slice(x0, x1))
        out = np.empty((y1 - y0, x1 - x0, 3), dtype=np.float32)
        wc = self.wc
        if wc is not None:
            K.render_reflectance(self.below_R[sl], self.base_K[sl], self.base_S[sl],
                                 self.film_K[sl], self.film_S[sl], True,
                                 wc.K[sl], wc.S[sl], wc.dK[sl], wc.dS[sl],
                                 wc.water[sl], wc.damp[sl], SPEC_TO_RGB, out)
        else:
            d3 = np.zeros((1, 1, NB), dtype=np.float32)
            d2 = np.zeros((1, 1), dtype=np.float32)
            K.render_reflectance(self.below_R[sl], self.base_K[sl], self.base_S[sl],
                                 self.film_K[sl], self.film_S[sl], False,
                                 d3, d3, d3, d3, d2, d2, SPEC_TO_RGB, out)
        return out

    def render(self, lighting=True, light_dir=(-0.55, -0.65, 0.75), relief=1.0,
               gloss=1.0, ambient=0.35, specular=0.25, shininess=40.0, exposure=1.0,
               region=None):
        """Render to an sRGB float image (h, w, 3) in 0..1.

        Lighting simulates a lamp at ``light_dir`` raking across the paint
        relief (impasto, canvas weave) with glossy highlights on wet oil.
        Pass ``region=(y0, y1, x0, x1)`` to render only a sub-rectangle.
        """
        y0, y1, x0, x1 = self._parse_region(region)
        rgb = self.render_linear(region=(y0, y1, x0, x1)) * exposure
        if lighting:
            pad = 6 if region is not None else 0
            py0, py1 = max(0, y0 - pad), min(self.H, y1 + pad)
            px0, px1 = max(0, x0 - pad), min(self.W, x1 + pad)
            psl = (slice(py0, py1), slice(px0, px1))
            crop = (slice(y0 - py0, y1 - py0), slice(x0 - px0, x1 - px0))
            hraw = self.surface_height(region=(py0, py1, px0, px1)).astype(np.float64)
            # wet paint has already levelled slightly; avoid razor-sharp ridges
            hgt = 0.35 * hraw + 0.65 * blur(hraw, 0.8)
            gy, gx = np.gradient(hgt)
            k = 0.9 * relief
            nx, ny, nz = -gx * k, -gy * k, np.ones_like(hgt)
            nl = np.sqrt(nx * nx + ny * ny + 1.0)
            nx, ny, nz = nx / nl, ny / nl, nz / nl
            L = np.asarray(light_dir, dtype=np.float64)
            L = L / np.linalg.norm(L)
            diff = np.maximum(nx * L[0] + ny * L[1] + nz * L[2], 0.0)
            shade = (ambient + (1 - ambient) * diff) / (ambient + (1 - ambient) * L[2])
            Hv = L + np.array([0, 0, 1.0])
            Hv /= np.linalg.norm(Hv)
            # specular from slightly smoothed relief (avoids pixel glitter)
            sgy, sgx = np.gradient(blur(hgt, 1.3))
            snx, sny = -sgx * k, -sgy * k
            snl = np.sqrt(snx * snx + sny * sny + 1.0)
            nh = np.maximum((snx * Hv[0] + sny * Hv[1] + Hv[2]) / snl, 0.0)
            # gloss: wet film, dried film, water surface
            fvol = self.film_vol[psl] + self.base_vol[psl]
            gl = np.where(fvol > 1e-4, self.gloss[psl], 0.0) * np.clip(fvol * 3, 0, 1)
            gl = np.maximum(gl, self.gloss_dry[psl] * np.clip(self.h_dry[psl] * 2, 0, 1))
            if self.wc is not None:
                gl = np.maximum(gl, 0.8 * np.clip(self.wc.water[psl] * 2, 0, 1))
            # specular only shows where the surface is tilted towards the half vector
            flat_nh = Hv[2]
            spec = specular * gloss * gl * np.maximum(nh ** shininess - flat_nh ** shininess * 0.9, 0.0)
            rgb = rgb * shade[crop][..., None].astype(np.float32) + spec[crop][..., None].astype(np.float32)
        return np.clip(linear_to_srgb(rgb), 0, 1).astype(np.float32)

    def to_image(self, **kw):
        from PIL import Image
        return Image.fromarray((self.render(**kw) * 255 + 0.5).astype(np.uint8))

    def save(self, path, **kw):
        img = self.to_image(**kw)
        img.save(path)
        return img

    # ------------------------------------------------------------------ timelapse
    def record(self, every=10, scale=0.5):
        """Capture a timelapse frame every ``every`` strokes."""
        self.record_every = int(every)
        self.record_scale = scale

    def preview(self, scale=0.5, light_dir=(-0.55, -0.65, 0.75)):
        """Fast, downsampled render with simplified lighting (for timelapses)."""
        from PIL import Image
        f = max(1, int(round(1.0 / scale)))
        h, w = (self.H // f) * f, (self.W // f) * f
        rgb = self.render_linear()[:h, :w].reshape(h // f, f, w // f, f, 3).mean(axis=(1, 3))
        hgt = self.surface_height()[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))
        gy, gx = np.gradient(hgt.astype(np.float64))
        k = 0.9 / f
        L = np.asarray(light_dir, float); L /= np.linalg.norm(L)
        nl = np.sqrt((gx * k) ** 2 + (gy * k) ** 2 + 1)
        diff = np.maximum((-gx * k * L[0] - gy * k * L[1] + L[2]) / nl, 0)
        shade = (0.35 + 0.65 * diff) / (0.35 + 0.65 * L[2])
        out = np.clip(linear_to_srgb(rgb * shade[..., None]), 0, 1)
        return Image.fromarray((out * 255 + 0.5).astype(np.uint8))

    def snapshot(self):
        img = self.preview(self.record_scale)
        self.frames.append(img)
        return img

    def save_timelapse(self, path, fps=12, hold_last=2.0, max_frames=90, width=None):
        """Write recorded frames as an animated GIF (subsampled to ``max_frames``)."""
        if not self.frames:
            return None
        from PIL import Image
        idx = np.unique(np.linspace(0, len(self.frames) - 1, min(max_frames, len(self.frames))).astype(int))
        frames = [self.frames[i] for i in idx]
        if width:
            frames = [f.resize((width, int(f.height * width / f.width)), Image.LANCZOS) for f in frames]
        frames = frames + [frames[-1]] * int(hold_last * fps)
        frames[0].save(path, save_all=True, append_images=frames[1:],
                       duration=int(1000 / fps), loop=0, optimize=False)
        return path

    # ------------------------------------------------------------------ checkpoints
    _STATE = ('below_R', 'film_K', 'film_S', 'film_vol', 'base_K', 'base_S', 'base_vol', 'top_id',
              'wet', 'gloss', 'body', 'drate', 'h_dry', 'gloss_dry', 'mask', 'tooth')

    def save_state(self, path):
        """Save the full painting state (re-render or keep painting later)."""
        st = self.get_state()
        wc = st.pop('_wc', None)
        extra = {f'wc__{k}': v for k, v in (wc or {}).items() if isinstance(v, np.ndarray)}
        if wc and wc.get('bbox') is not None:
            extra['wc__bbox'] = np.asarray(wc['bbox'], dtype=np.int64)
        np.savez(path, _time=st.pop('_time'), **st, **extra)

    def load_state(self, path):
        d = np.load(path)
        st = {k: d[k] for k in self._STATE}
        st['_time'] = float(d['_time'])
        if 'wax' in d:
            st['wax'] = d['wax']
        if '_strokes' in d:
            st['_strokes'] = int(d['_strokes'])
        wc = {k[4:]: d[k] for k in d.files if k.startswith('wc__')}
        if wc:
            if 'bbox' in wc:
                wc['bbox'] = wc['bbox'].tolist()
            else:
                # Older files saved water and pigment without the active box.
                ys, xs = np.nonzero((wc['water'] > 0) | (wc['damp'] > 0) | (wc['m'] > 0))
                wc['bbox'] = [int(ys.min()), int(ys.max()) + 1,
                              int(xs.min()), int(xs.max()) + 1] if ys.size else None
            st['_wc'] = wc
        self.set_state(st)

    def get_state(self):
        st = {k: getattr(self, k).copy() for k in self._STATE}
        if self.wax is not None:
            st['wax'] = self.wax.copy()
        if self.wc is not None:
            import copy
            st['_wc'] = {k: (v.copy() if isinstance(v, np.ndarray) else copy.deepcopy(v))
                         for k, v in self.wc.__dict__.items()}
        st['_time'] = self.time
        st['_strokes'] = self.strokes
        return st

    def set_state(self, st):
        self.wax = st['wax'].copy() if 'wax' in st else None
        for k in self._STATE:
            setattr(self, k, st[k].copy())
        if '_wc' in st:
            self.ensure_watercolor()
            import copy
            for k, v in st['_wc'].items():
                setattr(self.wc, k, v.copy() if isinstance(v, np.ndarray) else copy.deepcopy(v))
        else:
            self.wc = None
        self.time = st['_time']
        # Older checkpoints have layer IDs but no explicit stroke counter.
        self.strokes = int(st['_strokes'] if '_strokes' in st else self.top_id.max())
        self.stroke_id = self.strokes + 1

    # ------------------------------------------------------------------ undo / stack
    @property
    def checkpoints(self):
        """List of labels for saved in-memory checkpoints (oldest -> newest)."""
        return [lbl for lbl, _ in self._checkpoints]

    def checkpoint(self, label=None):
        """Push an in-memory snapshot onto the undo stack; returns its index."""
        self._checkpoints.append((label, self.get_state()))
        return len(self._checkpoints) - 1

    def undo(self, steps=1):
        """Pop ``steps`` checkpoint(s) from the stack and restore the target state."""
        steps = int(steps)
        if steps < 1 or steps > len(self._checkpoints):
            raise IndexError(f'cannot undo {steps} step(s); stack has {len(self._checkpoints)}')
        for _ in range(steps):
            label, st = self._checkpoints.pop()
        self.set_state(st)
        return label

    def restore(self, index_or_label):
        """Restore a saved checkpoint by integer index or string label without popping."""
        if isinstance(index_or_label, int):
            label, st = self._checkpoints[index_or_label]
        else:
            for label, st in reversed(self._checkpoints):
                if label == index_or_label:
                    break
            else:
                raise KeyError(f'no checkpoint named {index_or_label!r}')
        self.set_state(st)
        return label
