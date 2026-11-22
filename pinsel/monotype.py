"""A hand-inked glass plate for making a monotype and its successive ghost prints.

Supports both classic RGB colours (Beer-Lambert attenuation) and ``pinsel.Paint``
pigments (12-band Kubelka-Munk spectral compositing), and can pull an impression
either as a standalone PIL image (:meth:`pull`) or directly onto a
:class:`pinsel.Canvas` (:meth:`pull_onto`) for mixed-media painting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from .noise import spectral_noise
from .pigments import Paint
from .spectral import (NB, km_composite_np, parse_color, rgb_to_spectrum,
                       spectrum_to_srgb)


@dataclass
class _Mark:
    region: tuple[slice, slice]
    mass: np.ndarray
    rgb: Optional[np.ndarray]
    K: Optional[np.ndarray] = None
    S: Optional[np.ndarray] = None


class MonotypePlate:
    """Lay ink in plate coordinates; each impression transfers and reverses the image.

    Masks are floating point coverage in [0, 1], or a scalar covering the plate.
    Load is the thickness deposited at full coverage. At about 0.7, a fresh
    mark covers most of an underlying colour. Wiping exposes the glass.
    Pass a :class:`pinsel.Paint` to :meth:`ink` for 12-band spectral
    Kubelka-Munk mixing, or any sRGB colour for classic RGB Beer-Lambert inks.
    """

    def __init__(self, width, height, seed=0):
        self.shape = int(height), int(width)
        if min(self.shape) <= 0:
            raise ValueError('plate dimensions must be positive')
        self._rng = np.random.default_rng(seed)
        self._marks: list[_Mark] = []
        self._mass = np.zeros(self.shape, np.float32)
        self._pressure = spectral_noise(self.shape, beta=2.1, seed=self._rng)
        self._glass = self._substrate([218, 211, 192], self._pressure)
        self._preview = self._glass.copy()
        self._has_spectral = False

    def _substrate(self, color, grain):
        texture = .009 * grain + .004 * self._rng.standard_normal(self.shape)
        return ((1 + texture[..., None]) * (np.array(color) / 255)).astype(np.float32)

    def _spectral_substrate(self, color, grain):
        texture = .009 * grain + .004 * self._rng.standard_normal(self.shape)
        R0 = rgb_to_spectrum(np.array(color) / 255.0).astype(np.float32)
        return np.clip((1.0 + texture[..., None]) * R0[None, None, :], 1e-4, 0.999).astype(np.float32)

    def _coverage(self, mask):
        raw = np.asarray(mask, dtype=np.float32)
        if raw.ndim == 0:
            raw = np.broadcast_to(raw, self.shape)
        if raw.shape != self.shape or not np.isfinite(raw).all():
            raise ValueError('expected a finite scalar or coverage mask with plate dimensions')
        return np.clip(raw, 0, 1)

    @staticmethod
    def _over(surface, mass, rgb):
        # Beer-Lambert attenuation in three colour channels. A grain can carry
        # multiple colours; each successive mark partially obscures the last.
        alpha = -np.expm1(-6.2 * mass)[..., None]
        surface += (rgb - surface) * alpha

    @staticmethod
    def _over_spectral(surface_R, mass, K, S):
        # Kubelka-Munk spectral compositing in 12 bands (scaled so load ~ 0.7 covers well)
        t = (mass * 2.2)[..., None]
        surface_R[:] = km_composite_np(K[None, None, :] * t, S[None, None, :] * t, surface_R)

    @staticmethod
    def _image(surface):
        return Image.fromarray(np.rint(np.clip(surface, 0, 1) * 255).astype(np.uint8))

    def _mark_ks(self, mark: _Mark):
        if mark.K is not None and mark.S is not None:
            return mark.K, mark.S
        p = Paint.from_color(mark.rgb, opacity=0.75)
        return p.K.astype(np.float32), p.S.astype(np.float32)

    def ink(self, mask, color, load=1.0):
        """Deposit a mark. Keep its pigment and footprint for subsequent pulls.

        ``color`` may be a :class:`pinsel.Paint` (spectral Kubelka-Munk) or
        any sRGB colour specification (hex string or RGB tuple).
        """
        if not np.isfinite(load) or load < 0:
            raise ValueError('ink load must be finite and nonnegative')
        cov = self._coverage(mask)
        active = cov > 0
        rows = np.flatnonzero(active.any(axis=1))
        if not rows.size or load == 0:
            return
        cols = np.flatnonzero(active.any(axis=0))
        region = np.s_[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
        mass = cov[region] * float(load)
        if isinstance(color, Paint):
            self._has_spectral = True
            mark = _Mark(region, mass, None, color.K.astype(np.float32), color.S.astype(np.float32))
            self._marks.append(mark)
            self._mass[region] += mass
            self._preview = None
        else:
            rgb = np.asarray(parse_color(color), np.float32)
            self._marks.append(_Mark(region, mass, rgb))
            self._mass[region] += mass
            if self._preview is not None and not self._has_spectral:
                self._over(self._preview[region], mass, rgb)
            else:
                self._preview = None

    def wipe(self, mask, strength=1.0):
        """Remove ink from every colour touched by the rag, in proportion to coverage."""
        if not np.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError('wipe strength must be between zero and one')
        keep = 1 - self._coverage(mask) * strength
        for mark in self._marks:
            mark.mass *= keep[mark.region]
        self._mass *= keep
        self._preview = None

    def proof(self):
        """Look down at the current plate, without transferring any pigment."""
        if self._has_spectral:
            surf_R = self._spectral_substrate([218, 211, 192], self._pressure)
            for mark in self._marks:
                K, S = self._mark_ks(mark)
                self._over_spectral(surf_R[mark.region], mark.mass, K, S)
            return self._image(spectrum_to_srgb(surf_R))
        if self._preview is None:
            self._preview = self._glass.copy()
            for mark in self._marks:
                self._over(self._preview[mark.region], mark.mass, mark.rgb)
        return self._image(self._preview)

    def _split_ink(self):
        grain = spectral_noise(self.shape, beta=1.35, seed=self._rng)
        # A little ink adheres to glass at each contacting pixel. On a thicker
        # patch the sheet takes a larger fraction. The next sheet has new tooth.
        cling = np.clip(.11 + .025 * self._pressure + .034 * grain, .035, .23)
        reserve = np.minimum(.44 * self._mass, cling * (1 - np.exp(-self._mass)))
        retained = np.divide(reserve, self._mass, out=np.zeros_like(self._mass),
                             where=self._mass > 0)
        # In dry, thin passages some paper fibres never touch the plate.
        skip = .004 + .09 * np.exp(-7 * self._mass)
        retained[self._rng.random(self.shape) < skip] = 1
        return grain, retained

    def pull(self):
        """Press a new sheet against the glass; split every ink layer at its contact face."""
        grain, retained = self._split_ink()
        if self._has_spectral:
            paper_R = self._spectral_substrate([246, 233, 209], grain)
            for mark in self._marks:
                fraction = retained[mark.region]
                removed = mark.mass * (1 - fraction)
                K, S = self._mark_ks(mark)
                self._over_spectral(paper_R[mark.region], removed, K, S)
                mark.mass -= removed
            self._mass *= retained
            self._preview = None
            return self._image(spectrum_to_srgb(paper_R[:, ::-1]))
        paper = self._substrate([246, 233, 209], grain)
        for mark in self._marks:
            fraction = retained[mark.region]
            removed = mark.mass * (1 - fraction)
            self._over(paper[mark.region], removed, mark.rgb)
            mark.mass -= removed
        self._mass *= retained
        self._preview = None
        return self._image(paper[:, ::-1])

    def pull_onto(self, canvas, offset=(0, 0), bake=False):
        """Pull a reversed monotype impression directly onto a :class:`pinsel.Canvas`.

        Transfers the ink spectrally (Kubelka-Munk) into ``canvas`` at ``offset``
        ``(ox, oy)``, respecting ``canvas.mask`` and depleting the plate for
        subsequent ghost pulls. Set ``bake=True`` to dry the impression
        immediately into ``canvas.below_R``.
        """
        _, retained = self._split_ink()
        pH, pW = self.shape
        ox, oy = int(offset[0]), int(offset[1])
        # Overlap between plate (reversed horizontally) and canvas
        cx0, cy0 = max(0, ox), max(0, oy)
        cx1, cy1 = min(canvas.W, ox + pW), min(canvas.H, oy + pH)
        if cx1 <= cx0 or cy1 <= cy0:
            self._mass *= retained
            self._preview = None
            return canvas

        c_sl = (slice(cy0, cy1), slice(cx0, cx1))
        unmasked = (1.0 - canvas.mask[c_sl]).astype(np.float32)

        # Accumulate spectral optical depths in reversed-plate coordinates
        acc_K = np.zeros((pH, pW, NB), dtype=np.float32)
        acc_S = np.zeros((pH, pW, NB), dtype=np.float32)
        acc_vol = np.zeros((pH, pW), dtype=np.float32)
        for mark in self._marks:
            fraction = retained[mark.region]
            removed = mark.mass * (1 - fraction)
            K, S = self._mark_ks(mark)
            t = (removed * 2.2).astype(np.float32)
            acc_K[mark.region] += t[..., None] * K[None, None, :]
            acc_S[mark.region] += t[..., None] * S[None, None, :]
            acc_vol[mark.region] += removed.astype(np.float32)
            mark.mass -= removed

        self._mass *= retained
        self._preview = None

        # Flip horizontally (plate -> paper mirror) and slice into canvas overlap
        py0, py1 = cy0 - oy, cy1 - oy
        px0, px1 = cx0 - ox, cx1 - ox
        K_rev = acc_K[:, ::-1, :][py0:py1, px0:px1, :] * unmasked[..., None]
        S_rev = acc_S[:, ::-1, :][py0:py1, px0:px1, :] * unmasked[..., None]
        vol_rev = acc_vol[:, ::-1][py0:py1, px0:px1] * unmasked

        if bake:
            canvas.below_R[c_sl] = km_composite_np(K_rev, S_rev, canvas.below_R[c_sl]).astype(np.float32)
        else:
            # Push existing top stroke down where new ink lands
            hit = vol_rev > 1e-6
            if hit.any():
                sid = int(canvas.stroke_id)
                diff = hit & (canvas.top_id[c_sl] != sid) & (canvas.film_vol[c_sl] > 0)
                if diff.any():
                    canvas.base_K[c_sl][diff] += canvas.film_K[c_sl][diff]
                    canvas.base_S[c_sl][diff] += canvas.film_S[c_sl][diff]
                    canvas.base_vol[c_sl][diff] += canvas.film_vol[c_sl][diff]
                    canvas.film_K[c_sl][diff] = 0.0
                    canvas.film_S[c_sl][diff] = 0.0
                    canvas.film_vol[c_sl][diff] = 0.0
                canvas.top_id[c_sl][hit] = sid
                fv = canvas.film_vol[c_sl]
                nv = fv + vol_rev
                canvas.film_K[c_sl] += K_rev
                canvas.film_S[c_sl] += S_rev
                canvas.wet[c_sl] = np.where(hit, (canvas.wet[c_sl] * fv + vol_rev) / np.maximum(nv, 1e-8), canvas.wet[c_sl])
                canvas.gloss[c_sl] = np.where(hit, (canvas.gloss[c_sl] * fv + 0.15 * vol_rev) / np.maximum(nv, 1e-8), canvas.gloss[c_sl])
                canvas.body[c_sl] = np.where(hit, (canvas.body[c_sl] * fv + 0.08 * vol_rev) / np.maximum(nv, 1e-8), canvas.body[c_sl])
                canvas.drate[c_sl] = np.where(hit, (canvas.drate[c_sl] * fv + (1.0 / 3600.0) * vol_rev) / np.maximum(nv, 1e-8), canvas.drate[c_sl])
                canvas.film_vol[c_sl] = nv
        canvas._after_stroke()
        return canvas
