"""
Painterly auto-painting and limited-palette mixing.

Two tools live here:

:class:`PaletteSolver`
    Works like a painter with a limited palette: for a target colour it finds
    a *mixing recipe* (one to ``max_pigments`` tube colours plus optional
    white, with volume ratios) whose Kubelka-Munk masstone matches the target
    best in CIELAB.  A dense table of mixtures (log-ratio grids on every
    pigment combination x a geometric ladder of white) is indexed with a
    k-d tree in Lab; candidates are ranked with CIEDE2000 plus a small
    penalty for complicated mixtures, and the winner is polished by a batched
    Levenberg-Marquardt fit of the volume ratios.  Results are cached on a
    quantised Lab grid, so thousands of queries take a fraction of a second.

    >>> solver = PaletteSolver()                         # classic 9-colour palette
    >>> paint, recipe = solver.solve('#6a8caf')
    >>> recipe.describe()
    '26 parts titanium white + 11 parts phthalo blue + 5 parts alizarin crimson + 1 part cadmium red  (dE00 0.0)'

:class:`AutoPainter` / :func:`paint_image`
    A painterly renderer in the spirit of Hertzmann (1998), "Painterly
    Rendering with Curved Brush Strokes of Multiple Sizes", re-built around
    pinsel's physical bristle brushes.  The reference is painted coarse to
    fine; every layer blurs the reference, compares it with the *rendered*
    canvas in CIELAB, seeds strokes where the area error exceeds a threshold
    and grows curved strokes along the isophotes of a smoothed (colour)
    structure-tensor field.  Each stroke gets a colour from the reference,
    mixed from real pigments by the PaletteSolver (optionally jittered,
    divisionist or loaded as streaky, incompletely mixed paint), and the
    painter keeps several brushes per layer that get dirty, run dry and are
    wiped, re-dipped or rinsed as a painter would.

    >>> import pinsel as ps
    >>> from pinsel.painterly import paint_image
    >>> cv = ps.Canvas(1000, 750, ground='canvas')
    >>> stats = paint_image(cv, reference, style='impressionist', seed=1)
    >>> cv.save('painting.png')

Coordinates follow the rest of pinsel: x to the right, y down; angles are in
degrees measured from +x towards +y.
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field, fields, replace
from typing import Optional

import numpy as np
from numba import njit
from . import _filters as ndimage
from ._filters import KDTree as cKDTree

from . import pigments as _pg
from .pigments import Paint
from .spectral import (NB, SPEC_TO_RGB, LINEAR_SRGB_TO_XYZ, XYZ_TO_LINEAR_SRGB,
                       srgb_to_linear, linear_to_srgb, linear_rgb_to_lab,
                       km_composite_np, parse_color)

__all__ = ['PaletteSolver', 'Recipe', 'AutoPainter', 'LayerSpec', 'Style', 'STYLES',
           'PaintResult', 'paint_image', 'DEFAULT_PALETTE', 'srgb_to_lab', 'lab_to_srgb',
           'delta_e2000', 'orientation_field']


# ==========================================================================
# Colour helpers
# ==========================================================================
_XN = LINEAR_SRGB_TO_XYZ @ np.ones(3)


def _as_srgb_float(img):
    """uint8 / float image (or colour) -> float64 sRGB in 0..1."""
    a = np.asarray(img)
    if a.dtype == np.uint8:
        return a.astype(np.float64) / 255.0
    a = a.astype(np.float64)
    if a.size and a.max() > 1.0 + 1e-6:
        a = a / 255.0
    return np.clip(a, 0.0, 1.0)


def srgb_to_lab(rgb):
    """sRGB (float 0..1 or uint8, any leading shape) -> CIELAB (D65)."""
    return linear_rgb_to_lab(srgb_to_linear(_as_srgb_float(rgb)[..., :3]))


def lab_to_linear_rgb(lab):
    lab = np.asarray(lab, dtype=np.float64)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    e, k = 216 / 24389, 24389 / 27
    xr = np.where(fx ** 3 > e, fx ** 3, (116 * fx - 16) / k)
    yr = np.where(L > k * e, fy ** 3, L / k)
    zr = np.where(fz ** 3 > e, fz ** 3, (116 * fz - 16) / k)
    xyz = np.stack([xr, yr, zr], axis=-1) * _XN
    return xyz @ XYZ_TO_LINEAR_SRGB.T


def lab_to_srgb(lab):
    """CIELAB -> sRGB float in 0..1 (clipped)."""
    return np.clip(linear_to_srgb(lab_to_linear_rgb(lab)), 0.0, 1.0)


def delta_e2000(lab1, lab2):
    """CIEDE2000 colour difference (vectorised, broadcasting)."""
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cm7 = ((C1 + C2) * 0.5) ** 7
    G = 0.5 * (1.0 - np.sqrt(Cm7 / (Cm7 + 25.0 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0
    dLp = L2 - L1
    dCp = C2p - C1p
    zero = (C1p * C2p) == 0
    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, np.where(dhp < -180, dhp + 360, dhp))
    dhp = np.where(zero, 0.0, dhp)
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp * 0.5))
    Lpm = (L1 + L2) * 0.5
    Cpm = (C1p + C2p) * 0.5
    hs = h1p + h2p
    hpm = np.where(zero, hs, np.where(np.abs(h1p - h2p) <= 180, hs * 0.5,
                                      np.where(hs < 360, (hs + 360) * 0.5, (hs - 360) * 0.5)))
    T = (1 - 0.17 * np.cos(np.radians(hpm - 30)) + 0.24 * np.cos(np.radians(2 * hpm))
         + 0.32 * np.cos(np.radians(3 * hpm + 6)) - 0.20 * np.cos(np.radians(4 * hpm - 63)))
    dth = 30.0 * np.exp(-((hpm - 275.0) / 25.0) ** 2)
    Cpm7 = Cpm ** 7
    Rc = 2.0 * np.sqrt(Cpm7 / (Cpm7 + 25.0 ** 7))
    Sl = 1 + 0.015 * (Lpm - 50) ** 2 / np.sqrt(20 + (Lpm - 50) ** 2)
    Sc = 1 + 0.045 * Cpm
    Sh = 1 + 0.015 * Cpm * T
    Rt = -np.sin(np.radians(2 * dth)) * Rc
    tl, tc, th = dLp / Sl, dCp / Sc, dHp / Sh
    return np.sqrt(np.maximum(tl * tl + tc * tc + th * th + Rt * tc * th, 0.0))


def _fast_paint(base, K, S, name=None, **props):
    """Cheap Paint construction (no deepcopy) sharing ``base``'s medium props."""
    p = object.__new__(Paint)
    p.__dict__.update(base.__dict__)
    p.K = np.asarray(K, dtype=np.float64)
    p.S = np.asarray(S, dtype=np.float64)
    if name is not None:
        p.name = name
    for k, v in props.items():
        setattr(p, k, v)
    return p


# ==========================================================================
# Palette solver
# ==========================================================================
DEFAULT_PALETTE = ('titanium_white', 'cadmium_yellow', 'yellow_ochre', 'cadmium_red',
                   'alizarin_crimson', 'ultramarine', 'phthalo_blue', 'burnt_sienna',
                   'ivory_black')

_PROPS = ('body', 'gloss', 'gloss_dry', 'viscosity', 'granulation', 'staining')


@dataclass
class Recipe:
    """A mixing recipe found by :class:`PaletteSolver`."""
    pigments: list                 # [Paint, ...]
    weights: np.ndarray            # volume fractions (sum to 1)
    lab: np.ndarray                # predicted Lab of the mixture's masstone
    target_lab: np.ndarray
    delta_e: float                 # CIEDE2000 between prediction and target
    _paint: Optional[Paint] = None

    @property
    def names(self):
        return [p.name for p in self.pigments]

    @property
    def rgb(self):
        """Predicted sRGB of the mixture."""
        return lab_to_srgb(self.lab)

    @property
    def paint(self) -> Paint:
        if self._paint is None:
            self._paint = _pg.mix([(p, float(w)) for p, w in zip(self.pigments, self.weights)])
        return self._paint

    def parts(self, max_parts=24):
        """Express the volume fractions as small integer parts."""
        w = np.asarray(self.weights, float)
        best, best_err = None, 1e9
        for unit in range(1, 7):
            q = np.maximum(np.round(w / w.min() * unit), 1)
            if best is not None and q.sum() > max_parts * 8:
                break
            err = np.abs(q / q.sum() - w).max()
            if err < best_err - 1e-4:
                best, best_err = q, err
            if err < 0.01:
                break
        g = np.gcd.reduce(best.astype(int)) or 1
        return [(n, int(k // g)) for n, k in zip(self.names, best.astype(int))]

    def describe(self):
        parts = sorted(self.parts(), key=lambda t: -t[1])
        s = ' + '.join(f"{k} part{'s' if k != 1 else ''} {n}" for n, k in parts)
        return f'{s}  (dE00 {self.delta_e:.1f})'

    def __repr__(self):
        return f'<Recipe {self.describe()}>'


class PaletteSolver:
    """Find pigment mixtures from a limited palette that match target colours.

    Parameters
    ----------
    palette : sequence of Paint / pigment names (default :data:`DEFAULT_PALETTE`)
    max_pigments : int     max. number of non-white pigments in one mixture (1..4)
    include_white : bool   allow white *in addition* to ``max_pigments`` colours
                           (titanium white is added if the palette has none)
    medium : str           medium for pigments given by name ('oil', 'acrylic', ...)
    thickness : float      paint thickness (volume units) at which the masstone
                           is evaluated; ~3 matches well-covered canvas areas
    ground : array/colour  reflectance (NB) or colour of what lies beneath
                           (default: neutral 0.9)
    ratio_range : float    largest volume ratio between two pigments in the table
    grid : (int, int, int) log-ratio grid points per axis for 2-, 3-, 4-pigment mixes
    white_levels : int     number of steps on the white ladder
    complexity_penalty : float  CIEDE2000 units added per extra pigment
    refine : bool          polish ratios with Levenberg-Marquardt (default True)
    cache_step : float     Lab quantisation (dE) of the result cache
    candidates : int       nearest table entries examined per query
    n_sets : int           distinct pigment combinations refined per query
    """

    _TABLE_CACHE: dict = {}

    def __init__(self, palette=None, max_pigments=3, include_white=True, medium='oil',
                 thickness=3.0, ground=None, ratio_range=48.0, grid=(25, 13, 7),
                 white_levels=22, complexity_penalty=1.0, refine=True, cache_step=0.6,
                 candidates=32, n_sets=4):
        pal = list(DEFAULT_PALETTE if palette is None else palette)
        paints = []
        for p in pal:
            if isinstance(p, Paint):
                paints.append(p)
            else:
                paints.append(_pg.get(str(p), medium))
        self.max_pigments = int(np.clip(max_pigments, 1, 4))
        self.include_white = bool(include_white)
        self.white = None
        if include_white:
            wi = [i for i, p in enumerate(paints) if 'white' in p.name.lower()]
            if wi:
                self.white = paints.pop(wi[0])
            else:
                base_medium = paints[0].medium if paints else medium
                self.white = _pg.TITANIUM_WHITE if base_medium == 'oil' else \
                    _pg.TITANIUM_WHITE.as_medium(base_medium)
        if not paints:
            raise ValueError('palette needs at least one non-white pigment')
        self.colors = paints
        self.pigments = paints + ([self.white] if self.white is not None else [])
        self.n_colors = len(paints)
        self.thickness = float(thickness)
        if ground is None:
            self.ground = np.full(NB, 0.9)
        else:
            g = np.asarray(ground, dtype=np.float64) if not isinstance(ground, str) else None
            if g is None or g.shape != (NB,):
                from .spectral import rgb_to_spectrum
                g = rgb_to_spectrum(ground)
            self.ground = g
        self.complexity_penalty = float(complexity_penalty)
        self.refine_enabled = bool(refine)
        self.cache_step = float(cache_step)
        self.candidates = int(candidates)
        self.n_sets = max(1, int(n_sets))
        P = len(self.pigments)
        # pigment matrices with a zero row at index P (for padded slots = -1)
        self._K = np.vstack([np.array([p.K for p in self.pigments]), np.zeros(NB)])
        self._S = np.vstack([np.array([p.S for p in self.pigments]), np.zeros(NB)])
        self._props = np.vstack([np.array([[getattr(p, a) for a in _PROPS] for p in self.pigments]),
                                 np.zeros(len(_PROPS))])
        self._white_index = P - 1 if self.white is not None else -1
        self._slots = self.max_pigments + (1 if self.white is not None else 0)
        key = (tuple(p.K.tobytes() + p.S.tobytes() for p in self.pigments),
               self.max_pigments, self.white is not None, self.thickness,
               self.ground.tobytes(), float(ratio_range), tuple(grid), int(white_levels))
        tab = self._TABLE_CACHE.get(key)
        if tab is None:
            tab = self._build_table(float(ratio_range), tuple(grid), int(white_levels))
            if len(self._TABLE_CACHE) > 8:
                self._TABLE_CACHE.clear()
            self._TABLE_CACHE[key] = tab
        self._tidx, self._tw, self._tlab, self._tree = tab
        self._cache = {}
        self._paint_cache = {}

    # ------------------------------------------------------------------ table
    def _mix_lab(self, idx, w):
        """Lab of mixtures given slot indices (n,S) (-1 = empty) and weights (n,S)."""
        ii = np.where(idx < 0, len(self.pigments), idx)
        K = np.einsum('ns,nsb->nb', w, self._K[ii])
        S = np.einsum('ns,nsb->nb', w, self._S[ii])
        t = self.thickness
        R = km_composite_np(K * t, S * t, self.ground)
        return linear_rgb_to_lab(R @ SPEC_TO_RGB.T)

    def _build_table(self, ratio_range, grid, white_levels):
        C = self.n_colors
        S = self._slots
        zr = math.log(ratio_range)
        chrom_idx, chrom_w = [], []
        for k in range(1, self.max_pigments + 1):
            if k > C:
                break
            if k == 1:
                W = np.ones((1, 1))
            else:
                m = grid[min(k - 2, len(grid) - 1)]
                z = np.linspace(-zr, zr, m)
                Z = np.array(list(itertools.product(z, repeat=k - 1)))
                Z = np.column_stack([np.zeros(len(Z)), Z])
                E = np.exp(Z - Z.max(axis=1, keepdims=True))
                W = E / E.sum(axis=1, keepdims=True)
            for combo in itertools.combinations(range(C), k):
                ci = np.full((len(W), self.max_pigments), -1, dtype=np.int16)
                ci[:, :k] = combo
                cw = np.zeros((len(W), self.max_pigments))
                cw[:, :k] = W
                chrom_idx.append(ci)
                chrom_w.append(cw)
        chrom_idx = np.vstack(chrom_idx)
        chrom_w = np.vstack(chrom_w)
        if self.white is not None:
            # white fraction ladder: 0 and geometric steps toward pure white
            om = np.concatenate([[0.0], 1.0 - np.geomspace(0.93, 0.004, white_levels - 1)])
            n = len(chrom_idx)
            idx = np.full((n * len(om) + 1, S), -1, dtype=np.int16)
            w = np.zeros((n * len(om) + 1, S))
            for j, o in enumerate(om):
                sl = slice(j * n, (j + 1) * n)
                idx[sl, :self.max_pigments] = chrom_idx
                w[sl, :self.max_pigments] = chrom_w * (1 - o)
                if o > 0:
                    idx[sl, -1] = self._white_index
                    w[sl, -1] = o
            idx[-1, -1] = self._white_index      # pure white
            w[-1, -1] = 1.0
        else:
            idx, w = chrom_idx, chrom_w
        lab = np.empty((len(idx), 3))
        for s in range(0, len(idx), 65536):
            lab[s:s + 65536] = self._mix_lab(idx[s:s + 65536], w[s:s + 65536])
        tree = cKDTree(lab)
        return idx, w.astype(np.float32), lab, tree

    @property
    def table_size(self):
        return len(self._tidx)

    # ------------------------------------------------------------------ core
    def _n_pigments_cost(self, idx, w):
        active = (idx >= 0) & (w > 1e-6)
        cnt = active.sum(axis=-1).astype(float)
        if self.white is not None:
            hasw = active[..., -1]
            cnt = cnt - 0.6 * hasw          # white is cheap: it is always on the palette
        return np.maximum(cnt - 1.0, 0.0)

    def _refine(self, lab_t, idx, w, iters=12):
        n, S = w.shape
        active = (idx >= 0) & (w > 0)
        ii = np.where(idx < 0, len(self.pigments), idx)
        Kg = self._K[ii]                                   # n,S,NB
        Sg = self._S[ii]
        t = self.thickness
        Rg = self.ground
        lo = math.log(1e-3)

        def model(theta):
            z = np.where(active, theta, -np.inf)
            z = z - z.max(axis=1, keepdims=True)
            z = np.where(active, np.maximum(z, lo), -np.inf)
            e = np.exp(z)
            ww = e / e.sum(axis=1, keepdims=True)
            K = np.einsum('ns,nsb->nb', ww, Kg)
            Sx = np.einsum('ns,nsb->nb', ww, Sg)
            R = km_composite_np(K * t, Sx * t, Rg)
            return linear_rgb_to_lab(R @ SPEC_TO_RGB.T), ww

        theta = np.where(active, np.log(np.maximum(w, 1e-9)), 0.0)
        lab0, ww0 = model(theta)
        r = lab0 - lab_t
        err = (r * r).sum(axis=1)
        lam = np.full(n, 1e-2)
        h = 1e-4
        eye = np.eye(S)[None]
        nact = active.sum(axis=1)
        todo = nact > 1
        if not todo.any():
            return ww0, lab0
        for _ in range(iters):
            J = np.zeros((n, 3, S))
            for s in range(S):
                if not active[:, s].any():
                    continue
                th = theta.copy()
                th[:, s] += h
                ls, _ = model(th)
                J[:, :, s] = np.where(active[:, s, None], (ls - lab0) / h, 0.0)
            JtJ = np.einsum('nis,nit->nst', J, J)
            g = np.einsum('nis,ni->ns', J, r)
            dg = np.einsum('nss->ns', JtJ)
            A = JtJ + (lam[:, None] * (dg + 1e-3))[:, :, None] * eye
            A = A + (~active)[:, :, None] * eye
            step = -np.linalg.solve(A, g[..., None])[..., 0]
            step = np.clip(np.where(active, step, 0.0), -3.0, 3.0)
            th_new = theta + step
            lab_new, ww_new = model(th_new)
            rn = lab_new - lab_t
            en = (rn * rn).sum(axis=1)
            better = (en < err) & todo
            theta[better] = th_new[better]
            lab0[better] = lab_new[better]
            ww0[better] = ww_new[better]
            r[better] = rn[better]
            err[better] = en[better]
            lam = np.where(better, lam * 0.3, lam * 8.0)
            if (lam[todo] > 1e4).all():
                break
        return ww0, lab0

    def _solve_core(self, lab_t):
        """Uncached vectorised solve for Lab targets (n,3) -> idx, w, lab, dE.

        The k nearest table entries (CIE76) are ranked by CIEDE2000 + a
        complexity penalty; the best entries of up to ``n_sets`` *different*
        pigment combinations are refined, and the overall winner is kept.
        """
        lab_t = np.atleast_2d(np.asarray(lab_t, dtype=np.float64))
        n = len(lab_t)
        k = min(self.candidates, len(self._tlab))
        _, nn = self._tree.query(lab_t, k=k, workers=-1)
        nn = nn.reshape(n, k)
        cidx = self._tidx[nn]
        cw = self._tw[nn]
        de = delta_e2000(lab_t[:, None, :], self._tlab[nn])
        cplx = self._n_pigments_cost(cidx, cw)
        cost = de + self.complexity_penalty * cplx
        order = np.argsort(cost, axis=1)
        if not self.refine_enabled:
            best = nn[np.arange(n), order[:, 0]]
            idx = self._tidx[best].astype(np.int64)
            w = self._tw[best].astype(np.float64)
        else:
            # pigment-set signature (bit mask) of each candidate
            act = (cidx >= 0) & (cw > 1e-6)
            bits = np.where(act, np.left_shift(1, np.maximum(cidx, 0).astype(np.int64)), 0).sum(-1)
            M = self.n_sets
            chosen = np.full((n, M), -1, dtype=np.int64)       # candidate column
            cmask = np.full((n, M), -1, dtype=np.int64)
            cnt = np.zeros(n, dtype=np.int64)
            rows = np.arange(n)
            for j in range(k):
                col = order[:, j]
                b = bits[rows, col]
                new = (cnt < M) & ~(cmask == b[:, None]).any(axis=1)
                r = rows[new]
                chosen[r, cnt[new]] = col[new]
                cmask[r, cnt[new]] = b[new]
                cnt[new] += 1
            chosen = np.where(chosen < 0, chosen[:, :1], chosen)  # pad with the best
            sel = nn[rows[:, None], chosen].reshape(-1)
            idx_all = self._tidx[sel].astype(np.int64)
            w_all = self._tw[sel].astype(np.float64)
            tgt = np.repeat(lab_t, M, axis=0)
            w2, lab2 = self._refine(tgt, idx_all, w_all)
            de2 = delta_e2000(tgt, lab2)
            de1 = delta_e2000(tgt, self._tlab[sel])
            better = de2 < de1
            w_all[better] = w2[better]
            de_f = np.minimum(de1, de2)
            cost_f = (de_f + self.complexity_penalty * self._n_pigments_cost(idx_all, w_all)).reshape(n, M)
            pick = np.argmin(cost_f, axis=1)
            flat = rows * M + pick
            idx = idx_all[flat]
            w = w_all[flat]
        # tidy: drop vanishing components, canonical order (largest first)
        w = np.where(w < 1e-3, 0.0, w)
        idx = np.where(w > 0, idx, -1)
        w /= w.sum(axis=1, keepdims=True)
        order = np.argsort(-w, axis=1, kind='stable')
        idx = np.take_along_axis(idx, order, axis=1)
        w = np.take_along_axis(w, order, axis=1)
        lab = self._mix_lab(idx, w)
        return idx, w, lab, delta_e2000(lab_t, lab)

    def _keys(self, lab):
        q = np.round(np.asarray(lab, dtype=np.float64) / self.cache_step).astype(np.int64)
        return ((q[:, 0] + 2048) << 24) | ((q[:, 1] + 2048) << 12) | (q[:, 2] + 2048)

    def match_lab(self, lab):
        """Vectorised, cached solve in Lab.

        Returns ``(idx, w, lab_pred, delta_e, keys)`` where ``idx`` (n, slots)
        indexes :attr:`pigments` (-1 = unused slot) and ``w`` are volume
        fractions.  Targets are quantised to ``cache_step``.
        """
        lab = np.atleast_2d(np.asarray(lab, dtype=np.float64)).reshape(-1, 3)
        keys = self._keys(lab)
        uk, inv = np.unique(keys, return_inverse=True)
        missing = [k for k in uk.tolist() if k not in self._cache]
        if missing and len(self._cache) + len(missing) > 400000:
            keep = set(uk.tolist())
            self._cache = {k: v for k, v in self._cache.items() if k in keep}
        if missing:
            mk = np.array(missing, dtype=np.int64)
            ql = np.column_stack([((mk >> 24) & 4095) - 2048, ((mk >> 12) & 4095) - 2048,
                                  (mk & 4095) - 2048]).astype(np.float64) * self.cache_step
            for s in range(0, len(mk), 8192):
                idx, w, lp, de = self._solve_core(ql[s:s + 8192])
                for j, kk in enumerate(missing[s:s + 8192]):
                    self._cache[kk] = (idx[j], w[j], lp[j], de[j])
        rows = [self._cache[k] for k in uk.tolist()]
        S = self._slots
        U_idx = np.array([r[0] for r in rows]).reshape(-1, S)
        U_w = np.array([r[1] for r in rows]).reshape(-1, S)
        U_lab = np.array([r[2] for r in rows]).reshape(-1, 3)
        U_de = np.array([r[3] for r in rows])
        return U_idx[inv], U_w[inv], U_lab[inv], U_de[inv], keys

    def mixture_ks(self, idx, w):
        """Per-volume K, S (n, NB) and blended medium properties of recipes."""
        ii = np.where(idx < 0, len(self.pigments), idx)
        K = np.einsum('ns,nsb->nb', w, self._K[ii])
        S = np.einsum('ns,nsb->nb', w, self._S[ii])
        props = np.einsum('ns,nsp->np', w, self._props[ii])
        return K, S, props

    def _recipe(self, idx_row, w_row, lab, target, de):
        sel = idx_row >= 0
        return Recipe([self.pigments[i] for i in idx_row[sel]], w_row[sel].copy(),
                      np.asarray(lab, float), np.asarray(target, float), float(de))

    def paint_for(self, idx_row, w_row, name=None):
        """A :class:`Paint` for one recipe row (cached)."""
        key = (tuple(int(i) for i in idx_row), tuple(np.round(w_row, 4).tolist()))
        p = self._paint_cache.get(key)
        if p is None:
            K, S, props = self.mixture_ks(idx_row[None], w_row[None])
            sel = idx_row >= 0
            main = self.pigments[int(idx_row[sel][np.argmax(w_row[sel])])]
            if name is None:
                name = ' + '.join(self.pigments[i].name for i in idx_row[sel])
            p = _fast_paint(main, K[0], S[0], name, **dict(zip(_PROPS, props[0].tolist())))
            if len(self._paint_cache) > 50000:
                self._paint_cache.clear()
            self._paint_cache[key] = p
        return p

    # ------------------------------------------------------------------ public
    def solve(self, color):
        """Best mixture for one colour (hex string, sRGB 0..1 floats or 0..255).

        Returns ``(paint, recipe)``.
        """
        rgb = parse_color(color)
        target = srgb_to_lab(rgb)
        idx, w, lab, de, _ = self.match_lab(target[None])
        rec = self._recipe(idx[0], w[0], lab[0], target, de[0])
        return self.paint_for(idx[0], w[0]), rec

    def solve_lab(self, lab):
        """Like :meth:`solve` but takes a CIELAB colour."""
        target = np.asarray(lab, dtype=np.float64)
        idx, w, lp, de, _ = self.match_lab(target[None])
        rec = self._recipe(idx[0], w[0], lp[0], target, de[0])
        return self.paint_for(idx[0], w[0]), rec

    def solve_many(self, colors):
        """Vectorised solve for many sRGB colours ((n,3) floats 0..1 or uint8).

        Returns a list of ``(paint, recipe)`` tuples.  Repeated (or nearly
        identical) colours are served from the cache.
        """
        rgb = _as_srgb_float(colors).reshape(-1, 3)
        target = srgb_to_lab(rgb)
        idx, w, lab, de, _ = self.match_lab(target)
        return [(self.paint_for(idx[i], w[i]), self._recipe(idx[i], w[i], lab[i], target[i], de[i]))
                for i in range(len(rgb))]

    def mixed_rgb(self, colors):
        """Predicted sRGB (same shape as input) of the best mixtures: shows how
        an image looks when every colour is mixed from this palette."""
        a = _as_srgb_float(colors)
        shp = a.shape
        lab = srgb_to_lab(a.reshape(-1, 3))
        _, _, lp, _, _ = self.match_lab(lab)
        return lab_to_srgb(lp).reshape(shp)

    def delta_e(self, colors):
        """CIEDE2000 between targets and their best mixtures."""
        a = _as_srgb_float(colors).reshape(-1, 3)
        lab = srgb_to_lab(a)
        _, _, lp, _, _ = self.match_lab(lab)
        return delta_e2000(lab, lp)

    def swatch_image(self, colors, size=64, label=True):
        """Image (uint8) of target (left half) vs mixed (right half) swatches."""
        from PIL import Image, ImageDraw
        res = self.solve_many(colors)
        cols = _as_srgb_float(colors).reshape(-1, 3)
        n = len(cols)
        ncol = int(math.ceil(math.sqrt(n * 1.6)))
        nrow = int(math.ceil(n / ncol))
        lh = 14 if label else 0
        img = Image.new('RGB', (ncol * (size + 6) + 6, nrow * (size + 6 + lh) + 6), (245, 243, 238))
        dr = ImageDraw.Draw(img)
        for i, ((p, rec), c) in enumerate(zip(res, cols)):
            x = 6 + (i % ncol) * (size + 6)
            y = 6 + (i // ncol) * (size + 6 + lh)
            t = tuple(int(v * 255 + 0.5) for v in c)
            m = tuple(int(v * 255 + 0.5) for v in rec.rgb)
            dr.rectangle([x, y, x + size // 2 - 1, y + size - 1], fill=t)
            dr.rectangle([x + size // 2, y, x + size - 1, y + size - 1], fill=m)
            if label:
                dr.text((x + 2, y + size + 1), f'{rec.delta_e:.1f}', fill=(60, 60, 60))
        return np.asarray(img)


# ==========================================================================
# Orientation field
# ==========================================================================
def _gauss(a, sigma):
    if sigma <= 0.3:
        return a
    if a.ndim == 3:
        return ndimage.gaussian_filter(a, (sigma, sigma, 0), mode='nearest')
    return ndimage.gaussian_filter(a, sigma, mode='nearest')


def _tensor_dirvec(lab, sigma, chroma_weight=0.6, return_trace=False):
    """Smoothed colour structure tensor -> doubled-angle *tangent* vectors.

    Returns (vx, vy, anisotropy): the vector (vx, vy) points along twice the
    isophote angle; anisotropy = lambda1 - lambda2 of the tensor.
    """
    Jxx = np.zeros(lab.shape[:2])
    Jxy = np.zeros(lab.shape[:2])
    Jyy = np.zeros(lab.shape[:2])
    for c, wgt in ((0, 1.0), (1, chroma_weight), (2, chroma_weight)):
        ch = lab[..., c].astype(np.float64)
        gx = ndimage.sobel(ch, axis=1, mode='nearest') / 8.0
        gy = ndimage.sobel(ch, axis=0, mode='nearest') / 8.0
        Jxx += wgt * gx * gx
        Jxy += wgt * gx * gy
        Jyy += wgt * gy * gy
    Jxx, Jxy, Jyy = _gauss(Jxx, sigma), _gauss(Jxy, sigma), _gauss(Jyy, sigma)
    # gradient direction in doubled-angle form is (Jxx-Jyy, 2Jxy); the tangent
    # is rotated by 90 deg, i.e. 180 deg in doubled-angle space
    vx = -(Jxx - Jyy)
    vy = -2.0 * Jxy
    an = np.hypot(vx, vy)
    if return_trace:
        return vx, vy, an, Jxx + Jyy
    return vx, vy, an


def _angle_field(flow, shape):
    """User flow (angle in degrees, (H,W) degrees array, (H,W,2) vectors or
    callable(x, y) -> degrees) -> doubled-angle unit vectors (H,W,2)."""
    H, W = shape
    if callable(flow):
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        ang = np.radians(np.asarray(flow(xx, yy), dtype=np.float64))
    else:
        f = np.asarray(flow, dtype=np.float64)
        if f.ndim == 3 and f.shape[-1] == 2:
            ang = np.arctan2(f[..., 1], f[..., 0])
        else:
            ang = np.radians(np.broadcast_to(f, shape))
    return np.stack([np.cos(2 * ang), np.sin(2 * ang)], axis=-1)


def orientation_field(lab, radius, tensor_scale=1.0, fallback=None, flow=None,
                      flow_weight=0.0, strength_quantile=55.0, coherence_ref=0.5):
    """Stroke direction field for one layer as doubled-angle vectors (H,W,2).

    lab           blurred reference in CIELAB
    radius        brush radius of the layer (sets the tensor integration scale)
    fallback      (H,W,2) doubled-angle unit field used where the image has
                  little structure (e.g. from a coarse scale / default angle)
    flow          optional user flow (see :class:`AutoPainter`)
    flow_weight   scalar or (H,W) map: 1 = strokes follow ``flow`` only
    """
    sigma = max(1.5, tensor_scale * radius)
    vx, vy, an, tr = _tensor_dirvec(lab, sigma, return_trace=True)
    tau = float(np.percentile(an, strength_quantile)) * 0.6 + 1e-9
    coh = an / (tr + 1e-9)
    s = (1.0 - np.exp(-an / tau)) * np.clip(coh / coherence_ref, 0, 1) ** 0.75
    ux, uy = vx / (an + 1e-12), vy / (an + 1e-12)
    V = np.stack([s * ux, s * uy], axis=-1)
    if fallback is not None:
        V += (1.0 - s)[..., None] * fallback
    if flow is not None:
        fw = np.clip(np.broadcast_to(np.asarray(flow_weight, dtype=np.float64), lab.shape[:2]), 0, 1)
        if fw.max() > 0:
            V = (1.0 - fw)[..., None] * V + fw[..., None] * _angle_field(flow, lab.shape[:2])
    return V.astype(np.float32)


# ==========================================================================
# Stroke growth (numba)
# ==========================================================================
@njit(cache=True, fastmath=True, inline='always')
def _bil(img, x, y, c):
    H = img.shape[0]
    W = img.shape[1]
    if x < 0.0:
        x = 0.0
    elif x > W - 1.0:
        x = W - 1.0
    if y < 0.0:
        y = 0.0
    elif y > H - 1.0:
        y = H - 1.0
    x0 = int(x)
    y0 = int(y)
    x1 = x0 + 1 if x0 < W - 1 else x0
    y1 = y0 + 1 if y0 < H - 1 else y0
    fx = x - x0
    fy = y - y0
    return ((img[y0, x0, c] * (1 - fx) + img[y0, x1, c] * fx) * (1 - fy)
            + (img[y1, x0, c] * (1 - fx) + img[y1, x1, c] * fx) * fy)


@njit(cache=True, fastmath=True)
def _field_dir(V, x, y, rot):
    vx = _bil(V, x, y, 0)
    vy = _bil(V, x, y, 1)
    m = math.sqrt(vx * vx + vy * vy)
    if m < 0.04:
        return 0.0, 0.0, m
    th = 0.5 * math.atan2(vy, vx) + rot
    return math.cos(th), math.sin(th), m


@njit(cache=True, fastmath=True)
def _walk(x, y, px, py, nmax, count0, c0, c1, c2, ref, cvl, V, region,
          step, min_steps, fc, edge_stop, turn_cos, margin, outx, outy, use_hz, rot):
    """Grow one stroke branch; returns the number of points written."""
    H = region.shape[0]
    W = region.shape[1]
    n = 0
    for i in range(nmax):
        dx, dy, m = _field_dir(V, x, y, rot)
        if m < 0.04:
            dx, dy = px, py
        elif dx * px + dy * py < 0.0:
            dx, dy = -dx, -dy
        dx = fc * dx + (1.0 - fc) * px
        dy = fc * dy + (1.0 - fc) * py
        l = math.sqrt(dx * dx + dy * dy)
        if l < 1e-9:
            break
        dx /= l
        dy /= l
        if dx * px + dy * py < turn_cos:
            break
        nx = x + step * dx
        ny = y + step * dy
        if nx < -margin or ny < -margin or nx > W - 1 + margin or ny > H - 1 + margin:
            break
        xi = int(nx + 0.5)
        yi = int(ny + 0.5)
        if xi < 0:
            xi = 0
        elif xi > W - 1:
            xi = W - 1
        if yi < 0:
            yi = 0
        elif yi > H - 1:
            yi = H - 1
        if region[yi, xi] < 0.5:
            break
        r0 = _bil(ref, nx, ny, 0)
        r1 = _bil(ref, nx, ny, 1)
        r2 = _bil(ref, nx, ny, 2)
        dc = math.sqrt((r0 - c0) ** 2 + (r1 - c1) ** 2 + (r2 - c2) ** 2)
        if dc > edge_stop:
            break
        if use_hz and count0 + n >= min_steps:
            k0 = _bil(cvl, nx, ny, 0)
            k1 = _bil(cvl, nx, ny, 1)
            k2 = _bil(cvl, nx, ny, 2)
            dk = math.sqrt((r0 - k0) ** 2 + (r1 - k1) ** 2 + (r2 - k2) ** 2)
            if dk < dc:
                break
        outx[n] = nx
        outy[n] = ny
        n += 1
        x, y, px, py = nx, ny, dx, dy
    return n


@njit(cache=True, fastmath=True)
def _grow_strokes(sx, sy, rots, ref, cvl, V, region, step, min_steps, max_steps, back_steps,
                  fc, edge_stop, turn_cos, dab_len, margin, color_mix, ctrl, nctrl, cols):
    """Grow Hertzmann-style curved strokes from seed points.

    ctrl  (n, max_steps + 1, 2) out: control points
    nctrl (n,) out: number of control points
    cols  (n, 3) out: stroke colour (Lab), blend of seed colour and mean colour
    """
    n = sx.shape[0]
    bx = np.empty(max_steps + 2)
    by = np.empty(max_steps + 2)
    fx = np.empty(max_steps + 2)
    fy = np.empty(max_steps + 2)
    for s in range(n):
        x0 = sx[s]
        y0 = sy[s]
        c0 = _bil(ref, x0, y0, 0)
        c1 = _bil(ref, x0, y0, 1)
        c2 = _bil(ref, x0, y0, 2)
        rot = rots[s]
        d0x, d0y, m = _field_dir(V, x0, y0, rot)
        if m < 0.04:
            d0x, d0y = math.cos(rot), math.sin(rot)
        nb = 0
        if back_steps > 0 and max_steps > 0:
            nb = _walk(x0, y0, -d0x, -d0y, min(back_steps, max_steps), 0, c0, c1, c2,
                       ref, cvl, V, region, step, min_steps, fc, edge_stop, turn_cos, margin,
                       bx, by, False, rot)
        nf = _walk(x0, y0, d0x, d0y, max_steps - nb, nb, c0, c1, c2, ref, cvl, V, region,
                   step, min_steps, fc, edge_stop, turn_cos, margin, fx, fy, True, rot)
        k = 0
        for i in range(nb - 1, -1, -1):
            ctrl[s, k, 0] = bx[i]
            ctrl[s, k, 1] = by[i]
            k += 1
        ctrl[s, k, 0] = x0
        ctrl[s, k, 1] = y0
        k += 1
        for i in range(nf):
            ctrl[s, k, 0] = fx[i]
            ctrl[s, k, 1] = fy[i]
            k += 1
        if k == 1:
            # a dab: short press along the local direction
            ctrl[s, 0, 0] = x0 - 0.5 * dab_len * d0x
            ctrl[s, 0, 1] = y0 - 0.5 * dab_len * d0y
            ctrl[s, 1, 0] = x0 + 0.5 * dab_len * d0x
            ctrl[s, 1, 1] = y0 + 0.5 * dab_len * d0y
            k = 2
        nctrl[s] = k
        # stroke colour: seed colour blended with the mean along the stroke
        m0 = 0.0
        m1 = 0.0
        m2 = 0.0
        for i in range(k):
            m0 += _bil(ref, ctrl[s, i, 0], ctrl[s, i, 1], 0)
            m1 += _bil(ref, ctrl[s, i, 0], ctrl[s, i, 1], 1)
            m2 += _bil(ref, ctrl[s, i, 0], ctrl[s, i, 1], 2)
        cols[s, 0] = (1 - color_mix) * c0 + color_mix * m0 / k
        cols[s, 1] = (1 - color_mix) * c1 + color_mix * m1 / k
        cols[s, 2] = (1 - color_mix) * c2 + color_mix * m2 / k


@njit(cache=True, fastmath=True)
def _densify(ctrl, nctrl, sub, out, nout):
    """Uniform Catmull-Rom through control points -> dense polylines."""
    n = ctrl.shape[0]
    for s in range(n):
        k = nctrl[s]
        if k < 3:
            # straight segment
            for j in range(sub + 1):
                t = j / sub
                out[s, j, 0] = ctrl[s, 0, 0] + (ctrl[s, 1, 0] - ctrl[s, 0, 0]) * t
                out[s, j, 1] = ctrl[s, 0, 1] + (ctrl[s, 1, 1] - ctrl[s, 0, 1]) * t
            nout[s] = sub + 1
            continue
        m = 0
        for i in range(k - 1):
            i0 = i - 1 if i > 0 else 0
            i3 = i + 2 if i + 2 < k else k - 1
            for c in range(2):
                p0 = ctrl[s, i0, c]
                p1 = ctrl[s, i, c]
                p2 = ctrl[s, i + 1, c]
                p3 = ctrl[s, i3, c]
                if i == 0:
                    p0 = 2 * p1 - p2
                if i + 2 >= k:
                    p3 = 2 * p2 - p1
                for j in range(sub):
                    t = j / sub
                    t2 = t * t
                    t3 = t2 * t
                    out[s, m + j, c] = 0.5 * ((2 * p1) + (-p0 + p2) * t
                                              + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
            m += sub
        out[s, m, 0] = ctrl[s, k - 1, 0]
        out[s, m, 1] = ctrl[s, k - 1, 1]
        nout[s] = m + 1


from . import _backend as _B  # noqa: E402  (Rust port of the stroke-growing kernels)
_grow_strokes = _B.port('painterly_grow_strokes', _grow_strokes, mutable={17: 'ctrl', 18: 'nctrl', 19: 'cols'})
_densify = _B.port('painterly_densify', _densify, mutable={3: 'out', 4: 'nout'})


# ==========================================================================
# Styles
# ==========================================================================
@dataclass
class LayerSpec:
    """One layer of the coarse-to-fine painting process.

    Lengths are in pixels for a 1000 px wide/high image and are scaled with
    the canvas (see ``AutoPainter.scale``) unless ``absolute=True``.
    """
    radius: float = 10.0          # brush radius, or tip radius for an oil crayon
    shape: str = 'filbert'        # a bristle shape (pinsel.SHAPES) or 'oil_crayon'
    blur: float = 0.5             # reference blur sigma = blur * radius
    grid: float = 1.0             # seed grid spacing = grid * radius
    threshold: float = 10.0       # mean CIELAB error in a cell that triggers a stroke
    step: float = 1.0             # control point spacing = step * radius
    min_len: int = 2              # min. control steps before colour termination
    max_len: int = 10             # max. control steps
    back_frac: float = 0.35       # part of max_len that may grow backwards from the seed
    curvature: float = 0.7        # Hertzmann's filter fc (1 = follow field, 0 = straight)
    max_turn: float = 70.0        # stop when a step turns more than this (deg)
    edge_stop: float = 28.0       # stop when reference differs from stroke colour by > dE
    color_sample: float = 0.5     # 0 = colour at seed, 1 = mean along the stroke
    angle_jitter: float = 0.0     # random rotation of each stroke's direction (deg, std)
    dab_len: float = 1.2          # length of strokes that cannot grow (x radius)
    tensor_scale: float = 1.0     # structure tensor integration sigma = scale * radius
    compare_blur: float = 0.3     # canvas blur (x radius) before comparing with the reference
    edge_avoid: float = 0.4       # >0 keeps seeds off edge lines (no in-between colour
    #                               outlines); <0 seeks edges (contour drawing)
    coverage: float = 1.0         # fraction of the seeds actually painted (<1 leaves gaps)
    min_edge: float = 0.0         # only seed where the colour change across a radius >= this
    value: float = 0.0            # L* shift of this layer's colours (e.g. dark drawing lines)
    pressure: float = 0.75        # peak pressure
    pressure_var: float = 0.08    # per-stroke random variation of the peak
    profile: str = 'auto'         # 'auto', 'taper', 'swell', 'dab', 'flick', 'constant'
    pressure_jitter: float = 0.12  # slow pressure variation along a stroke
    amount: float = 1.0           # paint load (1 normal, 1.5+ impasto, <0.6 dry brush)
    thinning: float = 0.0         # dilute with medium (lean, transparent underpainting)
    impasto: float = 1.0          # multiply the paint's body (relief)
    speed: float = 1.0            # brush speed (>1: skips, less paint)
    angle: Optional[float] = None  # fixed brush angle (deg) or None (broad side forward)
    wobble: float = 0.0           # hand tremor (fraction of radius)
    size_var: float = 0.12        # size spread between the brushes of a layer
    brushes: Optional[int] = None  # number of brushes (colour families); None = style value
    brush: dict = field(default_factory=dict)   # tool kwargs (e.g. stiffness for Brush, side for OilCrayon)
    passes: int = 1               # re-render + re-plan passes with this brush
    dry_after: Optional[bool] = None   # dry the canvas after this layer (None = style)
    max_strokes: Optional[int] = None
    # per-layer overrides of style-level colour options (None = use style)
    jitter: Optional[tuple] = None        # (dL, dChroma_rel, dHue_deg)
    loose_mix: Optional[float] = None
    divisionism: Optional[float] = None
    chroma: Optional[float] = None
    absolute: bool = False


@dataclass
class Style:
    """Global settings of a painterly style plus its layers."""
    layers: list
    chroma: float = 1.0                 # saturation multiplier of target colours
    lightness: float = 1.0              # contrast of L* around 55
    jitter: tuple = (2.0, 0.06, 3.0)    # random (dL, relative dChroma, dHue deg) per stroke
    loose_mix: float = 0.3              # 0 = fully mixed paint, 1 = very streaky loads
    divisionism: float = 0.0            # dE of complementary colour splitting (pointillism)
    brush_care: str = 'auto'            # 'auto', 'clean', 'wipe'
    clean_threshold: float = 22.0       # dE above which the brush is rinsed
    reload_level: float = 0.3           # dip again when the load drops below this
    wipe: float = 0.7                   # fraction wiped off on colour changes
    brushes: int = 3                    # brushes per layer, one per value family
    default_angle: float = 0.0          # stroke direction in featureless areas (deg)
    angle_noise: float = 25.0           # smooth random variation of that direction (deg)
    coarse_scale: float = 0.06          # coarse orientation scale (fraction of image size)
    order: str = 'random'               # 'random', 'dark_to_light', 'light_to_dark', 'grouped'
    dry_between: bool = True            # dry the canvas between layers
    tone: Optional[tuple] = None        # (paint/name, thickness) imprimatura before painting
    edge_margin: float = 0.5            # how far strokes may run past the canvas edge (x radius)


def _L(**kw):
    return LayerSpec(**kw)


STYLES = {
    # Block in with the side of the stick, then turn onto the tip to follow edges.
    'oil_crayon': Style(
        layers=[
            _L(radius=21, shape='oil_crayon', brush={'side': 0.9}, grid=2.0,
               threshold=0.0, pressure=0.70, profile='swell', max_len=6,
               curvature=0.45, angle_jitter=14, edge_stop=20),
            _L(radius=10, shape='oil_crayon', brush={'side': 0.55}, grid=1.65,
               threshold=9.0, pressure=0.82, profile='swell', max_len=7,
               curvature=0.65, angle_jitter=9, edge_stop=17),
            _L(radius=5, shape='oil_crayon', brush={'side': 0.05}, grid=1.6,
               threshold=11.0, pressure=0.88, min_len=1, max_len=6,
               edge_avoid=-0.4, min_edge=5, coverage=0.8, edge_stop=16),
        ],
        chroma=1.09, jitter=(2.0, 0.07, 4.0), loose_mix=0.0, dry_between=False,
        order='dark_to_light', default_angle=-24.0, angle_noise=16.0),
    # broken colour, medium length strokes, streaky loads, clean colour
    'impressionist': Style(
        layers=[
            _L(radius=34, shape='filbert', blur=0.6, threshold=0.0, max_len=6, min_len=2,
               curvature=0.6, amount=0.9, thinning=0.2, impasto=0.4, pressure=0.72,
               edge_stop=15, angle_jitter=12),
            _L(radius=18, shape='flat', threshold=8.5, max_len=6, curvature=0.65,
               amount=0.9, impasto=0.75, edge_stop=18, angle_jitter=12),
            _L(radius=10, shape='flat', threshold=10.0, max_len=5, curvature=0.7,
               amount=0.9, impasto=0.95, edge_stop=20, angle_jitter=10, dab_len=1.8),
            _L(radius=5.5, shape='hog', threshold=12.0, max_len=5, min_len=1, curvature=0.75,
               amount=0.85, impasto=1.1, pressure=0.72, edge_stop=22, angle_jitter=8,
               dab_len=2.2),
        ],
        chroma=1.12, jitter=(3.0, 0.1, 5.0), loose_mix=0.35, divisionism=2.5,
        tone=('burnt_sienna', 0.12), default_angle=-20.0, angle_noise=30.0),
    # long, flowing, strongly curved impasto strokes, saturated colour, wet in wet
    'expressionist': Style(
        layers=[
            _L(radius=28, shape='hog', blur=0.6, threshold=0.0, max_len=12, min_len=3,
               curvature=0.85, amount=1.0, thinning=0.15, edge_stop=32),
            _L(radius=15, shape='hog', threshold=8.0, max_len=14, min_len=3, curvature=0.85,
               amount=1.4, impasto=1.6, edge_stop=30),
            _L(radius=8.5, shape='flat', threshold=11.0, max_len=12, min_len=3, curvature=0.9,
               amount=1.5, impasto=1.8, edge_stop=26),
            _L(radius=5, shape='round', threshold=13.0, max_len=9, min_len=2, curvature=0.9,
               amount=1.4, impasto=1.7, edge_stop=24),
        ],
        chroma=1.3, lightness=1.08, jitter=(4.0, 0.12, 7.0), loose_mix=0.55,
        dry_between=False, angle_noise=50.0, coarse_scale=0.04, brush_care='wipe'),
    # dabs of (nearly) pure colour placed side by side; optical mixing
    'pointillist': Style(
        layers=[
            _L(radius=22, shape='filbert', blur=0.6, threshold=0.0, max_len=5, min_len=2,
               amount=0.8, thinning=0.35, curvature=0.6),
            _L(radius=6.5, shape='round', grid=0.95, threshold=4.0, max_len=0, min_len=0,
               profile='dab', pressure=0.85, amount=1.2, impasto=1.4),
            _L(radius=4.2, shape='round', grid=0.95, threshold=5.0, max_len=0, min_len=0,
               profile='dab', pressure=0.85, amount=1.2, impasto=1.4),
        ],
        chroma=1.25, jitter=(3.0, 0.08, 6.0), loose_mix=0.0, divisionism=9.0, brushes=4,
        brush_care='clean'),
    # few, long, thin strokes; toned ground shows through; dark contour drawing
    'loose_sketch': Style(
        layers=[
            _L(radius=6, shape='round', blur=0.8, threshold=12.0, max_len=16, min_len=4,
               curvature=0.85, amount=0.7, thinning=0.55, impasto=0.5, pressure=0.6,
               edge_avoid=-0.8, min_edge=14.0, value=-22.0, chroma=0.6, edge_stop=40,
               coverage=0.85, grid=1.5, max_turn=55),
            _L(radius=20, shape='flat', blur=0.7, threshold=10.0, max_len=4, min_len=2,
               curvature=0.6, amount=0.6, thinning=0.5, impasto=0.4, pressure=0.6, grid=1.5,
               coverage=0.5, edge_stop=16, angle_jitter=12),
            _L(radius=9, shape='flat', threshold=14.0, max_len=4, min_len=1, curvature=0.7,
               amount=0.7, thinning=0.3, impasto=0.6, pressure=0.62, grid=1.5, coverage=0.5,
               edge_stop=18, angle_jitter=10),
            _L(radius=3.0, shape='rigger', threshold=14.0, max_len=12, min_len=3,
               curvature=0.85, amount=0.8, pressure=0.7, grid=1.6, edge_avoid=-0.6,
               min_edge=10.0, value=-14.0, coverage=0.6, max_strokes=2500),
        ],
        chroma=0.95, jitter=(2.0, 0.05, 3.0), loose_mix=0.15, brushes=2,
        tone=('raw_umber', 0.05), dry_between=True),
    # many layers down to small soft brushes; low thresholds; quiet colour, low relief
    'realist': Style(
        layers=[
            _L(radius=30, shape='filbert', blur=0.5, threshold=0.0, max_len=8, curvature=0.6,
               amount=0.9, thinning=0.2, impasto=0.35, edge_stop=16),
            _L(radius=16, shape='filbert', threshold=5.0, max_len=7, impasto=0.45, edge_stop=18),
            _L(radius=8, shape='sable', threshold=5.0, max_len=6, curvature=0.75, impasto=0.5),
            _L(radius=4.5, shape='sable', threshold=5.5, max_len=5, min_len=1, curvature=0.8,
               impasto=0.55, max_strokes=15000),
            _L(radius=2.6, shape='sable', threshold=6.5, max_len=4, min_len=1, curvature=0.8,
               impasto=0.6, max_strokes=10000),
        ],
        chroma=1.0, jitter=(1.0, 0.03, 1.5), loose_mix=0.12, divisionism=0.0,
        clean_threshold=15.0),
}


# ==========================================================================
# The painter
# ==========================================================================
@dataclass
class PaintResult:
    """Statistics returned by :meth:`AutoPainter.paint`."""
    layers: list
    total_strokes: int
    elapsed: float
    delta_e: float              # mean CIE76 dE between final canvas and reference
    delta_e_blurred: float      # same against the slightly blurred reference
    palette: list

    def __str__(self):
        lines = [f'painterly: {self.total_strokes} strokes in {self.elapsed:.1f}s, '
                 f'mean dE {self.delta_e:.2f} (blurred ref {self.delta_e_blurred:.2f})']
        for L in self.layers:
            lines.append(
                f"  layer {L['layer']}: r={L['radius']:.1f} {L['shape']:<8} strokes={L['strokes']:>6} "
                f"reloads={L['reloads']:>5} cleans={L['cleans']:>5}  dE {L['delta_e_before']:.1f}"
                f"->{L['delta_e_after']:.1f}  plan {L['t_plan']:.2f}s paint {L['t_paint']:.2f}s")
        return '\n'.join(lines)


class AutoPainter:
    """Paint a reference image onto a :class:`~pinsel.Canvas` with physical brushes.

    Parameters
    ----------
    reference : (H, W, 3) sRGB image, uint8 or float 0..1 (resized to the
                canvas if necessary)
    style : str or Style   preset name from :data:`STYLES` or a Style object
    palette : list of Paint / names, or a ready :class:`PaletteSolver`;
              ``None`` = :data:`DEFAULT_PALETTE`.  ``False`` = no palette:
              every stroke gets a fairly opaque paint fitted directly to its
              colour (smooth reflectance spectrum; fast, but not real pigments).
    layers : list of LayerSpec / dicts overriding the style's layers.  A dict
             updates the style layer with the same index.
    region : (H, W) mask, paint only where > 0.5
    clip_to_region : also protect the outside with ``canvas.mask`` while painting
    importance : (H, W) weights multiplying the error (more detail where > 1)
    flow : user stroke direction: angle (deg), (H, W) angle map, (H, W, 2)
           vector field or callable(x, y) -> deg
    flow_weight : scalar or (H, W) map in 0..1 – how strongly strokes follow
           ``flow`` instead of the image structure (1 = per-region override)
    max_strokes : global cap on the number of strokes
    seed : random seed
    progress : callable(info: dict) – keys 'fraction', 'message', 'layer',
           'layers', 'strokes_done', 'strokes_total', 'elapsed'
    medium : medium of the palette paints ('oil', 'acrylic', 'gouache')
    max_pigments : max. tube colours per mixture (plus white) for the solver
    scale : multiplier for all LayerSpec lengths (default: max(H, W) / 1000,
           i.e. presets are tuned for a 1000 px painting)
    layer_overrides : dict of LayerSpec fields applied to every layer
           (e.g. ``dict(amount=1.4, impasto=1.5)``)
    **style_overrides : any :class:`Style` field (e.g. ``chroma=1.2``,
           ``brush_care='clean'``, ``dry_between=False``, ``tone=None``)
    """

    def __init__(self, reference, style='impressionist', palette=None, layers=None,
                 region=None, clip_to_region=False, importance=None, flow=None,
                 flow_weight=0.0, max_strokes=None, seed=0, progress=None,
                 medium='oil', max_pigments=3, scale=None, layer_overrides=None,
                 **style_overrides):
        if isinstance(style, str):
            if style not in STYLES:
                raise ValueError(f'unknown style {style!r}; choose from {list(STYLES)}')
            st = STYLES[style]
            st = replace(st, layers=[replace(l) for l in st.layers])
        else:
            st = replace(style, layers=[replace(l) if isinstance(l, LayerSpec) else LayerSpec(**l)
                                        for l in style.layers])
        valid = {f.name for f in fields(Style)}
        for k, v in style_overrides.items():
            if k not in valid:
                raise TypeError(f'unknown option {k!r}')
            setattr(st, k, v)
        if layers is not None:
            new = []
            for i, l in enumerate(layers):
                if isinstance(l, LayerSpec):
                    new.append(replace(l))
                else:
                    base = st.layers[min(i, len(st.layers) - 1)] if st.layers else LayerSpec()
                    new.append(replace(base, **dict(l)))
            st.layers = new
        if layer_overrides:
            st.layers = [replace(l, **layer_overrides) for l in st.layers]
        self.style = st
        self.reference = _as_srgb_float(reference)[..., :3]
        self.region = None if region is None else np.asarray(region, dtype=np.float32)
        self.clip_to_region = clip_to_region
        self.importance = None if importance is None else np.asarray(importance, dtype=np.float32)
        self.flow = flow
        self.flow_weight = flow_weight
        self.max_strokes = max_strokes
        self.seed = seed
        self.progress = progress
        self.medium = medium
        self.scale = scale
        self._palette_arg = palette
        self.max_pigments = max_pigments
        self.solver = palette if isinstance(palette, PaletteSolver) else None

    # ------------------------------------------------------------------ helpers
    def _get_solver(self, canvas):
        if self.solver is not None:
            return self.solver
        if self._palette_arg is False:
            return None
        # masstone over the canvas' own ground
        self.solver = PaletteSolver(self._palette_arg, max_pigments=self.max_pigments,
                                    medium=self.medium, thickness=3.0, cache_step=1.0,
                                    ground=canvas.ground.reflectance.astype(np.float64))
        return self.solver

    def _prepare_reference(self, H, W):
        ref = self.reference
        if ref.shape[:2] != (H, W):
            from PIL import Image
            im = Image.fromarray((np.clip(ref, 0, 1) * 255 + 0.5).astype(np.uint8))
            ref = np.asarray(im.resize((W, H), Image.LANCZOS), dtype=np.float64) / 255.0
        lin = srgb_to_linear(ref).astype(np.float32)
        return lin

    def _resize_map(self, m, H, W):
        if m is None:
            return None
        if m.shape[:2] == (H, W):
            return m
        zoom = (H / m.shape[0], W / m.shape[1]) + ((1,) if m.ndim == 3 else ())
        return ndimage.zoom(m, zoom, order=1)

    def _report(self, **info):
        if self.progress is not None:
            info['elapsed'] = time.time() - self._t0
            self.progress(info)

    def _target_lab(self, lab, chroma, lightness):
        lab = lab.copy()
        if lightness != 1.0:
            lab[..., 0] = np.clip(55.0 + (lab[..., 0] - 55.0) * lightness, 0, 100)
        if chroma != 1.0:
            lab[..., 1:] *= chroma
        return lab

    def _fallback_field(self, lab_full, H, W, rng):
        st = self.style
        sigma = max(4.0, st.coarse_scale * max(H, W))
        small = max(1, int(sigma / 6))
        lab_s = lab_full[::small, ::small]
        vx, vy, an = _tensor_dirvec(_gauss(lab_s, sigma / small * 0.5), sigma / small)
        tau = float(np.percentile(an, 50)) + 1e-9
        s = 1.0 - np.exp(-an / tau)
        from .noise import fbm
        nz = fbm((H, W), scale=max(H, W) / 5.0, octaves=3, seed=rng)
        ang = np.radians(st.default_angle + st.angle_noise * nz)
        dvx, dvy = np.cos(2 * ang), np.sin(2 * ang)
        sx = ndimage.zoom(s * vx / (an + 1e-12), (H / s.shape[0], W / s.shape[1]), order=1)
        sy = ndimage.zoom(s * vy / (an + 1e-12), (H / s.shape[0], W / s.shape[1]), order=1)
        ss = ndimage.zoom(s, (H / s.shape[0], W / s.shape[1]), order=1)
        sx, sy, ss = sx[:H, :W], sy[:H, :W], ss[:H, :W]
        fx = 0.6 * sx + (1 - 0.6 * ss) * dvx
        fy = 0.6 * sy + (1 - 0.6 * ss) * dvy
        m = np.hypot(fx, fy) + 1e-9
        return np.stack([fx / m, fy / m], axis=-1)

    @staticmethod
    def _seeds(D, g, T, rng, score=None):
        """Hertzmann grid: one seed per cell whose mean error > T, placed at the
        maximum of ``score`` (default: the error itself) inside the cell."""
        H, W = D.shape
        g = max(1, int(round(g)))
        oy, ox = (int(v) for v in rng.integers(0, g, 2))
        nh = int(math.ceil((H + oy) / g))
        nw = int(math.ceil((W + ox) / g))
        Dp = np.zeros((nh * g, nw * g), dtype=np.float32)
        Cp = np.zeros((nh * g, nw * g), dtype=np.float32)
        Dp[oy:oy + H, ox:ox + W] = D
        Cp[oy:oy + H, ox:ox + W] = 1.0
        cells = Dp.reshape(nh, g, nw, g).transpose(0, 2, 1, 3).reshape(nh, nw, g * g)
        cnt = Cp.reshape(nh, g, nw, g).sum(axis=(1, 3))
        mean = cells.sum(axis=-1) / np.maximum(cnt, 1)
        Sp = np.full((nh * g, nw * g), -1e9, dtype=np.float32)
        Sp[oy:oy + H, ox:ox + W] = (D if score is None else score) + \
            rng.random(D.shape, dtype=np.float32) * 0.5
        am = Sp.reshape(nh, g, nw, g).transpose(0, 2, 1, 3).reshape(nh, nw, g * g).argmax(axis=-1)
        sel = (mean > T) & (cnt > 0.25 * g * g)
        ci, cj = np.nonzero(sel)
        a = am[ci, cj]
        ys = ci * g + a // g - oy
        xs = cj * g + a % g - ox
        err = mean[ci, cj]
        ok = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
        jit = rng.uniform(-0.3, 0.3, (2, ok.sum())) * min(g, 6)
        return (np.clip(xs[ok] + jit[0], 0, W - 1).astype(np.float64),
                np.clip(ys[ok] + jit[1], 0, H - 1).astype(np.float64), err[ok])

    # ------------------------------------------------------------------ colour
    def _stroke_colours(self, cols, spec, rng, xs, ys, H, W):
        """Apply style colour transforms + jitter + divisionism (Lab, (n,3))."""
        st = self.style
        chroma = st.chroma if spec.chroma is None else spec.chroma
        jit = st.jitter if spec.jitter is None else spec.jitter
        div = st.divisionism if spec.divisionism is None else spec.divisionism
        lab = self._target_lab(cols, chroma, st.lightness)
        if spec.value:
            lab[:, 0] = np.clip(lab[:, 0] + spec.value, 0, 100)
        n = len(lab)
        if n == 0:
            return lab
        dL, dC, dh = jit
        L = lab[:, 0] + rng.normal(0, 1, n) * dL
        a, b = lab[:, 1], lab[:, 2]
        C = np.hypot(a, b) * np.exp(rng.normal(0, 1, n) * dC)
        h = np.arctan2(b, a) + np.radians(rng.normal(0, 1, n) * dh)
        a, b = C * np.cos(h), C * np.sin(h)
        if div > 0:
            # complementary splitting along a slowly varying hue axis: dabs
            # alternate between c+d and c-d so that they average to c optically
            phi = self._div_phase[np.clip(ys.astype(int), 0, H - 1), np.clip(xs.astype(int), 0, W - 1)]
            sgn = np.where(rng.random(n) < 0.5, -1.0, 1.0)
            amp = div * (0.6 + 0.8 * rng.random(n))
            a = a + sgn * amp * np.cos(phi)
            b = b + sgn * amp * np.sin(phi)
            L = L + sgn * amp * 0.25 * rng.normal(0, 1, n)
        return np.column_stack([np.clip(L, 0, 100), a, b])

    # ------------------------------------------------------------------ main
    def paint(self, canvas) -> PaintResult:
        """Paint the reference onto ``canvas``.  Returns a :class:`PaintResult`."""
        self._t0 = time.time()
        st = self.style
        H, W = canvas.H, canvas.W
        rng = np.random.default_rng(self.seed)
        scale = self.scale or max(H, W) / 1000.0
        ref_lin = self._prepare_reference(H, W)
        ref_lab_full = linear_rgb_to_lab(ref_lin).astype(np.float32)
        region = self._resize_map(self.region, H, W)
        importance = self._resize_map(self.importance, H, W)
        reg = np.ones((H, W), dtype=np.float32) if region is None else \
            (region > 0.5).astype(np.float32)
        solver = self._get_solver(canvas)
        old_mask = None
        self._protect = None
        if self.clip_to_region and region is not None:
            old_mask = canvas.mask.copy()
            canvas.mask = np.maximum(canvas.mask, 1.0 - np.clip(region, 0, 1)).astype(np.float32)
            # the brush kernel's ploughing can push wet paint ~2 px into masked
            # pixels; keep a copy of the protected pixels to "peel the tape"
            prot = canvas.mask >= 0.999
            self._protect = (prot, {k: getattr(canvas, k)[prot].copy() for k in _CLIP_STATE})
        try:
            if st.tone is not None and float(canvas.h_dry.max()) == 0.0 and \
                    float(canvas.film_vol.max()) == 0.0:
                tp, tt = st.tone
                tpaint = tp if isinstance(tp, Paint) else _pg.get(str(tp), self.medium)
                _tone(canvas, tpaint, float(tt), None if region is None else reg,
                      int(rng.integers(1 << 30)))
            from .noise import fbm
            self._div_phase = (np.pi * (1.0 + fbm((H, W), scale=max(H, W) / 8.0, octaves=2,
                                                 seed=rng))).astype(np.float32)
            fallback = self._fallback_field(ref_lab_full, H, W, rng)
            stats = []
            total = 0
            nlayers = len(st.layers)
            for li, spec in enumerate(st.layers):
                if self.max_strokes is not None and total >= self.max_strokes:
                    break
                lst = self._paint_layer(canvas, spec, li, nlayers, ref_lin, reg, importance,
                                        fallback, solver, rng, scale, total)
                total += lst['strokes']
                stats.append(lst)
                dry = st.dry_between if spec.dry_after is None else spec.dry_after
                if dry and li < nlayers - 1:
                    self._peel(canvas)
                    canvas.dry()
        finally:
            self._peel(canvas)
            if old_mask is not None:
                canvas.mask = old_mask
        fin = linear_rgb_to_lab(canvas.render_linear())
        de = float(np.sqrt(((fin - ref_lab_full) ** 2).sum(-1))[reg > 0].mean())
        refb = linear_rgb_to_lab(_gauss(ref_lin, 2.0 * scale))
        deb = float(np.sqrt(((fin - refb) ** 2).sum(-1))[reg > 0].mean())
        self._report(fraction=1.0, message='done', layer=len(st.layers), layers=len(st.layers),
                     strokes_done=total, strokes_total=total)
        pal = [p.name for p in solver.pigments] if solver is not None else []
        return PaintResult(stats, total, time.time() - self._t0, de, deb, pal)

    def _peel(self, canvas):
        """Restore pixels protected by ``clip_to_region`` (remove leaked paint)."""
        if self._protect is None:
            return
        prot, saved = self._protect
        for k, v in saved.items():
            getattr(canvas, k)[prot] = v

    def _paint_layer(self, canvas, spec, li, nlayers, ref_lin, reg, importance, fallback,
                     solver, rng, scale, total_before):
        st = self.style
        H, W = canvas.H, canvas.W
        s = 1.0 if spec.absolute else scale
        R = max(1.0, spec.radius * s)
        t_plan = t_paint = 0.0
        tp0 = time.time()
        ref_lab = linear_rgb_to_lab(_gauss(ref_lin, spec.blur * R)).astype(np.float32)
        V = orientation_field(ref_lab, R, spec.tensor_scale, fallback, self.flow, self.flow_weight)
        # colour change across one brush radius: seeds avoid sitting right on edges
        edge = np.zeros((H, W), dtype=np.float32)
        if spec.edge_avoid != 0 or spec.min_edge > 0:
            for c in range(3):
                gx = ndimage.sobel(ref_lab[..., c], axis=1, mode='nearest') / 8.0
                gy = ndimage.sobel(ref_lab[..., c], axis=0, mode='nearest') / 8.0
                edge += gx * gx + gy * gy
            edge = (np.sqrt(edge) * R).astype(np.float32)
        nb_brushes = spec.brushes or st.brushes
        # brushes: one per value family, slightly different sizes
        sizes = R * (1.0 + spec.size_var * np.linspace(-1, 1, nb_brushes)) if nb_brushes > 1 \
            else np.array([R])
        crayon = spec.shape == 'oil_crayon'
        if crayon:
            from .tools import OilCrayon
            brushes = [OilCrayon(size=2.0 * sz, **spec.brush) for sz in sizes]
        else:
            from .brush import Brush
            brushes = [Brush(spec.shape, float(sz), seed=int(rng.integers(1 << 30)), **spec.brush)
                       for sz in sizes]
        bstate = [dict(key=None, lab=None) for _ in brushes]
        info = dict(layer=li + 1, radius=R, shape=spec.shape, strokes=0, reloads=0, cleans=0,
                    wipes=0, passes=0, delta_e_before=0.0, delta_e_after=0.0)
        t_plan += time.time() - tp0
        cap_layer = spec.max_strokes
        if self.max_strokes is not None:
            remaining = self.max_strokes - total_before
            left = nlayers - li
            share = remaining if left == 1 else int(math.ceil(remaining / left * 1.6))
            cap_layer = share if cap_layer is None else min(cap_layer, share)
        for ps_i in range(max(1, spec.passes)):
            tp0 = time.time()
            # compare at the scale of the brush: bristle streaks are not "error"
            cvl = linear_rgb_to_lab(_gauss(canvas.render_linear(), max(0.8, spec.compare_blur * R))
                                    ).astype(np.float32)
            D = np.sqrt(((cvl - ref_lab) ** 2).sum(-1)).astype(np.float32)
            if ps_i == 0:
                info['delta_e_before'] = float(D[reg > 0].mean())
            D *= reg
            if importance is not None:
                D *= importance
            score = D - spec.edge_avoid * edge if spec.edge_avoid != 0 else None
            xs, ys, err = self._seeds(D, spec.grid * R, spec.threshold, rng, score)
            if spec.min_edge > 0 and len(xs):
                keep = edge[ys.astype(int), xs.astype(int)] >= spec.min_edge
                xs, ys, err = xs[keep], ys[keep], err[keep]
            if spec.coverage < 1.0 and len(xs):
                keep = rng.random(len(xs)) < spec.coverage
                xs, ys, err = xs[keep], ys[keep], err[keep]
            if cap_layer is not None:
                room = cap_layer - info['strokes']
                if room <= 0:
                    break
                if len(xs) > room:
                    # error-weighted sampling without replacement (spatially spread)
                    pri = rng.random(len(xs)) ** (1.0 / np.maximum(err, 1e-3))
                    keep = np.argsort(-pri)[:room]
                    xs, ys, err = xs[keep], ys[keep], err[keep]
            n = len(xs)
            if n == 0:
                t_plan += time.time() - tp0
                break
            step = spec.step * R
            max_steps = int(spec.max_len)
            ctrl = np.zeros((n, max_steps + 2, 2))
            nctrl = np.zeros(n, dtype=np.int64)
            cols = np.zeros((n, 3))
            back = int(round(spec.back_frac * max_steps))
            rots = np.radians(rng.normal(0, 1, n) * spec.angle_jitter)
            _grow_strokes(xs, ys, rots, ref_lab, cvl, V, reg, float(step), int(spec.min_len),
                          max_steps, back, float(spec.curvature), float(spec.edge_stop),
                          math.cos(math.radians(spec.max_turn)), float(max(1.0, spec.dab_len * R)),
                          float(st.edge_margin * R), float(spec.color_sample), ctrl, nctrl, cols)
            sub = max(2, int(math.ceil(step / 1.5)))
            dense = np.zeros((n, (max_steps + 1) * sub + 2, 2))
            nd = np.zeros(n, dtype=np.int64)
            _densify(ctrl, nctrl, sub, dense, nd)
            # ---- colours and recipes
            lab_t = self._stroke_colours(cols, spec, rng, xs, ys, H, W)
            if solver is not None:
                ridx, rw, rlab, rde, rkeys = solver.match_lab(lab_t)
                Kpv, Spv, props = solver.mixture_ks(ridx, rw)
            else:
                ridx = rw = None
                rlab = lab_t
                rkeys = np.round(lab_t / 0.6).astype(np.int64) @ np.array([1 << 24, 1 << 12, 1])
                Kpv, Spv = _direct_ks(lab_t, canvas)
                props = None
            # ---- order
            Lr = rlab[:, 0]
            band = np.zeros(n, dtype=np.int64)
            if len(brushes) > 1:
                qs = np.quantile(Lr, np.linspace(0, 1, len(brushes) + 1)[1:-1]) if n > 4 else \
                    np.linspace(30, 70, len(brushes) - 1)
                band = np.searchsorted(qs, Lr)
            if st.order == 'dark_to_light':
                order = np.argsort(Lr + rng.normal(0, 6, n))
            elif st.order == 'light_to_dark':
                order = np.argsort(-Lr + rng.normal(0, 6, n))
            elif st.order == 'grouped':
                q = np.round(rlab / 12.0).astype(np.int64)
                gk = (q[:, 0] * 64 + q[:, 1] + 32) * 64 + q[:, 2] + 32
                order = np.lexsort((rng.random(n), gk, band))
            else:
                order = rng.permutation(n)
            t_plan += time.time() - tp0
            # ---- paint
            tq0 = time.time()
            loose = st.loose_mix if spec.loose_mix is None else spec.loose_mix
            base_paint = solver.pigments[0] if solver is not None else _pg.TITANIUM_WHITE
            thin = float(np.clip(1.0 - spec.thinning, 0.02, 1.0))
            nlayer_total = n
            for c_i, k in enumerate(order):
                bi = int(band[k])
                br = brushes[bi]
                bs = bstate[bi]
                key = int(rkeys[k])
                if crayon:
                    if key != bs['key']:
                        br.load(_fast_paint(base_paint, Kpv[k] * thin, Spv[k] * thin))
                        bs['key'] = key
                        info['reloads'] += 1
                elif key != bs['key'] or br.load_level < st.reload_level:
                    paints = self._make_load(base_paint, Kpv[k], Spv[k],
                                             None if props is None else props[k],
                                             ridx, rw, k, solver, loose, thin, spec, rng)
                    mixp = 1.0 - 0.85 * loose if loose > 0 else 1.0
                    if key != bs['key']:
                        dE = 1e9 if bs['lab'] is None else float(np.linalg.norm(rlab[k] - bs['lab']))
                        lighter = bs['lab'] is not None and rlab[k][0] - bs['lab'][0] > 0.6 * st.clean_threshold
                        if st.brush_care == 'clean' or bs['lab'] is None or \
                                (st.brush_care == 'auto' and (dE > st.clean_threshold or lighter)) or \
                                (st.brush_care == 'wipe' and dE > 2.5 * st.clean_threshold):
                            br.load(paints, amount=spec.amount, mix=mixp, clean=True)
                            info['cleans'] += 1
                        else:
                            br.wipe(st.wipe)
                            amt = max(0.05, spec.amount - br.load_level)
                            br.load(paints, amount=amt, mix=mixp, clean=False)
                            info['wipes'] += 1
                        bs['key'] = key
                        bs['lab'] = rlab[k].copy()
                    else:
                        amt = max(0.05, spec.amount - br.load_level)
                        br.load(paints, amount=amt, mix=mixp, clean=False)
                    info['reloads'] += 1
                pts = dense[k, :nd[k]]
                prof = self._pressure(spec, pts, R, rng)
                if crayon:
                    canvas.stroke(br, pts, pressure=prof, angle=spec.angle)
                else:
                    canvas.stroke(br, pts, pressure=prof, angle=spec.angle,
                                  wobble=spec.wobble * R, speed=spec.speed,
                                  pressure_jitter=spec.pressure_jitter,
                                  seed=int(rng.integers(1 << 30)))
                info['strokes'] += 1
                if (c_i & 255) == 0:
                    self._report(fraction=(li + c_i / max(1, nlayer_total)) / nlayers,
                                 message=f'layer {li + 1}/{nlayers} (r={R:.1f})',
                                 layer=li + 1, layers=nlayers, strokes_done=c_i,
                                 strokes_total=nlayer_total)
            t_paint += time.time() - tq0
            info['passes'] += 1
        cvl = linear_rgb_to_lab(canvas.render_linear())
        D = np.sqrt(((cvl - ref_lab) ** 2).sum(-1))
        info['delta_e_after'] = float(D[reg > 0].mean())
        info['t_plan'] = t_plan
        info['t_paint'] = t_paint
        return info

    def _make_load(self, base, K, S, props, ridx, rw, k, solver, loose, thin, spec, rng):
        """Paint(s) to load: the mixture, or streaky partially mixed variants."""
        pr = {} if props is None else dict(zip(_PROPS, props.tolist()))
        # lean (thinned) paint has little body; impasto paint has more
        bf = spec.impasto * (1.0 - 0.8 * spec.thinning)
        pr['body'] = pr.get('body', base.body) * bf
        if loose <= 0 or solver is None:
            return _fast_paint(base, K * thin, S * thin, 'mix', **pr)
        sel = ridx[k] >= 0
        idx = ridx[k][sel]
        w = rw[k][sel]
        if len(idx) == 1:
            return _fast_paint(base, K * thin, S * thin, 'mix', **pr)
        Kp = solver._K[idx]
        Sp = solver._S[idx]
        out = []
        nvar = 3
        for v in range(nvar):
            # incompletely mixed paint: concentrations wander log-normally
            wv = w * np.exp(rng.normal(0, 0.9 * loose, len(w)))
            wv /= wv.sum()
            out.append((_fast_paint(base, (wv @ Kp) * thin, (wv @ Sp) * thin, 'mix', **pr),
                        1.0 if v else 1.5))
        return out

    @staticmethod
    def _pressure(spec, pts, R, rng):
        peak = float(np.clip(spec.pressure * (1 + spec.pressure_var * rng.normal()), 0.2, 1.0))
        prof = spec.profile
        L = float(np.sqrt((np.diff(pts, axis=0) ** 2).sum(1)).sum()) if len(pts) > 1 else 0.0
        if prof == 'auto':
            prof = 'dab' if L < 0.9 * R else 'taper'
        if prof == 'dab':
            return _dab(peak)
        if prof == 'swell':
            lo = 0.35 * peak
            return lambda t: lo + (peak - lo) * np.exp(-((np.asarray(t) - 0.45) / 0.35) ** 2)
        if prof == 'flick':
            from .path import flick
            return flick(peak)
        if prof == 'constant':
            return peak
        # taper with attack/release relative to the stroke length
        att = float(np.clip(0.6 * R / max(L, 1e-6), 0.04, 0.35)) * rng.uniform(0.7, 1.3)
        rel = float(np.clip(1.2 * R / max(L, 1e-6), 0.08, 0.45)) * rng.uniform(0.7, 1.3)
        from .path import taper
        return taper(attack=att, release=rel, peak=peak, start=0.35 * peak, end=0.1 * peak)


_CLIP_STATE = ('film_K', 'film_S', 'film_vol', 'wet', 'gloss', 'body', 'drate',
               'below_R', 'h_dry', 'gloss_dry')


def _tone(canvas, paint, thickness, region, seed):
    """Imprimatura like ``Canvas.tone`` but limited to ``region`` and
    respecting ``canvas.mask``."""
    if region is None and float(canvas.mask.max()) == 0.0:
        canvas.tone(paint, thickness=thickness, seed=seed)
        return
    from .noise import fbm
    g = np.random.default_rng(seed)
    var = 1.0 + 0.25 * fbm(canvas.shape, scale=canvas.W / 6, octaves=5, seed=g)
    cover = (1.0 - canvas.mask) * (1.0 if region is None else region)
    t = (np.clip(thickness * var, 0, None) * cover).astype(np.float32)
    sel = t > 1e-6
    canvas.film_K += paint.K[None, None, :].astype(np.float32) * t[..., None]
    canvas.film_S += paint.S[None, None, :].astype(np.float32) * t[..., None]
    canvas.film_vol += t * 0.3
    canvas.body[sel] = 0.3
    canvas.gloss[sel] = paint.gloss
    canvas.wet[sel] = 1.0
    canvas._bake(sel)


def _dab(peak):
    def f(t):
        t = np.clip(np.asarray(t, float), 0, 1)
        return peak * (0.35 + 0.65 * np.sin(np.pi * t) ** 0.5)
    return f


def _smooth_projector():
    from .spectral import BAND_CENTERS
    lam = BAND_CENTERS
    Sig = np.exp(-0.5 * ((lam[:, None] - lam[None, :]) / 70.0) ** 2)
    M = SPEC_TO_RGB
    return Sig @ M.T @ np.linalg.inv(M @ Sig @ M.T)


_SMOOTH_P = _smooth_projector()


def smooth_reflectance(linear_rgb, iters=6, lo=0.003, hi=0.98):
    """Vectorised smooth reflectance spectra (n, NB) reproducing linear sRGB:
    the smoothest deviation from a grey of equal luminance, projected onto
    [lo, hi] (fast alternative to :func:`pinsel.spectral.rgb_to_spectrum`)."""
    t = np.clip(np.asarray(linear_rgb, dtype=np.float64).reshape(-1, 3), 1e-4, 1.0)
    y = t @ np.array([0.2126, 0.7152, 0.0722])
    R = np.repeat(y[:, None], NB, axis=1)
    for _ in range(iters):
        R = np.clip(R + (t - R @ SPEC_TO_RGB.T) @ _SMOOTH_P.T, lo, hi)
    return R


def _direct_ks(lab, canvas=None):
    """K/S per volume straight from colour (no palette): fairly opaque paint."""
    from .spectral import ks_from_reflectance, BAND_CENTERS
    R = smooth_reflectance(lab_to_linear_rgb(lab))
    S0 = np.full(NB, 3.5) * (560.0 / BAND_CENTERS) ** 0.3
    Sv = np.broadcast_to(S0, R.shape).copy()
    return Sv * ks_from_reflectance(R), Sv


def paint_image(canvas, reference, style='impressionist', **kw) -> PaintResult:
    """Convenience wrapper: ``AutoPainter(reference, style, **kw).paint(canvas)``."""
    return AutoPainter(reference, style=style, **kw).paint(canvas)
