"""
Watercolour: a physically inspired fluid simulation of water, paper and pigment.

The model follows the three-layer idea of Curtis et al. (1997, *Computer-
Generated Watercolor*) but replaces their velocity solver by a cheaper
virtual-pipe shallow-water scheme with porous-paper physics:

**Shallow-water layer** (``wc.water``, fluxes in ``wc.u``/``wc.v``)
    Free water flows down gradients of *water depth + paper relief* (+ board
    tilt) through "virtual pipes" between neighbouring pixels.  Pipe fluxes
    carry momentum (damped by viscosity, which grows in thin films) so water
    rushes into wet paper quickly and settles calmly.  Water can only enter
    a pixel that is already wet, sufficiently *damp* (capillary saturation
    above a fibre-dependent threshold) or that it overtops (a juicy brush
    load creeps into the paper valleys).  Dry paper therefore pins the
    contact line (crisp wet-on-dry edges) while damp paper lets fronts
    invade irregularly -- fractal, cauliflower *backruns*.

**Capillary layer** (``wc.damp``)
    Paper absorbs free water up to its capacity; moisture wicks through a
    rasterised network of paper fibres (anisotropic, irregular per-link
    conductances -> feathery fringes) with a saturation-dependent
    conductivity (wet paper conducts, dry paper hardly).

**Pigment**
    Suspended pigment (``wc.K, S, m, g, st``) is advected with the water
    (upwind, exactly mass conserving) and diffuses inside it.  It is
    adsorbed onto the paper (``wc.dK, dS, dm, dg, dst``) faster where the
    water is thin, and re-dissolved (desorbed) under free water.  The
    granulation ratio ``g/m`` biases adsorption towards paper valleys (and
    desorption from peaks) -- granulating ultramarine, sienna, cobalt settle
    into the tooth while phthalo stays smooth.  Staining pigments (``st/m``)
    resist desorption and lifting; deposits that have dried once become
    *bonded* (gum arabic) and re-wet only slowly, so glazes stay put.

**Evaporation**
    Free water evaporates, faster at the boundary of a wet area.  The
    replacement flow towards the pinned edge carries pigment that settles
    there: the classic *dark rim* (edge darkening / coffee ring).  When a
    pixel dries completely, all remaining pigment settles.

Time stepping is adaptive: ``canvas.wait(3)`` between strokes keeps a wash
wet (wet-in-wet); a normal wash is dry after roughly 2-4 minutes of
simulated time.  Only the bounding box of wet pixels is simulated.

Public helpers (all respect ``canvas.mask``):
:func:`wet_paper`, :func:`wash`, :func:`variegated_wash`, :func:`drop`,
:func:`spritz`, :func:`spatter`, :func:`salt`, :func:`blot`, :func:`lift`,
:func:`tilt`, :func:`hair_dryer`, :func:`wetness`, and region builders
:func:`rect`, :func:`ellipse`, :func:`polygon`.
"""
from __future__ import annotations

import math
import time as _time
import types
import numpy as np
from numba import njit, prange

from .spectral import NB
from . import noise as _nz

__all__ = ['simulate', 'settle', 'wet_paper', 'wash', 'variegated_wash', 'drop',
           'spritz', 'spatter', 'salt', 'blot', 'lift', 'tilt', 'hair_dryer',
           'wetness', 'is_dry', 'rect', 'ellipse', 'polygon', 'region_mask',
           'PARAMS', 'PIGMENT_PER_WATER']

#: pigment volume carried per unit water at ``concentration=1`` (matches
#: :meth:`pinsel.brush.Brush.load`, which loads ``water * amount * 0.35``).
PIGMENT_PER_WATER = 0.35

#: Tunable physical constants (rates are per simulated second).
PARAMS = dict(
    dt=1.0,              # macro time step while water rushes around (s)
    dt_calm=2.0,         # macro time step while free water is calm (s)
    dt_damp=2.0,         # macro time step when only damp paper is left (s)
    sub_min=2,           # flow sub-steps per macro step while busy
    sub_max=16,          # ... where water is rushing around (hot tiles)
    gravity=0.20,        # virtual-pipe acceleration per sub-step
    friction=0.14,       # flux damping per sub-step (viscosity)
    h_visc=0.06,         # thin films (< h_visc) are increasingly viscous
    kappa=0.2,           # weight of paper relief in the flow potential
    h_min=0.004,         # a pixel needs this much free water to push it out
    h_wet=0.012,         # free water below this counts as "no shine"
    theta=0.35,          # damp saturation above which free water can invade
    theta_var=0.15,      # spatial spread of that threshold (fibres, relief)
    wett_width=0.25,     # saturation margin above threshold for full invasion speed
    h_adv=0.95,          # free-water depth that overtops dry paper
    adv_width=0.35,
    absorb=0.16,         # paper absorption rate (1/s, scaled by ground.absorbency*2)
    capacity=0.30,       # water held by saturated paper (height units)
    evap=0.0055,         # free-water evaporation (height units / s)
    edge_evap=4.0,       # evaporation boost at the boundary of a wet area
    edge_radius=7,       # px, size of the boundary zone
    edge_dep=4.0,        # faster settling in the evaporating boundary zone
    evap_damp=1.0 / 55,  # paper drying (saturation / s)
    dep=0.0030,          # adsorption rate in free water (x 1/(h+h0))
    dep_h0=0.025,
    dep_damp=0.025,      # adsorption rate inside damp paper without shine
    des=0.03,            # desorption rate of fresh deposits under free water
    shear_lift=0.6,      # extra desorption per unit flow speed (backruns)
    bond=0.95,           # resistance of dried (bonded) deposits to re-wetting
    bond_time=6.0,       # time for a dry deposit to bond (s)
    soften=0.004,        # re-wetting slowly loosens bonded deposits (1/s)
    gran=0.45,           # strength of granulation (valley preference)
    mix=0.022,           # pigment diffusion inside free water (per 0.1 s)
    capillary=0.11,      # capillary wicking of paper moisture (per 0.5 s)
    salt_absorb=0.25,    # salt crystals soak up free water (per s and unit salt)
    salt_push=0.10,      # salt brine pushes pigment away (per sub-step)
    salt_diffuse=0.10,   # brine spreads through wet paper (per macro step)
)

# indices into the packed float64 parameter vector used by the kernels
(_I_G, _I_FR, _I_HV, _I_HMIN, _I_HWET, _I_WW, _I_HADV, _I_ADVW, _I_THETA,
 _I_ABS, _I_CAP, _I_EV, _I_EDGE, _I_EVD, _I_DEP, _I_H0, _I_DEPD, _I_DES,
 _I_BOND, _I_BT, _I_SOFT, _I_GRAN, _I_SABS, _I_EDEP, _I_SHEAR) = range(25)


def _param_vector(canvas):
    P = PARAMS
    a = np.zeros(32, dtype=np.float64)
    a[_I_G] = P['gravity']; a[_I_FR] = P['friction']; a[_I_HV] = P['h_visc']
    a[_I_HMIN] = P['h_min']; a[_I_HWET] = P['h_wet']; a[_I_WW] = P['wett_width']
    a[_I_HADV] = P['h_adv']; a[_I_ADVW] = P['adv_width']; a[_I_THETA] = P['theta']
    a[_I_ABS] = P['absorb'] * 2.0 * float(getattr(canvas.ground, 'absorbency', 0.5))
    a[_I_CAP] = P['capacity']; a[_I_EV] = P['evap']; a[_I_EDGE] = P['edge_evap']
    a[_I_EVD] = P['evap_damp']; a[_I_DEP] = P['dep']; a[_I_H0] = P['dep_h0']
    a[_I_DEPD] = P['dep_damp']; a[_I_DES] = P['des']; a[_I_BOND] = P['bond']
    a[_I_BT] = P['bond_time']; a[_I_SOFT] = P['soften']; a[_I_GRAN] = P['gran']
    a[_I_SABS] = P['salt_absorb']
    a[_I_EDEP] = P['edge_dep']
    a[_I_SHEAR] = P['shear_lift']
    return a


# ==========================================================================
# Paper structure (fibre network, wettability, granulation sites)
# ==========================================================================
@njit(cache=True)
def _k_fibres(H, W, xs, ys, ang, lens, curv, wts, cx, cy, cov):
    """Rasterise curved fibres 4-connectedly; count fibre links per pipe."""
    for f in range(xs.shape[0]):
        x = xs[f]
        y = ys[f]
        a = ang[f]
        L = lens[f]
        c = curv[f]
        wt = wts[f]
        ix = int(math.floor(x))
        iy = int(math.floor(y))
        t = 0.0
        while t < L:
            x += math.cos(a) * 0.5
            y += math.sin(a) * 0.5
            a += c * 0.5
            t += 0.5
            jx = int(math.floor(x))
            jy = int(math.floor(y))
            if jx != ix:
                lx = min(ix, jx)
                if 0 <= iy < H and 0 <= lx < W - 1:
                    cx[iy, lx] += wt
                ix = jx
            if jy != iy:
                ly = min(iy, jy)
                if 0 <= ix < W and 0 <= ly < H - 1:
                    cy[ly, ix] += wt
                iy = jy
            if 0 <= ix < W and 0 <= iy < H:
                cov[iy, ix] += wt


class _Paper:
    """Cached per-canvas paper properties used by the simulation."""

    def __init__(self, canvas):
        H, W = canvas.H, canvas.W
        g = np.random.default_rng(int(canvas.seed) * 7919 + 4242)
        hgt = canvas.h_ground.astype(np.float64)
        relief = float(getattr(canvas.ground, 'relief', 0.5)) or 1.0
        hn = np.clip(hgt / (relief + 1e-9), 0, 1)
        kind = getattr(canvas.ground, 'kind', 'cold_press')
        # ---- fibre network: ~2.5 fibres cross every pixel
        mean_len = {'rough': 26.0, 'cold_press': 22.0, 'hot_press': 16.0}.get(kind, 18.0)
        n = int(H * W * 2.6 / mean_len)
        xs = g.uniform(-20, W + 20, n)
        ys = g.uniform(-20, H + 20, n)
        ang = g.uniform(0, np.pi, n)
        lens = np.clip(g.lognormal(np.log(mean_len), 0.5, n), 3, 120)
        curv = g.normal(0, 0.04, n)
        wts = np.clip(g.lognormal(0.0, 0.45, n), 0.2, 4.0)
        cx = np.zeros((H, W), np.float64)
        cy = np.zeros((H, W), np.float64)
        cov = np.zeros((H, W), np.float64)
        _k_fibres(H, W, xs, ys, ang, lens, curv, wts, cx, cy, cov)
        # valleys conduct water a little better than peaks
        val_x = 1.0 - 0.25 * (hn + np.roll(hn, -1, 1))
        val_y = 1.0 - 0.25 * (hn + np.roll(hn, -1, 0))
        Cx = (0.18 + cx) * val_x
        Cy = (0.18 + cy) * val_y
        mean = 0.5 * (Cx[:, :-1].mean() + Cy[:-1].mean())
        Cx = np.clip(Cx / mean, 0.05, 4.0)
        Cy = np.clip(Cy / mean, 0.05, 4.0)
        Cx[:, -1] = 0
        Cy[-1, :] = 0
        f32 = lambda a: np.ascontiguousarray(a, dtype=np.float32)
        self.cx_flow = f32(0.75 + 0.25 * Cx)          # surface flow: mild fibre bias
        self.cy_flow = f32(0.75 + 0.25 * Cy)
        self.cx_cap = f32(np.minimum(Cx ** 1.5, 4.0))  # capillarity: strongly fibrous
        self.cy_cap = f32(np.minimum(Cy ** 1.5, 4.0))
        self.cx_cap[:, -1] = 0
        self.cy_cap[-1, :] = 0
        self.cx_flow[:, -1] = 0
        self.cy_flow[-1, :] = 0
        # ---- wettability threshold: fibre-rich valleys wet first
        covn = _nz.normalize(_nz.blur(cov, 1.2))
        hpn = _nz.normalize(hn - _nz.blur(hn, 6.0))
        hps = _nz.normalize(_nz.blur(hn, 1.2) - _nz.blur(hn, 6.0))
        white = _nz.normalize(_nz.blur(g.standard_normal((H, W)), 1.3))
        lowf = _nz.fbm((H, W), scale=40.0, octaves=3, seed=g)
        midf = _nz.fbm((H, W), scale=10.0, octaves=3, seed=g)
        z = _nz.normalize(-0.45 * covn + 0.45 * hps + 0.45 * white + 0.5 * midf + 0.3 * lowf)
        self.theta = f32(np.clip(PARAMS['theta'] + PARAMS['theta_var'] * z, 0.08, 0.97))
        # fibre resistance to invasion of paper without free water (heavy
        # tailed: a few easy paths, many slow pixels -> fingering fronts)
        zr = _nz.normalize(0.6 * _nz.normalize(_nz.blur(g.standard_normal((H, W)), 1.0))
                           + 0.5 * covn + 0.4 * midf)
        self.rho = f32(np.clip(np.exp(1.3 * (zr - 0.6)), 0.02, 1.0))
        # ---- granulation sites: 1 = valley (pigment settles), 0 = peak
        floc = _nz.normalize(_nz.blur(g.standard_normal((H, W)), 0.8))
        hps = _nz.normalize(_nz.blur(hn, 1.0) - _nz.blur(hn, 6.0))
        gz = _nz.normalize(-1.0 * hps - 0.3 * _nz.normalize(hn) + 0.7 * floc)
        self.gran = f32(1.0 / (1.0 + np.exp(-1.4 * gz)))
        # ---- flow potential from paper relief
        self.hp = f32(PARAMS['kappa'] * hgt)
        self.noise = f32(np.clip(0.5 + 0.18 * z, 0, 1))  # generic paper-tied noise 0..1
        en = _nz.fbm((H, W), scale=7.0, octaves=3, seed=g) + 0.6 * _nz.normalize(_nz.blur(hn, 1.5))
        self.edge_noise = f32(np.clip(_nz.normalize(en), -2.5, 2.5))   # smooth, ~N(0,1)
        self.key = (H, W, canvas.seed, id(canvas.h_ground))


def _paper(canvas):
    p = canvas.__dict__.get('_wc_paper')
    if p is None or p.key != (canvas.H, canvas.W, canvas.seed, id(canvas.h_ground)):
        p = _Paper(canvas)
        canvas.__dict__['_wc_paper'] = p
    return p


def _extra(wc, canvas):
    """Simulation-owned arrays stored on the WatercolorState (so that
    ``Canvas.get_state`` checkpoints them)."""
    H, W = canvas.H, canvas.W
    for name in ('bond', 'salt'):
        a = getattr(wc, name, None)
        if not isinstance(a, np.ndarray) or a.shape != (H, W):
            setattr(wc, name, np.zeros((H, W), dtype=np.float32))
    return wc


def _tilt_of(canvas):
    t = canvas.__dict__.get('_wc_tilt')
    return (0.0, 0.0) if t is None else t


# ==========================================================================
# Numba kernels
# ==========================================================================
_TS = 16          # activity tile size (px)
_MODE = [0]       # 0: parallel kernels, 1: serial kernels (see _pick_mode)


class _Kernel:
    """A kernel compiled twice: parallel (prange) and serial.  Small regions
    -- and large ones on a busy machine, where OpenMP barriers stall --
    run faster serially; :func:`_pick_mode` chooses at run time."""

    def __init__(self, f):
        self.par = njit(parallel=True, cache=True, fastmath=True)(f)
        g = types.FunctionType(f.__code__, f.__globals__, f.__name__ + '_serial',
                               f.__defaults__, f.__closure__)
        g.__qualname__ = f.__qualname__ + '_serial'
        g.__module__ = f.__module__
        self.ser = njit(parallel=False, cache=True, fastmath=True)(g)
        self.__doc__ = f.__doc__
        self.__name__ = f.__name__

    def __call__(self, *args):
        return (self.ser if _MODE[0] else self.par)(*args)


def _kernel(f):
    return _Kernel(f)


_TUNE = {'par': None, 'ser': None, 'n': 0}
_SERIAL_BELOW = 90_000      # cells: always serial below this


def _pick_mode(cells):
    """Choose parallel / serial kernels from measured cost per cell."""
    import numba
    if cells < _SERIAL_BELOW or numba.get_num_threads() <= 1:
        return 1
    p, s = _TUNE['par'], _TUNE['ser']
    _TUNE['n'] += 1
    if p is None:
        return 0
    if s is None:
        return 1
    best = 0 if p <= s else 1
    if _TUNE['n'] % 10 == 0:          # keep re-checking the other option
        return 1 - best
    return best


def _record_cost(mode, seconds, work):
    if work <= 0:
        return
    key = 'ser' if mode else 'par'
    c = seconds / work
    old = _TUNE[key]
    _TUNE[key] = c if old is None else 0.6 * old + 0.4 * c


@njit(cache=True, fastmath=True, inline='always')
def _smooth01(t):
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return t * t * (3.0 - 2.0 * t)


@njit(cache=True, fastmath=True, inline='always')
def _gate(h_src, s_dst, th_dst, rho_dst, mask_dst, prm):
    """How readily free water may enter a pixel that has none: damp paper
    above its (fibre dependent) threshold, or water deep enough to overtop
    the dry paper grain."""
    if mask_dst >= 0.5:
        return 0.0
    # invasion speed grows with the margin above the local threshold: paths
    # of easily wetted fibres race ahead -> irregular, cauliflower fronts
    w = (s_dst - th_dst) / prm[_I_WW]
    if w <= 0.0:
        w = 0.0
    elif w >= 1.0:
        w = 1.0
    else:
        w = w * w
    w *= rho_dst
    hadv = prm[_I_HADV] * (1.0 + 1.6 * (th_dst - prm[_I_THETA]))
    a = _smooth01((h_src - hadv) / prm[_I_ADVW])
    if a > w:
        w = a
    return w * (1.0 - mask_dst)


@njit(cache=True, fastmath=True, inline='always')
def _face(f, ha, hb, Ha, Hb, C, sa, sb, tha, thb, ra, rb, ma, mb, prm, G, damp):
    """New raw virtual-pipe flux a -> b over one face."""
    hmin = prm[_I_HMIN]
    if (ha <= hmin and hb <= hmin) or ma >= 0.5 or mb >= 0.5:
        return 0.0
    hav = 0.5 * (ha + hb)
    mob = hav / (hav + prm[_I_HV])           # thin films are viscous
    f = (f * damp + G * C * (Ha - Hb)) * mob
    hwet = prm[_I_HWET]
    if f > 0.0:
        if ha <= hmin:
            return 0.0
        if hb <= hwet:
            f *= _gate(ha, sb, thb, rb, mb, prm)
    elif f < 0.0:
        if hb <= hmin:
            return 0.0
        if ha <= hwet:
            f *= _gate(hb, sa, tha, ra, ma, prm)
    return f


@njit(cache=True, fastmath=True, inline='always')
def _tile_on(tiles, y, x):
    return tiles[y // _TS, x // _TS]


@_kernel
def _k_flow1(h, s, hp, cxf, cyf, th, rho, mask, u, v, ur, vr, r, prm, tx, ty, wall,
             use_tiles, tiles, rowtile):
    """Flow, launch 1: raw pipe fluxes (ur, vr) from the potential
    h + paper relief + tilt, and each pixel's outflow limiter ``r`` (no
    pixel gives away more than 60 % of its water per sub-step).

    Only faces touching an active tile are updated when ``use_tiles``.
    Records the peak |flux| per (row, tile column) in ``rowtile``."""
    n, w = h.shape
    G = prm[_I_G]
    damp = 1.0 - prm[_I_FR]
    for y in prange(n):
        for t in range(rowtile.shape[1]):
            rowtile[y, t] = 0.0
        if wall:
            u[y, 0] = 0.0
            u[y, w] = 0.0
            if y == 0:
                for x in range(w):
                    v[0, x] = 0.0
            if y == n - 1:
                for x in range(w):
                    v[n, x] = 0.0
        ur[y, 0] = u[y, 0]
        ur[y, w] = u[y, w]
        if y == 0:
            for x in range(w):
                vr[0, x] = v[0, x]
        # x faces of this row
        for x in range(w - 1):
            if use_tiles and not (_tile_on(tiles, y, x) or _tile_on(tiles, y, x + 1)):
                ur[y, x + 1] = u[y, x + 1]
                continue
            f = _face(u[y, x + 1], h[y, x], h[y, x + 1], h[y, x] + hp[y, x] + tx,
                      h[y, x + 1] + hp[y, x + 1], cxf[y, x], s[y, x], s[y, x + 1],
                      th[y, x], th[y, x + 1], rho[y, x], rho[y, x + 1], mask[y, x], mask[y, x + 1],
                      prm, G, damp)
            ur[y, x + 1] = f
            a = abs(f)
            t = x // _TS
            if a > rowtile[y, t]:
                rowtile[y, t] = a
        # y faces: bottom face owned by this row
        if y < n - 1:
            for x in range(w):
                if use_tiles and not (_tile_on(tiles, y, x) or _tile_on(tiles, y + 1, x)):
                    vr[y + 1, x] = v[y + 1, x]
                    continue
                fb = _face(v[y + 1, x], h[y, x], h[y + 1, x], h[y, x] + hp[y, x] + ty,
                           h[y + 1, x] + hp[y + 1, x], cyf[y, x], s[y, x], s[y + 1, x],
                           th[y, x], th[y + 1, x], rho[y, x], rho[y + 1, x], mask[y, x], mask[y + 1, x],
                           prm, G, damp)
                vr[y + 1, x] = fb
                a = abs(fb)
                t = x // _TS
                if a > rowtile[y, t]:
                    rowtile[y, t] = a
        else:
            for x in range(w):
                vr[n, x] = v[n, x]
    # outflow limiter: no pixel gives away more than 60 % of its water
    for y in prange(n):
        for x in range(w):
            out = 0.0
            f = ur[y, x + 1]
            if f > 0.0:
                out += f
            f = ur[y, x]
            if f < 0.0:
                out -= f
            f = vr[y + 1, x]
            if f > 0.0:
                out += f
            f = vr[y, x]
            if f < 0.0:
                out -= f
            avail = 0.6 * h[y, x]
            if out > avail and out > 0.0:
                r[y, x] = avail / out
            else:
                r[y, x] = 1.0


@_kernel
def _k_flow2(h, u, v, ur, vr, r, ax, ay, use_tiles, tiles):
    """Flow, launch 2: limit the raw fluxes, store them (momentum), add the
    moved volumes to the pigment accumulators and move the water.  Outer
    faces of the views are never applied (walls, or owned by an enclosing
    region)."""
    n, w = h.shape
    for y in prange(n):
        for x in range(w):
            d = 0.0
            on = (not use_tiles) or _tile_on(tiles, y, x)
            if x > 0 and (on or _tile_on(tiles, y, x - 1)):
                f = ur[y, x]
                f *= r[y, x - 1] if f > 0.0 else r[y, x]
                d += f
            if x < w - 1 and (on or _tile_on(tiles, y, x + 1)):
                f = ur[y, x + 1]
                f *= r[y, x] if f > 0.0 else r[y, x + 1]
                if -1e-8 < f < 1e-8:
                    f = 0.0
                u[y, x + 1] = f
                ax[y, x + 1] += f
                d -= f
            if y > 0 and (on or _tile_on(tiles, y - 1, x)):
                f = vr[y, x]
                f *= r[y - 1, x] if f > 0.0 else r[y, x]
                d += f
            if y < n - 1 and (on or _tile_on(tiles, y + 1, x)):
                f = vr[y + 1, x]
                f *= r[y, x] if f > 0.0 else r[y + 1, x]
                if -1e-8 < f < 1e-8:
                    f = 0.0
                v[y + 1, x] = f
                ay[y + 1, x] += f
                d -= f
            if d != 0.0:
                hv = h[y, x] + d
                h[y, x] = hv if hv > 0.0 else 0.0


@_kernel
def _k_outflow(h, ax, ay, den):
    """den = water before the accumulated outflows left (h + outflow)."""
    n, w = h.shape
    for y in prange(n):
        for x in range(w):
            o = 0.0
            if x < w - 1:
                f = ax[y, x + 1]
                if f > 0.0:
                    o += f
            if x > 0:
                f = ax[y, x]
                if f < 0.0:
                    o -= f
            if y < n - 1:
                f = ay[y + 1, x]
                if f > 0.0:
                    o += f
            if y > 0:
                f = ay[y, x]
                if f < 0.0:
                    o -= f
            den[y, x] = h[y, x] + o


@njit(cache=True, fastmath=True, inline='always')
def _load(buf, K, S, m, g, st, y, x, nb):
    for b in range(nb):
        buf[b] = K[y, x, b]
        buf[nb + b] = S[y, x, b]
    buf[2 * nb] = m[y, x]
    buf[2 * nb + 1] = g[y, x]
    buf[2 * nb + 2] = st[y, x]


@njit(cache=True, fastmath=True, inline='always')
def _link_terms(F, ha, hb, da, db, ma, mb, sa, sb, C, cmix, csalt, hwet, salt_on):
    """Pigment-carrying fractions over one link: upwind advection of the
    accumulated volume F, diffusion (mixing) and salt push.  ``da, db`` are
    the pre-transfer water volumes.  Returns (fa, fb), (0, 0) if negligible
    (the volume then stays accumulated for a later pass)."""
    if C <= 0.0:  # wax seals this face, including pigment pushed by brine
        return 0.0, 0.0
    fa = 0.0
    fb = 0.0
    # particles in a thin film drag on the fibres and drop out of the flow
    if F > 0.0:
        if da > 1e-9:
            fa = F / da * (ha / (ha + 0.05))
    elif F < 0.0:
        if db > 1e-9:
            fb = -F / db * (hb / (hb + 0.05))
    if ha > hwet and hb > hwet:
        if cmix > 0.0:
            ca = ma / ha
            cb = mb / hb
            dc = ca - cb
            tol = 0.015 * (ca + cb) + 1e-5
            if dc > tol or dc < -tol:
                q = cmix * C
                if q > 0.06:
                    q = 0.06
                Q = q * (ha if ha < hb else hb)
                fa += Q / da
                fb += Q / db
        if salt_on:
            d = csalt * (sa - sb)
            if d > 0.035:
                d = 0.035
            if d < -0.035:
                d = -0.035
            if d > 0.0:
                fa += d * ha / da
            else:
                fb -= d * hb / db
    if fa < 2.5e-3 and fb < 2.5e-3:
        return 0.0, 0.0
    return fa, fb


@_kernel
def _k_xpass(h, den, K, S, m, g, st, ax, salt, cxf, cmix, csalt, hwet, salt_on):
    """Move pigment over x faces (accumulated volumes ``ax``, consumed).

    Rows are independent; along a row the pre-update pigment of the current
    pixel is kept so every transfer uses old values (exact upwinding)."""
    n, w = h.shape
    nb = K.shape[2]
    nc = 2 * nb + 3
    for y in prange(n):
        A = np.empty(nc)
        B = np.empty(nc)
        have = False
        for x in range(w - 1):
            F = ax[y, x + 1]
            ha = h[y, x]
            hb = h[y, x + 1]
            if F == 0.0 and (ha <= hwet or hb <= hwet):
                have = False
                continue
            ma = A[2 * nb] if have else m[y, x]
            fa, fb = _link_terms(F, ha, hb, den[y, x], den[y, x + 1], ma, m[y, x + 1],
                                 salt[y, x], salt[y, x + 1], cxf[y, x], cmix, csalt, hwet, salt_on)
            if fa == 0.0 and fb == 0.0:
                have = False
                continue
            ax[y, x + 1] = 0.0
            if not have:
                _load(A, K, S, m, g, st, y, x, nb)
            _load(B, K, S, m, g, st, y, x + 1, nb)
            for b in range(nb):
                t = fa * A[b] - fb * B[b]
                K[y, x, b] -= t
                K[y, x + 1, b] += t
                t = fa * A[nb + b] - fb * B[nb + b]
                S[y, x, b] -= t
                S[y, x + 1, b] += t
            t = fa * A[2 * nb] - fb * B[2 * nb]
            m[y, x] -= t
            m[y, x + 1] += t
            t = fa * A[2 * nb + 1] - fb * B[2 * nb + 1]
            g[y, x] -= t
            g[y, x + 1] += t
            t = fa * A[2 * nb + 2] - fb * B[2 * nb + 2]
            st[y, x] -= t
            st[y, x + 1] += t
            for c in range(nc):
                A[c] = B[c]
            have = True


@_kernel
def _k_ypass(h, den, K, S, m, g, st, ay, salt, cyf, cmix, csalt, hwet, salt_on):
    """Move pigment over y faces (column blocks run in parallel)."""
    n, w = h.shape
    nb = K.shape[2]
    nc = 2 * nb + 3
    BW = 16
    nblk = (w + BW - 1) // BW
    for bi in prange(nblk):
        xa0 = bi * BW
        xb0 = min(w, xa0 + BW)
        A = np.empty((BW, nc))
        B = np.empty(nc)
        have = np.zeros(BW, dtype=np.bool_)
        for y in range(n - 1):
            y1 = y + 1
            for x in range(xa0, xb0):
                k = x - xa0
                F = ay[y1, x]
                ha = h[y, x]
                hb = h[y1, x]
                if F == 0.0 and (ha <= hwet or hb <= hwet):
                    have[k] = False
                    continue
                ma = A[k, 2 * nb] if have[k] else m[y, x]
                fa, fb = _link_terms(F, ha, hb, den[y, x], den[y1, x], ma, m[y1, x],
                                     salt[y, x], salt[y1, x], cyf[y, x], cmix, csalt, hwet, salt_on)
                if fa == 0.0 and fb == 0.0:
                    have[k] = False
                    continue
                ay[y1, x] = 0.0
                if not have[k]:
                    for b in range(nb):
                        A[k, b] = K[y, x, b]
                        A[k, nb + b] = S[y, x, b]
                    A[k, 2 * nb] = m[y, x]
                    A[k, 2 * nb + 1] = g[y, x]
                    A[k, 2 * nb + 2] = st[y, x]
                _load(B, K, S, m, g, st, y1, x, nb)
                for b in range(nb):
                    t = fa * A[k, b] - fb * B[b]
                    K[y, x, b] -= t
                    K[y1, x, b] += t
                    t = fa * A[k, nb + b] - fb * B[nb + b]
                    S[y, x, b] -= t
                    S[y1, x, b] += t
                t = fa * A[k, 2 * nb] - fb * B[2 * nb]
                m[y, x] -= t
                m[y1, x] += t
                t = fa * A[k, 2 * nb + 1] - fb * B[2 * nb + 1]
                g[y, x] -= t
                g[y1, x] += t
                t = fa * A[k, 2 * nb + 2] - fb * B[2 * nb + 2]
                st[y, x] -= t
                st[y1, x] += t
                for c in range(nc):
                    A[k, c] = B[c]
                have[k] = True


@_kernel
def _k_edge(h, mask, hwet, R, tmp, E):
    """E = 1 - box_blur(wet mask, R): how close a pixel is to a dry boundary."""
    n, w = h.shape
    inv = 1.0 / (2 * R + 1)
    for y in prange(n):
        acc = 0.0
        for x in range(-R, R + 1):
            xx = min(max(x, 0), w - 1)
            if h[y, xx] > hwet and mask[y, xx] < 0.5:
                acc += 1.0
        for x in range(w):
            tmp[y, x] = acc * inv
            xo = min(max(x - R, 0), w - 1)
            xi = min(max(x + R + 1, 0), w - 1)
            if h[y, xo] > hwet and mask[y, xo] < 0.5:
                acc -= 1.0
            if h[y, xi] > hwet and mask[y, xi] < 0.5:
                acc += 1.0
    BW = 32
    nblk = (w + BW - 1) // BW
    for bi in prange(nblk):
        xa0 = bi * BW
        xb0 = min(w, xa0 + BW)
        acc = np.zeros(BW)
        for x in range(xa0, xb0):
            a = 0.0
            for yy in range(-R, R + 1):
                a += tmp[min(max(yy, 0), n - 1), x]
            acc[x - xa0] = a
        for y in range(n):
            for x in range(xa0, xb0):
                k = x - xa0
                E[y, x] = 1.0 - acc[k] * inv
                yo = min(max(y - R, 0), n - 1)
                yi = min(max(y + R + 1, 0), n - 1)
                acc[k] += tmp[yi, x] - tmp[yo, x]


@njit(cache=True, fastmath=True, inline='always')
def _capw(D, c, si, sj):
    """Capillary exchange over one link (conductivity ~ saturation^2,
    weight capped for unconditional stability)."""
    sa = 0.5 * (si + sj)
    wgt = D * c * sa * sa * 2.0
    if wgt > 0.12:
        wgt = 0.12
    return wgt * (sj - si)


@_kernel
def _k_local(h, s, s2, K, S, m, g, st, dK, dS, dm, dg, dst, bond, salt, E, gr, mask,
             cxc, cyc, u, v, evm, prm, dt, Dcap, pdt):
    """Per-pixel processes: absorption, evaporation, capillary wicking (reads
    ``s``, writes ``s2``), salt; and -- when ``pdt > 0`` (the time since the
    last exchange) -- adsorption / desorption with granulation and staining
    and the bonding of dried deposits."""
    n, w = h.shape
    nb = K.shape[2]
    absr = prm[_I_ABS]
    cap = prm[_I_CAP]
    ev = prm[_I_EV]
    ke = prm[_I_EDGE]
    evd = prm[_I_EVD]
    kdep = prm[_I_DEP]
    h0 = prm[_I_H0]
    kdd = prm[_I_DEPD]
    kdes = prm[_I_DES]
    bstr = prm[_I_BOND]
    tb = prm[_I_BT]
    soft = prm[_I_SOFT]
    ag = prm[_I_GRAN]
    hwet = prm[_I_HWET]
    sab = prm[_I_SABS]
    kedge_dep = prm[_I_EDEP]
    kshear = prm[_I_SHEAR]
    for y in prange(n):
        for x in range(w):
            sv = s[y, x]
            # -- capillary wicking through the fibre network
            if mask[y, x] < 0.5:
                acc = 0.0
                if x > 0 and mask[y, x - 1] < 0.5:
                    acc += _capw(Dcap, cxc[y, x - 1], sv, s[y, x - 1])
                if x < w - 1 and mask[y, x + 1] < 0.5:
                    acc += _capw(Dcap, cxc[y, x], sv, s[y, x + 1])
                if y > 0 and mask[y - 1, x] < 0.5:
                    acc += _capw(Dcap, cyc[y - 1, x], sv, s[y - 1, x])
                if y < n - 1 and mask[y + 1, x] < 0.5:
                    acc += _capw(Dcap, cyc[y, x], sv, s[y + 1, x])
                sv += acc
            hv = h[y, x]
            mv = m[y, x]
            if hv <= 0.0 and sv <= 0.0 and mv <= 0.0:
                s2[y, x] = 0.0
                continue
            em = evm[y, x]
            ed = E[y, x]
            # -- absorption into the paper
            if hv > 0.0 and sv < 1.0:
                a = dt * absr * (1.0 - sv) * cap
                if a > hv:
                    a = hv
                hv -= a
                sv += a / cap
                if sv > 1.0:
                    sv = 1.0
            # -- evaporation (faster near the boundary of the wet area)
            if hv > 0.0:
                hv -= dt * ev * em * (1.0 + ke * ed)
                if hv < 2e-4:
                    if hv > 0.0:
                        sv += hv / cap
                        if sv > 1.0:
                            sv = 1.0
                    hv = 0.0
            else:
                sv -= dt * evd * em * (1.0 + 0.4 * ke * ed)
                if sv < 0.004:
                    sv = 0.0
            # -- salt soaks up free water
            sl = salt[y, x]
            if sl > 0.0 and hv > 0.0:
                take = dt * sab * sl
                if take > hv:
                    take = hv
                hv -= take
            if pdt <= 0.0:
                h[y, x] = hv
                s2[y, x] = sv
                continue
            # -- pigment adsorption / desorption
            dmv = dm[y, x]
            gam = 0.0
            if mv > 1e-9:
                gam = g[y, x] / mv
                if gam > 1.0:
                    gam = 1.0
            gamd = 0.0
            stnd = 0.0
            if dmv > 1e-9:
                gamd = dg[y, x] / dmv
                stnd = dst[y, x] / dmv
                if gamd > 1.0:
                    gamd = 1.0
                if stnd > 1.0:
                    stnd = 1.0
            val = 2.0 * gr[y, x] - 1.0          # +1 valley, -1 peak
            fdes = 0.0
            if hv > hwet:
                fdep = pdt * kdep * (1.0 + ag * gam * val) * (1.0 + kedge_dep * ed) / (hv + h0)
                if dmv > 1e-9:
                    wf = hv / 0.08
                    if wf > 1.0:
                        wf = 1.0
                    # flowing water lifts loosely settled pigment (shear)
                    spd = (abs(u[y, x]) + abs(u[y, x + 1]) + abs(v[y, x]) + abs(v[y + 1, x])) \
                        / (hv + 0.05)
                    if spd > 2.0:
                        spd = 2.0
                    fdes = pdt * (kdes + kshear * spd) * (1.0 - 0.92 * stnd) \
                        * (1.0 - bstr * bond[y, x]) * (1.0 - ag * gamd * val) * wf
                if sl > 0.0:
                    fdep /= (1.0 + 8.0 * sl)
            elif sv > 0.03:
                fdep = pdt * kdd * (1.0 + ag * gam * val) * (1.25 - sv)
            else:
                fdep = 1.0
            if fdep > 1.0:
                fdep = 1.0
            if fdep < 0.0:
                fdep = 0.0
            if fdes > 0.5:
                fdes = 0.5
            if fdes < 0.0:
                fdes = 0.0
            if (fdep > 0.0 and mv > 0.0) or fdes > 0.0:
                for b in range(nb):
                    t = K[y, x, b] * fdep - dK[y, x, b] * fdes
                    K[y, x, b] -= t
                    dK[y, x, b] += t
                    t = S[y, x, b] * fdep - dS[y, x, b] * fdes
                    S[y, x, b] -= t
                    dS[y, x, b] += t
                md = mv * fdep
                mu = dmv * fdes
                t = g[y, x] * fdep - dg[y, x] * fdes
                g[y, x] -= t
                dg[y, x] += t
                t = st[y, x] * fdep - dst[y, x] * fdes
                st[y, x] -= t
                dst[y, x] += t
                m[y, x] = mv - md + mu
                ndm = dmv + md - mu
                dm[y, x] = ndm
                if ndm > 1e-9:
                    # fresh deposit is unbonded: mass-weighted bond
                    bond[y, x] = bond[y, x] * (dmv - mu) / ndm
            # -- bonding of dried deposits / slow loosening under water
            if hv <= hwet and sv < 0.06:
                bond[y, x] += min(pdt / tb, 1.0) * (1.0 - bond[y, x])
            elif hv > hwet:
                bond[y, x] -= pdt * soft * bond[y, x]
            h[y, x] = hv
            s2[y, x] = sv


@_kernel
def _k_salt(salt, s2, h, s, cxc, cyc, D, hwet):
    """Brine spreads along fibres through wet paper (spiky, star-like)."""
    n, w = salt.shape
    for y in prange(n):
        for x in range(w):
            si = salt[y, x]
            wi = h[y, x] > hwet or s[y, x] > 0.35
            if not wi:
                s2[y, x] = si
                continue
            acc = 0.0
            if x > 0 and (h[y, x - 1] > hwet or s[y, x - 1] > 0.35):
                c = min(D * cxc[y, x - 1] ** 2, 0.12)
                acc += c * (salt[y, x - 1] - si)
            if x < w - 1 and (h[y, x + 1] > hwet or s[y, x + 1] > 0.35):
                c = min(D * cxc[y, x] ** 2, 0.12)
                acc += c * (salt[y, x + 1] - si)
            if y > 0 and (h[y - 1, x] > hwet or s[y - 1, x] > 0.35):
                c = min(D * cyc[y - 1, x] ** 2, 0.12)
                acc += c * (salt[y - 1, x] - si)
            if y < n - 1 and (h[y + 1, x] > hwet or s[y + 1, x] > 0.35):
                c = min(D * cyc[y, x] ** 2, 0.12)
                acc += c * (salt[y + 1, x] - si)
            s2[y, x] = si + acc


@_kernel
def _k_active(h, s, m, rowx0, rowx1):
    """Per-row extent of pixels that still hold water or suspended pigment."""
    n, w = h.shape
    for y in prange(n):
        a = w
        b = -1
        for x in range(w):
            if h[y, x] > 1e-6 or s[y, x] > 2e-3 or m[y, x] > 1e-9:
                if x < a:
                    a = x
                b = x
        rowx0[y] = a
        rowx1[y] = b


@_kernel
def _k_settle_all(K, S, m, g, st, dK, dS, dm, dg, dst, h, s):
    """Deposit every bit of suspended pigment and remove all water."""
    n, w = h.shape
    nb = K.shape[2]
    for y in prange(n):
        for x in range(w):
            if m[y, x] != 0.0:
                dm[y, x] += m[y, x]
                dg[y, x] += g[y, x]
                dst[y, x] += st[y, x]
                m[y, x] = 0.0
                g[y, x] = 0.0
                st[y, x] = 0.0
            for b in range(nb):
                if K[y, x, b] != 0.0 or S[y, x, b] != 0.0:
                    dK[y, x, b] += K[y, x, b]
                    dS[y, x, b] += S[y, x, b]
                    K[y, x, b] = 0.0
                    S[y, x, b] = 0.0
            h[y, x] = 0.0
            s[y, x] = 0.0


# ==========================================================================
# Driver
# ==========================================================================
_PAD = 14
_TRACE = None      # set to a list to record per-step diagnostics


def _region(wc, H, W, pad=_PAD):
    y0, y1, x0, x1 = wc.bbox
    return max(0, y0 - pad), min(H, y1 + pad), max(0, x0 - pad), min(W, x1 + pad)


def _update_bbox(wc, canvas, y0, y1, x0, x1):
    n = y1 - y0
    rx0 = np.empty(n, np.int64)
    rx1 = np.empty(n, np.int64)
    _k_active(wc.water[y0:y1, x0:x1], wc.damp[y0:y1, x0:x1], wc.m[y0:y1, x0:x1], rx0, rx1)
    rows = np.nonzero(rx1 >= 0)[0]
    if len(rows) == 0:
        wc.bbox = None
        return False
    ny0, ny1 = y0 + rows[0], y0 + rows[-1] + 1
    nx0, nx1 = x0 + int(rx0[rows].min()), x0 + int(rx1[rows].max()) + 1
    H, W = canvas.H, canvas.W
    wc.bbox = [int(max(0, ny0 - 2)), int(min(H, ny1 + 2)), int(max(0, nx0 - 2)), int(min(W, nx1 + 2))]
    return True


def _scratch(canvas):
    """Canvas-sized scratch buffers (raw fluxes, pigment accumulators)."""
    sc = canvas.__dict__.get('_wc_scratch')
    if sc is None or sc['ax'].shape != (canvas.H, canvas.W + 1):
        H, W = canvas.H, canvas.W
        sc = dict(ax=np.zeros((H, W + 1), np.float32), ay=np.zeros((H + 1, W), np.float32),
                  ur=np.zeros((H, W + 1), np.float32), vr=np.zeros((H + 1, W), np.float32))
        canvas.__dict__['_wc_scratch'] = sc
    return sc


def _finish(canvas, y0, y1, x0, x1):
    wc = canvas.wc
    sl = (slice(y0, y1), slice(x0, x1))
    sc = _scratch(canvas)
    sc['ax'][y0:y1, x0:x1 + 1] = 0
    sc['ay'][y0:y1 + 1, x0:x1] = 0
    _k_settle_all(wc.K[sl], wc.S[sl], wc.m[sl], wc.g[sl], wc.st[sl], wc.dK[sl], wc.dS[sl],
                  wc.dm[sl], wc.dg[sl], wc.dst[sl], wc.water[sl], wc.damp[sl])
    wc.u[y0:y1, x0:x1 + 1] = 0
    wc.v[y0:y1 + 1, x0:x1] = 0


class _Views:
    """Views of all simulation arrays on a sub-rectangle of the canvas."""

    def __init__(self, canvas, paper, y0, y1, x0, x1):
        wc = canvas.wc
        sl = (slice(y0, y1), slice(x0, x1))
        self.shape = n, w = (y1 - y0, x1 - x0)
        self.h, self.s = wc.water[sl], wc.damp[sl]
        self.K, self.S, self.m, self.g, self.st = wc.K[sl], wc.S[sl], wc.m[sl], wc.g[sl], wc.st[sl]
        self.dK, self.dS = wc.dK[sl], wc.dS[sl]
        self.dm, self.dg, self.dst = wc.dm[sl], wc.dg[sl], wc.dst[sl]
        self.bond, self.salt = wc.bond[sl], wc.salt[sl]
        self.mask = canvas.watercolor_mask(sl)
        self.u = wc.u[y0:y1, x0:x1 + 1]
        self.v = wc.v[y0:y1 + 1, x0:x1]
        sc = _scratch(canvas)
        self.ax, self.ur = sc['ax'][y0:y1, x0:x1 + 1], sc['ur'][y0:y1, x0:x1 + 1]
        self.ay, self.vr = sc['ay'][y0:y1 + 1, x0:x1], sc['vr'][y0:y1 + 1, x0:x1]
        self.hp, self.th, self.gr = paper.hp[sl], paper.theta[sl], paper.gran[sl]
        self.rho = paper.rho[sl]
        self.cxf, self.cyf = paper.cx_flow[sl], paper.cy_flow[sl]
        self.cxc, self.cyc = paper.cx_cap[sl], paper.cy_cap[sl]
        if canvas.wax is not None:
            # Partial wax narrows the conducting paper fibres. Use the
            # less permeable of the two pixels sharing each face so that
            # exchanging water or pigment across it is symmetric.
            wax = canvas.wax[sl]
            open_paper = np.where(wax >= 0.5, 0.0, 1.0 - wax)
            xface = np.ones_like(wax)
            yface = np.ones_like(wax)
            xface[:, :-1] = np.minimum(open_paper[:, :-1], open_paper[:, 1:])
            yface[:-1, :] = np.minimum(open_paper[:-1, :], open_paper[1:, :])
            self.cxf, self.cyf = self.cxf * xface, self.cyf * yface
            self.cxc, self.cyc = self.cxc * xface, self.cyc * yface
        self.r = np.empty((n, w), np.float32)
        self.den = np.empty((n, w), np.float32)
        self.s2 = np.empty((n, w), np.float32)
        self.E = np.zeros((n, w), np.float32)
        self.tmp = np.empty((n, w), np.float32)
        self.nty, self.ntx = (n + _TS - 1) // _TS, (w + _TS - 1) // _TS
        self.rowtile = np.zeros((n, self.ntx), np.float32)
        self.tiles = np.ones((self.nty, self.ntx), np.bool_)
        self.tile_act = np.zeros((self.nty, self.ntx), np.float32)


def _flow(V, prm, tx, ty, use_tiles=False):
    """One virtual-pipe sub-step (water only; pigment volumes accumulate)."""
    _k_flow1(V.h, V.s, V.hp, V.cxf, V.cyf, V.th, V.rho, V.mask, V.u, V.v, V.ur, V.vr, V.r, prm,
             tx, ty, True, use_tiles, V.tiles, V.rowtile)
    _k_flow2(V.h, V.u, V.v, V.ur, V.vr, V.r, V.ax, V.ay, use_tiles, V.tiles)


def _activity(V):
    """Peak flux per tile from the last (full) sub-step."""
    n = V.shape[0]
    pad = V.nty * _TS - n
    rt = V.rowtile if pad == 0 else np.vstack([V.rowtile, np.zeros((pad, V.ntx), np.float32)])
    V.tile_act[:] = rt.reshape(V.nty, _TS, V.ntx).max(axis=1)
    return float(V.tile_act.max()) if V.tile_act.size else 0.0


def _pigment(V, cmix, salt_on, parity):
    """Move suspended pigment with the accumulated water volumes (+ mixing)."""
    P = PARAMS
    hwet = P['h_wet']
    cs = P['salt_push']
    _k_outflow(V.h, V.ax, V.ay, V.den)
    if parity % 2 == 0:
        _k_xpass(V.h, V.den, V.K, V.S, V.m, V.g, V.st, V.ax, V.salt, V.cxf, cmix, cs, hwet, salt_on)
        _k_ypass(V.h, V.den, V.K, V.S, V.m, V.g, V.st, V.ay, V.salt, V.cyf, cmix, cs, hwet, salt_on)
    else:
        _k_ypass(V.h, V.den, V.K, V.S, V.m, V.g, V.st, V.ay, V.salt, V.cyf, cmix, cs, hwet, salt_on)
        _k_xpass(V.h, V.den, V.K, V.S, V.m, V.g, V.st, V.ax, V.salt, V.cxf, cmix, cs, hwet, salt_on)


def _run(canvas, seconds, evap=1.0, fast=False):
    """Advance the watercolour layer by ``seconds`` of simulated time."""
    wc = canvas.wc
    if wc is None or wc.bbox is None or seconds <= 0:
        return
    _extra(wc, canvas)
    paper = _paper(canvas)
    prm = _param_vector(canvas)
    P = PARAMS
    H, W = canvas.H, canvas.W
    tx, ty = _tilt_of(canvas)
    hwet = P['h_wet']
    remaining = float(seconds)
    evap_map = isinstance(evap, np.ndarray)
    athr = P.get('act_tile', 0.006)
    busy = True
    while remaining > 1e-9 and wc.bbox is not None:
        y0, y1, x0, x1 = _region(wc, H, W)
        V = _Views(canvas, paper, y0, y1, x0, x1)
        n, w = V.shape
        if evap_map:
            evm = np.ascontiguousarray(evap[y0:y1, x0:x1], dtype=np.float32)
        else:
            evm = np.full((n, w), float(evap), dtype=np.float32)
        free = bool(V.h.max() > hwet)
        salt_on = bool(V.salt.max() > 1e-6)
        chunk = min(remaining, (10.0 if free else 30.0) * (2.0 if fast else 1.0))
        t = 0.0
        step = 0
        pdt_acc = 0.0
        while t < chunk - 1e-9:
            if free:
                dt = P['dt'] if busy else P['dt_calm']
            else:
                dt = P['dt_damp']
            if fast:
                dt *= 2.0
            dt = min(dt, chunk - t)
            n_extra = 0
            mode = _pick_mode(n * w)
            _MODE[0] = mode
            t_step = _time.perf_counter()
            pdt_acc += dt
            do_pig = True
            if free:
                _flow(V, prm, tx, ty, False)
                act = _activity(V)
                hot = V.tile_act > athr
                nhot = int(hot.sum())
                wet_tiles = max(1, int((V.tile_act > 0).sum()))
                busy = nhot > 0.3 * wet_tiles and not fast
                n_extra = int(np.clip(round(act / 0.008), 0, P['sub_max'] - 1))
                if busy:
                    n_extra = max(n_extra, int(P['sub_min']) - 1)
                if n_extra > 0 and nhot > 0:
                    # extra sub-steps only where water moves fast (tiles
                    # dilated by one so fronts can advance)
                    d = hot.copy()
                    d[1:] |= hot[:-1]; d[:-1] |= hot[1:]
                    d[:, 1:] |= hot[:, :-1]; d[:, :-1] |= hot[:, 1:]
                    V.tiles[:] = d
                    every = 1 if act > 0.05 else (2 if act > 0.015 else n_extra + 1)
                    for k in range(n_extra):
                        if k % every == 0 and k > 0:
                            _pigment(V, 0.0, salt_on, step + k)
                        _flow(V, prm, tx, ty, True)
                    V.tiles[:] = True
                else:
                    n_extra = 0
                _pigment(V, min(P['mix'] * pdt_acc / 0.1, 0.06), salt_on, step)
                if _TRACE is not None:
                    _TRACE.append((n * w, 1 + n_extra, act, nhot))
            if step % 2 == 0 or not free:
                _k_edge(V.h, V.mask, hwet, int(P['edge_radius']), V.tmp, V.E)
            Dcap = P['capillary'] * dt / 0.5
            _k_local(V.h, V.s, V.s2, V.K, V.S, V.m, V.g, V.st, V.dK, V.dS, V.dm, V.dg, V.dst,
                     V.bond, V.salt, V.E, V.gr, V.mask, V.cxc, V.cyc, V.u, V.v, evm, prm, dt, Dcap,
                     pdt_acc if do_pig else 0.0)
            if do_pig:
                pdt_acc = 0.0
            V.s[...] = V.s2
            if salt_on:
                _k_salt(V.salt, V.s2, V.h, V.s, V.cxc, V.cyc, P['salt_diffuse'] * dt, hwet)
                V.salt[...] = V.s2
            work = n * w * ((2.5 + 0.3 * n_extra) if free else 1.0)
            _record_cost(mode, _time.perf_counter() - t_step, work)
            if free and step % 3 == 2 and not (V.h.max() > hwet):
                free = False
            t += dt
            step += 1
        remaining -= chunk
        if not _update_bbox(wc, canvas, y0, y1, x0, x1):
            _finish(canvas, y0, y1, x0, x1)
            wc.bbox = None


def simulate(canvas, seconds):
    """Let the watercolour layer flow, soak in and dry for ``seconds``.

    Called by :meth:`Canvas.wait`.  Simulates only the bounding box of wet
    pixels (``wc.bbox``), shrinks it as areas dry and sets it to ``None``
    when everything is dry."""
    _run(canvas, float(seconds))


def settle(canvas):
    """Finish drying quickly but plausibly (called by :meth:`Canvas.dry`).

    Runs an accelerated simulation (stronger evaporation, longer steps) so
    edges still darken and pigment still settles into the paper, then
    deposits whatever is left."""
    wc = canvas.wc
    if wc is None:
        return
    _extra(wc, canvas)
    for _ in range(40):
        if wc.bbox is None:
            break
        _run(canvas, 30.0, evap=3.0, fast=True)
    H, W = canvas.H, canvas.W
    _finish(canvas, 0, H, 0, W)
    wc.bbox = None
    wc.salt[:] = 0
    wc.bond[:] = 1.0


# ==========================================================================
# Regions
# ==========================================================================
def rect(x0, y0, x1, y1):
    """Rectangular region (for the ``region`` argument of the helpers)."""
    def f(canvas):
        m = np.zeros(canvas.shape, np.float32)
        m[max(0, int(round(y0))):max(0, int(round(y1))), max(0, int(round(x0))):max(0, int(round(x1)))] = 1
        return m
    return f


def ellipse(cx, cy, rx, ry=None, angle=0.0):
    """Elliptical region, ``angle`` in degrees."""
    ry = rx if ry is None else ry

    def f(canvas):
        yy, xx = np.mgrid[0:canvas.H, 0:canvas.W].astype(np.float32)
        a = math.radians(angle)
        dx, dy = xx - cx, yy - cy
        u = (dx * math.cos(a) + dy * math.sin(a)) / rx
        v = (-dx * math.sin(a) + dy * math.cos(a)) / ry
        d = np.sqrt(u * u + v * v)
        return np.clip((1.0 - d) * min(rx, ry) + 0.5, 0, 1).astype(np.float32)
    return f


def polygon(points):
    """Polygon region from (x, y) points."""
    pts = [tuple(map(float, p[:2])) for p in points]

    def f(canvas):
        from PIL import Image, ImageDraw
        ss = 2
        img = Image.new('L', (canvas.W * ss, canvas.H * ss), 0)
        ImageDraw.Draw(img).polygon([(x * ss, y * ss) for x, y in pts], fill=255)
        img = img.resize((canvas.W, canvas.H), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0
    return f


def region_mask(canvas, region, soft=0.0, rough=0.0, respect_mask=True):
    """Coverage 0..1 (H, W) of ``region``.

    ``region`` may be ``None`` (whole canvas), an (H, W) boolean / float
    array, a polygon ``[(x, y), ...]``, a rectangle ``(x0, y0, x1, y1)``, or
    a callable ``f(canvas) -> array`` (see :func:`rect`, :func:`ellipse`,
    :func:`polygon`).  ``soft`` feathers the edge (px); ``rough`` makes the
    edge follow the paper texture (0..1).  Masked pixels are removed."""
    H, W = canvas.shape
    if region is None:
        m = np.ones((H, W), np.float32)
    elif callable(region):
        m = np.asarray(region(canvas), dtype=np.float32)
    else:
        arr = np.asarray(region)
        if arr.shape == (H, W):
            m = arr.astype(np.float32)
        elif arr.ndim == 1 and arr.size == 4:
            m = rect(*arr.tolist())(canvas)
        elif arr.ndim == 2 and arr.shape[1] >= 2 and arr.shape[0] >= 3:
            m = polygon(arr)(canvas)
        else:
            raise ValueError('region must be None, an (H, W) mask, a polygon, (x0, y0, x1, y1) or a callable')
    m = np.clip(m, 0, 1)
    if rough > 0 and soft <= 0:
        # edge wanders a pixel or two along the paper grain, anti-aliased,
        # without single-pixel whiskers
        p = _paper(canvas)
        ms = _nz.blur(m, 2.0)
        thr = 0.5 - rough * 0.3 * p.edge_noise
        hard = ms > thr
        from scipy import ndimage as _ndi
        hard = _ndi.binary_opening(hard, iterations=1) | (ms > 0.97)
        hard = _ndi.binary_closing(hard, iterations=1)
        m = np.where(hard, np.clip((ms - thr) * 2.5 + 0.75, 0.5, 1.0), 0.0)
        m = np.minimum(m, np.clip(_nz.blur(hard.astype(np.float32), 0.6) * 1.6, 0, 1))
    elif soft > 0:
        m = _nz.blur(m, soft)
    if respect_mask:
        m = m * (1.0 - canvas.watercolor_mask())
    return m.astype(np.float32)


def _grow(canvas, cov, pad=10):
    ys, xs = np.nonzero(cov > 1e-4)
    if len(ys) == 0:
        return False
    canvas.wc.grow_bbox(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()),
                        canvas.H, canvas.W, pad=pad)
    return True


def _paint_ks(paint):
    """K, S per unit pigment volume plus granulation / staining of ``paint``
    (a Paint or a list of (Paint, weight))."""
    from .pigments import Paint, mix
    if isinstance(paint, (list, tuple)):
        pairs = [(p, 1.0) if isinstance(p, Paint) else p for p in paint]
        paint = mix(pairs)
    return (np.asarray(paint.K, np.float32), np.asarray(paint.S, np.float32),
            float(paint.granulation), float(paint.staining))


def _add(canvas, water, pig=None, paint=None, *, origin=(0, 0)):
    """Add a water/pigment patch whose top-left canvas pixel is (x, y)."""
    wc = canvas.ensure_watercolor()
    _extra(wc, canvas)
    cov = water if pig is None else np.maximum(water, pig)
    active = cov > 1e-6
    ys = np.flatnonzero(active.any(axis=1))
    if len(ys) == 0:
        return
    xs = np.flatnonzero(active.any(axis=0))
    y0, y1, x0, x1 = ys[0], ys[-1] + 1, xs[0], xs[-1] + 1
    src = (slice(y0, y1), slice(x0, x1))
    ox, oy = origin
    dst = (slice(oy + y0, oy + y1), slice(ox + x0, ox + x1))
    wc.water[dst] += water[src].astype(np.float32, copy=False)
    if pig is not None and paint is not None:
        Kp, Sp, gr, stn = _paint_ks(paint)
        pv = pig[src].astype(np.float32, copy=False)
        wc.K[dst] += pv[..., None] * Kp
        wc.S[dst] += pv[..., None] * Sp
        wc.m[dst] += pv
        wc.g[dst] += pv * gr
        wc.st[dst] += pv * stn
    wc.grow_bbox(int(ox + x0), int(oy + y0), int(ox + x1 - 1), int(oy + y1 - 1),
                 canvas.H, canvas.W, pad=10)


def _rng(seed, canvas):
    return np.random.default_rng(canvas.rng.integers(1 << 31) if seed is None else seed)


def _gradient(canvas, gradient):
    """Evaluate a gradient spec to an (H, W) float array."""
    H, W = canvas.shape
    if gradient is None:
        return np.ones((H, W), np.float32)
    if callable(gradient):
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        return np.asarray(gradient(xx, yy), np.float32) * np.ones((H, W), np.float32)
    if isinstance(gradient, np.ndarray) and gradient.shape == (H, W):
        return gradient.astype(np.float32)
    g = list(gradient)
    (xa, ya), (xb, yb) = g[0], g[1]
    c0, c1 = (g[2] if len(g) > 2 else (1.0, 0.0))
    gamma = g[3] if len(g) > 3 else 1.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx, dy = xb - xa, yb - ya
    L2 = dx * dx + dy * dy + 1e-9
    t = np.clip(((xx - xa) * dx + (yy - ya) * dy) / L2, 0, 1) ** gamma
    return (c0 + (c1 - c0) * t).astype(np.float32)


# ==========================================================================
# Painting helpers
# ==========================================================================
def wet_paper(canvas, region=None, water=0.6, soak=0.6, evenness=0.85, seed=None):
    """Wet the paper with clear water (for wet-in-wet work).

    water    : free-water depth (0.3 = damp sheen, 0.6 = wet, 1.2 = puddle)
    soak     : how saturated the paper already is (0..1)
    evenness : 1 = perfectly even, lower = patchy brush-applied water
    """
    wc = canvas.ensure_watercolor()
    _extra(wc, canvas)
    g = _rng(seed, canvas)
    cov = region_mask(canvas, region, rough=0.4)
    var = _nz.fbm(canvas.shape, scale=canvas.W / 8, octaves=4, seed=g)
    wv = water * cov * np.clip(1.0 + (1.0 - evenness) * 1.5 * var, 0.2, 2.0)
    wc.damp[:] = np.maximum(wc.damp, (soak * cov).astype(np.float32))
    _add(canvas, wv.astype(np.float32))
    canvas._after_stroke()


def wash(canvas, region, paint, concentration=0.5, water=0.8, gradient=None,
         variation=0.08, soft=0.0, rough=0.5, seed=None):
    """Lay a flat or graded wash of ``paint`` over ``region``.

    concentration : pigment strength on the brush scale (1 = like
                    ``brush.load(paint, amount=1)``: ``0.35`` pigment volume
                    per unit water); 0.1 = pale tint, 1.5 = strong
    water         : free-water depth laid down (0.5 damp .. 1.2 juicy)
    gradient      : None (flat), ``((x0, y0), (x1, y1))`` for a graded wash
                    fading from full strength at p0 to nothing at p1,
                    ``((x0, y0), (x1, y1), (c0, c1)[, gamma])`` for custom
                    end strengths, an (H, W) array or ``f(xx, yy)``
    variation     : subtle unevenness of the brushed-on wash
    soft          : feather the region edge (px), for soft-edged washes
    rough         : how much the edge follows the paper texture
    paint         : a watercolour Paint or ``[(paint, weight), ...]``
    """
    g = _rng(seed, canvas)
    cov = region_mask(canvas, region, soft=soft, rough=rough)
    gf = np.clip(_gradient(canvas, gradient), 0, None)
    n1 = _nz.fbm(canvas.shape, scale=max(canvas.W, canvas.H) / 6, octaves=4, seed=g)
    streak = _nz.spectral_noise(canvas.shape, beta=2.2, seed=g, aniso=(1.0, 0.12))
    wv = water * cov * np.clip(1.0 + variation * (0.6 * n1 + 0.4 * streak), 0.3, 2.0)
    pig = wv * PIGMENT_PER_WATER * concentration * gf * \
        np.clip(1.0 + variation * (0.8 * n1 - 0.6 * streak), 0.2, 2.0)
    _add(canvas, wv.astype(np.float32), pig.astype(np.float32), paint)
    canvas._after_stroke()


def variegated_wash(canvas, region, stops, direction, concentration=0.6, water=0.8,
                    variation=0.12, rough=0.5, seed=None):
    """Wash whose colour changes across the region (colours mingle wet-in-wet).

    stops     : ``[(paint, position), ...]`` with positions 0..1 along
                ``direction`` (a ``((x0, y0), (x1, y1))`` pair), or
                ``[(paint, position, strength), ...]``
    """
    g = _rng(seed, canvas)
    cov = region_mask(canvas, region, rough=rough)
    (xa, ya), (xb, yb) = direction
    H, W = canvas.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx, dy = xb - xa, yb - ya
    t = np.clip(((xx - xa) * dx + (yy - ya) * dy) / (dx * dx + dy * dy + 1e-9), 0, 1)
    t = np.clip(t + 0.04 * _nz.fbm(canvas.shape, scale=W / 10, octaves=3, seed=g), 0, 1)
    pos = np.array([float(s[1]) for s in stops])
    n1 = _nz.fbm(canvas.shape, scale=max(W, H) / 6, octaves=4, seed=g)
    wv = water * cov * np.clip(1.0 + variation * n1, 0.3, 2.0)
    _add(canvas, wv.astype(np.float32))
    for i, st in enumerate(stops):
        e = np.zeros(len(stops)); e[i] = 1.0
        wgt = np.interp(t, pos, e).astype(np.float32)
        strength = float(st[2]) if len(st) > 2 else 1.0
        pig = wv * PIGMENT_PER_WATER * concentration * strength * wgt
        _add(canvas, np.zeros_like(wv, dtype=np.float32), pig.astype(np.float32), st[0])
    canvas._after_stroke()


def _blob(canvas, cx, cy, r, g, wobble=0.18):
    """Irregular disc coverage (local window) -> (y0, y1, x0, x1, cov)."""
    H, W = canvas.shape
    R = r * (1 + wobble) + 3
    x0, x1 = max(0, int(cx - R)), min(W, int(cx + R) + 2)
    y0, y1 = max(0, int(cy - R)), min(H, int(cy + R) + 2)
    if x1 <= x0 or y1 <= y0:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    ang = np.arctan2(dy, dx)
    k = g.integers(3, 7)
    ph = g.uniform(0, 2 * np.pi, 3)
    amp = g.uniform(0.3, 1.0, 3) * wobble / 1.5
    rr = r * (1 + amp[0] * np.sin(k * ang + ph[0]) + amp[1] * np.sin((k + 2) * ang + ph[1])
              + amp[2] * np.sin(2 * ang + ph[2]))
    d = np.sqrt(dx * dx + dy * dy)
    pn = _paper(canvas).noise[y0:y1, x0:x1]
    cov = np.clip((rr - d) + 0.8 * (pn - 0.5) + 0.5, 0, 1)
    return y0, y1, x0, x1, cov


def drop(canvas, x, y, paint=None, radius=12.0, concentration=1.2, water=0.7, seed=None):
    """Charge a drop of paint (or clear water if ``paint`` is None) into the
    paper at (x, y).  Into a wet wash it blooms out softly; into a damp,
    half-dry wash clear water causes a cauliflower backrun."""
    wc = canvas.ensure_watercolor()
    _extra(wc, canvas)
    g = _rng(seed, canvas)
    b = _blob(canvas, x, y, radius, g)
    if b is None:
        return
    y0, y1, x0, x1, cov = b
    cov = cov * (1.0 - canvas.watercolor_mask(np.s_[y0:y1, x0:x1]))
    water_patch = (cov * water).astype(np.float32, copy=False)
    pigment = None if paint is None else water_patch * PIGMENT_PER_WATER * concentration
    _add(canvas, water_patch, pigment, paint, origin=(x0, y0))
    canvas._after_stroke()


def spritz(canvas, region=None, density=0.004, size=(0.8, 4.0), water=0.55, paint=None,
           concentration=0.8, seed=None):
    """Spray water droplets (or, with ``paint``, spatter paint) over a region.

    density : droplets per pixel of region area
    size    : (min, max) droplet radius in px (power-law distributed)
    On a damp wash every droplet makes a tiny bloom; on dry paper clear
    droplets leave nothing, paint droplets dry with dark rims."""
    wc = canvas.ensure_watercolor()
    _extra(wc, canvas)
    g = _rng(seed, canvas)
    cov = region_mask(canvas, region)
    area = float(cov.sum())
    n = int(g.poisson(max(area * density, 0.0)))
    if n == 0:
        return
    flat = cov.ravel() / max(cov.sum(), 1e-9)
    idx = g.choice(flat.size, size=n, p=flat)
    ys, xs = np.divmod(idx, canvas.W)
    a, b = size
    u = g.uniform(0, 1, n)
    radii = a * (b / a) ** (u ** 2.2)      # many small, few big droplets
    full = np.zeros(canvas.shape, np.float32)
    for x, y, r in zip(xs + g.uniform(-.5, .5, n), ys + g.uniform(-.5, .5, n), radii):
        bl = _blob(canvas, x, y, r, g, wobble=0.12)
        if bl is None:
            continue
        y0, y1, x0, x1, c = bl
        full[y0:y1, x0:x1] = np.maximum(full[y0:y1, x0:x1], c * water * (0.8 + 0.4 * min(r / b, 1)))
    full *= (1.0 - canvas.watercolor_mask())
    if paint is None:
        _add(canvas, full)
    else:
        _add(canvas, full, full * PIGMENT_PER_WATER * concentration, paint)
    canvas._after_stroke()


def spatter(canvas, region, paint, density=0.002, size=(0.6, 3.5), concentration=1.0,
            water=0.5, seed=None):
    """Flick paint droplets from a loaded brush (see :func:`spritz`)."""
    spritz(canvas, region, density=density, size=size, water=water, paint=paint,
           concentration=concentration, seed=seed)


def salt(canvas, region=None, density=0.0012, size=(0.6, 2.0), strength=1.0, seed=None):
    """Sprinkle salt crystals onto a damp wash.

    Every crystal soaks up the surrounding moisture; the brine creeps out
    along the paper fibres and pushes pigment away, leaving pale, star-
    shaped, feathery spots with a slightly darker fringe.  The effect
    depends on the wetness under each crystal: strongest when the wash has
    just lost its shine, softer and larger on a very wet wash, nothing on
    dry paper.  Remaining brine (``wc.salt``) keeps the spot from refilling
    and keeps pushing pigment while there is free water.

    density : crystals per pixel of region area;  size : crystal radius (px)
    """
    wc = canvas.ensure_watercolor()
    _extra(wc, canvas)
    g = _rng(seed, canvas)
    cov = region_mask(canvas, region)
    area = float(cov.sum())
    n = int(g.poisson(max(area * density, 0.0)))
    if n == 0:
        return
    flat = cov.ravel() / max(cov.sum(), 1e-9)
    idx = g.choice(flat.size, size=n, p=flat)
    ys, xs = np.divmod(idx, canvas.W)
    H, W = canvas.shape
    a, b = size
    pn = _paper(canvas).noise
    for x0, y0 in zip(xs, ys):
        r = a * (b / a) ** g.uniform(0, 1)
        cx, cy = x0 + g.uniform(-.5, .5), y0 + g.uniform(-.5, .5)
        iy, ix = int(round(cy)), int(round(cx))
        sv = float(wc.damp[iy, ix])
        hv = float(wc.water[iy, ix])
        wet = min(1.0, sv * 1.25) * (1.0 - 0.5 * min(1.0, max(0.0, (hv - 0.1) / 0.5)))
        if wet <= 0.05 or canvas.watercolor_mask((iy, ix)) >= 0.5:
            continue
        R = r * strength * (3.0 + 4.0 * wet) * (1.0 + 0.6 * min(1.0, hv / 0.4))
        Rm = int(R * 1.6) + 3
        ya, yb, xa, xb = max(0, iy - Rm), min(H, iy + Rm + 1), max(0, ix - Rm), min(W, ix + Rm + 1)
        yy, xx = np.mgrid[ya:yb, xa:xb].astype(np.float32)
        dx, dy = xx - cx, yy - cy
        d = np.sqrt(dx * dx + dy * dy)
        th = np.arctan2(dy, dx)
        # star: a few long fibre-guided spikes plus a ragged core
        k = int(g.integers(5, 10))
        prof = np.full(d.shape, 0.45, np.float32)
        for _ in range(k):
            t0 = g.uniform(-np.pi, np.pi)
            ln = g.uniform(0.6, 1.0)
            wdt = g.uniform(0.12, 0.3)
            dth = np.angle(np.exp(1j * (th - t0)))
            prof = np.maximum(prof, ln * np.exp(-(dth / wdt) ** 2))
        Rt = R * prof * (0.8 + 0.4 * pn[ya:yb, xa:xb])
        q = d / np.maximum(Rt, 0.5)
        inner = np.clip(1.0 - q, 0, 1)                    # 1 at centre, 0 at the star edge
        take = (0.9 * wet * inner ** 0.6).astype(np.float32)
        ring = np.clip(1.0 - np.abs(q - 1.15) / 0.35, 0, 1).astype(np.float32)
        sl = (slice(ya, yb), slice(xa, xb))
        ring *= (1.0 - canvas.watercolor_mask(sl))
        if canvas.wax is not None:
            take *= 1.0 - canvas.wax[sl]
        loose = take * (1.0 - wc.bond[sl])               # settled, unbonded pigment moves too
        rs = float(ring.sum())
        if rs <= 1e-6:
            continue
        rn = ring / rs
        for arr, darr in ((wc.K, wc.dK), (wc.S, wc.dS)):
            moved = (arr[sl] * take[..., None]).sum(axis=(0, 1)) + (darr[sl] * loose[..., None]).sum(axis=(0, 1))
            arr[sl] *= (1 - take)[..., None]
            darr[sl] *= (1 - loose)[..., None]
            darr[sl] += rn[..., None] * moved[None, None, :]
        for arr, darr in ((wc.m, wc.dm), (wc.g, wc.dg), (wc.st, wc.dst)):
            moved = float((arr[sl] * take).sum() + (darr[sl] * loose).sum())
            arr[sl] *= (1 - take)
            darr[sl] *= (1 - loose)
            darr[sl] += rn * moved
        wc.water[sl] *= (1 - 0.8 * take)
        wc.damp[sl] *= (1 - 0.5 * take)
        wc.salt[sl] += (0.6 * strength * inner).astype(np.float32)
    wc.salt *= (1.0 - canvas.watercolor_mask())
    canvas._after_stroke()


def _tissue(canvas, cov, g, scale=10.0):
    """Crumpled-tissue contact pattern within cov."""
    n1 = _nz.fbm(canvas.shape, scale=scale, octaves=4, seed=g)
    n2 = _nz.fbm(canvas.shape, scale=scale * 2.5, octaves=3, seed=g)
    ridges = 1.0 - np.abs(n1)            # creases
    t = np.clip(0.55 * ridges + 0.45 * (0.5 + 0.35 * n2), 0, 1)
    return np.clip((t - 0.22) / 0.33, 0, 1) * cov


def blot(canvas, region, strength=0.8, texture='tissue', scale=10.0, seed=None):
    """Press a tissue / sponge onto the paper to lift paint.

    Removes free water with its suspended pigment, part of the paper
    moisture and loosely settled pigment.  Works best on a wet wash (soft
    cloud shapes in a sky); staining pigments and dried deposits resist.
    texture : 'tissue' (crumpled creases), 'sponge' (pitted) or 'flat'."""
    wc = canvas.ensure_watercolor()
    _extra(wc, canvas)
    g = _rng(seed, canvas)
    cov = region_mask(canvas, region, soft=1.5)
    if texture == 'tissue':
        c = _tissue(canvas, cov, g, scale)
    elif texture == 'sponge':
        nn = _nz.fbm(canvas.shape, scale=scale * 0.4, octaves=3, seed=g)
        c = np.clip((nn + 0.4) / 0.8, 0, 1) * cov
    else:
        c = cov
    c = (np.clip(strength, 0, 1) * c).astype(np.float32)
    ys, xs = np.nonzero(c > 1e-4)
    if len(ys) == 0:
        return
    pad = 8
    sl = (slice(max(0, ys.min() - pad), ys.max() + pad + 1),
          slice(max(0, xs.min() - pad), xs.max() + pad + 1))
    cc = c[sl]
    halo = np.maximum(cc, 0.8 * _nz.blur(cc, 3.0))      # the tissue also wicks nearby water
    h = wc.water[sl]
    wet = np.clip(h / 0.15, 0, 1) * 0.6 + np.clip(wc.damp[sl], 0, 1) * 0.4
    fw = np.clip(halo * 0.97, 0, 0.99)
    fp = np.clip(cc * (0.45 + 0.55 * np.clip(h / 0.15, 0, 1)), 0, 0.98)
    fp = np.maximum(fp, fw * 0.6)
    for arr in (wc.K, wc.S):
        arr[sl] *= (1 - fp)[..., None]
    for arr in (wc.m, wc.g, wc.st):
        arr[sl] *= (1 - fp)
    wc.water[sl] *= (1 - fw)
    wc.damp[sl] *= (1 - 0.7 * halo)
    _lift_deposit(wc, sl, cc * wet * 0.9)
    wc.u[:] *= 0.5
    wc.v[:] *= 0.5
    canvas._after_stroke()


def _lift_deposit(wc, sl, f):
    dm = wc.dm[sl]
    stn = np.where(dm > 1e-9, wc.dst[sl] / np.maximum(dm, 1e-9), 0.0)
    ease = (1.0 - 0.9 * np.clip(stn, 0, 1)) * (1.0 - 0.75 * wc.bond[sl])
    fl = np.clip(f * ease, 0, 0.97).astype(np.float32)
    for arr in (wc.dK, wc.dS):
        arr[sl] *= (1 - fl)[..., None]
    for arr in (wc.dm, wc.dg, wc.dst):
        arr[sl] *= (1 - fl)


def lift(canvas, region, strength=0.7, water=0.25, soft=3.0, seed=None):
    """Lift paint with a damp brush and tissue (also works on dried, not yet
    ``canvas.dry()``-baked paint): re-wets, loosens and removes settled
    pigment -- easily for non-staining pigments, hardly for staining ones --
    leaving a soft-edged lighter area."""
    wc = canvas.ensure_watercolor()
    _extra(wc, canvas)
    g = _rng(seed, canvas)
    cov = region_mask(canvas, region, soft=soft)
    tex = 0.85 + 0.15 * _nz.fbm(canvas.shape, scale=6, octaves=3, seed=g)
    c = (np.clip(strength, 0, 1) * cov * tex).astype(np.float32)
    ys, xs = np.nonzero(c > 1e-4)
    if len(ys) == 0:
        return
    sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    cc = c[sl]
    # scrubbing loosens the bond, then the tissue takes paint away
    wc.bond[sl] *= (1 - 0.8 * cc)
    _lift_deposit(wc, sl, cc * 0.9)
    for arr in (wc.K, wc.S):
        arr[sl] *= (1 - 0.9 * cc)[..., None]
    for arr in (wc.m, wc.g, wc.st):
        arr[sl] *= (1 - 0.9 * cc)
    # a little water stays behind and lets the edge soften
    wc.water[sl] = wc.water[sl] * (1 - 0.9 * cc) + water * cc
    wc.damp[sl] = np.maximum(wc.damp[sl], 0.7 * cc)
    _grow(canvas, c)
    canvas._after_stroke()


def tilt(canvas, angle=0.0, direction=90.0):
    """Tilt the board by ``angle`` degrees; water runs towards ``direction``
    (degrees in image space, 90 = down).  Painters tilt the board for graded
    washes so that a bead of paint collects at the lower edge."""
    a = math.radians(direction)
    s = 0.12 * math.sin(math.radians(angle))   # potential drop per px
    canvas.__dict__['_wc_tilt'] = (s * math.cos(a), s * math.sin(a))


def hair_dryer(canvas, seconds=10.0, region=None, power=6.0):
    """Force-dry with a hair dryer: evaporation ``power`` times faster inside
    ``region`` (whole canvas if None) for ``seconds``.  Freezes flows before
    they turn into backruns."""
    if canvas.wc is None or canvas.wc.bbox is None:
        return
    cov = region_mask(canvas, region, soft=20.0, respect_mask=False)
    ev = (1.0 + (power - 1.0) * cov).astype(np.float32)
    _run(canvas, float(seconds), evap=ev)
    canvas.time += float(seconds)


def wetness(canvas):
    """Summary of the watercolour layer: dict with ``wet_fraction`` (free
    water), ``damp_fraction``, ``water`` (total volume), ``bbox``."""
    wc = canvas.wc
    if wc is None:
        return dict(wet_fraction=0.0, damp_fraction=0.0, water=0.0, bbox=None)
    return dict(wet_fraction=float((wc.water > PARAMS['h_wet']).mean()),
                damp_fraction=float((wc.damp > 0.05).mean()),
                water=float(wc.water.sum()), bbox=wc.bbox)


def is_dry(canvas):
    """True when no free water, moisture or suspended pigment remains."""
    return canvas.wc is None or canvas.wc.bbox is None
