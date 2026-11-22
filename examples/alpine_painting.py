"""
"Alpenglühen am Bergsee" -- an oil painting made with pinsel.

The painting is built up the way a painter would work:
  0. warm toned ground (imprimatura)
  1. thin, loose block-in with big brushes (lets the ground glow through)
  2. sky wet-in-wet, blended; clouds
  3. far mountains: planes of light, snow, crisp ridge
  4. forested slopes, then individual pines
  5. boathouse
  6. lake: reflections dragged down, wind lines
  7. meadow, rocks, grass, flowers
  8. drying, glazes and final impasto highlights
"""
import sys, time, math
import numpy as np
from scipy import ndimage

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pinsel as ps
from pinsel import pigments as pg, path as P
from pinsel.pigments import Paint, mix
from pinsel.noise import blur, noise1d
from alpine_scene import make_scene

OUT = ROOT / 'output'
W, H = 1600, 1100
QUICK = '--quick' in sys.argv
T0 = time.time()


def log(msg):
    print(f'[{time.time() - T0:6.1f}s] {msg}', flush=True)


# ----------------------------------------------------------------------------- colour
# Every stroke is mixed from a real, limited palette of 12 tube colours.
PALETTE = ['titanium_white', 'cadmium_yellow', 'yellow_ochre', 'cadmium_red', 'permanent_rose',
           'quinacridone_magenta', 'ultramarine', 'phthalo_blue', 'burnt_sienna', 'dioxazine_purple',
           'ivory_black', 'sap_green']
_solver = None
_paint_cache = {}
USED = {}


def paint_of(rgb, opacity=0.75, q=40):
    """sRGB colour -> a paint mixed from the palette (cached on a quantised grid).
    ``opacity`` < 0.7 gives lean, thinned paint for the underpainting."""
    global _solver
    if _solver is None:
        from pinsel.painterly import PaletteSolver
        _solver = PaletteSolver(PALETTE, max_pigments=3)
    key = tuple(np.clip(np.round(np.asarray(rgb) * q), 0, q).astype(int)) + (round(opacity, 2),)
    p = _paint_cache.get(key)
    if p is None:
        p, recipe = _solver.solve(np.array(key[:3]) / q)
        if opacity < 0.7:
            p = p.thinned(0.7 - opacity)
        _paint_cache[key] = (p, recipe)
    else:
        p, recipe = p
    USED[recipe.describe()] = USED.get(recipe.describe(), 0) + 1
    return p


def jitter_color(rgb, g, amount=0.04, warm=0.0):
    c = np.asarray(rgb, float) * (1 + g.normal(0, amount))      # value
    c = c + g.normal(0, amount * 0.6, 3)                         # hue/chroma
    c[0] += warm; c[2] -= warm
    return np.clip(c, 0, 1)


# ----------------------------------------------------------------------------- fields
def orientation_field(img, sigma=3.0, tsigma=8.0):
    """Isophote angle (radians) and coherence from the structure tensor."""
    lum = img @ np.array([0.3, 0.59, 0.11])
    lum = blur(lum, sigma)
    gy, gx = np.gradient(lum)
    jxx, jxy, jyy = blur(gx * gx, tsigma), blur(gx * gy, tsigma), blur(gy * gy, tsigma)
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-9)
    return theta, coh


def blend_angle(theta, coh, default, strength=1.0):
    """Mix structure orientation with a default angle field (doubled-angle)."""
    w = np.clip(coh * strength, 0, 1)
    c = w * np.cos(2 * theta) + (1 - w) * np.cos(2 * default)
    s = w * np.sin(2 * theta) + (1 - w) * np.sin(2 * default)
    return 0.5 * np.arctan2(s, c)


def trace(x, y, angle, mask, length, step=2.0, g=None, wiggle=0.0, overrun=0.0):
    """Trace a stroke path through an orientation field in both directions."""
    H_, W_ = mask.shape

    def walk(sign):
        pts = []
        cx, cy = x, y
        prev = None
        travelled = 0.0
        while travelled < length / 2:
            xi, yi = int(round(cx)), int(round(cy))
            if not (0 <= xi < W_ and 0 <= yi < H_):
                break
            if mask[yi, xi] < 0.5:
                if travelled > overrun or overrun == 0:
                    break
            a = angle[yi, xi] if np.ndim(angle) else angle
            dx, dy = math.cos(a), math.sin(a)
            if prev is None:
                dx, dy = dx * sign, dy * sign
            elif dx * prev[0] + dy * prev[1] < 0:
                dx, dy = -dx, -dy
            if wiggle and g is not None:
                aa = math.atan2(dy, dx) + g.normal(0, wiggle)
                dx, dy = math.cos(aa), math.sin(aa)
            cx += dx * step; cy += dy * step
            prev = (dx, dy)
            travelled += step
            pts.append((cx, cy))
        return pts
    fwd = walk(1)
    bwd = walk(-1)
    pts = bwd[::-1] + [(x, y)] + fwd
    return np.array(pts)


def grid_points(mask, spacing, g, jitter=0.7, thresh=0.5):
    H_, W_ = mask.shape
    ys = np.arange(spacing / 2, H_, spacing)
    xs = np.arange(spacing / 2, W_, spacing)
    X, Y = np.meshgrid(xs, ys)
    X = X + g.uniform(-jitter, jitter, X.shape) * spacing
    Y = Y + g.uniform(-jitter, jitter, Y.shape) * spacing
    X = np.clip(X, 0, W_ - 1); Y = np.clip(Y, 0, H_ - 1)
    keep = mask[Y.astype(int), X.astype(int)] > thresh
    pts = np.column_stack([X[keep], Y[keep]])
    g.shuffle(pts)
    return pts


def fill_gaps(cv, ref, g, brush, spacing=5, thresh=0.05):
    """Cover spots the block-in missed (tiny pockets between shapes)."""
    bare = ((cv.film_vol + cv.base_vol) < thresh).astype(float)
    bare = ndimage.binary_opening(bare > 0.5, iterations=1).astype(float)
    pts = grid_points(bare, spacing, g, jitter=0.4)
    for x, y in pts:
        brush.load(paint_of(sample(ref, x, y), 0.6), 0.8)
        a = g.uniform(0, np.pi)
        cv.stroke(brush, P.line((x - 4 * math.cos(a), y - 4 * math.sin(a)), (x + 4 * math.cos(a), y + 4 * math.sin(a))),
                  pressure=0.8)
    return len(pts)


def signature(cv, x0, y0, h=22.0, seed=0):
    """Sign the painting with a rigger, in a simple cursive hand."""
    def S(pts):
        return np.array([(x0 + px * h, y0 + py * h) for px, py in pts])
    letters = [
        P.arc((0, 0), 1, -50, -310, n=60) * [0.38, 0.5] + [0.45, 0.5],                        # C
        np.array([(0.9, 0.95), (1.15, 0.4), (1.25, -0.1), (1.1, 0.0), (1.05, 0.6), (1.2, 1.0)]),  # l
        P.arc((1.55, 0.75), 0.25, 10, -350, n=48),                                              # a
        np.array([(1.8, 0.5), (1.8, 1.0), (1.92, 0.95)]),
        np.array([(2.02, 0.5), (2.0, 0.9), (2.2, 1.0), (2.4, 0.85), (2.45, 0.5), (2.45, 0.95), (2.57, 1.0)]),  # u
        P.arc((2.88, 0.75), 0.25, 10, -350, n=48),                                              # d
        np.array([(3.13, -0.15), (3.13, 0.95), (3.25, 1.0)]),
        np.array([(3.35, 0.8), (3.62, 0.72), (3.58, 0.5), (3.38, 0.6), (3.37, 0.9), (3.62, 1.0), (3.8, 0.85)]),  # e
    ]
    rig = ps.Brush('rigger', 1.25, seed=seed)
    ink = mix(pg.CADMIUM_RED, 2, pg.BURNT_SIENNA, 1, pg.TITANIUM_WHITE, 0.8)
    for k, L in enumerate(letters):
        pts = S(L) if len(L) > 8 or k in (0, 2, 5) else S(P.spline(L, samples_per_px=40))
        rig.load(ink, 1.0)
        cv.stroke(rig, pts, pressure=P.taper(0.1, 0.25, 0.75))


def sample(img, x, y):
    return img[int(np.clip(y, 0, img.shape[0] - 1)), int(np.clip(x, 0, img.shape[1] - 1))]


def paint_region(cv, brush, mask, ref, spacing, length, angle, g, pressure=None,
                 amount=1.0, opacity=0.75, cjit=0.035, min_len=4, overrun=0.0,
                 wiggle=0.0, step=2.0, speed=1.0, dirty=0.0, angle_mode=None, warm=0.0,
                 limit=None, len_jit=0.35, coverage=None):
    """Fill a region with strokes.  ``coverage`` = average number of coats:
    when given, the grid spacing is derived from brush width and stroke length
    (long strokes need far fewer starting points than short dabs)."""
    if coverage is not None:
        spacing = max(brush.size * 0.8, math.sqrt(2 * brush.size * length / coverage))
    pts = grid_points(mask, spacing, g)
    if limit:
        pts = pts[:limit]
    n = 0
    for x, y in pts:
        L = length * (1 + g.uniform(-len_jit, len_jit))
        path = trace(x, y, angle, mask, L, step=step, g=g, wiggle=wiggle, overrun=overrun)
        if len(path) < 2 or P.arclength(path)[-1] < min_len:
            path = np.array([(x - 1, y), (x + 1, y)])
        col = jitter_color(sample(ref, x, y), g, cjit, warm)
        p = paint_of(col, opacity)
        if dirty > 0 and g.uniform() < dirty:
            brush.dip(p, amount * 0.7)
        else:
            brush.load(p, amount)
        cv.stroke(brush, path, pressure=pressure, speed=speed, angle=angle_mode,
                  seed=int(g.integers(1 << 30)))
        n += 1
    return n


# ============================================================================= main
def main():
    g = np.random.default_rng(2024)
    log('building reference')
    ref, S = make_scene(W, H)
    horizon = S['horizon']
    yy, xx = np.mgrid[0:H, 0:W]
    ref_b12 = np.stack([blur(ref[..., k], 12) for k in range(3)], -1)
    ref_b5 = np.stack([blur(ref[..., k], 5) for k in range(3)], -1)
    ref_b2 = np.stack([blur(ref[..., k], 1.8) for k in range(3)], -1)
    ref_lake = np.stack([ndimage.gaussian_filter(ref[..., k], (2.5, 18.0)) for k in range(3)], -1)

    cv = ps.Canvas(W, H, ground='canvas', seed=11, texture_scale=1.1)
    cv.record(every=150, scale=0.4)

    # masks
    sky = 1 - np.clip(S['far_mtn'] + S['mid'] + S['trees'] + S['house'], 0, 1)
    sky[horizon:] = 0
    far = S['far_mtn'] * (1 - np.clip(S['mid'] + S['trees'], 0, 1))
    mid = S['mid'] * (1 - S['trees'])
    mid[horizon:] = 0
    water = S['water'] * (1 - S['meadow']) * (1 - S['rocks'])
    meadow = np.clip(S['meadow'] + S['rocks'], 0, 1)

    theta, coh = orientation_field(ref, 2.5, 6)

    # --------------------------------------------------------------- 0. imprimatura
    log('0 toned ground')
    cv.tone(mix(pg.BURNT_SIENNA, 2, pg.YELLOW_OCHRE, 1).thinned(0.2), thickness=0.45, seed=3)
    cv.snapshot()

    # --------------------------------------------------------------- 1. block-in
    log('1 block-in')
    big = ps.Brush('filbert', 26, seed=1)
    a_sky = np.zeros((H, W)) + 0.05 * blur(g.normal(size=(H, W)), 30) * 30
    n = paint_region(cv, big, sky, ref_b5, 34, 160, a_sky, g, amount=0.7, opacity=0.55, coverage=2.2,
                     pressure=P.taper(0.1, 0.2, 0.75), overrun=12)
    a_far = blend_angle(theta, coh, np.full((H, W), np.pi / 2 - 0.4), 1.5)
    n += paint_region(cv, ps.Brush('flat', 20, seed=2), far, ref_b12, 26, 70, a_far, g, coverage=2.2,
                      amount=0.7, opacity=0.55, pressure=P.taper(0.1, 0.25, 0.75), overrun=6)
    n += paint_region(cv, ps.Brush('flat', 18, seed=3), np.clip(mid + S['trees'], 0, 1), ref_b12, 24, 40,
                      np.pi / 2, g, amount=0.7, opacity=0.55, pressure=P.taper(0.1, 0.25, 0.75), overrun=4)
    ref_lake_b = np.stack([ndimage.gaussian_filter(ref[..., k], (4.0, 20.0)) for k in range(3)], -1)
    n += paint_region(cv, big, water, ref_lake_b, 26, 180, 0.0, g, amount=0.8, opacity=0.65, coverage=2.2,
                      pressure=P.taper(0.1, 0.2, 0.75), overrun=10)
    n += paint_region(cv, ps.Brush('hog', 18, seed=4), meadow, ref_b12, 26, 50, -np.pi / 2 + 0.3, g,
                      amount=0.8, opacity=0.6, pressure=P.taper(0.1, 0.25, 0.8), overrun=6)
    n += fill_gaps(cv, ref_b5, g, ps.Brush('round', 3, seed=40))
    log(f'  block-in strokes: {n}')
    cv.dry()
    cv.snapshot()

    # --------------------------------------------------------------- 2. sky
    log('2 sky')
    hake = ps.Brush('hake', 24, seed=5)
    sky_a = np.full((H, W), 0.0) + 0.06 * np.sin(xx / 260.0 + yy / 400.0)
    n = paint_region(cv, hake, sky, ref_b5, 17, 130, sky_a, g, amount=1.0, opacity=0.9, coverage=8,
                     pressure=P.jittered(P.taper(0.08, 0.2, 0.8), 0.15, seed=1), cjit=0.02, overrun=10,
                     dirty=0.25, wiggle=0.02, len_jit=0.6)
    # soften: a clean, dry hake drawn lightly across in criss-cross strokes
    soft = ps.Brush('hake', 22, seed=6, pickup=0.6)
    for i in range(240):
        soft.clean()
        y = g.uniform(5, horizon - 140)
        x0 = g.uniform(-100, W)
        L = g.uniform(150, 450)
        a = g.normal(0, 0.12)
        cv.stroke(soft, P.line((x0, y), (x0 + L * math.cos(a), y + L * math.sin(a))),
                  pressure=P.taper(0.2, 0.3, 0.45), speed=1.3)
    # clouds
    cloud = S['cloud'] * sky
    fil = ps.Brush('filbert', 8, seed=7)
    cl_a = np.full((H, W), 0.0) + 0.06 * blur(g.normal(size=(H, W)), 20) * 20
    ref_cloud = np.stack([ndimage.gaussian_filter(ref[..., k], (1.5, 5.0)) for k in range(3)], -1)
    n += paint_region(cv, fil, (blur(cloud, 2) > 0.3).astype(float), ref_cloud, 8, 80, cl_a, g, amount=1.1, coverage=3.5,
                      opacity=0.9, pressure=P.taper(0.2, 0.4, 0.7), cjit=0.03, dirty=0.3, overrun=8,
                      len_jit=0.6)
    # warm undersides catching the glow: thin horizontal streaks
    under = np.clip(cloud - ndimage.shift(cloud, (-5, 0), order=1), 0, 1) * (yy > 150)
    fil2 = ps.Brush('flat', 4, seed=8)
    n += paint_region(cv, fil2, (blur(under, 1.5) > 0.25).astype(float),
                      np.clip(ref_cloud * 1.1 + [0.06, 0.02, -0.02], 0, 1), 10, 60, 0.0, g, amount=1.2, coverage=2.5,
                      opacity=0.9, pressure=P.taper(0.2, 0.4, 0.6), angle_mode=90, overrun=10)
    log(f'  sky strokes: {n}')

    # --------------------------------------------------------------- 3. far mountains
    log('3 mountains')
    # protect the sky so the ridge stays crisp-ish (like a painter cutting in)
    sky_guard = blur(sky, 1.2) * 0.85
    cv.set_mask(sky_guard)
    a_m = blend_angle(theta, coh, np.full((H, W), np.pi / 2 - 0.5), 2.0)
    br = ps.Brush('bright', 9, seed=9)
    n = paint_region(cv, br, far, ref_b2, 8, 24, a_m, g, amount=1.1, opacity=0.9, coverage=4.5,
                     pressure=P.taper(0.1, 0.3, 0.8), cjit=0.035, overrun=3, dirty=0.2)
    # light planes and snow with heavier paint
    snowlit = far * np.clip(S['snow'] * (0.4 + S['lit']), 0, 1)
    br2 = ps.Brush('bright', 6, seed=10)
    snow_ref = np.clip(ref_b2 * 1.06 + [0.03, 0.01, 0.0], 0, 1)
    n += paint_region(cv, br2, (snowlit > 0.5).astype(float), snow_ref, 6, 16, a_m, g, amount=1.5, coverage=4,
                      opacity=0.95, pressure=P.taper(0.1, 0.4, 0.85), cjit=0.02)
    # shadow accents in the gullies
    shad = far * (1 - S['lit']) * (1 - S['snow'])
    rg = ps.Brush('round', 3.5, seed=11)
    n += paint_region(cv, rg, (shad > 0.6).astype(float), ref_b2 * 0.92, 10, 20, a_m, g, amount=1.0,
                      opacity=0.9, pressure=P.taper(0.1, 0.4, 0.8), limit=2000)
    cv.clear_mask()
    log(f'  mountain strokes: {n}')

    # --------------------------------------------------------------- 4. forested slopes + pines
    log('4 slopes & pines')
    hog = ps.Brush('hog', 7, seed=12)
    n = paint_region(cv, hog, mid, ref_b2, 8, 16, np.pi / 2, g, amount=1.0, opacity=0.9,
                     pressure=P.taper(0.1, 0.4, 0.8), cjit=0.05, dirty=0.3)
    # pines, back to front
    rig = ps.Brush('rigger', 1.6, seed=13)
    tb = ps.Brush('round', 3.0, seed=14)
    dark = [mix(pg.CHROMIUM_OXIDE_GREEN, 1, pg.PHTHALO_GREEN, 0.6, pg.IVORY_BLACK, 1.1, pg.ULTRAMARINE, 0.5),
            mix(pg.SAP_GREEN, 2, pg.IVORY_BLACK, 1.2, pg.PRUSSIAN_BLUE, 0.6, pg.CHROMIUM_OXIDE_GREEN, 0.5),
            mix(pg.TERRE_VERTE, 1, pg.PHTHALO_GREEN, 0.5, pg.IVORY_BLACK, 1, pg.BURNT_UMBER, 0.3)]
    cool_light = mix(pg.TITANIUM_WHITE, 1, pg.CERULEAN_BLUE, 1, pg.SAP_GREEN, 1, pg.DIOXAZINE_PURPLE, 0.3)
    trees = S['tree_list'] if not QUICK else S['tree_list'][::3]
    for k, (tx, base, h, wdt) in enumerate(trees):
        top = base - h
        dk = dark[k % 3]
        rig.load(dk, 1.0)
        cv.stroke(rig, P.line((tx, top + 2), (tx + g.normal(0, 1), base)), pressure=P.ramp(0.3, 0.9))
        nb = int(h / 4.5)
        for i in range(nb):
            t = (i + 0.5) / nb
            yb = top + t * h
            half = wdt * t ** 0.9 * (0.8 + 0.3 * g.uniform())
            for side in (-1, 1):
                tb.load(dk if g.uniform() > 0.12 else mix(dk, 3, cool_light, 1), 1.2)
                ex = tx + side * half
                ey = yb + half * 0.35 + g.normal(0, 1.5)
                cv.stroke(tb, P.quad((tx, yb), (tx + side * half * 0.5, yb - 1), (ex, ey)),
                          pressure=P.flick(0.85, 0.2))
        # a few cool sky-lit touches on the left flank
        for i in range(int(h / 18)):
            t = g.uniform(0.2, 0.95)
            yb = top + t * h
            half = wdt * t ** 0.9
            tb.load(mix(dk, 2, cool_light, 1), 0.6)
            cv.stroke(tb, P.line((tx - half * 0.3, yb), (tx - half * 0.9, yb + half * 0.3)),
                      pressure=P.flick(0.6, 0.2))
    log(f'  pines: {len(trees)}')

    # --------------------------------------------------------------- 5. boathouse
    log('5 boathouse')
    bx, by, bw, bh = S['house_geom']
    fl = ps.Brush('flat', 4, seed=15)
    wood = [mix(pg.BURNT_SIENNA, 2, pg.BURNT_UMBER, 1), mix(pg.BURNT_UMBER, 2, pg.ULTRAMARINE, 1),
            mix(pg.BURNT_SIENNA, 2, pg.CADMIUM_ORANGE, 0.3, pg.BURNT_UMBER, 1)]
    for i in range(int(bw / 5)):
        x = bx - bw / 2 + 2.5 + i * 5
        fl.load(wood[i % 3], 1.0)
        cv.stroke(fl, P.line((x, by - bh), (x, by)), pressure=0.7, angle=0)
    roofc = mix(pg.IVORY_BLACK, 1, pg.BURNT_UMBER, 1, pg.ALIZARIN_CRIMSON, 0.3, pg.TITANIUM_WHITE, 0.6)
    for i in range(14):
        t = i / 13
        y = by - bh - 34 + t * 36
        half = (t * 34) / 0.55 * 0.5 + 8
        fl.load(roofc if i % 4 else mix(roofc, 2, pg.TITANIUM_WHITE, 1), 1.0)
        cv.stroke(fl, P.line((bx - min(half, bw / 2 + 12), y), (bx + min(half, bw / 2 + 12), y)), pressure=0.7)
    fl.load(mix(pg.IVORY_BLACK, 1, pg.BURNT_UMBER, 2), 1.0)
    cv.stroke(fl, P.line((bx - 20, by - 33), (bx - 20, by)), pressure=0.9, angle=0)
    cv.stroke(fl, P.line((bx - 27, by - 33), (bx - 27, by)), pressure=0.9, angle=0)
    glow = ps.Brush('round', 4, seed=16)
    glow.load(mix(pg.CADMIUM_YELLOW, 2, pg.CADMIUM_ORANGE, 1, pg.TITANIUM_WHITE, 1), 1.8)
    cv.stroke(glow, P.line((bx + 13, by - 24), (bx + 23, by - 24)), pressure=0.9)
    cv.stroke(glow, P.line((bx + 14, by - 20), (bx + 22, by - 28)), pressure=0.8)

    # --------------------------------------------------------------- 6. lake
    log('6 lake')
    flat = ps.Brush('filbert', 9, seed=17)
    wa = np.zeros((H, W)) + 0.02 * np.sin(xx / 150.0)
    n = paint_region(cv, flat, water, ref_lake, 6, 55, wa, g, amount=1.0, opacity=0.9, coverage=5.5,
                     pressure=P.taper(0.15, 0.3, 0.7), cjit=0.02, overrun=4, dirty=0.2, len_jit=0.5)
    # drag the reflections down a little with a clean flat (classic technique)
    # gentle horizontal wet blend, then a few downward drags (classic reflection technique)
    blend = ps.Brush('flat', 12, seed=28, pickup=0.45)
    for i in range(260):
        blend.clean()
        y = g.uniform(horizon + 4, H)
        x0 = g.uniform(-60, W)
        cv.stroke(blend, P.line((x0, y), (x0 + g.uniform(80, 220), y + g.normal(0, 1))),
                  pressure=P.taper(0.3, 0.3, 0.4), speed=1.3)
    drag = ps.Brush('flat', 8, seed=18, pickup=0.5)
    for x in np.arange(5, W, 13):
        drag.clean()
        ytop = horizon + 3 + g.uniform(0, 200)
        cv.stroke(drag, P.line((x + g.normal(0, 2), ytop), (x + g.normal(0, 2), ytop + g.uniform(40, 110))),
                  pressure=0.35, angle=0, speed=1.4)
    # wind lines and the shore glint
    rgl = ps.Brush('flat', 5, seed=19)
    st = S['streak'] * water
    pts = grid_points((st > 0.4).astype(float), 22, g)
    for x, y in pts:
        col = np.clip(sample(ref_b2, x, y) * 1.08 + 0.03, 0, 1)
        rgl.load(paint_of(col, 0.8), 0.9)
        L = g.uniform(40, 140)
        cv.stroke(rgl, P.line((x - L / 2, y), (x + L / 2, y + g.normal(0, 0.5))), pressure=P.taper(0.2, 0.3, 0.55),
                  angle=90)
    shore = ps.Brush('rigger', 1.5, seed=20)
    for i in range(70):
        x = g.uniform(0, W)
        if 0.42 < x / W < 0.47:
            continue
        shore.load(mix(pg.TITANIUM_WHITE, 3, pg.DIOXAZINE_PURPLE, 0.2, pg.CERULEAN_BLUE, 0.4), 0.8)
        cv.stroke(shore, P.line((x, horizon + 1 + g.uniform(0, 3)), (x + g.uniform(10, 60), horizon + 1 + g.uniform(0, 3))),
                  pressure=P.taper(0.3, 0.3, 0.6))
    # window glow reflection
    glow.load(mix(pg.CADMIUM_YELLOW, 2, pg.CADMIUM_ORANGE, 1, pg.TITANIUM_WHITE, 1), 1.2)
    for i in range(6):
        y = by + 14 + i * 6
        cv.stroke(glow, P.line((bx + 12 + g.normal(0, 2), y), (bx + 24 + g.normal(0, 2), y)), pressure=0.6 - i * 0.07)
    log(f'  lake strokes: {n}')

    # --------------------------------------------------------------- 6b. jetty, boat, chimney smoke
    log('6b jetty, boat & smoke')
    wood_lit = [mix(pg.BURNT_SIENNA, 2, pg.YELLOW_OCHRE, 1, pg.TITANIUM_WHITE, 1),
                mix(pg.RAW_UMBER, 2, pg.YELLOW_OCHRE, 1, pg.TITANIUM_WHITE, 1.5),
                mix(pg.BURNT_SIENNA, 1, pg.RAW_UMBER, 1, pg.TITANIUM_WHITE, 0.8, pg.PERMANENT_ROSE, 0.15)]
    wood_dark = mix(pg.BURNT_UMBER, 2, pg.ULTRAMARINE, 1)
    # posts and their wobbly reflections first (planks overlap them)
    post = ps.Brush('round', 2.0, seed=41)
    for (px_, cy, ln, wd) in S['posts']:
        post = ps.Brush('round', max(1.2, wd * 0.9), seed=41)
        post.load(wood_dark, 1.2)
        cv.stroke(post, P.line((px_, cy), (px_, cy + ln)), pressure=0.85)
        post.load(mix(wood_dark, 2, pg.ULTRAMARINE, 1, pg.TITANIUM_WHITE, 0.5), 0.9)
        refl = np.column_stack([px_ + 1.5 * np.sin(np.linspace(0, 9, 12)), np.linspace(cy + ln, cy + 2.3 * ln, 12)])
        cv.stroke(post, refl, pressure=P.taper(0.1, 0.6, 0.7), wobble=1.0)
    for i, (x0, x1, cy, th) in enumerate(S['planks']):
        pb = ps.Brush('flat', max(1.6, th * 0.6), seed=42 + i)
        pb.load(wood_lit[i % 3], 1.1)
        cv.stroke(pb, P.line((x0, cy), (x1, cy + g.normal(0, 0.3))), pressure=0.75)
        if i % 2 == 0:   # plank seams
            seam = ps.Brush('rigger', 0.8, seed=60 + i)
            seam.load(wood_dark, 0.8)
            cv.stroke(seam, P.line((x0 + 2, cy + th * 0.5), (x1 - 2, cy + th * 0.5)), pressure=0.5)
    # the rowing boat: red hull, white stripe, wooden inside, broken reflection
    bcx, bcy, blen = S['boat']
    hullc = mix(pg.CADMIUM_RED, 2, pg.ALIZARIN_CRIMSON, 1)
    hb = ps.Brush('filbert', 3.5, seed=50)
    for k in range(7):
        off = -4 + k * 2.1
        xs = np.linspace(-1, 1, 30)
        top = bcy - 6 + 3 * xs ** 2
        bot = bcy + 9 - 7 * xs ** 4
        ys = top + (bot - top) * (k + 0.5) / 7
        hb.load(hullc if k < 5 else mix(hullc, 2, pg.BURNT_UMBER, 1), 1.2)
        cv.stroke(hb, np.column_stack([bcx + xs * blen / 2 * (0.97 - 0.08 * k / 7), ys]), pressure=0.8)
    st_b = ps.Brush('rigger', 1.1, seed=51)
    st_b.load(mix(pg.TITANIUM_WHITE, 4, pg.NAPLES_YELLOW, 1), 1.2)
    xs = np.linspace(-0.92, 0.92, 30)
    cv.stroke(st_b, np.column_stack([bcx + xs * blen / 2, bcy + 1 + 3 * xs ** 2]), pressure=0.8)
    ins = ps.Brush('flat', 2.0, seed=52)
    ins.load(mix(pg.BURNT_SIENNA, 1, pg.RAW_UMBER, 1, pg.NAPLES_YELLOW, 1), 1.1)
    xs = np.linspace(-0.8, 0.8, 30)
    cv.stroke(ins, np.column_stack([bcx + xs * blen / 2, bcy - 5 + 3 * xs ** 2]), pressure=0.7)
    rb = ps.Brush('flat', 3, seed=53)
    for k in range(6):
        y = bcy + 13 + k * 3.5
        w_ = blen / 2 * (0.9 - 0.1 * k) * g.uniform(0.7, 1.0)
        rb.load(mix(hullc, 2, pg.BURNT_UMBER, 1, pg.ULTRAMARINE, 0.5), 0.9)
        x0 = bcx + g.normal(0, 4)
        cv.stroke(rb, P.line((x0 - w_, y), (x0 + w_, y)), pressure=P.taper(0.2, 0.3, 0.6), wobble=0.8)
    # mooring rope
    rope = ps.Brush('rigger', 0.7, seed=54)
    rope.load(mix(pg.RAW_UMBER, 1, pg.NAPLES_YELLOW, 1), 0.9)
    jx, jy = S['planks'][-1][0] + 4, S['planks'][-1][2]
    cv.stroke(rope, P.quad((bcx + blen / 2 - 6, bcy - 4), ((bcx + blen / 2 + jx) / 2, max(bcy, jy) + 8), (jx, jy)),
              pressure=0.6)
    # chimney and smoke
    cx_, cy_ = S['chimney']
    ch = ps.Brush('flat', 4, seed=55)
    ch.load(mix(pg.IVORY_BLACK, 1, pg.TITANIUM_WHITE, 1.5, pg.BURNT_SIENNA, 0.4), 1.2)
    cv.stroke(ch, P.line((cx_, cy_), (cx_, cy_ + 22)), pressure=0.85, angle=0)
    smoke_p = mix(pg.TITANIUM_WHITE, 3, pg.DIOXAZINE_PURPLE, 0.15, pg.CERULEAN_BLUE, 0.2).thinned(0.55).with_body(0.1)
    sp = S['smoke_path']
    for k in range(4):
        sm = ps.Brush('filbert', 4 + k * 3, seed=56 + k)
        seg = sp[int(k * 12):int(k * 12 + 26)]
        if len(seg) < 3:
            continue
        sm.load(smoke_p, 0.8 - 0.12 * k)
        cv.stroke(sm, seg, pressure=P.taper(0.3, 0.5, 0.55 - 0.08 * k), wobble=2.0)
    soft2 = ps.Brush('mop', 10, seed=60, pickup=0.5)
    cv.stroke(soft2, sp[6:], pressure=0.35, wobble=3.0)

    # --------------------------------------------------------------- 7. meadow
    log('7 meadow')
    hg = ps.Brush('hog', 10, seed=21)
    n = paint_region(cv, hg, meadow * (1 - S['rocks']), ref_b5 * 0.9, 13, 28, -np.pi / 2 + 0.25, g,
                     amount=1.2, opacity=0.8, pressure=P.flick(0.85, 0.3), cjit=0.06, dirty=0.4)
    # rocks: dark base then light tops
    rk = ps.Brush('bright', 6, seed=22)
    rock_a = blend_angle(theta, coh, np.full((H, W), 0.2), 1.0)
    n += paint_region(cv, rk, S['rocks'], ref_b2, 6, 16, rock_a, g, amount=1.4, opacity=0.85, coverage=4,
                      pressure=P.taper(0.1, 0.4, 0.9), cjit=0.03)
    # grass blades
    grass_cols = [mix(pg.SAP_GREEN, 2, pg.YELLOW_OCHRE, 1), mix(pg.SAP_GREEN, 1, pg.CADMIUM_YELLOW, 1, pg.TITANIUM_WHITE, 0.5),
                  mix(pg.YELLOW_OCHRE, 2, pg.TITANIUM_WHITE, 1), mix(pg.SAP_GREEN, 2, pg.ULTRAMARINE, 1, pg.BURNT_SIENNA, 0.5),
                  mix(pg.CHROMIUM_OXIDE_GREEN, 2, pg.NAPLES_YELLOW, 1)]
    blade = ps.Brush('rigger', 1.7, seed=23)
    edge = S['edge']
    nbl = 2600 if not QUICK else 700
    for i in range(nbl):
        x = g.uniform(0, 0.63 * W)
        top = np.interp(x, np.arange(W), edge)
        if top > H - 6:
            continue
        y = g.uniform(top - 4, H + 10)
        depthf = (y - top) / max(H - top, 1)
        L = 18 + 50 * depthf * g.uniform(0.6, 1.3)
        a = -np.pi / 2 + g.normal(0.15, 0.3)
        cx = x + math.cos(a) * L * 0.5 + g.normal(0, 4)
        blade.load(grass_cols[int(g.integers(len(grass_cols)))], 0.9)
        cv.stroke(blade, P.quad((x, y), (cx, y + math.sin(a) * L * 0.6), (x + math.cos(a) * L, y + math.sin(a) * L)),
                  pressure=P.flick(0.8, 0.15))
    # flowers: little dabs with a highlight
    fb = ps.Brush('round', 3.2, seed=24)
    for fx, fy, col in S['flowers']:
        top = np.interp(fx, np.arange(W), edge)
        sz = 0.6 + 1.2 * (fy - top) / max(H - top, 1)
        base_p = paint_of(np.array([int(col[i:i + 2], 16) / 255 for i in (1, 3, 5)]), 0.85)
        fb.load(base_p, 1.4)
        for j in range(3):
            a = g.uniform(0, 2 * np.pi)
            r = 2.5 * sz
            cv.stroke(fb, P.line((fx, fy), (fx + r * math.cos(a), fy + r * math.sin(a))), pressure=P.dab(0.5 + 0.3 * sz))
    log(f'  meadow strokes: {n} + {nbl} blades + {len(S["flowers"])} flowers')

    # --------------------------------------------------------------- 8. dry, glaze, highlights
    log('8 drying, glazes, highlights')
    cv.dry()
    # a thin warm glaze over the glowing peaks unifies the alpenglow
    gl = ps.Brush('hake', 30, seed=25)
    glaze = mix(pg.PERMANENT_ROSE, 1, pg.QUINACRIDONE_GOLD, 1).thinned(0.975).with_body(0.1)
    cv.set_mask(1 - blur(far * np.clip(S['lit'] * 1.4 - 0.2, 0, 1), 8))
    for y in np.arange(150, horizon - 120, 26):
        gl.load(glaze, 0.5)
        cv.stroke(gl, P.line((600, y), (1500, y + g.normal(0, 6))), pressure=0.6)
    cv.clear_mask()
    # final impasto light on the highest snow
    hi = (S['snow'] * S['lit'] * far > 0.8) & (yy < 330)
    imp = ps.Brush('bright', 3.5, seed=26)
    imp = ps.Brush('bright', 2.6, seed=26)
    pts = grid_points(hi.astype(float), 7, g)
    # palette-knife slabs of thick, sunlit snow on the main faces
    from pinsel import tools as T
    knife = T.PaletteKnife(length=11, angle=70, thickness=1.8, seed=29)
    kpts = grid_points(((S['snow'] * S['lit'] * far > 0.85) & (yy < 300)).astype(float), 16, g)
    for x, y in kpts[:160]:
        knife.load(mix(pg.TITANIUM_WHITE, 5, pg.PERMANENT_ROSE, 0.35, pg.NAPLES_YELLOW, 0.8), 1.0)
        a = a_m[int(y), int(x)]
        L = g.uniform(10, 20)
        knife.spread(cv, P.line((x, y), (x + L * math.cos(a), y + L * math.sin(a))), pressure=0.5)
    # knife-laid light on the tops of the shoreline rocks
    rk_knife = T.PaletteKnife(length=10, angle=80, thickness=1.2, seed=30)
    rtop = S['rocks'] * (ndimage.shift(S['rocks'], (6, 0), order=0) > 0.5) * (1 - ndimage.shift(S['rocks'], (-5, 0), order=0))
    for x, y in grid_points((blur(rtop, 1) > 0.3).astype(float), 16, g)[:35]:
        rk_knife.load(mix(pg.TITANIUM_WHITE, 2, pg.DIOXAZINE_PURPLE, 0.35, pg.BURNT_SIENNA, 0.25,
                          pg.ULTRAMARINE, 0.2, pg.PERMANENT_ROSE, 0.15), 1.0)
        L = g.uniform(8, 16)
        rk_knife.spread(cv, P.line((x - L / 2, y), (x + L / 2, y + g.normal(0, 2))), pressure=0.5)
    # a few flicked petals and seed heads in the meadow
    sp = T.Splatter(size=1.4, seed=31)
    for col, (cx, cy) in [('#e0507f', (180, 1010)), ('#f3c844', (420, 1050)), ('#f4efe6', (90, 940)),
                          ('#f3c844', (260, 930))]:
        sp.load(paint_of(np.array([int(col[i:i + 2], 16) / 255 for i in (1, 3, 5)]), 0.9).thinned(0.3), 1.0)
        sp.tap(cv, (cx, cy), radius=45, count=18)
    sp.load(mix(pg.TITANIUM_WHITE, 3, pg.CADMIUM_YELLOW, 1).thinned(0.3), 1.0)
    sp.toothbrush(cv, (250, 1000), radius=120, count=700, size=0.55)
    for x, y in pts[:500]:
        imp.load(mix(pg.TITANIUM_WHITE, 4, pg.PERMANENT_ROSE, 0.3, pg.NAPLES_YELLOW, 0.6).impasto(1.6), 1.3)
        a = a_m[int(y), int(x)]
        L = g.uniform(4, 10)
        cv.stroke(imp, P.line((x, y), (x + L * math.cos(a), y + L * math.sin(a))), pressure=P.taper(0.2, 0.5, 0.7))
    # glints of light on the water near the boathouse
    glint = ps.Brush('rigger', 1.4, seed=27)
    for i in range(10):
        glint.load(mix(pg.CADMIUM_YELLOW, 1, pg.TITANIUM_WHITE, 2, pg.CADMIUM_ORANGE, 0.3), 1.2)
        x = bx + g.normal(18, 8); y = horizon + 10 + i * 6 + g.uniform(0, 4)
        cv.stroke(glint, P.line((x - g.uniform(3, 9), y), (x + g.uniform(3, 9), y)), pressure=P.taper(0.3, 0.3, 0.6))
    signature(cv, W - 135, H - 50)
    cv.snapshot()

    log('rendering')
    cv.save_state(OUT / 'alpine_state.npz')
    cv.save(f'{OUT}/alpine_lake.png')
    cv.save(f'{OUT}/alpine_lake_raking.png', light_dir=(-0.9, -0.3, 0.35), relief=1.6, specular=0.6)
    cv.save_timelapse(f'{OUT}/alpine_lake_timelapse.gif', fps=10, max_frames=80, width=560)
    log(f'done -- {len(USED)} different mixtures from {len(PALETTE)} tube colours')
    top = sorted(USED.items(), key=lambda kv: -kv[1])[:12]
    for desc, cnt in top:
        print(f'   {cnt:5d}x  {desc}')
    return cv


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    main()
