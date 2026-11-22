"""
Painterly auto-painting demo.

Synthesises a landscape reference procedurally (no photo needed), then lets
pinsel's AutoPainter paint it with physical bristle brushes and a limited
palette of nine real pigments, mixed per stroke by the PaletteSolver.

    python examples/painterly_demo.py [style] [--small]

Writes
    output/painterly_reference.png     the synthetic reference
    output/painterly_demo.png          the painting (lit render)
    output/painterly_palette.png       palette-solver check: target | mixed swatches
    output/painterly_detail.png        2x crop of the painting
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

import pinsel as ps
from pinsel import noise as nz
from pinsel.painterly import AutoPainter, PaletteSolver, DEFAULT_PALETTE

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output')


# --------------------------------------------------------------------------
# Procedural landscape
# --------------------------------------------------------------------------
def _hex(c):
    c = c.lstrip('#')
    return np.array([int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def _lerp(a, b, t):
    t = np.asarray(t, dtype=np.float64)[..., None]
    return a * (1 - t) + b * t


def _smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def _ridge(W, base, amp, scale, seed, octaves=5, sharp=0.0):
    g = np.random.default_rng(seed)
    x = np.arange(W)
    y = np.zeros(W)
    a, s = 1.0, float(scale)
    for _ in range(octaves):
        knots = g.normal(0, 1, int(W / s) + 4)
        y += a * np.interp(x / s, np.arange(len(knots)), knots)
        a *= 0.5
        s /= 2.1
    y = y / np.abs(y).max()
    if sharp:
        y = np.sign(y) * np.abs(y) ** (1 - sharp)
    return base - amp * y


def landscape(W=1000, H=750, seed=3, return_masks=False):
    """A summer lake landscape at late afternoon, sRGB float (H, W, 3).

    The scene is synthesised at 1000 x 750 and resampled to (W, H), so the
    composition does not depend on the resolution.  With
    ``return_masks=True`` also returns a dict of soft region masks
    ('sky', 'mountains', 'lake', 'trees', 'meadow', 'flowers').
    """
    img, masks = _landscape(1000, 750, seed)
    if (W, H) != (1000, 750):
        im = Image.fromarray((img * 255 + 0.5).astype(np.uint8)).resize((W, H), Image.LANCZOS)
        img = np.asarray(im, dtype=np.float64) / 255.0
        masks = {k: np.clip(ndimage.zoom(m, (H / m.shape[0], W / m.shape[1]), order=1), 0, 1)[:H, :W]
                 for k, m in masks.items()}
    return (img, masks) if return_masks else img


def _landscape(W, H, seed):
    g = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    v = yy / H
    u = xx / W
    img = np.zeros((H, W, 3))

    # ---- sky: gradient + sun glow + clouds
    horizon = 0.46
    t = np.clip(v / horizon, 0, 1)
    sky = _lerp(_hex('#2f62a8'), _hex('#7fa7d2'), t ** 0.8)
    sky = _lerp(sky, _hex('#f1dcb8'), _smooth(0.55, 1.0, t) * 0.85)
    sx, sy = 0.72, 0.30
    d = np.hypot((u - sx) * W / H, v - sy)
    glow = np.exp(-d / 0.16)
    sky = _lerp(sky, _hex('#fff0c4'), np.clip(glow * 1.1, 0, 1))
    sky = _lerp(sky, _hex('#fffbea'), np.clip(np.exp(-d / 0.025) * 1.4, 0, 1))
    # cumulus banks: horizontally stretched noise inside two height bands
    cl = nz.spectral_noise((H, W), beta=3.2, seed=g, aniso=(0.55, 2.2))
    cl += 0.4 * nz.spectral_noise((H, W), beta=2.4, seed=g, aniso=(0.7, 1.6))
    band = 0.9 * np.exp(-((v - 0.16) / 0.07) ** 2) + 1.0 * np.exp(-((v - 0.34) / 0.045) ** 2)
    cover = _smooth(0.55, 1.25, cl * 0.6 + band * 1.1 - 0.25) * _smooth(0.0, 0.08, v)
    cover = ndimage.gaussian_filter(cover, 1.2)
    # shading: lit tops, violet-grey bellies (compare with the cover a bit lower)
    cs = ndimage.gaussian_filter(cover, 5)
    belly = np.clip(2.2 * (cs - np.roll(cs, -9, axis=0)) + 0.15, 0, 1)
    c_lit = _lerp(_hex('#fbf5e9'), _hex('#ffe3a6'), np.clip(glow * 1.6, 0, 1))
    c_dark = _lerp(_hex('#a49cb4'), _hex('#d0a08e'), np.clip(glow * 1.3, 0, 1))
    cloud = _lerp(c_lit, c_dark, np.clip(1.0 - belly * 1.4 + 0.25 * (1 - cover), 0, 1) * 0.85)
    sky = _lerp(sky, cloud, cover * _smooth(horizon + 0.02, horizon - 0.06, v))
    img[:] = sky

    # ---- distant mountains (aerial perspective)
    r1 = _ridge(W, horizon * H - 20, 105, 260, seed + 1, sharp=0.25)
    m1 = _smooth(-1.5, 1.5, yy - r1[None, :])
    tex = nz.fbm((H, W), scale=40, octaves=4, seed=g)
    mcol = _lerp(_hex('#8f98bf'), _hex('#a8a4c0'), np.clip(0.5 + 0.3 * tex, 0, 1))
    # snow / light on the sunny flanks
    slope = np.gradient(r1)[None, :]
    lit = np.clip(-slope * 0.25 + 0.15 * tex, 0, 1) * _smooth(r1[None, :] + 45, r1[None, :], yy)
    mcol = _lerp(mcol, _hex('#e6dbe0'), lit * 0.8)
    haze = _smooth(r1[None, :], horizon * H + 10, yy)
    mcol = _lerp(mcol, _hex('#c9c3cf'), haze * 0.55)
    img = _lerp(img, mcol, m1)

    r2 = _ridge(W, horizon * H + 18, 55, 180, seed + 2, sharp=0.1)
    m2 = _smooth(-1.5, 1.5, yy - r2[None, :])
    tex2 = nz.fbm((H, W), scale=22, octaves=4, seed=g)
    c2 = _lerp(_hex('#4f6683'), _hex('#6f7f8c'), np.clip(0.5 + 0.35 * tex2, 0, 1))
    c2 = _lerp(c2, _hex('#8e9aa6'), _smooth(r2[None, :] + 50, r2[None, :] + 90, yy) * 0.4)
    img = _lerp(img, c2, m2)

    # ---- middle-ground hills (warm greens, sunlit)
    r3 = _ridge(W, 0.585 * H, 38, 300, seed + 3, octaves=4)
    m3 = _smooth(-1.5, 1.5, yy - r3[None, :])
    tex3 = nz.fbm((H, W), scale=18, octaves=5, seed=g)
    c3 = _lerp(_hex('#5d7a3a'), _hex('#a9a552'), np.clip(0.45 + 0.35 * tex3 + 0.3 * u, 0, 1))
    c3 = _lerp(c3, _hex('#3e5a33'), np.clip(-tex3 * 0.6, 0, 1))
    img = _lerp(img, c3, m3)

    # ---- lake reflecting the sky and the mountains
    shore = 0.64 * H + 14 * np.sin(u[0] * 5.0 + 0.7) + 8 * nz.noise1d(W, 120, seed=g)
    lake_bot = 0.80 * H + 10 * nz.noise1d(W, 160, seed=g)
    lake = _smooth(-1.5, 1.5, yy - shore[None, :]) * _smooth(1.5, -1.5, yy - lake_bot[None, :])
    mirror = np.clip(2 * shore[None, :] - yy, 0, H - 1)
    rip = nz.spectral_noise((H, W), beta=2.2, seed=g, aniso=(0.12, 3.5))
    my = np.clip(mirror + rip * 4.0 * (1 + (yy - shore[None, :]) / 60.0), 0, H - 1)
    refl = np.stack([ndimage.map_coordinates(img[..., c], [my, xx + rip * 2], order=1, mode='nearest')
                     for c in range(3)], axis=-1)
    refl = _lerp(refl, _hex('#35587a'), np.clip((yy - shore[None, :]) / 140.0, 0, 0.55))
    sparkle = np.clip((rip - 1.2) * 1.5, 0, 1) * np.exp(-((u - sx) / 0.07) ** 2) * \
        _smooth(shore[None, :], shore[None, :] + 30, yy)
    refl = _lerp(refl, _hex('#fff4d0'), sparkle)
    img = _lerp(img, refl, lake)

    # ---- trees: clumps of foliage, lit from the sun side (upper right)
    tree_mask = np.zeros((H, W))
    tcol = np.zeros((H, W, 3))
    trees = []
    # distant treeline: a bumpy band sitting on the far shore
    bumps = np.zeros(W)
    for i in range(60):
        bx, br = g.uniform(0, W), g.uniform(5, 16)
        bumps = np.maximum(bumps, br * np.sqrt(np.clip(1 - ((np.arange(W) - bx) / (br * 1.3)) ** 2, 0, 1)))
    top = shore - 3 - bumps - 4 * nz.noise1d(W, 30, seed=g)
    band_m = _smooth(-1.2, 1.2, yy - top[None, :]) * _smooth(1.0, -1.0, yy - shore[None, :] - 3)
    bl = np.clip(0.45 + 0.6 * nz.fbm((H, W), scale=6, octaves=3, seed=g) * 0.5
                 + 0.4 * (np.gradient(top)[None, :] < 0) - 0.3 * (yy - top[None, :]) / 25.0, 0, 1)
    band_c = _lerp(_hex('#2c4234'), _hex('#6f8a48'), bl)
    band_c = _lerp(band_c, _hex('#6c7f86'), 0.3)
    img = _lerp(img, band_c, band_m)
    trees.append((0.11 * W, 0.73 * H, 120, 1.15, True))   # big trees in front
    trees.append((0.26 * W, 0.74 * H, 62, 1.35, True))
    trees.append((0.92 * W, 0.72 * H, 96, 1.3, True))
    fol = nz.fbm((H, W), scale=7, octaves=3, seed=g)
    Ldir = np.array([0.75, -0.66])
    for (x, y, r, el, trunk) in trees:
        cy = y - r * el
        if trunk:
            tr = (np.abs(xx - x - 0.04 * (yy - y)) < max(3, r * 0.06)) & (yy > cy) & (yy < y + 4)
            img[tr] = _hex('#3a2c22')
        nclump = int(6 + r / 6)
        cxs = x + g.normal(0, 0.45, nclump) * r
        cys = cy + g.normal(0, 0.5, nclump) * r * el
        rcs = r * g.uniform(0.35, 0.6, nclump)
        order = np.argsort(g.random(nclump))
        y0, y1 = int(max(0, cy - 2.6 * r * el)), int(min(H, y + r))
        x0, x1 = int(max(0, x - 2.4 * r)), int(min(W, x + 2.4 * r))
        sub_x, sub_y = xx[y0:y1, x0:x1], yy[y0:y1, x0:x1]
        f = fol[y0:y1, x0:x1]
        for k in order:
            dx, dy = (sub_x - cxs[k]) / rcs[k], (sub_y - cys[k]) / rcs[k]
            dd = np.hypot(dx, dy)
            a = _smooth(1.0, 0.8, dd + 0.18 * f)
            if a.max() <= 0:
                continue
            # clump shading: sphere-ish normal . light + tree-level light
            lam = np.clip(0.55 + 0.45 * (dx * Ldir[0] + dy * Ldir[1]) / np.maximum(dd, 0.3), 0, 1)
            glob = np.clip(0.5 + 0.5 * ((cxs[k] - x) / r * 0.6 - (cys[k] - cy) / (r * el) * 0.5), 0, 1)
            sh = np.clip(0.65 * lam + 0.35 * glob + 0.12 * f, 0, 1)
            c = _lerp(_hex('#1c3022'), _hex('#4f7030'), sh)
            c = _lerp(c, _hex('#b6bd62'), np.clip(sh - 0.78, 0, 1) * 2.5)
            if not trunk:   # distant trees: bluish haze
                c = _lerp(c, _hex('#6d8088'), 0.35)
            tm = tree_mask[y0:y1, x0:x1]
            tc = tcol[y0:y1, x0:x1]
            tcol[y0:y1, x0:x1] = _lerp(tc, c, a)
            tree_mask[y0:y1, x0:x1] = np.maximum(tm, a)
    img = _lerp(img, tcol, tree_mask)

    # ---- foreground meadow with grass texture and flowers
    fg_top = np.maximum(lake_bot, 0.0)
    fg = _smooth(-2, 2, yy - fg_top[None, :])
    fg = np.maximum(fg, (1 - lake) * _smooth(-2, 2, yy - shore[None, :]))
    fg = np.clip(fg - tree_mask, 0, 1) * _smooth(0.6 * H, 0.62 * H, yy)
    grass_n = nz.spectral_noise((H, W), beta=1.7, seed=g, aniso=(1.6, 0.6))
    depth = np.clip((yy - 0.6 * H) / (0.4 * H), 0, 1)
    gcol = _lerp(_hex('#86a04a'), _hex('#4e7034'), depth)
    gcol = _lerp(gcol, _hex('#c6c26a'), np.clip(grass_n * 0.35 + 0.2 - depth * 0.3, 0, 1))
    gcol = _lerp(gcol, _hex('#2c4a26'), np.clip(-grass_n * 0.45, 0, 1) * (0.4 + depth))
    img = _lerp(img, gcol, fg)
    # flowers
    flower_cols = [_hex('#d8312c'), _hex('#f2c230'), _hex('#f4f0e6'), _hex('#8a5bb8'), _hex('#e8803a')]
    nfl = 900
    fx = g.uniform(0, W, nfl)
    fy = 0.8 * H + (H - 0.8 * H) * g.uniform(0, 1, nfl) ** 0.7
    cluster = nz.fbm((H, W), scale=60, octaves=2, seed=g)
    flowers = np.zeros((H, W))
    for i in range(nfl):
        x, y = fx[i], fy[i]
        if cluster[int(min(y, H - 1)), int(x)] < -0.2 or tree_mask[int(min(y, H - 1)), int(x)] > 0.3:
            continue
        r = 1.5 + 6.5 * ((y - 0.8 * H) / (0.2 * H)) ** 1.5
        c = flower_cols[g.integers(len(flower_cols))]
        y0, y1 = int(max(0, y - r - 2)), int(min(H, y + r + 3))
        x0, x1 = int(max(0, x - r - 2)), int(min(W, x + r + 3))
        dd = np.hypot(xx[y0:y1, x0:x1] - x, (yy[y0:y1, x0:x1] - y) * 1.3) / r
        a = _smooth(1.0, 0.6, dd)
        cc = _lerp(c * 1.05, c * 0.7, np.clip((yy[y0:y1, x0:x1] - y) / r, 0, 1))
        img[y0:y1, x0:x1] = _lerp(img[y0:y1, x0:x1], np.clip(cc, 0, 1), a)
        flowers[y0:y1, x0:x1] = np.maximum(flowers[y0:y1, x0:x1], a)
    img = np.clip(img, 0, 1)
    land = np.clip(np.maximum.reduce([m1, m2, m3]), 0, 1)
    tree_all = np.maximum(tree_mask, band_m)
    masks = dict(sky=np.clip(1 - land - tree_all, 0, 1), mountains=np.clip(np.maximum(m1, m2) - m3, 0, 1),
                 lake=np.clip(lake - tree_all, 0, 1), trees=tree_all, meadow=fg,
                 flowers=ndimage.gaussian_filter(flowers, 3))
    return img, masks


# --------------------------------------------------------------------------
# Palette-solver check image
# --------------------------------------------------------------------------
def palette_check(solver, ref, path, n=48, seed=0):
    """Swatches of reference colours (left half) vs. the solver's mixtures
    (right half), with recipe text, plus the palette itself."""
    g = np.random.default_rng(seed)
    H, W = ref.shape[:2]
    # pick representative colours: k-means-ish via random + farthest point
    pix = ref.reshape(-1, 3)[g.integers(0, H * W, 20000)]
    chosen = [pix[0]]
    dmin = np.linalg.norm(pix - pix[0], axis=1)
    for _ in range(n - 1):
        i = int(np.argmax(dmin))
        chosen.append(pix[i])
        dmin = np.minimum(dmin, np.linalg.norm(pix - pix[i], axis=1))
    cols = np.array(chosen)
    cols = cols[np.argsort(ps.spectral.linear_rgb_to_lab(ps.spectral.srgb_to_linear(cols))[:, 0])]
    res = solver.solve_many(cols)
    sw, th = 70, 24
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    cw = 430
    pal_h = 70
    img = Image.new('RGB', (ncol * cw + 20, nrow * (th + 8) + 70 + pal_h), (246, 243, 236))
    dr = ImageDraw.Draw(img)
    dr.text((12, 8), 'PaletteSolver check: reference colour | Kubelka-Munk mixture  (CIEDE2000, recipe in volume parts)',
            fill=(30, 30, 30))
    # palette row
    x = 12
    for p in solver.pigments:
        c = tuple(int(v * 255 + 0.5) for v in p.rgb(3.0))
        dr.rectangle([x, 28, x + 40, 58], fill=c)
        dr.text((x, 60), p.name.replace('titanium ', 'ti. ').replace('cadmium ', 'cad. ')[:14],
                fill=(40, 40, 40))
        x += 110
    y0 = 28 + pal_h + 10
    des = []
    for i, ((paint, rec), c) in enumerate(zip(res, cols)):
        x = 12 + (i % ncol) * cw
        y = y0 + (i // ncol) * (th + 8)
        t = tuple(int(v * 255 + 0.5) for v in c)
        m = tuple(int(v * 255 + 0.5) for v in rec.rgb)
        dr.rectangle([x, y, x + sw // 2, y + th], fill=t)
        dr.rectangle([x + sw // 2, y, x + sw, y + th], fill=m)
        txt = ' + '.join(f'{k} {nm.replace("titanium white", "white").replace("cadmium ", "cad ").replace("ultramarine blue", "ultram.").replace("alizarin crimson", "alizarin").replace("burnt sienna", "b.sienna").replace("phthalo blue", "phthalo").replace("yellow ochre", "ochre").replace("ivory black", "black")}'
                         for nm, k in sorted(rec.parts(), key=lambda t: -t[1]))
        dr.text((x + sw + 6, y + 1), f'dE {rec.delta_e:4.1f}', fill=(80, 40, 40))
        dr.text((x + sw + 6, y + 12), txt[:58], fill=(40, 40, 40))
        des.append(rec.delta_e)
    img.save(path)
    return np.array(des)


# --------------------------------------------------------------------------
def region_flow(masks, H, W, seed=0):
    """Per-region stroke direction overrides: near-vertical grass strokes in
    the meadow, horizontal strokes on the water; elsewhere the strokes follow
    the image structure."""
    g = np.random.default_rng(seed)
    wob = nz.fbm((H, W), scale=60, octaves=3, seed=g)
    angle = np.zeros((H, W))
    angle = np.where(masks['meadow'] > 0.5, -78.0 + 14.0 * wob, angle)
    angle = np.where(masks['lake'] > 0.5, 0.0 + 4.0 * wob, angle)
    weight = 0.6 * masks['meadow'] * (1 - np.clip(masks['flowers'] * 2, 0, 1)) + 0.75 * masks['lake']
    weight = ndimage.gaussian_filter(weight, 4)
    return angle, weight


def paint_style(ref, masks, style, W, H, seed=11, solver=None, progress=None):
    cv = ps.Canvas(W, H, ground='canvas', seed=7)
    if solver is None:
        solver = PaletteSolver(DEFAULT_PALETTE, max_pigments=3, thickness=3.0,
                               ground=cv.ground.reflectance.astype(np.float64), cache_step=1.0)
    angle, weight = region_flow(masks, H, W)
    importance = 1.0 + 0.5 * np.clip(masks['flowers'] * 2, 0, 1) + 0.15 * masks['trees']
    painter = AutoPainter(ref, style=style, palette=solver, seed=seed, progress=progress,
                          flow=angle, flow_weight=weight, importance=importance)
    res = painter.paint(cv)
    return cv, res


RENDER = dict(relief=0.8, specular=0.4)


def main():
    style = 'impressionist'
    small = '--small' in sys.argv
    for a in sys.argv[1:]:
        if not a.startswith('--'):
            style = a
    W, H = (500, 375) if small else (1000, 750)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    ref, masks = landscape(W, H, seed=3, return_masks=True)
    Image.fromarray((ref * 255 + 0.5).astype(np.uint8)).save(os.path.join(OUT, 'painterly_reference.png'))
    print(f'reference synthesised in {time.time() - t0:.1f}s')

    # limited palette: titanium white, cadmium yellow, yellow ochre, cadmium red,
    # alizarin crimson, ultramarine, phthalo blue, burnt sienna, ivory black
    ground = ps.Canvas(8, 8, ground='canvas', seed=7).ground.reflectance.astype(np.float64)
    t1 = time.time()
    solver = PaletteSolver(DEFAULT_PALETTE, max_pigments=3, thickness=3.0, ground=ground,
                           cache_step=1.0)
    print(f'palette solver: {solver.table_size} tabulated mixtures in {time.time() - t1:.2f}s')
    des = palette_check(solver, ref, os.path.join(OUT, 'painterly_palette.png'))
    print(f'palette check: mean dE00 {des.mean():.2f}, max {des.max():.2f}')

    last = [0.0]

    def progress(info):
        if info['elapsed'] - last[0] > 15 or info['message'] == 'done':
            last[0] = info['elapsed']
            print(f"  {100 * info['fraction']:5.1f}%  {info['message']}  ({info['elapsed']:.0f}s)")

    cv, res = paint_style(ref, masks, style, W, H, solver=solver, progress=progress)
    print(res)
    out = os.path.join(OUT, 'painterly_demo.png')
    cv.save(out, **RENDER)
    img = Image.open(out)
    cx, cy = int(0.04 * W), int(0.62 * H)
    crop = img.crop((cx, cy, cx + W // 3, cy + H // 3)).resize((2 * (W // 3), 2 * (H // 3)), Image.LANCZOS)
    crop.save(os.path.join(OUT, 'painterly_detail.png'))
    print(f'painting written to {out}  ({time.time() - t0:.1f}s total)')

    if '--styles' in sys.argv:
        # contact sheet of all presets at half resolution
        w2, h2 = W // 2, H // 2
        ref2, masks2 = landscape(w2, h2, seed=3, return_masks=True)   # same scene, half size
        from pinsel.painterly import STYLES
        tiles = []
        for name in STYLES:
            t = time.time()
            c2, r2 = paint_style(ref2, masks2, name, w2, h2, solver=solver)
            print(f'  style {name}: {r2.total_strokes} strokes, {time.time() - t:.1f}s')
            tile = Image.fromarray((c2.render(**RENDER) * 255 + 0.5).astype(np.uint8))
            ImageDraw.Draw(tile).text((8, 6), name, fill=(255, 255, 255))
            tiles.append(tile)
        tiles.insert(0, Image.fromarray((ref2 * 255 + 0.5).astype(np.uint8)))
        sheet = Image.new('RGB', (3 * w2, 2 * h2))
        for i, tile in enumerate(tiles):
            sheet.paste(tile, ((i % 3) * w2, (i // 3) * h2))
        sheet.save(os.path.join(OUT, 'painterly_styles.png'))
        print('styles contact sheet written')


if __name__ == '__main__':
    main()
