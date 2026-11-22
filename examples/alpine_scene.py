"""
Procedural reference for "Alpenglühen am Bergsee" (alpine lake at dusk).

Produces an sRGB reference image plus region masks and helper geometry that
the painting script uses for composition and stroke direction.
"""
import numpy as np
from scipy import ndimage

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pinsel.noise import fbm, spectral_noise, noise1d, blur


def hexc(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def lerp(a, b, t):
    t = np.asarray(t)[..., None] if np.ndim(t) else t
    return a + (b - a) * t


def smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def gradient_stops(y, stops):
    ys = np.array([s[0] for s in stops])
    cs = np.array([hexc(s[1]) for s in stops])
    return np.stack([np.interp(y, ys, cs[:, k]) for k in range(3)], axis=-1)


def make_scene(W=1600, H=1100, seed=7):
    g = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    yn, xn = yy / H, xx / W
    horizon = int(0.585 * H)
    S = {}

    # ------------------------------------------------------------- sky
    sky = gradient_stops(yn, [(0.0, '#1f2f6a'), (0.12, '#2f4688'), (0.26, '#5f7cbd'),
                              (0.34, '#9aa6cf'), (0.40, '#dcaab8'), (0.46, '#f0bca8'),
                              (0.52, '#d8b0b8'), (0.585, '#a9a3c4')])
    # soft glow toward the lower right (afterglow)
    glow = np.exp(-(((xn - 0.75) / 0.45) ** 2 + ((yn - 0.47) / 0.16) ** 2))
    sky = lerp(sky, hexc('#f7c9a6'), 0.35 * glow)
    # streaky clouds lit from below
    cn = spectral_noise((H, W), beta=3.1, seed=g, aniso=(1.0, 4.0))
    band = np.exp(-((yn - 0.2) / 0.09) ** 2) + 0.6 * np.exp(-((yn - 0.33) / 0.04) ** 2)
    cloud = smooth(0.9, 1.9, cn + 1.1 * band - 0.75) * (yn < 0.42)
    cloud_top = np.clip(ndimage.shift(cloud, (6, 0), order=1) - cloud * 0.5, 0, 1)
    ccol = lerp(hexc('#b49cc0'), hexc('#f6b7a0'), smooth(0.05, 0.38, yn))
    sky = lerp(sky, ccol, 0.85 * cloud)
    sky = lerp(sky, hexc('#ffd9c0'), 0.35 * np.clip(cloud - cloud_top, 0, 1) * smooth(0.1, 0.35, yn))
    S['cloud'] = cloud

    # ------------------------------------------------------------- far mountains
    x = np.arange(W) / W
    peaks = [(0.08, 0.36, 1.5), (0.22, 0.27, 1.25), (0.36, 0.33, 1.4), (0.50, 0.24, 1.15),
             (0.62, 0.135, 1.05), (0.71, 0.20, 1.3), (0.82, 0.255, 1.2), (0.95, 0.33, 1.4)]
    ridge = np.full(W, 0.5)
    for px, py, k in peaks:
        ridge = np.minimum(ridge, py + k * np.abs(x - px) ** 1.05)
    ridge += 0.018 * noise1d(W, scale=60, seed=g) + 0.006 * noise1d(W, scale=9, seed=g)
    ridge = np.minimum(ridge, 0.47)
    ridge_px = ridge * H
    mtn = (yy > ridge_px[None, :]).astype(np.float64)
    depth = (yy - ridge_px[None, :])                    # px below the ridge
    # rock structure: vertical gullies + fractal detail
    gul = spectral_noise((H, W), beta=2.2, seed=g, aniso=(4.0, 1.0))
    det = fbm((H, W), scale=40, octaves=5, seed=g)
    hfield = -depth / H * 6 + 0.35 * gul + 0.12 * det
    gy, gx = np.gradient(blur(hfield, 2.0))
    lit = np.clip(0.5 + 9.0 * (-gx * 1.0 - gy * 0.3) * W / 1600, 0, 1)   # faces turned left/front catch the glow
    snowline = 0.10 * H + 0.06 * H * fbm((H, W), scale=120, octaves=3, seed=g)
    gul_f = spectral_noise((H, W), beta=1.7, seed=g, aniso=(3.0, 1.0))
    snow = smooth(0.0, 0.35, 0.55 * gul_f + 0.3 * det + 0.25 * (lit - 0.5) + 0.25) \
        * smooth(snowline + 50, snowline - 40, depth)
    snow *= mtn
    rock = lerp(hexc('#545287'), hexc('#b47f86'), lit)
    snowc = lerp(hexc('#9d9ccc'), hexc('#fbd0bd'), lit ** 0.8)
    mcol = lerp(rock, snowc, snow)
    # haze toward the valley
    haze = smooth(0.33 * H, 0.50 * H, yy) * 0.55
    mcol = lerp(mcol, hexc('#9f97bf'), haze)
    S['far_mtn'] = mtn
    S['snow'] = snow
    S['lit'] = lit
    S['ridge'] = ridge_px

    # ------------------------------------------------------------- mid forested slopes
    r2 = 0.475 + 0.035 * noise1d(W, scale=150, seed=g) + 0.012 * noise1d(W, scale=25, seed=g)
    r2 += 0.05 * np.exp(-((x - 0.12) / 0.12) ** 2) * -1 + 0.04 * np.exp(-((x - 0.9) / 0.1) ** 2) * -1
    r2_px = r2 * H
    mid = (yy > r2_px[None, :]).astype(np.float64)
    tn = spectral_noise((H, W), beta=1.6, seed=g, aniso=(2.0, 1.0))
    midc = lerp(hexc('#2c4450'), hexc('#3d5a5c'), np.clip(0.5 + 0.3 * tn, 0, 1))
    toplight = smooth(30, 0, yy - r2_px[None, :]) * 0.6
    midc = lerp(midc, hexc('#7a7a72'), toplight)
    midc = lerp(midc, hexc('#6c7698'), 0.35 * smooth(0.60 * H, 0.47 * H, yy))
    S['mid'] = mid
    S['mid_ridge'] = r2_px

    # ------------------------------------------------------------- pines along the shore
    trees = np.zeros((H, W))
    tree_list = []
    dens = 0.65 + 0.35 * np.sin(x * 6.3 + 1.0)
    for _ in range(320):
        tx = g.uniform(0, W)
        if g.uniform() > dens[int(tx) % W] * (0.25 if 0.42 < tx / W < 0.58 else 1.0):
            continue
        base = horizon + g.uniform(-4, 6)
        h = g.uniform(45, 150) * (1.25 if (tx / W < 0.25 or tx / W > 0.85) else 1.0)
        wdt = h * g.uniform(0.22, 0.3)
        tree_list.append((tx, base, h, wdt))
    tree_list.sort(key=lambda t: t[1] - t[2])
    for tx, base, h, wdt in tree_list:
        top = base - h
        y0, y1 = max(0, int(top)), min(H, int(base) + 1)
        x0, x1 = max(0, int(tx - wdt * 1.1)), min(W, int(tx + wdt * 1.1) + 1)
        sy, sx = yy[y0:y1, x0:x1], xx[y0:y1, x0:x1]
        rel = np.clip((sy - top) / h, 0, 1)
        jag = 0.25 * np.sin((sy - top) * 0.35 + tx) ** 2
        half = wdt * (rel ** 0.9) * (0.75 + jag)
        m = (np.abs(sx - tx) < half) & (sy > top) & (sy < base)
        trees[y0:y1, x0:x1][m] = 1.0
    treec = lerp(hexc('#16251f'), hexc('#24372e'), np.clip(0.5 + 0.4 * tn, 0, 1))
    S['trees'] = trees
    S['tree_list'] = tree_list

    # ------------------------------------------------------------- boathouse
    bx, bw, bh = 0.735 * W, 96, 46
    by = horizon - 2
    house = np.zeros((H, W))
    roof = np.zeros((H, W))
    walls = (np.abs(xx - bx) < bw / 2) & (yy > by - bh) & (yy < by)
    roofm = (yy > by - bh - 40 + np.abs(xx - bx) * 0.55) & (yy < by - bh + 6) & (np.abs(xx - bx) < bw / 2 + 12)
    house[walls] = 1
    roof[roofm] = 1
    win = (np.abs(xx - (bx + 18)) < 7) & (np.abs(yy - (by - 24)) < 8)
    door = (np.abs(xx - (bx - 20)) < 13) & (yy > by - 34) & (yy < by)
    chim = (np.abs(xx - (bx + 30)) < 5) & (yy > by - bh - 38) & (yy < by - bh - 14)
    house[chim] = 1
    S['house'] = np.clip(house + roof, 0, 1)
    S['house_geom'] = (bx, by, bw, bh)
    S['chimney'] = (bx + 30, by - bh - 38)
    # a wisp of smoke drifting up and to the right
    t = np.linspace(0, 1, 60)
    smoke = np.column_stack([bx + 30 + 120 * t ** 1.4 + 10 * np.sin(t * 7),
                             by - bh - 40 - 190 * t + 8 * np.sin(t * 5)])
    S['smoke_path'] = smoke

    # ------------------------------------------------------------- compose upper half
    img = sky.copy()
    img = lerp(img, mcol, mtn)
    img = lerp(img, midc, mid)
    img = lerp(img, treec, trees)
    img[walls] = hexc('#5b3a29')
    img[roofm] = hexc('#3b3236')
    img[door] = hexc('#2a1d18')
    img[win] = hexc('#ffb54a')
    img[chim] = hexc('#5a5258')
    for i, (sx, sy) in enumerate(smoke):
        r = 5 + 22 * (i / len(smoke))
        a = 0.45 * (1 - i / len(smoke)) ** 0.8
        m = ((xx - sx) ** 2 + (yy - sy) ** 2) < r * r
        img[m] = lerp(img[m], hexc('#d9d0dc'), a * 0.25)

    # ------------------------------------------------------------- lake
    water = (yy >= horizon)
    ry = np.clip(2 * horizon - yy, 0, H - 1)
    rip = 3.0 * noise1d(H, scale=6, seed=g)[:, None] * smooth(horizon, H, yy)
    rx = np.clip(xx + rip * np.sin(yy * 0.9), 0, W - 1)
    refl = np.stack([ndimage.map_coordinates(img[..., k], [ry, rx], order=1) for k in range(3)], -1)
    refl = np.stack([ndimage.gaussian_filter1d(refl[..., k], 2.5, axis=0) for k in range(3)], -1)
    dist = smooth(horizon, H, yy)
    watc = lerp(hexc('#2f5560'), hexc('#1d3438'), dist)
    lake = lerp(refl * 0.86, watc, 0.15 + 0.5 * dist)
    # wind lines: pale horizontal streaks
    wl = spectral_noise((H, W), beta=2.0, seed=g, aniso=(1.0, 30.0))
    streak = smooth(1.1, 2.0, wl) * (yy > horizon + 15)
    lake = lerp(lake, hexc('#b9b2cf'), 0.45 * streak)
    img = np.where(water[..., None], lake, img)
    S['water'] = water.astype(np.float64)
    S['horizon'] = horizon
    S['streak'] = streak

    # ------------------------------------------------------------- jetty & rowing boat
    # jetty: planks run from the hut towards the viewer (simple perspective)
    j_far = (bx - 30, horizon + 1)       # centre of the far end (at the hut)
    j_near = (bx - 130, horizon + 74)    # centre of the near end
    planks = []
    for i in range(16):
        t = i / 15
        cx = j_far[0] + (j_near[0] - j_far[0]) * t
        cy = j_far[1] + (j_near[1] - j_far[1]) * t
        half = 16 + 26 * t
        planks.append((cx - half, cx + half, cy, (3.6 + 1.6 * t) * 0.85))
    jetty = np.zeros((H, W), bool)
    jetty_refl = np.zeros((H, W), bool)
    for x0, x1, cy, th in planks:
        m = (xx >= x0) & (xx <= x1) & (np.abs(yy - cy) <= th / 2 + 0.5)
        jetty |= m
    posts = []
    for t in (0.15, 0.45, 0.75, 1.0):
        cx = j_far[0] + (j_near[0] - j_far[0]) * t
        cy = j_far[1] + (j_near[1] - j_far[1]) * t
        half = 16 + 26 * t
        for sx in (-1, 1):
            px_ = cx + sx * (half - 3)
            ln = 6 + 14 * t
            posts.append((px_, cy, ln, 1.5 + 2 * t))
            m = (np.abs(xx - px_) <= 1.5 + 2 * t) & (yy >= cy) & (yy <= cy + ln)
            jetty |= m
            jetty_refl |= (np.abs(xx - px_) <= 1.5 + 2 * t) & (yy > cy + ln) & (yy <= cy + 2.4 * ln)
    shade_j = np.clip((yy - horizon) / 80, 0, 1)
    img[jetty] = lerp(hexc('#8b6a4c'), hexc('#5a4030'), shade_j)[jetty]
    img[jetty_refl] = lerp(img[jetty_refl], hexc('#2a2420'), 0.6)
    S['jetty'] = jetty.astype(np.float64)
    S['planks'] = planks
    S['posts'] = posts
    # rowing boat moored at the end of the jetty
    bcx, bcy, blen = j_near[0] - 95, j_near[1] + 30, 84
    u = (xx - bcx) / (blen / 2)
    hull_top = bcy - 6 + 3 * u ** 2
    hull_bot = bcy + 9 - 7 * u ** 4
    hull = (np.abs(u) <= 1) & (yy >= hull_top) & (yy <= hull_bot)
    inside = (np.abs(u) <= 0.85) & (yy >= hull_top) & (yy <= hull_top + 4)
    stripe = hull & (np.abs(yy - (hull_top + 7)) <= 1.3)
    img[hull] = hexc('#a83226')
    img[stripe] = hexc('#efe6d8')
    img[inside] = hexc('#6a4a34')
    hull_ref = (np.abs(u) <= 1) & (yy > hull_bot) & (yy <= hull_bot + (hull_bot - hull_top) * 0.9)
    img[hull_ref] = lerp(img[hull_ref], hexc('#5a2420'), 0.55)
    S['boat'] = (bcx, bcy, blen)
    S['boat_mask'] = (hull | inside).astype(np.float64)

    # ------------------------------------------------------------- foreground meadow
    edge = H * (0.80 + 0.22 * (x / 0.55) ** 1.6) + 10 * noise1d(W, scale=40, seed=g)
    edge = np.where(x < 0.62, edge, H + 50)
    meadow = (yy > edge[None, :]).astype(np.float64)
    mn = fbm((H, W), scale=30, octaves=4, seed=g)
    grass = lerp(hexc('#3a4626'), hexc('#7b8a3e'), np.clip(0.5 + 0.35 * mn, 0, 1))
    grass = lerp(grass, hexc('#a39a55'), 0.4 * smooth(edge[None, :] + 60, edge[None, :], yy))
    img = lerp(img, grass, meadow)
    # shoreline rocks
    rocks = np.zeros((H, W))
    for _ in range(9):
        cx = g.uniform(0.02, 0.5) * W
        cy = np.interp(cx, np.arange(W), edge) + g.uniform(-8, 10)
        rx_, ry_ = g.uniform(20, 55), g.uniform(12, 26)
        m = ((xx - cx) / rx_) ** 2 + ((yy - cy) / ry_) ** 2 < 1
        rocks[m] = 1
        shade = np.clip(0.5 - (xx - cx) / rx_ * 0.4 - (yy - cy) / ry_ * 0.4, 0, 1)
        img[m] = lerp(hexc('#4a4658'), hexc('#a89aa6'), shade[m])
    S['meadow'] = meadow
    S['rocks'] = rocks
    S['edge'] = edge
    flowers = []
    for _ in range(260):
        fx = g.uniform(0, 0.6 * W)
        top = np.interp(fx, np.arange(W), edge)
        if top >= H - 14:
            continue
        fy = g.uniform(top + 8, H - 4)
        col = g.choice(['#e0507f', '#f3c844', '#f3c844', '#4a5fc8', '#f4efe6', '#d9433a'])
        flowers.append((fx, fy, col))
    S['flowers'] = flowers
    img = np.clip(img, 0, 1)
    return img, S


if __name__ == '__main__':
    (ROOT / 'output').mkdir(exist_ok=True)
    from PIL import Image
    img, S = make_scene()
    Image.fromarray((img * 255).astype(np.uint8)).save(ROOT / 'output' / 'alpine_reference.png')
    print('ok')
