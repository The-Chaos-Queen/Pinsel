"""
"Alpenglühen am Bergsee" -- the same view as a pen-and-watercolour sketch.

Classic watercolour order of work: masking fluid on the highlights, a
wet-in-wet variegated sky with clouds lifted out by tissue, granulating
mountain washes charged wet-in-wet, remove the mask, glaze the alpenglow,
then darker shapes wet-on-dry, the lake with soft reflections, details with
a round brush, spatter, and finally a few sepia ink lines.
"""
import sys, time, math
import numpy as np
from scipy import ndimage

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pinsel as ps
from pinsel import pigments as pg, path as P, watercolor as wc, tools as T
from pinsel.pigments import mix
from pinsel.noise import blur
from alpine_scene import make_scene

W, H = 1200, 825
OUT = ROOT / 'output'
T0 = time.time()


def log(m):
    print(f'[{time.time() - T0:6.1f}s] {m}', flush=True)


def w(name):
    return pg.get(name, 'watercolor')


def main():
    g = np.random.default_rng(5)
    ref, S = make_scene(W, H)
    hz = S['horizon']
    yy, xx = np.mgrid[0:H, 0:W]
    cv = ps.Canvas(W, H, ground='cold_press', seed=8)
    cv.record(every=40, scale=0.5)

    far = S['far_mtn'] * (1 - np.clip(S['mid'] + S['trees'], 0, 1))
    upper = (yy < hz).astype(float)
    sky = upper * (1 - np.clip(S['far_mtn'] + S['mid'] + S['trees'] + S['house'], 0, 1))
    snowlit = ndimage.binary_opening((S['snow'] * S['lit'] * far) > 0.55, iterations=1)
    bx, by, bw, bh = S['house_geom']
    window = (np.abs(xx - (bx + 14)) < 6) & (np.abs(yy - (by - 18)) < 6)

    # ------------------------------------------------------------ masking fluid
    log('masking fluid on the sunlit snow and the window')
    cv.set_mask(np.clip(blur((snowlit | window).astype(float), 0.7) * 1.4, 0, 1))

    # ------------------------------------------------------------ sky, wet-in-wet
    log('sky: variegated wash wet-in-wet, clouds lifted with tissue')
    wc.wet_paper(cv, upper, water=0.5)
    wc.variegated_wash(cv, upper, [(w('ultramarine'), 0.0, 1.1), (w('cobalt_blue'), 0.3, 0.8),
                                   (w('permanent_rose'), 0.62, 0.45), (w('quinacridone_gold'), 0.9, 0.3)],
                       direction=((0, 0), (0, hz)), concentration=0.55, water=0.7)
    cv.wait(6)
    clouds = blur(S['cloud'] * sky, 2) > 0.35
    wc.blot(cv, clouds, strength=0.8, texture='tissue', scale=9)
    cv.wait(240)

    # ------------------------------------------------------------ mountains
    log('mountains: granulating wash, charged wet-in-wet')
    mtn_paint = mix(w('ultramarine'), 2, w('permanent_rose'), 1, w('burnt_sienna'), 0.3)
    wc.wash(cv, far > 0.5, mtn_paint, concentration=0.55, water=0.6,
            gradient=((0, 140), (0, hz - 60), (0.7, 1.1)))
    cv.wait(4)
    shade = (far > 0.5) & (S['lit'] < 0.35) & (yy < hz - 80)
    cand = np.argwhere(shade)
    for y, x in cand[g.choice(len(cand), size=min(14, len(cand)), replace=False)]:
        wc.drop(cv, x, y, mix(w('ultramarine'), 2, w('burnt_sienna'), 1), radius=g.uniform(10, 18),
                concentration=0.6, water=0.45)
    cv.wait(240)

    # ------------------------------------------------------------ remove the masking fluid, alpenglow glaze
    log('peel off the masking fluid, glaze the alpenglow')
    cv.clear_mask()
    glow = blur((S['snow'] * S['lit'] * far > 0.4).astype(float), 2)
    wc.wash(cv, glow > 0.3, mix(w('permanent_rose'), 1, w('quinacridone_gold'), 0.6), concentration=0.12, water=0.4)
    cv.wait(200)

    # ------------------------------------------------------------ forested slopes and pines
    log('forested slopes')
    mid = S['mid'] * (1 - S['trees'])
    mid[hz:] = 0
    wc.wash(cv, np.clip(mid + S['trees'], 0, 1) > 0.5, mix(w('phthalo_green'), 1, w('ultramarine'), 1.5, w('burnt_sienna'), 1),
            concentration=0.45, water=0.6, gradient=((0, 430), (0, hz), (0.6, 1.2)))
    cv.wait(200)
    log('pines with a round brush')
    rb = ps.Brush('round', 4.5, seed=3)
    dark = mix(w('phthalo_green'), 1, w('burnt_sienna'), 1.4, w('ultramarine'), 0.6)
    trees = [t for t in S['tree_list'] if t[2] > 70]
    for k, (tx, base, h, wdt) in enumerate(trees):
        top = base - h
        rb.load(dark, amount=1.0 + 0.4 * g.uniform(), water=0.8)
        # the spine, pressing harder towards the bottom
        cv.stroke(rb, P.line((tx, top), (tx, base)), pressure=P.ramp(0.15, 0.7))
        # drooping branch tiers, alternating sides
        n = int(h / 8)
        for i in range(n):
            t = (i + 1) / n
            yb = top + t * h * 0.95
            half = wdt * t ** 0.9 * g.uniform(0.75, 1.05)
            side = 1 if i % 2 else -1
            if i % 3 == 0:
                rb.load(dark, amount=1.1, water=0.7)
            cv.stroke(rb, P.quad((tx, yb - 3), (tx + side * half * 0.6, yb - 2), (tx + side * half, yb + half * 0.3)),
                      pressure=P.flick(0.75, 0.25))
            cv.stroke(rb, P.quad((tx, yb), (tx - side * half * 0.5, yb + 1), (tx - side * half * 0.9, yb + half * 0.35)),
                      pressure=P.flick(0.7, 0.25))
        if k % 10 == 9:
            cv.wait(2)
    cv.wait(180)

    # ------------------------------------------------------------ lake
    log('lake: graded wash, soft reflections into the damp wash')
    # leave the shore as white paper (watercolour is transparent), with a small overlap
    shore = ndimage.binary_erosion(np.clip(S['meadow'] + S['rocks'], 0, 1) > 0.5, iterations=6)
    water = (yy > hz).astype(float) * (1 - shore)
    wc.wet_paper(cv, water > 0.5, water=0.45)
    wc.variegated_wash(cv, water > 0.5, [(mix(w('phthalo_green'), 1, w('ultramarine'), 2), 0.0, 0.9),
                                         (mix(w('ultramarine'), 1, w('permanent_rose'), 0.7), 0.45, 0.6),
                                         (mix(w('ultramarine'), 1, w('phthalo_blue'), 0.5), 1.0, 0.8)],
                       direction=((0, hz), (0, H)), concentration=0.5, water=0.6)
    cv.wait(8)
    fl = ps.Brush('flat', 7, seed=4)
    for (tx, base, h, wdt) in trees[::2]:
        fl.load(dark, amount=0.8, water=0.7)
        L = h * 0.8
        cv.stroke(fl, P.line((tx, hz + 3), (tx + g.normal(0, 2), hz + L)), pressure=P.taper(0.1, 0.5, 0.6), angle=0)
    cv.wait(240)
    log('wind lines lifted with a damp brush')
    for i in range(10):
        y = g.uniform(hz + 60, H - 40)
        x0 = g.uniform(300, W - 300)
        L = g.uniform(120, 380)
        wc.lift(cv, wc.rect(x0, y, x0 + L, y + 2.5), strength=0.6, soft=2)
    cv.wait(60)

    # ------------------------------------------------------------ hut, jetty, boat
    log('hut, jetty and the red boat')
    sm = ps.Brush('round', 2.2, seed=6)
    wc.wash(cv, wc.rect(bx - bw / 2, by - bh, bx + bw / 2, by), w('burnt_sienna'), concentration=0.5, water=0.5)
    cv.wait(60)
    for i in range(int(bw / 7)):
        sm.load(mix(w('burnt_sienna'), 2, w('burnt_umber'), 1), amount=0.8, water=0.5)
        x = bx - bw / 2 + 3 + i * 7
        cv.stroke(sm, P.line((x, by - bh + 2), (x, by)), pressure=0.5)
    cv.wait(30)
    sm.load(mix(w('burnt_umber'), 1, w('ultramarine'), 1), amount=1.4, water=0.6)
    for i in range(10):
        t = i / 9
        y = by - bh - 30 + t * 30
        half = (t * 30) / 0.55 * 0.5 + 6
        cv.stroke(sm, P.line((bx - min(half, bw / 2 + 10), y), (bx + min(half, bw / 2 + 10), y)), pressure=0.8)
    sm.load(w('cadmium_yellow'), amount=1.0, water=0.7)
    cv.stroke(sm, P.line((bx + 10, by - 18), (bx + 18, by - 18)), pressure=0.7)
    for (x0, x1, cy, th) in S['planks']:
        sm.load(mix(w('raw_sienna'), 1, w('burnt_sienna'), 0.5), amount=1.0, water=0.6)
        cv.stroke(sm, P.line((x0, cy), (x1, cy)), pressure=0.7)
    bcx, bcy, blen = S['boat']
    sm.load(w('cadmium_red'), amount=1.5, water=0.6)
    for k in range(4):
        xs = np.linspace(-1, 1, 20)
        cv.stroke(sm, np.column_stack([bcx + xs * blen / 2 * 0.95, bcy - 3 + 3 * xs ** 2 + k * 3]), pressure=0.8)
    cv.wait(90)

    # ------------------------------------------------------------ meadow
    log('rocks: granulating grey')
    wc.wash(cv, S['rocks'] > 0.5, mix(w('ultramarine'), 1, w('burnt_sienna'), 1, w('permanent_rose'), 0.3),
            concentration=0.35, water=0.5)
    cv.wait(120)
    log('meadow: wash, dry-brush grass, spattered flowers')
    meadow = S['meadow'] > 0.5
    wc.wash(cv, meadow & (S['rocks'] < 0.5), mix(w('sap_green'), 2, w('hansa_yellow'), 1), concentration=0.6, water=0.6,
            gradient=((0, H * 0.8), (0, H), (0.6, 1.2)))
    cv.wait(5)
    wc.salt(cv, meadow, density=0.0015)
    cv.wait(200)
    grass = ps.Brush('rigger', 2.0, seed=7)
    for i in range(260):
        x = g.uniform(0, 0.6 * W)
        top = np.interp(x, np.arange(W), S['edge'])
        if top > H - 6:
            continue
        y = g.uniform(top, H)
        grass.load(mix(w('sap_green'), 1, w('ultramarine'), 0.5, w('burnt_sienna'), 0.3), amount=1.0, water=0.35)
        L = g.uniform(15, 40)
        a = -np.pi / 2 + g.normal(0.1, 0.3)
        cv.stroke(grass, P.line((x, y), (x + L * math.cos(a), y + L * math.sin(a))), pressure=P.flick(0.8, 0.2))
    for name in ('quinacridone_magenta', 'cadmium_yellow', 'cobalt_blue', 'cadmium_red'):
        wc.spatter(cv, wc.polygon([(0, H * 0.82), (W * 0.5, H), (0, H)]), w(name), density=0.0004, size=(0.8, 2.6))
    cv.wait(120)

    # ------------------------------------------------------------ ink lines
    log('a few sepia ink lines')
    pen = T.Pen('pointed', width=0.9, flex=1.6, seed=9)
    ink = mix(w('burnt_umber'), 2, w('ivory_black'), 1).as_medium('ink')
    pen.load(ink)
    pen.line(cv, [(bx - bw / 2, by), (bx - bw / 2, by - bh), (bx + bw / 2, by - bh), (bx + bw / 2, by)], pressure=0.5)
    pen.line(cv, [(bx - bw / 2 - 10, by - bh + 4), (bx, by - bh - 34), (bx + bw / 2 + 10, by - bh + 4)], pressure=0.6)
    for (x0, x1, cy, th) in S['planks'][::3]:
        pen.load(ink)
        pen.line(cv, [(x0, cy + th / 2), (x1, cy + th / 2)], pressure=0.4)
    xs = np.linspace(-1, 1, 24)
    pen.line(cv, np.column_stack([bcx + xs * blen / 2, bcy + 9 - 7 * xs ** 4]), pressure=0.5)
    cv.wait(60)

    log('drying')
    cv.dry()
    cv.snapshot()
    cv.save(f'{OUT}/alpine_watercolor.png')
    cv.save_timelapse(f'{OUT}/alpine_watercolor_timelapse.gif', fps=8, max_frames=70, width=600)
    log('done')


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    main()
