"""
Demo of the non-brush tools in :mod:`pinsel.tools`.

Writes
  output/tools_demo.png        1200 x 800 on primed canvas: palette knife
                               (impasto over dried brushwork, scraping,
                               sgraffito), airbrush with frisket, sea sponge
                               foliage, splatter, finger smudge, blender,
                               glaze and masking tape
  output/tools_demo_paper.png  1200 x 560 on cold-press watercolour paper:
                               technical / dip pens, stippling, hatching and
                               the fluid (watercolour / ink) variants of
                               airbrush, splatter, sponge and glaze

Run:  python examples/tools_demo.py
"""
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import pinsel as ps
from pinsel import path as P, pigments as pg, tools as T

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output')
os.makedirs(OUT, exist_ok=True)

FONT = None
for f in ('/usr/share/fonts/opentype/inter/Inter-Medium.otf',
          '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
    if os.path.exists(f):
        FONT = ImageFont.truetype(f, 13)
        break
if FONT is None:
    FONT = ImageFont.load_default()


def label(img, panels):
    d = ImageDraw.Draw(img, 'RGBA')
    for (x0, y0, x1, y1), text in panels:
        w = d.textlength(text, font=FONT)
        d.rectangle([x0 + 4, y1 - 22, x0 + 14 + w, y1 - 4], fill=(255, 255, 255, 200))
        d.text((x0 + 9, y1 - 21), text, fill=(40, 40, 40, 255), font=FONT)
    return img


def oil(p):
    return p.as_medium('oil') if p.medium != 'oil' else p


def acr(p):
    return p.as_medium('acrylic')


def wc(p):
    return p.as_medium('watercolor')


# ==========================================================================
# Canvas image
# ==========================================================================
def panel_knife_landscape(cv, x0, y0, g):
    """Knife-painted mountain, sky and meadow over dried brushwork."""
    W, H = 380, 370
    # -- brushed underpainting (sky, hills, field), allowed to dry
    b = ps.Brush.hog(13, seed=11)
    for i in range(10):
        t = i / 9
        col = pg.mix(pg.TITANIUM_WHITE, 2.0 + 3 * t, pg.CERULEAN_BLUE, 1.2 - t, pg.NAPLES_YELLOW, 0.2 + t)
        b.load(col, 1.3)
        y = y0 + 12 + i * 26
        cv.stroke(b, P.wobble(P.line((x0 - 5, y), (x0 + W + 5, y + g.normal(0, 4))), 2, seed=i),
                  pressure=0.75)
    b = ps.Brush.flat(14, seed=12)
    for i in range(6):
        b.load(pg.mix(pg.SAP_GREEN, 1, pg.YELLOW_OCHRE, 2, pg.TITANIUM_WHITE, 1), 1.4)
        y = y0 + 265 + i * 19
        cv.stroke(b, P.line((x0, y), (x0 + W, y + g.normal(0, 3))), pressure=0.8)
    cv.dry()
    kn = T.PaletteKnife(length=52, seed=13)

    def stiff(p):
        return oil(p).impasto(1.4)
    def bent(p0, p1, bend):
        p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
        d = p1 - p0
        m = (p0 + p1) / 2 + np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9) * bend
        return P.quad(p0, m, p1)
    # -- sky: a mosaic of overlapping knife strokes, cool at the top, warm low
    rows = 8
    for r in range(rows):
        t = r / (rows - 1)
        y = y0 + 14 + r * 19
        x = x0 - 40 + g.uniform(0, 50)
        while x < x0 + W:
            c1 = pg.mix(pg.TITANIUM_WHITE, 2 + 5 * t, pg.CERULEAN_BLUE, 1.5 - t, pg.ULTRAMARINE, 0.35 * (1 - t),
                        pg.NAPLES_YELLOW, 0.8 * t, pg.PERMANENT_ROSE, 0.3 * t * g.uniform())
            c2 = pg.mix(c1, 2, pg.TITANIUM_WHITE, 1)
            kn.load([(stiff(c1), 1), (stiff(c2), 0.5)], g.uniform(0.6, 1.0), mix=0.3)
            L = g.uniform(90, 150)
            kn.spread(cv, bent((x, y + g.normal(0, 3)), (x + L, y + g.normal(0, 4)), g.normal(0, 5)),
                      pressure=g.uniform(0.3, 0.5), rel_angle=g.uniform(76, 88))
            x += L * g.uniform(0.6, 0.85)
    # -- mountain: each face pulled down the fall line from its ridge; the
    #    blade is held parallel to the ridge, so its touch-down edge draws
    #    the crisp silhouette
    peak = np.array([x0 + 170.0, y0 + 70.0])
    base_y = y0 + 262.0
    lx, rx = x0 - 10.0, x0 + 390.0
    lit = pg.mix(pg.TITANIUM_WHITE, 4, pg.NAPLES_YELLOW, 2, pg.PERMANENT_ROSE, 0.5, pg.CADMIUM_ORANGE, 0.15)
    litb = pg.mix(pg.TITANIUM_WHITE, 5, pg.NAPLES_YELLOW, 1, pg.PERMANENT_ROSE, 0.3)
    shadow = pg.mix(pg.ULTRAMARINE, 1, pg.BURNT_UMBER, 0.45, pg.DIOXAZINE_PURPLE, 0.2, pg.TITANIUM_WHITE, 1.3)
    deep = pg.mix(shadow, 2, pg.ULTRAMARINE, 0.6, pg.IVORY_BLACK, 0.15)
    faces = [(np.linspace(peak[0] + 30, rx - 10, 8), rx, [(shadow, 1), (deep, 0.7)]),
             (np.linspace(peak[0] - 18, lx + 25, 9), lx, [(lit, 1), (litb, 0.55)])]
    for xs, xend, cols in faces:
        ang = np.degrees(np.arctan2(base_y - 20 - peak[1], xend - peak[0]))
        for xc in xs:
            yr = peak[1] + (base_y - 20 - peak[1]) * (xc - peak[0]) / (xend - peak[0])
            kn.load([(stiff(c), w) for c, w in cols], g.uniform(0.9, 1.2), mix=0.35)
            p0 = np.array([xc, yr + 2])
            p1 = np.array([xc + g.normal(0, 6), base_y + g.uniform(-5, 15)])
            kn.spread(cv, bent(p0, p1, g.normal(0, 4)), pressure=g.uniform(0.28, 0.4),
                      angle=ang + g.normal(0, 2))
    # sunlit snow accents near the peak
    ang = np.degrees(np.arctan2(base_y - 20 - peak[1], lx - peak[0]))
    for k in range(3):
        kn.load(stiff(pg.TITANIUM_WHITE), 0.7)
        xc = peak[0] - 8 - 26 * k
        yr = peak[1] + (base_y - 20 - peak[1]) * (xc - peak[0]) / (lx - peak[0])
        kn.spread(cv, P.line((xc, yr + 1), (xc + 4, yr + 30 + 10 * k)), pressure=0.45, angle=ang)
    # -- foreground meadow: rows of curved strokes with broken colour
    for r in range(8):
        y = y0 + 258 + r * 15
        x = x0 - 50 + g.uniform(0, 40)
        while x < x0 + W:
            c = pg.mix(pg.SAP_GREEN, 0.6 + g.uniform(0, 0.8) + 0.1 * r, pg.CADMIUM_YELLOW, 1.8 - 0.12 * r,
                       pg.TITANIUM_WHITE, 0.5 * g.uniform(), pg.BURNT_SIENNA, 0.04 * r)
            kn.load([(stiff(c), 1), (stiff(pg.mix(pg.CADMIUM_YELLOW_LIGHT, 1, pg.TITANIUM_WHITE, 0.5)), 0.3)],
                    g.uniform(0.6, 1.1), mix=0.4)
            L = g.uniform(80, 150)
            kn.spread(cv, bent((x, y + g.normal(0, 3)), (x + L, y + g.normal(0, 5)), g.normal(0, 4)),
                      pressure=g.uniform(0.3, 0.5), rel_angle=g.uniform(74, 88))
            x += L * g.uniform(0.6, 0.9)


def panel_scrape(cv, x0, y0, g):
    """Scraping wet impasto off a dried warm layer; sgraffito."""
    W, H = 380, 370
    b = ps.Brush.flat(22, seed=21)
    warm = [pg.CADMIUM_YELLOW, pg.CADMIUM_ORANGE, pg.CADMIUM_RED_LIGHT, pg.CADMIUM_YELLOW_LIGHT]
    for i in range(10):
        b.load(warm[i % 4], 1.4)
        y = y0 + 12 + i * 37
        cv.stroke(b, P.line((x0, y), (x0 + W, y + g.normal(0, 3))), pressure=0.85)
    cv.dry()
    # thick wet dark impasto over everything
    hb = ps.Brush.hog(18, seed=22)
    dark = pg.mix(pg.PHTHALO_BLUE, 1, pg.BURNT_UMBER, 1.2, pg.TITANIUM_WHITE, 0.25)
    for i in range(13):
        hb.load([(dark, 1), (pg.mix(dark, 1, pg.PHTHALO_GREEN, 0.4), 0.4)], 2.2, mix=0.5)
        y = y0 + 10 + i * 28
        cv.stroke(hb, P.wobble(P.line((x0 - 5, y), (x0 + W + 5, y + g.normal(0, 4))), 2, seed=40 + i),
                  pressure=0.95)
    # three diagonal scrapes: the knife collects the dark paint ...
    kn = T.PaletteKnife(length=54, seed=23)
    for k in range(3):
        xs = x0 + 40 + k * 100
        kn.scrape(cv, P.line((xs, y0 + 15), (xs + 70, y0 + 230)), pressure=0.92)
    # ... and smears it into the next mark: white taken up with the dirty knife
    kn.load(oil(pg.TITANIUM_WHITE).impasto(1.4), 0.5, clean=False)
    kn.spread(cv, P.line((x0 + 25, y0 + 290), (x0 + 195, y0 + 300)), pressure=0.35)
    kn.spread(cv, P.line((x0 + 35, y0 + 335), (x0 + 190, y0 + 340)), pressure=0.35)
    # sgraffito: grass blades and a spiral scratched with the tip
    for k in range(14):
        bx = x0 + 215 + k * 11 + g.normal(0, 2)
        by = y0 + 352
        top = (bx + g.normal(0, 18), by - g.uniform(60, 110))
        mid = ((bx + top[0]) / 2 + g.normal(0, 6), (by + top[1]) / 2)
        kn.sgraffito(cv, P.spline([(bx, by), mid, top]), width=2.6, pressure=P.taper(0.05, 0.6))
    kn.sgraffito(cv, P.spiral((x0 + 300, y0 + 190), 3, 34, turns=3.2), width=2.4)


def panel_airbrush(cv, x0, y0, g):
    """Airbrushed sky gradient, frisket-masked sphere with cast shadow."""
    W, H = 380, 370
    ab = T.Airbrush(seed=31)
    # sky: passes getting lighter towards the horizon
    ab.load(acr(pg.mix(pg.ULTRAMARINE, 1, pg.PHTHALO_BLUE, 0.3)))
    for i in range(9):
        y = y0 + 10 + i * 13
        pts = P.line((x0 - 20, y), (x0 + W + 20, y))
        ab.spray(cv, pts if i % 2 == 0 else pts[::-1], distance=55, pressure=0.95 - i * 0.09,
                 wobble=2)
    ab.load(acr(pg.mix(pg.CADMIUM_ORANGE, 1, pg.TITANIUM_WHITE, 1)))
    for i in range(3):
        y = y0 + 150 + i * 14
        ab.spray(cv, P.line((x0 - 20, y), (x0 + W + 20, y)), distance=60, pressure=0.35 + 0.1 * i)
    # sphere through a frisket (only the inside receives paint)
    cx, cy, r = x0 + 190, y0 + 215, 95
    ab.load(acr(pg.mix(pg.CADMIUM_RED, 1, pg.ALIZARIN_CRIMSON, 0.5)))
    # cast shadow first (soft, no frisket)
    ab.load(acr(pg.mix(pg.ULTRAMARINE, 1, pg.BURNT_UMBER, 1.5)))
    for i in range(4):
        ab.spray(cv, P.arc((cx + 45, cy + r - 5), 85 - i * 12, 200, 340, ry=20 - i * 4),
                 distance=45, pressure=0.7)
    with T.Stencil.ellipse(cv, cx, cy, r, r, invert=True, leak=0.0):
        ab.load(acr(pg.mix(pg.CADMIUM_RED_LIGHT, 1, pg.CADMIUM_ORANGE, 0.3)))
        ab.fill(cv, cx - r, cy - r, cx + r, cy + r, distance=45, pressure=0.6)
        ab.load(acr(pg.mix(pg.ALIZARIN_CRIMSON, 1, pg.ULTRAMARINE, 0.35)))
        for i in range(9):     # form shadow: arcs around the light, darkest far from it
            rr = r * (1.3 - i * 0.07)
            ab.spray(cv, P.arc((cx - 40, cy - 40), rr, -25, 115),
                     distance=48, pressure=0.85 - 0.06 * i)
        ab.load(acr(pg.TITANIUM_WHITE))
        ab.dot(cv, cx - 38, cy - 40, duration=0.35, distance=26, pressure=0.8)
        ab.dot(cv, cx - 38, cy - 40, duration=0.15, distance=10, pressure=0.7)
    # free-hand lines: distance increasing -> crisp to soft and speckled
    ab.load(acr(pg.mix(pg.ULTRAMARINE, 1, pg.IVORY_BLACK, 0.5)))
    ab.spray(cv, P.wave((x0 + 20, y0 + 345), (x0 + W - 20, y0 + 345), amplitude=8, wavelength=120),
             distance=lambda t: 8 + 90 * t, pressure=0.9)


def panel_sponge(cv, x0, y0, g):
    """Tree with sea-sponge foliage."""
    W, H = 380, 370
    # sky and ground, brushed and dried
    b = ps.Brush.flat(24, seed=41)
    for i in range(8):
        b.load(pg.mix(pg.TITANIUM_WHITE, 5, pg.CERULEAN_BLUE, 1, pg.NAPLES_YELLOW, 0.5 * (i / 7)), 1.0)
        cv.stroke(b, P.line((x0, y0 + 15 + i * 36), (x0 + W, y0 + 15 + i * 36)), pressure=0.7)
    cv.dry()
    # trunk and branches (round brush, umber)
    rb = ps.Brush.round(7, seed=42)
    trunk = pg.mix(pg.BURNT_UMBER, 2, pg.ULTRAMARINE, 0.5)
    rb.load(trunk, 1.6)
    cv.stroke(rb, P.spline([(x0 + 190, y0 + 345), (x0 + 186, y0 + 260), (x0 + 195, y0 + 180)]),
              pressure=P.constant(0.95))
    for (ex, ey) in [(110, 120), (270, 110), (150, 70), (240, 160), (200, 60)]:
        rb.load(trunk, 1.2)
        cv.stroke(rb, P.spline([(x0 + 192, y0 + 230), ((x0 + 192 + x0 + ex) / 2, y0 + 190),
                                (x0 + ex, y0 + ey)]), pressure=P.taper(0.05, 0.6, 0.7))
    # foliage: dark -> mid -> light, the sponge turned at every dab
    crown = (x0 + 190, y0 + 145)
    sg = T.Sponge(size=28, porosity=0.55, seed=43)
    layers = [
        (acr(pg.mix(pg.SAP_GREEN, 1, pg.ULTRAMARINE, 0.6, pg.BURNT_UMBER, 0.3)), (12, 22), 40, 0.8, 1.0),
        (acr(pg.mix(pg.SAP_GREEN, 2, pg.CADMIUM_YELLOW, 1)), (-6, 4), 32, 0.65, 0.85),
        (acr(pg.mix(pg.CADMIUM_YELLOW, 2, pg.SAP_GREEN, 1, pg.TITANIUM_WHITE, 0.6)), (-25, -18), 20, 0.5, 0.65),
        (acr(pg.mix(pg.CADMIUM_YELLOW_LIGHT, 2, pg.TITANIUM_WHITE, 1)), (-38, -30), 9, 0.4, 0.45),
    ]
    for paint, (ox, oy), n, pmax, spread in layers:
        sg.load(paint, 1.0)
        for i in range(n):
            a = g.uniform(0, 2 * np.pi)
            rr = 100 * spread * np.sqrt(g.uniform(0, 1))
            x = crown[0] + ox + rr * np.cos(a) * 1.2
            y = crown[1] + oy + rr * np.sin(a) * 0.8
            sg.dab(cv, x, y, pressure=g.uniform(0.3, pmax))
            if sg.load_level < 0.45:
                sg.load(paint, 0.8, clean=False)
    # grass: the sponge dragged along the ground
    sg2 = T.Sponge(size=18, porosity=0.6, coarseness=0.8, seed=44)
    for k in range(3):
        sg2.load(acr(pg.mix(pg.SAP_GREEN, 1, pg.YELLOW_OCHRE, 1 + k)), 0.9)
        y = y0 + 330 + k * 10
        sg2.drag(cv, P.line((x0 + 10, y), (x0 + W - 10, y - 6)), pressure=0.35)


def panel_splatter(cv, x0, y0, g):
    """Toothbrush stars on a night block, flicked and tapped drops."""
    W, H = 380, 370
    # a dark block laid with the knife
    kn = T.PaletteKnife(length=60, seed=51)
    night = pg.mix(pg.PRUSSIAN_BLUE, 2, pg.IVORY_BLACK, 1, pg.DIOXAZINE_PURPLE, 0.4)
    for k in range(8):
        kn.load([(night, 1), (pg.mix(night, 2, pg.TITANIUM_WHITE, 0.3), 0.3)], 1.1, mix=0.4)
        y = y0 + 15 + k * 22
        kn.spread(cv, P.line((x0 + 5, y), (x0 + 190, y + g.normal(0, 3))), pressure=0.55)
    sp = T.Splatter(seed=52)
    sp.load(acr(pg.TITANIUM_WHITE).thinned(0.15))
    sp.toothbrush(cv, (x0 + 100, y0 + 95), radius=95, count=2500, size=0.45)
    sp.toothbrush(cv, (x0 + 70, y0 + 60), radius=40, count=250, size=0.8)
    # flicked drops on the light canvas
    for col, dirn, org in [(pg.CADMIUM_RED, -12, (x0 + 200, y0 + 90)),
                           (pg.CADMIUM_YELLOW, 8, (x0 + 210, y0 + 150)),
                           (pg.ULTRAMARINE, -30, (x0 + 30, y0 + 330))]:
        sp.load(acr(col).thinned(0.1))
        sp.flick(cv, org, direction=dirn, count=45, distance=(15, 170), size=2.4)
    # tapped drops (round, with crown satellites), oil paint with relief
    sp.load(oil(pg.mix(pg.PHTHALO_GREEN, 1, pg.TITANIUM_WHITE, 1)))
    sp.tap(cv, (x0 + 290, y0 + 285), radius=55, count=18, size=4.2)
    sp.load(oil(pg.CADMIUM_ORANGE))
    sp.tap(cv, (x0 + 120, y0 + 260), radius=40, count=12, size=3.4)


def panel_smudge(cv, x0, y0, g):
    """Masking tape, colour bands smudged with a finger / blended, glaze."""
    W, H = 380, 370
    tape = T.Stencil.tape(cv, (x0 - 10, y0 + 300), (x0 + W + 10, y0 + 120), width=26, seed=61)
    b = ps.Brush.flat(20, seed=62)
    bands = [pg.CADMIUM_YELLOW_LIGHT, pg.CADMIUM_ORANGE, pg.CADMIUM_RED, pg.QUINACRIDONE_MAGENTA,
             pg.mix(pg.DIOXAZINE_PURPLE, 1, pg.TITANIUM_WHITE, 1), pg.ULTRAMARINE,
             pg.mix(pg.CERULEAN_BLUE, 1, pg.TITANIUM_WHITE, 1), pg.mix(pg.VIRIDIAN, 1, pg.TITANIUM_WHITE, 0.6)]
    tape.apply()
    for i, c in enumerate(bands):
        b.load(c, 1.6)
        y = y0 + 15 + i * 30
        cv.stroke(b, P.line((x0, y), (x0 + W, y)), pressure=0.9)
    # finger smudges down through the bands (left)
    sm = T.Smudge(size=9, strength=0.75)
    for k in range(5):
        xs = x0 + 25 + k * 30
        sm.stroke(cv, P.spline([(xs, y0 + 5), (xs + 8, y0 + 120), (xs - 4, y0 + 260)]), pressure=0.85)
    # a soft blender brush (right), criss-cross
    bl = T.Blender(size=12, shape='filbert', seed=63)
    for k in range(5):
        xs = x0 + 215 + k * 32
        bl.stroke(cv, P.line((xs, y0 + 10), (xs + 6, y0 + 250)))
    # transparent glaze over the lower right with a soft edge
    T.glaze(cv, pg.mix(pg.PHTHALO_BLUE, 1, pg.ALIZARIN_CRIMSON, 0.4), 0.16,
            region=('ellipse', x0 + 300, y0 + 330, 140, 70), feather=10, seed=64)
    tape.remove()       # peel the tape: crisp, clean stripe


def canvas_image():
    g = np.random.default_rng(7)
    cv = ps.Canvas(1200, 800, ground='canvas', seed=3)
    cv.tone(pg.mix(pg.TITANIUM_WHITE, 12, pg.YELLOW_OCHRE, 0.4, pg.BURNT_UMBER, 0.1), thickness=0.25,
            seed=1)
    panels = [((10, 10, 390, 380), 'Palette knife: impasto over dried brushwork', panel_knife_landscape),
              ((410, 10, 790, 380), 'Knife scraping + sgraffito', panel_scrape),
              ((810, 10, 1190, 380), 'Airbrush + frisket', panel_airbrush),
              ((10, 420, 390, 790), 'Sea sponge', panel_sponge),
              ((410, 420, 790, 790), 'Splatter: toothbrush, flick, tap', panel_splatter),
              ((810, 420, 1190, 790), 'Tape, finger smudge, blender, glaze', panel_smudge)]
    for (bx0, by0, bx1, by1), name, fn in panels:
        t = time.time()
        # each panel is painted inside a window stencil (crisp panel edges)
        with T.Stencil.rect(cv, bx0, by0, bx1, by1, invert=True, leak=0.0):
            fn(cv, bx0, by0, g)
        print(f'  {name}: {time.time() - t:.1f}s')
    img = cv.to_image()
    return label(img, [(p[0], p[1]) for p in panels]), cv


# ==========================================================================
# Watercolour paper image
# ==========================================================================
def paper_image():
    g = np.random.default_rng(11)
    cv = ps.Canvas(1200, 560, ground='cold_press', seed=5)
    ink = pg.LAMP_BLACK.as_medium('ink')
    sepia = pg.SEPIA.as_medium('ink')
    panels = []

    # -- sponge: lifting clouds out of a damp wash, then foliage prints -----
    x0, y0 = 410, 290
    T.glaze(cv, wc(pg.mix(pg.CERULEAN_BLUE, 1, pg.ULTRAMARINE, 0.6)), 0.16,
            region=(x0 + 10, y0 + 10, x0 + 370, y0 + 125), feather=2, water=0.9, seed=73)
    cv.wait(150)                         # lift from the (nearly) dry wash
    blot = T.Sponge(size=24, porosity=0.5, seed=74).wet(0.1)
    for i in range(9):
        blot.dab(cv, x0 + 35 + i * 40 + g.normal(0, 6), y0 + 55 + g.normal(0, 16),
                 pressure=g.uniform(0.5, 0.8))
    sg = T.Sponge(size=24, porosity=0.55, seed=75)
    for col, n, dy in [(pg.mix(pg.SAP_GREEN, 1, pg.INDIGO, 0.5), 10, 20), (pg.SAP_GREEN, 9, 5),
                       (pg.mix(pg.HANSA_YELLOW, 1, pg.SAP_GREEN, 0.5), 6, -8)]:
        sg.load(wc(col), 1.0, water=0.9)
        for i in range(n):
            sg.dab(cv, x0 + 40 + g.uniform(0, 300), y0 + 160 + dy + g.uniform(0, 60),
                   pressure=g.uniform(0.35, 0.7))
    panels.append(((410, 290, 790, 550), 'Sponge (watercolour): blotting + prints'))

    # -- fluid airbrush: sky + frisket sphere ---------------------------------
    x0, y0 = 810, 10
    ab = T.Airbrush(seed=71, flow=1.6).load(wc(pg.ULTRAMARINE), water=3.0)
    for i in range(6):
        y = y0 + 10 + i * 14
        ab.spray(cv, P.line((x0 - 10, y), (x0 + 390, y)), distance=55, pressure=0.9 - 0.13 * i)
    with T.Stencil.ellipse(cv, x0 + 190, y0 + 170, 70, 70, invert=True, leak=0.0):
        ab.load(wc(pg.QUINACRIDONE_GOLD), water=3.0)
        ab.fill(cv, x0 + 120, y0 + 100, x0 + 260, y0 + 240, distance=40, pressure=0.6)
        ab.load(wc(pg.mix(pg.BURNT_SIENNA, 1, pg.ULTRAMARINE, 0.6)), water=3.0)
        for i in range(5):
            ab.spray(cv, P.arc((x0 + 180 + i * 2, y0 + 160 + i * 2), 72 - i * 5, -20, 130),
                     distance=40, pressure=0.8)
    panels.append(((810, 10, 1190, 270), 'Airbrush (watercolour) + frisket'))

    # -- wet splatter -----------------------------------------------------------
    x0, y0 = 10, 290
    sp = T.Splatter(seed=72)
    for col, org, dirn in [(pg.BURNT_SIENNA, (x0 + 10, y0 + 110), -6),
                           (pg.PHTHALO_BLUE, (x0 + 10, y0 + 190), 4),
                           (pg.PERMANENT_ROSE, (x0 + 10, y0 + 40), 3)]:
        sp.load(wc(col), water=1.0)
        sp.flick(cv, org, direction=dirn, count=30, distance=(20, 250), spread=22, size=2.4)
    sp.load(wc(pg.QUINACRIDONE_GOLD))
    sp.tap(cv, (x0 + 320, y0 + 150), radius=30, count=9, size=3.5)
    panels.append(((10, 290, 390, 550), 'Splatter (watercolour)'))

    # -- glaze / wash with masking tape ----------------------------------------
    x0, y0 = 810, 290
    with T.Stencil.tape(cv, (x0, y0 + 200), (x0 + 380, y0 + 60), width=22, seed=76):
        T.glaze(cv, wc(pg.mix(pg.PERMANENT_ROSE, 1, pg.QUINACRIDONE_GOLD, 1)), 0.18,
                region=(x0 + 15, y0 + 15, x0 + 365, y0 + 240), feather=1.5, water=1.0, seed=77)
    panels.append(((810, 290, 1190, 550), 'Wash glaze + masking tape'))

    # let the water flow and dry before inking
    t = time.time()
    cv.wait(40)
    cv.dry()
    print(f'  watercolour simulation: {time.time() - t:.1f}s')

    # -- technical pen: hatched cube + stippled sphere --------------------------
    x0, y0 = 10, 10
    tp = T.Pen('technical', width=1.3, seed=81).load(ink)
    A = np.array([x0 + 60, y0 + 90]); B = A + [110, -40]; C = B + [110, 40]; D = A + [110, 40]
    up = np.array([0, 120])
    for p0, p1 in [(A, B), (B, C), (C, D), (D, A), (A, A + up), (D, D + up), (C, C + up),
                   (A + up, D + up), (D + up, C + up)]:
        tp.line(cv, P.line(p0, p1), wobble=0.5)
    yy, xx = np.mgrid[0:cv.H, 0:cv.W]

    def poly_mask(pts):
        im = Image.new('L', (cv.W, cv.H), 0)
        ImageDraw.Draw(im).polygon([tuple(map(float, p)) for p in pts], fill=1)
        return np.asarray(im) > 0
    # left face (A, D, D+up, A+up) and right face (D, C, C+up, D+up)
    face_l = poly_mask([A, D, D + up, A + up])
    face_r = poly_mask([D, C, C + up, D + up])
    tp.hatch(cv, A[0], A[1], D[0], (D + up)[1], angle=60, spacing=6.5, region=face_l, seed=82)
    tp.hatch(cv, D[0], B[1], C[0], (C + up)[1], angle=-30, spacing=5.5, cross=True, region=face_r, seed=83)
    d = np.hypot(xx - (x0 + 310), yy - (y0 + 160))
    shade = np.clip(((xx - (x0 + 280)) + (yy - (y0 + 130))) / 150, 0, 1)
    tone = np.where(d < 62, 0.08 + 0.92 * shade ** 1.3, 0.0)
    tp2 = T.Pen('technical', width=1.6, seed=84).load(ink)
    T.stipple(cv, tp2, tone, seed=85)
    panels.append(((10, 10, 390, 270), 'Technical pen: hatching, stippling'))

    # -- dip pens ---------------------------------------------------------------
    x0, y0 = 410, 10
    nib = T.Pen('nib', nib=11, nib_angle=40, seed=86).load(sepia)
    # calligraphic loops ("ovals and pushes")
    for k in range(5):
        cx = x0 + 45 + k * 70
        nib.line(cv, P.spline([(cx - 20, y0 + 95), (cx - 25, y0 + 40), (cx + 5, y0 + 25),
                               (cx + 25, y0 + 55), (cx + 10, y0 + 100), (cx + 30, y0 + 105)]))
        if nib.level < 0.4:
            nib.dip(sepia)
    nib.dip(sepia)
    nib.line(cv, P.wave((x0 + 20, y0 + 135), (x0 + 360, y0 + 135), amplitude=12, wavelength=55))
    # pointed flexible nib: swelling strokes; the last ones run dry (railroading)
    pt = T.Pen('pointed', flex=6.0, capacity=2600, seed=87).load(ink)
    for k in range(9):
        cx = x0 + 30 + k * 38
        pt.line(cv, P.spline([(cx, y0 + 250), (cx + 6, y0 + 200), (cx + 18, y0 + 168)]),
                pressure=P.swell(0.05, 1.0, center=0.4, width=0.3))
    pt.load(ink, 0.3)
    pt.line(cv, P.spline([(x0 + 20, y0 + 165), (x0 + 120, y0 + 150), (x0 + 220, y0 + 175),
                          (x0 + 360, y0 + 160)]), pressure=P.swell(0.2, 1.0, width=0.5), speed=1.8)
    panels.append(((410, 10, 790, 270), 'Dip pens: broad nib, pointed nib'))

    cv.wait(20)
    cv.dry()
    img = cv.to_image()
    return label(img, panels), cv


if __name__ == '__main__':
    which = sys.argv[1:] or ['canvas', 'paper']
    if 'canvas' in which:
        t = time.time()
        print('canvas image ...')
        img, _ = canvas_image()
        img.save(os.path.join(OUT, 'tools_demo.png'))
        print(f'  -> output/tools_demo.png ({time.time() - t:.1f}s)')
    if 'paper' in which:
        t = time.time()
        print('paper image ...')
        img, _ = paper_image()
        img.save(os.path.join(OUT, 'tools_demo_paper.png'))
        print(f'  -> output/tools_demo_paper.png ({time.time() - t:.1f}s)')
