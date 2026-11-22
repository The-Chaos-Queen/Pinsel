"""The Night Fisher — oil-crayon resist and watercolour on cold-press paper.

Run from the project root with python examples/moonlit_pond.py.
"""
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pinsel as ps
from pinsel import path as P, watercolor as wc
from pinsel.pigments import Paint


def paint(before_wash=None):
    rng = np.random.default_rng(102)
    width, height = 720, 510
    cv = ps.Canvas(width, height, ground='cold_press', ground_color='#f0e5ca',
                   texture_scale=.80, seed=35)
    y, x = np.mgrid[:height, :width]

    @lru_cache(None)
    def color(hexcode, medium='oil'):
        return Paint.from_color(hexcode, opacity=.91, medium=medium)

    def rub(points, hexcode, size, pressure, side=.18, angle=None):
        stick = ps.OilCrayon(size=size, side=side).load(color(hexcode))
        cv.stroke(stick, points, pressure=pressure, angle=angle)

    # Rub the moon in short overlapping arcs. The lower rim barely touches
    # the tooth. A few horizontal skips interrupt the light on the water.
    mx, my = 472, 113
    for offset in np.arange(-33, 34, 2.6):
        half = np.sqrt(max(0, 34**2 - offset**2))
        pts = P.quad((mx-half, my+offset), (mx-4, my+offset-2.3),
                     (mx+half, my+offset+rng.uniform(-1, 1)))
        # Warm the rim gradually, varying the touch over the tooth.
        warmth = np.clip((offset+15)/56 + rng.uniform(-.06,.06), 0, 1)
        rgb = np.rint(np.array([251,235,178]) - warmth*np.array([12,24,20])).astype(int)
        tint = '#%02x%02x%02x' % tuple(rgb)
        rub(pts, tint, 4.1, rng.uniform(.64,.87)-.18*warmth, side=.27)
    for dx, dy, length in [(-17,8,8),(-12,11,4),(-10,15,7),(8,-8,5),(12,-5,4)]:
        rub(P.quad((mx+dx,my+dy),(mx+dx+2,my+dy+1),(mx+dx+length,my+dy+2)),
            '#efd99e', 2.0, P.taper(.3,.4,.47))

    horizon = 232
    for j in range(88):
        t = (j + rng.uniform(0, .8)) / 88
        yy = horizon + 4 + 269*t
        spread = 5 + 42*t
        centre = mx + 8*np.sin(6*t) - 36*t + rng.normal(0, spread*.65)
        length = rng.uniform(.7, 1.4)*(1.7 + 14*t)
        shade = rng.choice(['#f6db9b', '#efca84', '#e8b774', '#e5c595'])
        rub(P.quad((centre-length, yy), (centre, yy-1.5*t),
                   (centre+length, yy+rng.uniform(-.9, .9))),
            shade, rng.uniform(1.2, 2.4)+1.4*t,
            P.taper(.19, .32, rng.uniform(.55, .91)), side=.20)
    # Glints along curled reed leaves and on the bird's brow and bill.
    for points in [[(71,503),(56,442),(54,413)], [(76,504),(103,435),(129,408)],
                   [(637,504),(652,440),(674,415)], [(675,499),(678,412),(667,364)]]:
        rub(P.quad(*points), '#c6a276', 2.8, P.taper(.18, .3, .70))
    rub([(206,306),(214,305),(222,307)], '#ddb87f', 2.4, .80)
    rub([(225,309),(253,310)], '#cb935e', 2, .74)
    # Bake the colour into the sheet before the wash. Its wax stays on the tooth.
    cv.dry()
    if before_wash is not None:
        cv.save(before_wash, relief=.52, specular=.08)

    # The sky and water are one continuous indigo wash. Its warmest band
    # sits at the distant bank and its coolest blue collects above the moon.
    stops = [('#202947', 0), ('#344763', .25), ('#ab8382', .45),
             ('#70858d', .50), ('#3e6977', .60), ('#243b53', 1)]
    wc.variegated_wash(cv, None, [(color(c, 'watercolor'), t) for c,t in stops],
                       direction=((0,0),(0,height)), water=.68,
                       concentration=3.8, variation=.15, rough=0, seed=10)

    # Thin windblown veils of violet run diagonally under the moon.
    for cx, cy, length, depth in [(130,66,230,7),(270,125,240,6),(540,153,245,5)]:
        line = cy - .026*(x-cx) + 4*np.sin(x/83)
        veil = np.exp(-((x-cx)/length)**2 - ((y-line)/depth)**2)
        wc.wash(cv, veil, color('#635671', 'watercolor'), concentration=.87,
                water=.21, rough=0, seed=cy)

    # Two washes for the far shore. Fine uneven tree crowns interrupt the
    # second ridge and recur, in softer and narrower bands, in its reflection.
    xx = np.arange(width)
    distant = 208 + 14*np.sin(xx/124-.5) + 5*np.sin(xx/43) - .016*xx
    ridge = 218 - 4*np.sin(xx/62) - 7*np.cos(xx/127)
    for _ in range(190):
        cx = rng.uniform(-8, width+8)
        peak = rng.uniform(1.5, 13)*(1-.70*np.exp(-((cx-mx)/100)**2))
        ridge -= peak*.52*np.maximum(0, 1-np.abs(xx-cx)/rng.uniform(2,6))
    bank = horizon + 1.8*np.sin(xx/115)
    for top, tint, strength in [(distant, '#655f77', 1.6), (ridge, '#273e4c', 3.5)]:
        area = (y >= top) & (y <= bank)
        wc.wash(cv, area, color(tint, 'watercolor'), concentration=strength,
                water=.60, variation=.16, rough=.35, seed=int(strength*100))
    reflected = (y > bank) & (y < 2*bank-ridge + 2.8*np.sin(x/11+y/3))
    wc.wash(cv, reflected, color('#344653','watercolor'), concentration=1.35,
            water=.36, variation=.25, soft=1, seed=18)

    # A few dark, shallow troughs give the unlit water its own movement.
    ripples = np.zeros(cv.shape, np.float32)
    for _ in range(225):
        cy = rng.uniform(238, height+2)
        t = (cy-horizon)/(height-horizon)
        cx = rng.uniform(-30, width+30)
        length, radius = rng.uniform(4, 13)+18*t, rng.uniform(.32, .85)+.75*t
        xa, xb = max(0, int(cx-length*2)), min(width, int(cx+length*2)+1)
        ya, yb = max(0, int(cy-4*radius-3)), min(height, int(cy+4*radius+4))
        if xb <= xa or yb <= ya:
            continue
        xo = (xx[xa:xb]-cx)/length
        centre = cy + rng.uniform(-1,1)*xo + 1.5*t*xo**2
        d = (y[ya:yb,xa:xb]-centre)/radius
        patch = np.exp(-d*d - 2.3*xo**2)*rng.uniform(.42, .95)
        np.maximum(ripples[ya:yb,xa:xb], patch, out=ripples[ya:yb,xa:xb])
    wc.wash(cv, ripples, color('#243846','watercolor'), concentration=2.3,
            water=.47, rough=0, variation=.12, seed=32)

    # The heron waits motionless. Its folded wing is a separate cool glaze,
    # and the near edge of the cheek is just a touch of warm wax.
    bird = [(99,380),(114,373),(131,360),(146,347),(159,340),(170,338),
            (176,331),(177,321),(180,311),(189,304),(204,301),(215,302),
            (221,306),(227,308),(257,310),(225,313),(218,314),(210,318),
            (202,320),(196,326),(195,335),(200,343),(201,351),(197,360),
            (186,370),(173,374),(158,376),(148,378),(135,378),(119,384)]
    area = wc.polygon(P.spline(bird, closed=True))(cv)
    wc.wash(cv, area, color('#293841','watercolor'), concentration=4.1,
            water=.65, rough=0, variation=.09, seed=77)
    wing = [(115,376),(134,358),(151,348),(166,344),(174,347),(167,355),
            (153,364),(137,372)]
    wc.wash(cv, wc.polygon(P.spline(wing, closed=True)), color('#425364','watercolor'),
            concentration=1.6, water=.33, rough=0, seed=84)

    cv.wait(1.6)
    cv.dry()
    # Dark pencil-like strokes finish the feet, fine plumes and foreground
    # reed beds; the occasional dull gold edge picks up light from the water.
    for xx0,yy0,rr in [(142,425,22),(164,416,12)]:
        rub(P.quad((xx0-rr,yy0),(xx0,yy0+3),(xx0+rr,yy0+.4)),
            '#557481', 1.7, P.taper(.25,.29,.70))
    for points in [[(151,370),(150,390),(142,408),(143,424),(135,425)],
                   [(164,370),(168,389),(160,401),(165,415),(175,415)]]:
        rub(P.spline(points), '#1c303d', 2.7, .92)
    rub([(150,384),(149,391),(143,405)], '#78827e', 1.1, .74)
    rub([(160,401),(165,413)], '#7e8076', 1.1, .70)
    rub([(144,423),(152,424)], '#243642', 1.35, .84)
    for points, tint, weight in [
        ([(105,380),(135,356),(159,344)], '#30434e', 2.8),
        ([(215,307),(207,303),(196,305)], '#344853', 3.1),
        ([(203,305),(189,304),(175,311)], '#434f5d', 2.0),
        ([(207,306),(192,308),(181,314)], '#677078', 1.4),
        ([(213,316),(199,319),(195,328)], '#a99b83', 1.6),
        ([(195,329),(191,338),(196,345)], '#7b8988', 1.6),
        ([(197,343),(200,351),(193,359)], '#425560', 2.1)]:
        rub(P.quad(*points), tint, weight, P.taper(.16,.31,.81))
    for k in range(11):
        xx0=167-1.8*k
        rub(P.quad((xx0,348+.7*k),(xx0-11,359+.7*k),(xx0-30,371+.7*k)),
            rng.choice(['#435667','#50626e','#596570']), 1.25,
            P.taper(.22,.38,rng.uniform(.52,.67)))
    rub([(210,307.5),(211,307.5)], '#f0ca80', 2.5, .94)
    rub([(210.7,307.3)], '#293641', 1, .90)

    for side in (0,1):
        for i in range(45):
            base = rng.normal(37,37) if side==0 else rng.normal(688,27)
            rise = rng.uniform(20,115)*(1 if side==0 else 1.3)
            lean = rng.normal(12 if side==0 else -10, 27)
            root = (base, height+rng.uniform(0,12))
            tip = (base+lean,height-rise)
            bend = (base+.22*lean,height-.65*rise)
            tone = rng.choice(['#293d47','#344950','#455051','#535752'])
            rub(P.quad(root,bend,tip), tone, rng.uniform(2,5),
                P.taper(.05,.58,.91,start=.66), side=.22)
            if i%7==0 and rise>60:
                head = (tip[0]+lean*.09,tip[1]-rng.uniform(9,18))
                rub(P.line(bend,tip), '#4b5050', 1.65, .83)
                rub([tip,head], '#494449', 4.7,.88)
                rub([(tip[0]+1.0,tip[1]-3),head], '#97775c', 1.8,.66)
    return cv


if __name__ == '__main__':
    target = ROOT/'output'/'moonlit_pond.png'
    target.parent.mkdir(exist_ok=True)
    paint(target.with_name('moonlit_pond_before_wash.png')).save(target, relief=.52, specular=.08)
    print(target)
