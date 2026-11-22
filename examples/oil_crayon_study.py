"""The Last Pear — a night-window still life, drawn entirely with oil crayons.

Run from the project root with python examples/oil_crayon_study.py.
"""
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pinsel as ps
from pinsel import path as P
from pinsel.pigments import Paint


def paint():
    rng = np.random.default_rng(41)
    cv = ps.Canvas(640, 496, ground="cold_press", ground_color="#26313d",
                   texture_scale=0.82, seed=12)

    @lru_cache(None)
    def pigment(color):
        return Paint.from_color(color, opacity=1.0)

    def mark(points, color, size, pressure=0.82, side=0.6, angle=None):
        stick = ps.OilCrayon(size=size, side=side).load(pigment(color))
        stick.stroke(cv, points, pressure=pressure, angle=angle)

    # The last light outside. Vertical rubbings leave dark paper in the gaps.
    for x in range(-20, 684, 11):
        sky = rng.choice(["#244653", "#28545d", "#344858", "#2a4c58"])
        mark([(x, -30), (x - 8, 78), (x - 23, 154), (x - 36, 255)],
             sky, 19, rng.uniform(.61, .88), angle=4)
    for x in range(35, 620, 24):
        y = rng.uniform(24, 125)
        mark([(x + 14, y), (x + 1, y + 46), (x - 12, y + 88)],
             "#437077", rng.uniform(1.6, 3.5), P.taper(.30, .32, rng.uniform(.24, .43)))
    # Opposite rooftops and a few lamps along the street.
    for x in range(-10, 667, 28):
        y = 172 + 13 * np.sin(x / 64) + rng.normal(0, 9)
        mark([(x, y), (x - 4, 213), (x - 8, 254)], "#253743", 35, .91, angle=0)
    # Chimneys interrupt the long horizon above the opposite houses.
    for x, y in [(57, 157), (177, 151), (414, 153), (548, 161)]:
        mark([(x, y), (x, 197)], "#293943", 7, .91, angle=0)
        mark([(x - 2, y), (x + 4, y)], "#435158", 2, .70)
    for x, y in [(51, 193), (70, 194), (134, 202), (403, 191), (419, 191),
                 (565, 185), (579, 185)]:
        mark([(x, y), (x, y + 5)], "#dfae73", 3, .77, side=.15)
    for x in (99, 496):
        mark([(x + 8, -18), (x + 1, 110), (x - 6, 245)],
             "#282f39", 19, .98, angle=0)
        mark([(x + 15, -8), (x + 8, 111), (x + 1, 241)], "#645557", 3, .69)

    # A red sill slants up to the right, with dry swipes of reflected blue.
    for y in range(232, 571, 8):
        tone = rng.choice(["#9f584a", "#ad664f", "#93524b", "#ad614b"])
        mark([(-25, y + 23), (180, y + 8), (403, y - 9), (665, y - 30)],
             tone, 22, rng.uniform(.75, .97), angle=82)
    for _ in range(60):
        x, y = rng.uniform(-55, 610), rng.uniform(276, 519)
        mark([(x, y), (x + 30, y - 3), (x + rng.uniform(55, 134), y - 8)],
             rng.choice(["#da8b5a", "#773f43", "#ba7c63"]), rng.uniform(1.5, 5),
             P.taper(.18, .26, rng.uniform(.29, .67)), angle=75)

    # The plate is a series of broken, nested elliptical gestures.
    def oval(x, y, rx, ry, start=0, end=360):
        curve = P.arc((x, y), rx, start, end, ry=ry)
        curve[:, 1] -= .075 * (curve[:, 0] - x)
        return curve

    for y in range(374, 420, 4):
        reach = 212 * np.sqrt(max(0, 1 - ((y - 395) / 24) ** 2))
        mark([(361 - reach, y + .07 * reach), (361 + reach, y - .07 * reach)],
             "#553940", 10, .83, angle=80)
    for r in np.arange(1, .015, -.047):
        mark(oval(316, 360, r * 228, r * 60), "#29415b", 19, .97)
    for r in np.arange(.98, .02, -.045):
        col = "#436c9b" if r >= .80 else "#365c83"
        mark(oval(313, 352, r * 227, r * 53), col, 18, .89)
    for r, start, end, col, width in [
        (.98, 7, 162, "#93b4bb", 5), (.98, 166, 257, "#adced0", 4),
        (.97, 270, 352, "#7898ad", 3), (.78, 15, 155, "#8ca1ae", 3),
        (.75, 81, 186, "#263f62", 5), (.88, 7, 62, "#558db8", 6),
    ]:
        mark(oval(313, 352, 227 * r, 53 * r, start, end), col, width, .82)
    for x in np.arange(130, 481, 11):
        y = 352 + 34 * np.sqrt(max(0, 1 - ((x - 314) / 196) ** 2)) - .075*(x - 314)
        mark([(x, y - 1), (x + 5, y - 2)], "#6284a0", 1.45, .48)

    # Cast shadows tie the round bodies to the shallow dish.
    for x, y, rx, ry in [(275, 351, 69, 12), (365, 361, 54, 8)]:
        for t in np.linspace(-.96, .96, 9):
            half = rx * np.sqrt(1 - t*t)
            mark([(x-half, y+t*ry+half*.07), (x+half, y+t*ry-half*.07)],
                 "#22374d", 5, .83, angle=84)

    # Width of a pear at each height. Turn the same form to reveal its cut face.
    radius = PchipInterpolator([0, .055, .17, .33, .51, .68, .80, .92, .98, 1],
                               [.025, .16, .24, .38, .76, 1, .99, .76, .37, .03])

    def pear(base, height, width, lean, cut):
        angle = np.radians(lean)
        across = np.array([np.cos(angle), np.sin(angle)])
        up = np.array([np.sin(angle), -np.cos(angle)])

        def curve(t, u):
            t = np.asarray(t, dtype=float)
            x = width * (u * radius(t) + .15 * np.sin(2 * np.pi * t))
            return base + np.outer(x, across) + np.outer(height * (1 - t), up)

        values = np.linspace(0, 1, 55)
        outer = "#b99535" if cut else "#a28a39"
        for u in np.linspace(-1, 1, 21):
            mark(curve(values, u), outer, 10, .96)
        bands = (["#e5cc81", "#f3dfa0", "#f6e4ad", "#efda93", "#e4c681"]
                 if cut else ["#c5b34f", "#e4c550", "#f1cc53", "#dba438", "#ab803b"])
        # Colour follows the curvature, tapering near the stalk and at the blossom.
        for u in np.linspace(-.92, .90, 27):
            band = min(4, int((u + 1) / 2 * 5))
            t = np.linspace(.043 if cut else .02, .974 if cut else .986, 39)
            mark(curve(t, u), bands[band], 8, rng.uniform(.78, .96))
        for _ in range(120 if not cut else 53):
            u, t = rng.uniform(-.89, .87), rng.uniform(.07, .93)
            dist = rng.uniform(.014, .076)
            color = rng.choice(["#d8b968", "#fff0b7"] if cut else
                               ["#9e7a3e", "#d4a13b", "#ffe37a", "#b19845"])
            mark(curve(np.linspace(t, min(.98, t + dist), 6), u), color,
                 rng.uniform(1.1, 3.2), rng.uniform(.37, .72), side=.18)
        mark(curve(np.linspace(.27, .76, 32), -.80), "#fff2aa" if cut else "#f8dc70",
             3.4, .77)
        if cut:
            mark(curve(np.linspace(.12, .68, 40), .06), "#c9a66a", 2.7, .69)
            for u, top, bottom in [(-.20, .58, .72), (.16, .56, .67)]:
                t = np.linspace(top, bottom, 26)
                pip = curve(t, u * np.linspace(.6, 1.2, len(t)))
                mark(pip, "#886144", 7.5, P.taper(.55, .15, .98), side=.12)
                mark(pip[6:-2], "#533a31", 4.7, P.taper(.38, .2, .97), side=.12)
                mark(pip[12:18] + [-1, 0], "#c49a66", 1.6, .73, side=0)
        tip = curve([0], 0)[0]
        if not cut:
            # A single olive leaf folds along its midrib.
            center = P.bezier(tip + [-1, -13], tip + [-20, -40],
                              tip + [-51, -24], tip + [-73, -39], n=40)
            along = np.linspace(0, 1, len(center))
            for offset in np.linspace(-1, 1, 11):
                leaf = center.copy()
                leaf[:, 1] += offset * 9 * np.sin(np.pi * along) ** 1.3
                mark(leaf, "#7d874c" if offset < 0 else "#596844", 3.8, .83)
            mark(center[6:-3], "#baa360", 1.7, .74)
        twig = [tip - up, tip + 12 * up + 2 * across, tip + 24 * up + 10 * across]
        mark(twig, "#5a4140", 6 if not cut else 4, .96, side=.24)
        mark(twig[1:], "#c08f60", 2, .78)

    pear(np.array([266, 353]), 184, 73, -9, False)
    pear(np.array([339, 365]), 134, 51, 57, True)
    return cv


if __name__ == "__main__":
    target = ROOT / "output" / "oil_crayon_study.png"
    target.parent.mkdir(exist_ok=True)
    paint().save(target, relief=.38, specular=.09)
    print(target)

