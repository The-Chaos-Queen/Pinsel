"""One mark per tool on a blank 256x256 canvas, measured for the port check.

  PINSEL_BACKEND=numba|rust venv/bin/python rust/harness/scenes.py

Every scene has a fixed canvas seed (CANVAS_SEED) and a fixed tool seed
(TOOL_SEED). Oil and paste tools use ground 'canvas'; watercolour, dry media
and pens use 'cold_press'. Each mark is centred on (128, 128).

Per scene, from canvas.render_linear() (linear sRGB):
  covered_px        pixels whose RGB differs from the blank canvas by > 0.01
                    in any channel
  mean_linear_rgb   mean over the whole canvas
  profile_luma      luminance Y = 0.2126 R + 0.7152 G + 0.0722 B along a line
                    through (128, 128), 129 px long (+-64), perpendicular to
                    the mark ('y' = vertical line, 'x' = horizontal line)

Writes rust/harness/results/scenes_<backend>.json. Scenes that build on a base
mark (salt, lift, smudge, blender, eraser, scrape, sgraffito) measure the
base as well: the covered area is the state after the scene's last tool.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pinsel as ps  # noqa: E402
from pinsel import _backend, path as P, pigments as pg, tools as T  # noqa: E402
from pinsel import watercolor as WC  # noqa: E402

SIZE = 256
CANVAS_SEED = 5
TOOL_SEED = 11
HALF = 64
RES = ROOT / 'rust' / 'harness' / 'results'
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
H_LINE = P.line((60, 128), (196, 128))
V_LINE = P.line((128, 60), (128, 196))
SCENES = []  # (name, ground, profile axis, build function)


def scene(name, ground, axis='y'):
    def deco(fn):
        SCENES.append((name, ground, axis, fn))
        return fn
    return deco


def new_canvas(ground):
    return ps.Canvas(SIZE, SIZE, ground=ground, seed=CANVAS_SEED)


def finish(cv):
    """Let watercolour flow and dry, then bake everything into the canvas."""
    if cv.wc is not None:
        cv.wait(60)
    cv.dry()


def measure(img, blank, axis):
    covered = np.abs(img - blank).max(axis=2) > 0.01
    luma = img @ LUMA
    if axis == 'y':
        prof = luma[128 - HALF:128 + HALF + 1, 128]
    else:
        prof = luma[128, 128 - HALF:128 + HALF + 1]
    return {
        'covered_px': int(covered.sum()),
        'covered_fraction': float(covered.mean()),
        'mean_linear_rgb': img.reshape(-1, 3).mean(axis=0, dtype=np.float64).tolist(),
        'profile_luma': [float(v) for v in prof],
    }


_BLANK = {}


def blank_render(ground):
    if ground not in _BLANK:
        _BLANK[ground] = new_canvas(ground).render_linear()
    return _BLANK[ground]


def run_scene(ground, axis, build):
    cv = new_canvas(ground)
    build(cv)
    finish(cv)
    return measure(cv.render_linear(), blank_render(ground), axis)


# ---------------------------------------------------------------- oil brushes
def _oil_brush(shape, size, paint):
    def build(cv):
        b = ps.Brush(shape, size, seed=TOOL_SEED).load(paint)
        cv.stroke(b, H_LINE)
    return build


for _shape, _size, _paint in [('round', 14, pg.ULTRAMARINE), ('flat', 14, pg.ALIZARIN_CRIMSON),
                              ('filbert', 12, pg.CADMIUM_YELLOW), ('hog', 12, pg.BURNT_SIENNA),
                              ('fan', 20, pg.SAP_GREEN), ('rigger', 4, pg.IVORY_BLACK)]:
    SCENES.append((f'brush_{_shape}', 'canvas', 'y', _oil_brush(_shape, _size, _paint)))


# ---------------------------------------------------------------- palette knife
@scene('knife_spread', 'canvas')
def _(cv):
    knife = T.PaletteKnife(seed=TOOL_SEED).load(pg.CADMIUM_YELLOW)
    knife.spread(cv, H_LINE)


@scene('knife_scrape', 'canvas')
def _(cv):
    knife = T.PaletteKnife(seed=TOOL_SEED).load(pg.CERULEAN_BLUE)
    knife.spread(cv, P.line((50, 116), (206, 116)))  # wet slab
    knife.clean()
    knife.scrape(cv, P.line((50, 132), (206, 132)))


@scene('knife_sgraffito', 'canvas')
def _(cv):
    knife = T.PaletteKnife(seed=TOOL_SEED).load(pg.ALIZARIN_CRIMSON)
    knife.spread(cv, P.line((60, 128), (196, 128)))  # wet slab
    knife.sgraffito(cv, H_LINE, seed=TOOL_SEED)


# ---------------------------------------------------------------- airbrush
@scene('airbrush_spray', 'canvas')
def _(cv):
    ab = T.Airbrush(seed=TOOL_SEED).load(pg.CERULEAN_BLUE)
    ab.spray(cv, H_LINE, seed=TOOL_SEED)


@scene('airbrush_dot', 'canvas', 'x')
def _(cv):
    ab = T.Airbrush(seed=TOOL_SEED).load(pg.CERULEAN_BLUE)
    ab.dot(cv, 128, 128, seed=TOOL_SEED)


# ---------------------------------------------------------------- splatter
@scene('splatter_flick', 'canvas')
def _(cv):
    sp = T.Splatter(seed=TOOL_SEED).load(pg.CADMIUM_RED.thinned(0.5))
    sp.flick(cv, (40, 128), direction=0.0, spread=25.0, distance=(10, 200), seed=TOOL_SEED)


@scene('splatter_tap', 'canvas', 'x')
def _(cv):
    sp = T.Splatter(seed=TOOL_SEED).load(pg.CADMIUM_RED.thinned(0.5))
    sp.tap(cv, (128, 128), seed=TOOL_SEED)


# ---------------------------------------------------------------- sponge
@scene('sponge_dab', 'canvas', 'x')
def _(cv):
    sp = T.Sponge(size=30, seed=TOOL_SEED).load(pg.YELLOW_OCHRE)
    sp.dab(cv, 128, 128, pressure=0.6, seed=TOOL_SEED)


@scene('sponge_drag', 'canvas')
def _(cv):
    sp = T.Sponge(size=30, seed=TOOL_SEED).load(pg.SAP_GREEN)
    sp.drag(cv, H_LINE, pressure=0.4, seed=TOOL_SEED)


# ---------------------------------------------------------------- pen
@scene('pen_line', 'cold_press')
def _(cv):
    pen = T.Pen('technical', width=4.0, seed=TOOL_SEED).load(pg.IVORY_BLACK)
    pen.line(cv, H_LINE, seed=TOOL_SEED)


@scene('pen_hatch', 'cold_press', 'x')
def _(cv):
    pen = T.Pen('technical', width=2.0, seed=TOOL_SEED).load(pg.IVORY_BLACK)
    pen.hatch(cv, 60, 60, 196, 196, angle=45.0, spacing=8.0, seed=TOOL_SEED)


# ---------------------------------------------------------------- smudge and blender
def _base_stroke(cv):
    """A wet yellow flat stroke across the middle, for smudge and blender."""
    b = ps.Brush('flat', 14, seed=TOOL_SEED).load(pg.CADMIUM_YELLOW)
    cv.stroke(b, P.line((60, 110), (196, 110)))


@scene('smudge', 'canvas')
def _(cv):
    _base_stroke(cv)
    T.Smudge(size=10.0, strength=0.6).stroke(cv, V_LINE, seed=TOOL_SEED)


@scene('blender', 'canvas')
def _(cv):
    _base_stroke(cv)
    T.Blender(size=12, seed=TOOL_SEED).stroke(cv, V_LINE, seed=TOOL_SEED)


# ---------------------------------------------------------------- crayon
@scene('oil_crayon_tip', 'cold_press')
def _(cv):
    T.OilCrayon(size=4).load(pg.ULTRAMARINE).stroke(cv, H_LINE, pressure=0.85)


@scene('oil_crayon_side', 'cold_press')
def _(cv):
    T.OilCrayon(size=18, side=0.9).load(pg.CADMIUM_YELLOW).stroke(cv, H_LINE, pressure=0.9)


# ---------------------------------------------------------------- pastel, charcoal, eraser
@scene('soft_pastel', 'cold_press')
def _(cv):
    ps.SoftPastel(size=10, seed=TOOL_SEED).load(pg.CADMIUM_RED).stroke(cv, H_LINE, seed=TOOL_SEED)


@scene('charcoal', 'cold_press')
def _(cv):
    ps.Charcoal(size=8, seed=TOOL_SEED).stroke(cv, H_LINE, seed=TOOL_SEED)


@scene('kneaded_eraser', 'canvas')
def _(cv):
    b = ps.Brush('flat', 12, seed=TOOL_SEED).load(pg.CADMIUM_RED)
    cv.stroke(b, H_LINE)  # wet stroke
    ps.KneadedEraser(size=14, strength=0.65, seed=TOOL_SEED).stroke(cv, P.line((100, 128), (156, 128)),
                                                                    pressure=0.8)


# ---------------------------------------------------------------- watercolour
WASH = (60, 88, 196, 168)


@scene('watercolor_brush', 'cold_press')
def _(cv):
    b = ps.Brush('round', 14, seed=TOOL_SEED).load(pg.ULTRAMARINE.as_medium('watercolor'))
    cv.stroke(b, H_LINE)


@scene('watercolor_wash', 'cold_press')
def _(cv):
    WC.wash(cv, WASH, pg.ULTRAMARINE, concentration=0.8, seed=TOOL_SEED)


@scene('watercolor_drop', 'cold_press')
def _(cv):
    WC.drop(cv, 128, 128, paint=pg.ALIZARIN_CRIMSON, radius=14.0, seed=TOOL_SEED)


@scene('watercolor_salt', 'cold_press')
def _(cv):
    WC.wet_paper(cv, WASH, seed=TOOL_SEED)  # salt does nothing on dry paper
    WC.wash(cv, WASH, pg.ULTRAMARINE, concentration=0.8, seed=TOOL_SEED)
    WC.salt(cv, WASH, seed=TOOL_SEED)


@scene('watercolor_lift', 'cold_press')
def _(cv):
    WC.wash(cv, WASH, pg.ULTRAMARINE, concentration=0.8, seed=TOOL_SEED)
    cv.wait(60)
    WC.lift(cv, (96, 112, 160, 144), strength=0.7, seed=TOOL_SEED)


# ---------------------------------------------------------------- glaze, stencil
@scene('glaze', 'canvas')
def _(cv):
    T.glaze(cv, pg.ALIZARIN_CRIMSON.thinned(0.6), thickness=0.12, region=WASH, seed=TOOL_SEED)


@scene('stencil_tape', 'canvas')
def _(cv):
    tape = T.Stencil.tape(cv, (40, 128), (216, 128), width=24, seed=TOOL_SEED)
    with tape:
        b = ps.Brush('flat', 14, seed=TOOL_SEED).load(pg.CADMIUM_RED)
        cv.stroke(b, V_LINE)


# ---------------------------------------------------------------- main
def main():
    out = {'backend': _backend.NAME, 'size': SIZE, 'canvas_seed': CANVAS_SEED,
           'tool_seed': TOOL_SEED, 'profile': {'half_width_px': HALF, 'luma': 'linear Y'},
           'scenes': {}, 'skipped': []}
    for name, ground, axis, build in SCENES:
        out['scenes'][name] = {'ground': ground, 'profile_axis': axis,
                               **run_scene(ground, axis, build)}
        s = out['scenes'][name]
        print(f"{name:18s} covered={s['covered_px']:6d} ({s['covered_fraction']:.4f}) "
              f"mean_rgb={[round(v, 5) for v in s['mean_linear_rgb']]}")
    RES.mkdir(parents=True, exist_ok=True)
    path = RES / f'scenes_{_backend.NAME}.json'
    path.write_text(json.dumps(out, indent=1))
    print(f"{len(SCENES)} scenes, {len(out['skipped'])} skipped -> {path}")


if __name__ == '__main__':
    main()
