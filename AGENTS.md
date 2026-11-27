# pinsel for AI painters

You can paint with this. Not generate an image: paint, stroke by stroke, in
Python, with paint that behaves like paint. Bristles split and run dry,
colours mix by real pigment physics (ultramarine + burnt sienna gives a
neutral grey, not mud), glazes tint what is under them, watercolour flows,
pools and blooms, and the finished canvas is lit so you can see the impasto.

No image model, no GPU. pinsel runs on an ordinary cloud sandbox: a
1200 x 800 painting needs well under 1 GB of memory and seconds to minutes of CPU.
Several AI painters already paint with it.

## Install (one line)

On Linux x86_64 (most sandboxes):

    pip install https://github.com/The-Chaos-Queen/Pinsel/releases/download/v0.2.0/pinsel-0.2.0-py3-none-any.whl https://github.com/The-Chaos-Queen/Pinsel/releases/download/v0.2.0/pinsel_core-0.2.0-cp310-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl

Anywhere else (Linux ARM, Windows, Mac), or to get the newest release:

    curl -fsSL https://raw.githubusercontent.com/The-Chaos-Queen/Pinsel/main/install.py | python -

(`install.py` is 70 short lines; read it first if you like.) This needs
Python 3.10 or newer and installs only pinsel, its Rust kernels, numpy and
Pillow. Check it with:

    python -c "import pinsel, pinsel._backend as B; print(pinsel.__version__, B.NAME)"   # 0.2.0 rust

## Your first painting

This is `examples/quickstart.py`: a toned canvas, a sky of broken flat-brush
strokes, an impasto sun, a hill of 900 hog-brush flicks, and a warm glaze over
the dried paint. About 15 seconds on a laptop.

```python
import numpy as np
import pinsel as ps
from pinsel import pigments as pg, path as P

rng = np.random.default_rng(0)
cv = ps.Canvas(900, 600, ground='canvas', seed=1)
cv.tone(pg.mix(pg.BURNT_SIENNA, 2, pg.YELLOW_OCHRE, 1).thinned(0.3), thickness=0.25)

# sky: overlapping strokes with streaky, partially mixed blues (lighter towards the horizon)
flat = ps.Brush('flat', 24, seed=2)
for y in np.arange(10, 430, 14):
    t = y / 430
    for x in np.arange(-100, 900, 260):
        flat.load([(pg.ULTRAMARINE, 1.2 - t), (pg.TITANIUM_WHITE, 1.5 + 4 * t), (pg.CERULEAN_BLUE, 0.8)],
                  mix=0.6)
        x0 = x + rng.uniform(-60, 60)
        cv.stroke(flat, P.wave((x0, y), (x0 + rng.uniform(250, 380), y + rng.normal(0, 4)), amplitude=5,
                               wavelength=400), pressure=P.taper(0.1, 0.2, 0.8))

# a sun, painted wet into the sky
sun = ps.Brush('round', 16, seed=4).load(pg.mix(pg.CADMIUM_YELLOW, 2, pg.TITANIUM_WHITE, 3), 1.6)
cv.stroke(sun, P.spiral((650, 140), 2, 26, turns=3), pressure=0.9)

# rolling hill in thick hog-bristle impasto, dabbed with varied greens
block = ps.Brush('flat', 30, seed=6)
for x in np.arange(-20, 920, 40):
    block.load(pg.mix(pg.SAP_GREEN, 2, pg.ULTRAMARINE, 0.6, pg.TITANIUM_WHITE, 0.8), 1.0)
    top = 370 + 40 * np.sin(x / 180) + 15 * np.sin(x / 57)
    cv.stroke(block, P.line((x, top + 8), (x + 8, 620)), pressure=0.85)
hog = ps.Brush('hog', 9, seed=3)
greens = [pg.mix(pg.SAP_GREEN, 2, pg.CADMIUM_YELLOW, 1, pg.TITANIUM_WHITE, 1),
          pg.mix(pg.SAP_GREEN, 2, pg.YELLOW_OCHRE, 1, pg.TITANIUM_WHITE, 0.6),
          pg.mix(pg.CADMIUM_YELLOW_LIGHT, 1, pg.SAP_GREEN, 1, pg.TITANIUM_WHITE, 1.5)]
for i in range(900):
    x = rng.uniform(-10, 910)
    top = 370 + 40 * np.sin(x / 180) + 15 * np.sin(x / 57)
    y = top + 4 + (620 - top) * rng.uniform() ** 1.6
    hog.load(greens[i % 3].impasto(1.5), amount=1.3)
    a = -np.pi / 2 + rng.normal(0.2, 0.35)
    L = 18 + 30 * (y - top) / 250
    cv.stroke(hog, P.quad((x, y), (x + 0.4 * L * np.cos(a) + 4, y + 0.5 * L * np.sin(a)),
                         (x + L * np.cos(a), y + L * np.sin(a))), pressure=P.flick(0.85, 0.25))

# let it dry, then glaze warm evening light over the hill
cv.dry()
glaze = ps.Brush('hake', 40, seed=5)
for y in range(380, 620, 50):
    glaze.load(pg.QUINACRIDONE_GOLD.thinned(0.85), 0.6)
    cv.stroke(glaze, P.line((-20, y), (920, y - 10)), pressure=0.7)

cv.save('quickstart.png')
```

## Look at your work

You can read images, so look often. That is the whole method.

- `cv.save('step3.png')` at every stage, then open the PNG and judge it like a painter: values first, then colour, then edges.
- `cv.preview(0.5)` returns a fast half-size PIL image (good for checking mid-painting).
- `cv.save('raking.png', light_dir=(-0.9, -0.3, 0.35), relief=1.6)` shows the paint relief under raking light.
- `cv.checkpoint('before sky')` then `cv.undo()` lets you try a passage and take it back. Each checkpoint is a full in-memory copy of the canvas, so on big canvases keep only one or two.
- `cv.save_state('work.npz')` / `cv.load_state('work.npz')` keeps a painting across sessions, when your sandbox resets.
- `seed=` on the canvas and brushes makes a script paint the same picture every run, so you can change one passage and compare.

## Work like a painter

- Tone the ground, block in big shapes with big brushes, dry, then refine. Details last, small brushes (`'rigger'`, `'round'`).
- Oil: paint stays wet and mixes on the canvas until you `cv.dry()`. Glazes (`thinned(...)`, `tools.glaze`) only look right over dry paint.
- Watercolour: `from pinsel import watercolor as wc`, paints via `pg.ULTRAMARINE.as_medium('watercolor')`, then `wc.wet_paper`, `wc.wash`, `wc.variegated_wash`, `wc.drop`, `wc.salt`, `wc.lift`, and `cv.wait(seconds)` to let the water move and dry.
- Brushes: `'round' 'sable' 'hog' 'flat' 'bright' 'filbert' 'angular' 'fan' 'rigger' 'mop' 'hake'`.
- Paths: `P.line, polyline, bezier, quad, spline, arc, spiral, wave, zigzag, scribble`; pressure: `P.taper, swell, flick, dab`.
- Pigments: `pg.PIGMENTS` lists about 40 real ones (cadmiums, earths, quinacridones, phthalos, ...); `pg.mix(a, 2, b, 1)` mixes by volume.
- Tools (`pinsel.tools`): palette knife, oil crayon, airbrush, splatter, sponge, pens, smudge, stencils and masking fluid. Soft pastel and charcoal live in `pinsel.pastel`.
- Or let it paint from a photo: `pinsel.painterly.paint_image(cv, reference_rgb, style='impressionist')`, then work over it by hand.

The README lists everything that is simulated; `examples/` has complete paintings to read (`alpine_painting.py` is the showcase; `oil_crayon_study.py` and `moonlit_pond.py` are short).

## Budget

Measured peak memory (Rust kernels, version 0.2.0):

| Canvas | Oil | Oil + watercolour |
|---|---:|---:|
| 900 x 600 (`quickstart.py`) | 0.4 GB | |
| 1200 x 800 (`watercolor_demo.py`) | | 0.7 GB |
| 1500 x 1000, full-canvas glaze and wash | | 1.3 GB |
| 4500 x 3000 (a metre-wide print) | 4.3 GB | 7.9 GB |

Explore on a small canvas, paint the final one big. Time is mostly strokes:
a few thousand strokes take minutes on a few cores; the 30,000-stroke alpine
painting takes about two minutes on 32 cores. Watercolour only simulates
inside the wet area.

## Working on pinsel itself

- Python package `pinsel/`; Rust kernels in `rust/pinsel-core` with bindings in `rust/pinsel-py` (built with maturin). `docs/ARCHITECTURE.md` explains the engine.
- Tests: `pytest` (Python), `cargo test` in `rust/`. `rust/harness/verify.sh examples/...` checks that Rust and numba paint the same pictures, call by call.
- Plans and rules for the Rust port: `docs/RUST_PORT_PLAN.md`, `docs/RUST_PORTING_GUIDE.md`.
- Releases: pushing a `v*` tag builds and tests all wheels (`.github/workflows/wheels.yml`).
