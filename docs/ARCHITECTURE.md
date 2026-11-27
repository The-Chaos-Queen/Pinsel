# pinsel – architecture & contracts

Python painting engine (numpy + Pillow; kernels in Rust via `pinsel_core`, or numba). Package root:
`pinsel/`. From the project root, install with `python -m pip install -e .`.

## Colour model (`spectral.py`, `pigments.py`)
* 12 spectral bands (`spectral.NB`), 380–740 nm.
* Paint colour = Kubelka-Munk absorption `K` and scattering `S` **per unit
  volume** (`Paint.K`, `Paint.S`, shape `(NB,)`).
* Mixing is linear in K and S weighted by volume. On the canvas we store
  *extensive* optical depths (K·x, S·x), so mixing on canvas = adding.
* A layer with optical depths (K, S) over a substrate with reflectance Rg:
  `spectral.km_over(K, S, Rg)` (numba scalar) / `km_composite_np` (numpy).
* `spectral.SPEC_TO_RGB` (3×NB) maps reflectance → linear sRGB.
* `Paint` attributes: `K, S, name, medium ('oil','acrylic','gouache','watercolor','ink'),
  fluid (bool), gloss, gloss_dry, body (height per volume), dry_time (s),
  granulation, staining, viscosity`. Helpers: `mix(a, 2, b, 1)`,
  `paint.thinned(f)`, `paint.as_medium('watercolor')`, `Paint.from_color(hex, opacity)`,
  `paint.rgb(thickness)`, library in `pigments.PIGMENTS` / `pigments.get(name, medium)`.

## Canvas state (`canvas.py`) — all float32
| array | shape | meaning |
|---|---|---|
| `h_ground` | H,W | ground relief (height units; 1 unit ≈ one normal oil stroke) |
| `tooth` | H,W | normalised local relief (≈ -1.6..1.6) for dry-brush contact |
| `below_R` | H,W,NB | reflectance of everything already dried |
| `h_dry`, `gloss_dry` | H,W | dried paint height / gloss |
| `film_K`, `film_S` | H,W,NB | wet paste-paint optical depths (additive) |
| `film_vol` | H,W | wet paint volume |
| `wet` | H,W | wetness 0..1 (volume-weighted); 0 → gets baked into below_R |
| `gloss`, `body`, `drate` | H,W | volume-weighted props of the wet film (drate = 1/dry_time) |
| `mask` | H,W | 0..1 protection (1 = fully masked) – every tool must respect it |
| `wc` | – | `WatercolorState` or None (`canvas.ensure_watercolor()`) |

`WatercolorState` (see `canvas.py`): `water` (surface water), `damp` (paper
moisture 0..1), `K,S` (suspended pigment, NB), `m,g,st` (suspended mass,
granulation-weighted mass, staining-weighted mass), `dK,dS` (deposited pigment,
NB), `dm,dg,dst`, `u (H,W+1)`, `v (H+1,W)` (staggered velocities, free for the
sim), `bbox=[y0,y1,x0,x1]` active region or None, `grow_bbox(x0,y0,x1,y1,H,W,pad)`.

Render order (Kubelka-Munk, bottom → top): `below_R` → watercolour deposit
(`wc.dK,dS`) → wet film + suspended pigment (`film_K + wc.K`, ...). Wet paper
is darkened slightly. Lighting uses `canvas.surface_height()` (ground texture
levelled by paint + `h_dry` + `film_vol*body`) for impasto shading and
specular highlights on glossy/wet paint.

Pixel (y, x) has its centre at integer coordinates (x, y).

## Time
* `canvas.wait(seconds)` → calls `watercolor.simulate(canvas, seconds)` when
  `wc.bbox` is not None, then dries paste films by `drate`; dry pixels are
  baked (`canvas._bake(sel)`).
* `canvas.dry()` → `watercolor.settle(canvas)`, bakes `wc.dK/dS` into
  `below_R`, bakes all wet film.

## Brushes (`brush.py`, kernels in `_kernels.py`)
`Brush(shape, size, ...)`, shapes: round, sable, hog, flat, bright, filbert,
angular, fan, rigger, mop, hake. `brush.load(paint, amount, mix, jitter, water)`,
`brush.load([(p1, w1), (p2, w2)], mix=0.3)` (streaky multi-colour),
`side_load`, `dip`, `wipe`, `clean`, `add_water`.
`canvas.stroke(brush, path, pressure=None|float|array|fn(t), angle=None|deg|fn,
angle_offset=0, speed=1, wobble=0, pressure_jitter=0, seed=None)`.
Paste paints → `paste_stroke` kernel exchanges with film arrays; fluid paints
(watercolour / ink) → `fluid_stroke` kernel exchanges water + pigment with
`wc.water, wc.K, wc.S, wc.m, wc.g, wc.st` and grows `wc.bbox`.

Reusable numba helpers in `_kernels.py`:
* `deposit_px(film_K, film_S, film_vol, wet, gloss, body, drate, y, x, dv, Kpv, Spv, p_gloss, p_body, p_drate)`
  – add `dv` volume of paint with per-volume K/S to the *top* layer at one pixel
  (note: blends properties by top-layer volume only; `tools.py` has
  two-layer-aware `_push/_dep/_take` helpers).
* Wet film has two sub-layers: `film_*` (top, latest stroke) and
  `base_K/base_S/base_vol` (older wet paint); `top_id`/`canvas.stroke_id`
  decide when a stroke pushes the old top down. `canvas.LEVELING` is the
  ground-levelling constant used by `surface_height`.
* `move_film(film_K, film_S, film_vol, y0, x0, y1, x1, frac)` – move a fraction of wet film.
* `bake_film`, `bake_deposit`, `render_reflectance`.

Paste-paint props to pass when depositing: `paint.gloss`, `paint.body`,
`1/paint.dry_time`.

## Paths (`path.py`)
Builders return (N,2) arrays: `line, polyline, bezier, quad, spline, arc,
circle, spiral, wave, zigzag, scribble, concat`; modifiers `resample, wobble,
transform, reverse, arclength`. Pressure profiles (functions of t∈[0,1]):
`constant, taper, ramp, swell, flick, dab, jittered`.

## Rules for contributors
* Own your module; **do not edit** `spectral.py, pigments.py, canvas.py,
  brush.py, _kernels.py, path.py, grounds.py, noise.py` (the core is being
  tuned concurrently; its public API above is stable). If you need a core
  change, work around it in your module and mention it in your report.
* Heavy per-pixel loops → numba `@njit(cache=True, fastmath=True)`; restrict
  work to bounding boxes. Keep float32 canvas arrays float32.
* Every tool must respect `canvas.mask` and call `canvas._after_stroke()`
  once per user-level mark (keeps timelapse recording working).
* Put demos in `examples/`, images in `output/`.
