# pinsel 🖌️

*A physically inspired painting engine for Python.* You paint by placing
brushstrokes on a virtual canvas in code; pinsel simulates individual
bristles, paint loads, pigment mixing, canvas texture, drying, watercolour
flow and lighting of the paint relief.

```python
import pinsel as ps
from pinsel import pigments as pg, path as P

cv = ps.Canvas(1200, 800, ground='canvas')           # primed cotton canvas
cv.tone(pg.mix(pg.BURNT_SIENNA, 2, pg.YELLOW_OCHRE, 1).thinned(0.2))  # imprimatura

brush = ps.Brush('filbert', size=14)
brush.load(pg.mix(pg.ULTRAMARINE, 1, pg.TITANIUM_WHITE, 3))
cv.stroke(brush, P.spline([(100, 400), (400, 250), (800, 420), (1100, 300)]),
          pressure=P.taper(0.1, 0.3, 0.85))

cv.dry()                                              # let it dry ...
glaze = ps.Brush('flat', 30).load(pg.ALIZARIN_CRIMSON.thinned(0.8))
cv.stroke(glaze, P.line((100, 500), (1100, 500)))     # ... and glaze over it
cv.save('painting.png')
```

Requirements: Python ≥ 3.10, numpy, scipy, numba, Pillow.

## What is simulated

### Colour: spectral Kubelka-Munk pigments
* Paint colour lives in **12 spectral bands** (380–740 nm), not RGB. Every
  paint has an absorption `K` and scattering `S` spectrum per unit volume.
* Mixing adds `K` and `S` by volume → real subtractive behaviour: yellow +
  phthalo blue = vivid green, ultramarine + burnt sienna = neutral grey,
  a dab of phthalo dominates a mix, cadmium red + white = salmon pink.
* Layers are composited with the Kubelka-Munk layer equations, so thin
  transparent glazes tint what is underneath, opaque paints hide it, and
  thin edges of dark strokes reveal their hue.
* A library of ~40 pigments with realistic spectral shapes (cadmiums,
  earths, alizarin, quinacridones, ultramarine, phthalos, …) with
  granulation and staining properties; `Paint.from_color()` makes a paint
  from any sRGB colour, `paint.thinned()`, `.impasto()`, `.as_medium()`.
* Media: oil, acrylic, gouache, watercolour, ink (gloss, body/height,
  drying time, opacity boost).

### Brushes: individually simulated bristles
* Shapes: `round, sable, hog, flat, bright, filbert, angular, fan, rigger,
  mop, hake`.
* Each bristle has its own root position, length (pressure at which it
  touches), stiffness (tips lag behind in curves), contact radius and **its
  own paint reservoir**.
* Paint runs out bristle by bristle → strokes fade and break up into
  streaks; nearly empty brushes only touch the peaks of the canvas weave
  (dry brush & scumbling happen by themselves).
* Brushes **pick up wet paint** from the canvas into a loosely held tip
  reservoir (it smears forward and then slowly contaminates the brush), and
  stiff bristles **plough** wet paint into ridges.
* Pressure profiles, flat-brush angles (calligraphy), splay under pressure,
  speed, hand tremor, streaky multi-colour loading, side-loading,
  dipping a dirty brush, wiping, rinsing.

### Canvas
* Grounds: `canvas, linen, coarse_canvas, cold_press/paper, rough,
  hot_press, panel, smooth`, procedurally generated (weave with slubs,
  cellular paper tooth with fibres, gesso brush marks).
* Two wet paint layers (fresh strokes lie on top of older wet paint), dried
  layers baked into a spectral reflectance stack, paint relief height,
  gloss, drying by medium, masking (tape / masking fluid), toned grounds.
* Rendering with raking-light impasto shading and glossy highlights on wet
  oil, a fast preview renderer, timelapse recording (GIF), checkpoints.

### Watercolour (`pinsel.watercolor`)
A fluid simulation in the spirit of Curtis et al. (1997): virtual-pipe
shallow-water flow over the paper relief, capillary wicking through a fibre
raster (crisp wet-on-dry edges, feathery wet-in-wet), evaporation-driven
**edge darkening**, pigment transport/settling/re-wetting, **granulation**
into paper valleys, staining pigments that resist lifting, **backruns**.
`canvas.wait(seconds)` lets it flow and dry; helpers: `wet_paper`, `wash`
(flat/graded), `variegated_wash`, `drop` (charging), `spritz`, `spatter`,
`salt`, `blot` (tissue), `lift` (damp brush), `tilt`, `hair_dryer`.
Watercolour brushes (`paint.as_medium('watercolor')`) carry water and
pigment and a damp brush lifts paint back up.

### Tools beyond brushes (`pinsel.tools`)
* `PaletteKnife` – `spread`/`lay` thick slabs with crisp edges, squeeze-out
  ridges and skips over relief; `scrape` wet paint down to the layer below;
  `sgraffito` scratches. Marbled multi-colour knife loads.
* `Airbrush` – gaussian spray cone with droplet grain and overspray
  (`spray`, `dot`, `fill`), paste or watercolour.
* `Splatter` – `flick` (elongated drops, satellites, streaks), `tap`,
  `toothbrush` mist, `drops`.
* `Sponge` – porous sea-sponge texture (`dab`, `dabs`, `scatter`, `drag`),
  and lifting wet watercolour with a damp sponge.
* `OilCrayon` – draw on the tip for fine lines or roll onto the side for a
  broad rubbing. Light pressure catches the paper peaks. A firmer pass
  fills the grain and lays fresh pigment over the previous colour.
* `Pen` – technical, broad-nib and pointed (flex) dip pens: direction- and
  pressure-dependent width, ink pooling, running dry, feathering, `hatch`.
* `Smudge` (finger) and `Blender` (clean brush), `glaze(...)` helper,
  `Stencil` (tape, masking fluid, frisket; leaks into the paper tooth),
  `stipple` with blue-noise dot placement.

```python
import pinsel as ps
from pinsel import path as P, pigments as pg

cv = ps.Canvas(720, 320, ground='cold_press', seed=12)
stick = ps.OilCrayon(size=18, side=0.9).load(pg.ULTRAMARINE)
cv.stroke(stick, P.line((55, 155), (665, 155)), pressure=[0.2, 0.95])
edge = ps.OilCrayon(size=4).load(pg.YELLOW_OCHRE)
cv.stroke(edge, P.spline([(65, 170), (355, 146), (645, 164)]), pressure=0.85)
cv.save('oil_crayon.png')
```

`size` is the tip diameter in pixels at full pressure. Set `side` between
0 (tip) and 1 (side). The long axis follows the path crosswise by default.
Pass `angle=0` to keep it horizontal as the stroke turns.

**Wax resist.** Draw on the bare paper with an oil crayon, then run a
watercolour wash straight across the drawing. The wax follows the pigment
transferred by the stick. Press hard to reserve a bright shape, or skim
the surface to let the wash settle between flecks of colour.

```python
cv = ps.Canvas(360, 220, ground='cold_press', seed=17)
cream = pg.mix(pg.TITANIUM_WHITE, 12, pg.YELLOW_OCHRE, 1)
cv.stroke(ps.OilCrayon(36).load(cream), [(240, 68)], pressure=1.0)
cv.dry()
ps.watercolor.wash(cv, None, pg.ULTRAMARINE, concentration=1.4, seed=3)
cv.wait(8)
cv.dry()
cv.save('moon_wash.png')
```

### Painterly auto-painter & palette solver (`pinsel.painterly`)
* `PaletteSolver(palette)` works like a painter with a limited palette: for
  any target colour it finds a mixing recipe of 1–3 tube colours (+ white)
  whose Kubelka-Munk masstone matches best (CIEDE2000), e.g.
  *"26 parts titanium white + 11 parts phthalo blue + 5 parts alizarin"*.
  ~220k tabulated mixtures, k-d tree + Levenberg-Marquardt refinement, cached.
* `AutoPainter(reference, style=...)` / `paint_image(canvas, reference)`
  paints an image with the physical brushes, coarse to fine, in the spirit of
  Hertzmann's curved-stroke painterly rendering: error-driven stroke seeding,
  strokes grown along a structure-tensor flow field, colours mixed from real
  pigments, brushes that get dirty, run dry and are wiped or rinsed.
  Styles: `impressionist`, `expressionist`, `pointillist`, `loose_sketch`,
  `realist`, and `oil_crayon`. The crayon preset blocks in colour with the
  side of the stick and finishes edges with the tip. All styles support
  region masks, custom flow fields and importance maps.

```python
from pinsel.painterly import paint_image
cv = ps.Canvas(1000, 750, ground='canvas')
paint_image(cv, reference_rgb, style='impressionist', seed=1)
```

## Modules
| module | contents |
|---|---|
| `pinsel.canvas` | `Canvas`: state, time/drying, masks, rendering, timelapse |
| `pinsel.brush` | `Brush`: bristle simulation, loading, strokes |
| `pinsel.pigments` | `Paint`, `mix`, pigment library, media |
| `pinsel.path` | stroke paths (`line, bezier, spline, arc, spiral, wave, scribble…`) and pressure profiles (`taper, swell, flick, dab…`) |
| `pinsel.spectral` | spectral ↔ RGB, Kubelka-Munk |
| `pinsel.watercolor` | watercolour fluid simulation and wash helpers |
| `pinsel.tools` | `OilCrayon`, `PaletteKnife`, `Airbrush`, `Splatter`, `Sponge`, `Pen`, `Smudge`, `Blender`, `Stencil`, `glaze`, `stipple` |
| `pinsel.painterly` | `PaletteSolver`, `AutoPainter`, `paint_image`, style presets |
| `pinsel.grounds`, `pinsel.noise` | procedural grounds and noise |

## Examples
From the repository root, install with `python -m pip install -e .` and run `python examples/quickstart.py`. To run `two_readings.py`, first generate its base sheet with `python examples/untitled_marbling.py`.

* `examples/brush_sampler.py` – tour of brush & paint behaviour
* `examples/quickstart.py` – a minimal script
* `examples/oil_crayon_study.py` – two pears on a cobalt plate, drawn stroke by stroke
* `examples/moonlit_pond.py` – a cream-crayon moon and its reflection under an indigo wash
* `examples/watercolor_demo.py` – washes, wet-in-wet, granulation, backruns, salt, lifting
* `examples/tools_demo.py` – knife, airbrush, splatter, sponge, pens, stencils
* `examples/painterly_demo.py` – automatic painting of a reference in five styles
* `examples/alpine_painting.py` – the showcase painting
  ("Alpenglühen am Bergsee"), built up like a real oil painting from a
  12-colour palette
* `examples/alpine_watercolor.py` – the same view as a pen-and-watercolour sketch

## Gallery outputs
The examples generate images locally in `output/`. Rendered artwork is excluded from version control.

| file | what |
|---|---|
| `alpine_lake.png` | "Alpenglühen am Bergsee", oil on canvas (≈ 22 000 strokes, 12 pigments) |
| `alpine_lake_raking.png` | the same under raking light (shows the impasto) |
| `alpine_lake_timelapse.gif` | how it was painted |
| `alpine_watercolor.png` | the same view in pen & watercolour |
| `oil_crayon_study.png` | a night-window still life in oil crayon on cold-press paper |
| `moonlit_pond.png` | the moonlit pond in oil crayon and watercolour |
| `alpine_oil_crayon.png` | the alpine reference rendered with the `oil_crayon` preset on rough paper, using seven pigments |
| `brush_sampler.png`, `tools_demo*.png`, `watercolor_demo.png`, `painterly_*.png` | feature tours |

## Performance notes
Kernels are numba-compiled (first run compiles and caches, ~1 min). A
1600×1100 oil painting with ~20k strokes takes about 3–4 minutes on 4 cores.
Watercolour simulation runs only inside the wet bounding box.
