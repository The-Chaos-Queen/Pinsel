# Stage A results (2026-11-27)

Stage A of `docs/RUST_PORT_PLAN.md`: every numba kernel the engine calls and
every scipy/FFT call now has a Rust version in the `pinsel_core` wheel, behind
the unchanged Python API. Done by Tern (core) and five subagents (watercolor,
tools-film, pastel-painterly, filters, tools-spray: Sonnet; test scenes: Haiku),
each on its own branch, merged into `rust-port`.

## What is in the wheel

48 Rust functions, 71 cargo tests.

| Module | Kernels |
|---|---|
| core (`paste.rs`, `fluid.rs`, `render.rs`, `spectral.rs`) | `paste_stroke`, `fluid_stroke`, `render_reflectance` (rayon rows), `bake_film`, `bake_deposit`; Kubelka-Munk |
| watercolor | `wc_k_flow1`, `flow2`, `outflow`, `xpass`, `ypass`, `edge`, `local`, `salt`, `active`, `settle_all` (rayon above 90,000 cells, like numba), `fibres` |
| tools (film) | `knife`, `sgraffito`, `smudge`, `crayon`, `apply_volume_map`, `apply_fluid_map`; shared helpers in `tools_common.rs` |
| tools (spray) | `spray_accumulate`, `splat_discs`, `splat_drops`, `sponge_dab`, `pen_raster`, `feather` |
| pastel, painterly | `pastel_stroke`, `eraser_stroke`, `grow_strokes`, `densify` |
| filters (`pinsel/_filters.py`) | `gaussian_filter`, `sobel`, `uniform_filter`, `maximum_filter`, `zoom` (order 1 / 3 grid-wrap), binary erosion/dilation/opening/closing, `KDTree`, `fftfreq`, `rfftfreq`, `rfft2`, `irfft2` |

No numba helper is called from plain Python any more (checked with an AST
scan), so with the wheel installed nothing runs in numba.

## The bar, and the result

Bar: **bit-identical to the numba engine compiled with fastmath off**, over
whole example runs, with per-call traces (`rust/harness/verify.sh`).

Final run, `rust-port` at `1643f31` (all branches merged), ML-WS:
- 15 scripts (13 examples + `pastel_demo.py`, `spray_demo.py`, `two_readings_chain.py`), 29 PNGs: **all identical** (md5).
- 480 per-kernel trace lines: **0 with any difference** (first differing call `None` everywhere).
- The numba side ran with scipy filters, the Rust side with the Rust filters (`PINSEL_FILTERS`), so this covers the filters too.
- Against the shipped engine (numba with fastmath on), the pictures differ by a handful of pixels by 1 level of 255 (quickstart 2 px, tools_demo 3 px): fastmath flips, not the port.

Per-porter evidence (A/B against fastmath-on numba, fuzzing, conservation and
serial == parallel tests) is in each branch's report; summary in Tern's
journal 2026-11-27.

## Windows

- `rust/harness/build_windows.sh`: cross-built on ML-WS with zig for `x86_64-pc-windows-gnu` (no MSVC, no licence prompt), 1 MB wheel.
- **numba and scipy are no longer needed** (`pinsel/_jit.py`): with the wheel, pinsel needs only numpy and Pillow. Tested on Laura's laptop (CPython 3.11, numpy + Pillow + wheel, nothing else): quickstart, moonlit_pond and tools_demo run.
- quickstart and moonlit_pond paint the same pixels as Linux (540,000 and 367,200 of each).
- tools_demo differs between Linux and Windows in 1.2 % of pixels, but **the old numba engine differs between the two machines by exactly the same 11,294 pixels**. numpy's float64 `sin/cos/exp/log/power` give different last bits on the two platforms (Linux numpy uses SIMD/SVML code paths), and threshold decisions in the glue (splatter, fibres) amplify that. On each machine, Rust matches the shipped engine within 3 pixels by 1 level. Cross-platform identical output would need the glue's transcendentals in Rust (stage C), if wanted.

## Memory items pulled into stage A (Derek)

- **Strip-wise output** (`d09a13c`): `save` / `to_image` / `preview` render ~1 MP strips; `render(region=...)` pads by the lighting filters' full reach (10 px), so strips give the same pixels (`rust/harness/strip_render_check.py`: render, to_image, preview at 3 scales, bit-identical; examples' md5 unchanged). Saving a 3000 x 2000 canvas: peak 1,459 MB -> 280 MB.

## Stage B baseline (`rust/harness/stageB_acceptance.py`, 4500 x 3000, ML-WS)

| Scene | Peak resident memory | Wall |
|---|---:|---:|
| oil: tone, 40 strokes, full-canvas wet glaze, dry, save | **4.33 GB** | 2:08 |
| same + full-canvas watercolour wash | **7.94 GB** | 2:48 |

- The oil scene is already under Derek's 7 GB acceptance line (with the old full-size save it would have peaked at about 7.6 GB). Nives Cargnel's *Soglia* was oil, a ~4.4 GB canvas, killed during a full-canvas glaze in a 7 GB workspace.
- The watercolour scene is not: the watercolour state adds ~1.4 GB when the wash goes on (pages are committed on first write), and `dry` adds another ~2.3 GB (the settle simulation's scratch arrays plus the deposit layers coming alive).
- Floors at 4500 x 3000 with dense float32: five 12-band oil arrays 3.24 GB, four 12-band watercolour arrays 2.6 GB, ~20 single-band planes ~1.1 GB. So everything resident at once is ~7 GB before any scratch: stage B needs lazy (absent) tiles and idle-f16, not only fewer temporaries.

## Known deviations and follow-ups

- Watercolour kernels receive non-contiguous windows; the porter added stride-aware views (`Pl/PlM/Sp/SpM`) in `watercolor.rs`. `grid::Plane`/`Spec` should gain a row stride in stage B so there is one view type.
- `tools_sponge_dab` takes the 12 watercolour arguments as generic objects (paste paints pass one shared 1x1 dummy for several of them, which typed mutable borrows reject).
- `KDTree` ties resolve to the lower index (cKDTree's order depends on its tree layout); no ties occur in the palette tables.
- Verification harness bugs found on the way (all fixed): examples without a `sys.path` line imported the fastmath-on install (found by two porters independently); harness scripts outside `examples/` were not found; `two_readings` needs `untitled_marbling`'s PNG.
- pinsel's `pyproject.toml` still lists numba and scipy as dependencies; making them optional is a packaging decision for when the wheel is distributed.
