# Stage 0 results (2026-11-27, ML-WS)

Stage 0 delivers: the canvas access type (`pinsel_core::grid`: `WetCanvas`, with the dense backend over today's numpy arrays and the tiled backend of stage B), `paste_stroke` ported against it, the Python module `pinsel_core` with the same 50-argument signature as the numba kernel, `pinsel._backend` (Rust when the wheel is installed, numba otherwise; A/B mode), and the same-seed harness.

## 1. Rust tests (`cargo test -p pinsel-core --release`)

| Test | Result |
|---|---|
| Same stroke on `DenseWet` and on `TiledWet` (150×190 canvas, partial tiles, mask, older wet paint, ploughing) | **bit-identical** canvas and brush state, same bbox |
| Paint conservation: brush + canvas optical depth K, before vs after | relative error **6.1e-10** (f32 rounding) |
| Lazy tiles | 6 of 9 tiles allocated; an untouched canvas allocates none |

## 2. Per-stroke A/B on identical inputs (`PINSEL_AB=N`)

Every N-th stroke runs through both kernels on copies of the same inputs. The painting continues with numba's result.

| Example | Strokes compared | Canvas arrays (film/base K,S, volumes, wet, gloss, body, drate, top_id) | Brush state (f64) | Out of tolerance | bbox mismatches |
|---|---:|---|---|---:|---:|
| quickstart (every 5th) | 210 of 1,050 | **bit-identical** | max \|diff\| 1e-9 | 0 | 0 |
| alpine_painting (every 50th) | 615 of 30,763 | **bit-identical** | max \|diff\| 1e-10 | 0 | 0 |

Tolerance: `|a − b| ≤ 1e-5 + 1e-4·|b|` (Derek). The brush differences are float reordering: numba compiles with `fastmath`.

## 3. Whole-image same-seed test

**Correction (same day, 12:55).** My first reading here was "the engine is chaotic": numba vs Rust differed in 0.3% of the alpine pixels (max ΔE 55), and so did numba with fastmath ON vs OFF. That was wrong. Derek asked for the index of the first stroke at which two runs differ. The trace (`PINSEL_TRACE`, `first_divergence.py`) showed that **even Rust vs Rust and numba vs numba differed between runs**: quickstart from stroke 1, alpine from stroke 26,901. The cause was a bug in the Python engine, not the port. `Canvas.tone`, `Brush.stroke` (wobble, pressure jitter), `SoftPastel.stroke`, `Smudge.stroke`, `glaze`, `Stencil.tape`/`masking_fluid` and `stipple` used fresh OS entropy when called without a seed, instead of the canvas's, brush's or tool's own seeded stream. So the same seeded script never painted the same picture twice. Fixed in commit "reproducibility fix" on this branch. The first table below is from before the fix; the numbers in it measure that randomness, not the port.

Before the reproducibility fix (superseded):

| Image | Comparison | Identical pixels | ΔE2000 > 1 | max ΔE |
|---|---|---:|---:|---:|
| quickstart | numba vs Rust | 96.55% | 0.021% | 7.1 |
| alpine_lake | numba vs Rust | 99.47% | 0.329% | 55.3 |
| alpine_lake | numba fastmath ON vs OFF | 99.48% | 0.317% | 55.8 |

**After the fix: first divergence per stroke** (canvas region touched by each stroke; full-canvas hash every 1,000 strokes):

| Comparison | quickstart (1,050 strokes) | alpine (30,763 strokes) |
|---|---|---|
| numba, 1 thread vs many | identical | identical |
| Rust run 1 vs run 2 | identical | identical |
| **numba fastmath OFF vs Rust** | **identical (every stroke, every checkpoint)** | **identical (every stroke, every checkpoint)** |
| numba fastmath ON vs Rust | first at stroke 13; 11 strokes differ | first at stroke 1; 42 strokes differ |
| numba fastmath ON vs OFF | first at stroke 13; 11 strokes differ | first at stroke 1; 42 strokes differ |

**After the fix: final images**

| Image | Comparison | Result |
|---|---|---|
| quickstart.png, alpine_lake.png | numba fastmath OFF vs Rust | **byte-identical files** (same MD5) |
| quickstart.png, alpine_lake.png, alpine_lake_raking.png | numba (fastmath ON, the default) vs Rust | **100.0000% identical pixels**, max ΔE 0.00 |

Reading:
- **The Rust kernel computes exactly what the numba kernel computes without fastmath**, over the whole painting.
- The only remaining difference is numba's fastmath rounding. It flips a threshold in 11 of 1,050 and 42 of 30,763 strokes; the same strokes, at the same first index, as numba against itself with fastmath switched. Those differences stay below 8-bit quantization: the delivered PNGs are pixel-identical.
- Derek's criteria: (a) first divergence of Rust vs numba = first divergence of numba ON vs OFF (stroke 13 and stroke 1); (b) the canvas is bit-identical up to there. Both hold. The floor over three numba variants (fastmath on/off, 1 vs many threads) is: threads 0, fastmath 11/42 strokes.

**Criterion for stage A (replaces the noise-floor criterion of the first reading):** every ported kernel must be **bit-identical to its numba kernel compiled without fastmath**, over the whole example (trace and PNG checksum). Against default numba, the first-divergence index and the number of differing strokes must equal numba's own fastmath ON-vs-OFF numbers. Per-stroke A/B, single-mark, conservation and seed-distribution tests come as planned.

## 4. Speed (not a stage 0 goal)

| Example | numba | Rust |
|---|---:|---:|
| quickstart | 7.1 s | 8.5 s |
| alpine_painting | 85.7 s | 93.3 s (rerun after the fix) |

No speedup yet, and the Rust kernel is even about 9% slower on alpine. Likely causes, to check in stage C: numba's fastmath vectorizes the 12-band loops and Rust's exact arithmetic doesn't; and each call extracts 33 numpy arrays through PyO3 with borrow checks. Speed isn't a stage 0 or A goal; memory (B) and speed (C) are. The wheel removes the compile minute on fresh machines (not visible here, warm cache). Speed comes in stage C.

## Files

- `rust/pinsel-core/src/grid.rs` access type and both backends; `paste.rs` the kernel; `tests/paste_backends.rs`.
- `rust/pinsel-py/` the bindings and the maturin config (wheel: `pinsel_core-0.1.0-cp310-abi3-manylinux_2_34_x86_64.whl`).
- `pinsel/_backend.py` backend switch and A/B; `pinsel/brush.py` calls `_backend.paste_stroke`.
- `rust/harness/same_seed.py` the whole-image test; results JSON in `rust/harness/results/`.
