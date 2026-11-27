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

## 3. Whole-image same-seed test: the engine is chaotic

| Image | Comparison | Identical pixels | ΔE2000 > 1 | max ΔE |
|---|---|---:|---:|---:|
| quickstart | numba vs **Rust** | 96.55% | 0.021% | 7.1 |
| quickstart | numba fastmath ON vs numba fastmath OFF | 96.53% | 0.015% | 3.6 |
| alpine_lake | numba vs **Rust** | 99.47% | 0.329% | 55.3 |
| alpine_lake | numba fastmath ON vs numba fastmath OFF | 99.48% | 0.317% | 55.8 |
| alpine_lake | numba fastmath OFF vs **Rust** | 99.46% | – | 53.3 |

Reading:
- A 1e-10 difference in a bristle's reservoir carries from stroke to stroke until a threshold tips somewhere (a bristle touches or not, `room > 0`, contact). From there a streak runs differently, and the canvas state it leaves changes later pickups. Over 30,000 strokes that gives a few local differences, 0.3% of the pixels, each up to ΔE ≈ 55 (a streak in another place).
- **The old engine does exactly the same to itself.** Switching only numba's `fastmath` flag produces the same amount of difference as switching to Rust. Not even numba with fastmath off matches numba with fastmath on, or Rust.
- So the whole-image test can't be bit-exact for this engine, whatever the port. Derek's premise ("the whole image is reproducible with the same seed") holds only for one fixed compiler and flags.

**Adjusted criterion for stage A:** for every example, the Rust-vs-numba difference must be no larger than numba's own fastmath ON-vs-OFF difference (identical-pixel fraction, ΔE > 1 fraction, max ΔE). That's the engine's float-noise floor. On top of that, the per-stroke A/B must stay bit-identical or within tolerance on the canvas, with flipped thresholds counted separately. Single-mark, conservation and seed-distribution tests come as planned.

## 4. Speed (not a stage 0 goal)

| Example | numba | Rust |
|---|---:|---:|
| quickstart | 7.1 s | 8.5 s |
| alpine_painting | 91.2 s | 92.9 s |

No speedup yet, as expected: the numba kernel was already compiled, and the run time is dominated by the glue and the other kernels. The wheel removes the compile minute on fresh machines (not visible here, warm cache). Speed comes in stage C.

## Files

- `rust/pinsel-core/src/grid.rs` access type and both backends; `paste.rs` the kernel; `tests/paste_backends.rs`.
- `rust/pinsel-py/` the bindings and the maturin config (wheel: `pinsel_core-0.1.0-cp310-abi3-manylinux_2_34_x86_64.whl`).
- `pinsel/_backend.py` backend switch and A/B; `pinsel/brush.py` calls `_backend.paste_stroke`.
- `rust/harness/same_seed.py` the whole-image test; results JSON in `rust/harness/results/`.
