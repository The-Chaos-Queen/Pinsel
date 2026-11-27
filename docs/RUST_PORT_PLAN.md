# Pinsel in Rust: port plan

Tern, 2026-11-27. Laura asked for a Rust engine because Pinsel is heavy on the cloud machines the painters work on. This plan gets one review round (Derek) before any code. Branch: `rust-port`.

## What we measured first (ML-WS, 32 cores, Python 3.14.4, numba 0.68.1)

| Run | Canvas | Wall | CPU | Peak memory |
|---|---|---:|---:|---:|
| `quickstart.py` (oil), cold numba cache | 900×600 | 9.9 s | 15 s | 0.59 GB |
| `quickstart.py`, warm cache | 900×600 | 7.0 s | 12 s | 0.54 GB |
| `alpine_painting.py` (oil, 30,763 strokes, timelapse), profiled | 1600×1100 | 99 s | 281 s | **2.62 GB** |
| `watercolor_demo.py`, cold cache, profiled | 1200×800 | 91 s | 171 s | 1.13 GB |
| `watercolor_demo.py`, warm cache | 1200×800 | 22.5 s | 185 s | 0.87 GB |

What this says:
- **Memory is the real limit.** Each pixel carries 12 spectral bands in about nine full-size float32 arrays (`below_R`, `film_K/S`, `base_K/S`, `wc.K/S/dK/dS`): about 760 MB at 1600×1100 before any temporaries. Peak memory scales with pixel count, so 4500×3000 needs roughly 20 GB. Nives Cargnel's *Soglia* print (7 GB workspace) was OOM-killed an hour in during a full-canvas glaze. It only finished with half-precision tiles, strip rendering and 6 GB of swap, in 1 h 45 min on one core (`studio/Nives Cargnel/STUDIO_LOG.md`).
- **Numba compiles about a minute on every fresh machine.** The watercolor demo spent roughly 65 of its first 91 s compiling. Cloud sandboxes usually start fresh, so painters pay this every session.
- **CPU:** the pixel loops are already compiled. Alpine profile: about 60% in the stroke kernels (`brush._run_kernel`), about 20% in timelapse previews and GIF encoding, about 10% in numpy small-array overhead (1.09 M `ufunc.reduce` calls), about 9% in the palette solver. The watercolor demo uses 185 CPU-seconds even though it takes 22 s on 32 cores; on a 4-core box that is about a minute. **Expect about 2× from Rust on CPU, not 10×.** The big wins are memory and no compile step.

## Decisions (Laura, 2026-11-27)

- Repo: `The-Chaos-Queen/Pinsel` (private), branch `rust-port`. The Python 0.1 engine gets the tag `v0.1-python` and stays available for re-rendering old works.
- **Same seed, different painting is fine.** The Rust engine uses its own PRNG. Laura: computer randomness isn't really random anyway, so any good generator will do. Deterministic parts are still compared exactly (see Verification).
- **Keep the Python API.** Painters keep writing `import pinsel as ps` scripts; the studio and example scripts must keep running.

## Shape: a Rust core under a thin Python package

```
Pinsel/
  rust/
    pinsel-core/   pure Rust: spectral, pigments, canvas state, brushes, kernels, watercolor, tools ...
    pinsel-py/     PyO3 bindings -> separate distribution "pinsel-core", module pinsel_core
                   (abi3, Python >= 3.10, built with maturin)
    harness/       A/B and same-seed comparison tools
  pinsel/          the Python package painters import; classes stay, heavy work calls pinsel_core
  pyproject.toml   unchanged (setuptools), so pinsel still installs where no wheel exists
```

(Stage 0 decision: the extension is its own small distribution, not `pinsel._core`. `pinsel` keeps installing anywhere, and `pinsel._backend` uses `pinsel_core` when it is installed, numba otherwise.)

The Python package stays the public face. A prebuilt wheel means `pip install` takes seconds: no compiler, no numba, no scipy. numpy stays a dependency because painters pass and get numpy arrays.

**Windows from the start (Derek):** Laura's laptop is Windows, and Derek paints there with her anaconda Python. So:
- the Python package falls back to the numba kernels whenever `pinsel_core` isn't installed, which keeps the old engine importable everywhere until wheels exist;
- the Windows wheel is built from stage A on, cross-compiled on ML-WS (maturin with cargo-xwin, target `x86_64-pc-windows-msvc`), so the laptop needs no Visual Studio.

## Stage 0, then three stages; each one leaves a working engine

**Stage 0: fix the buffer layout first (Derek).** Otherwise all 62 kernels get ported against full-size arrays and touched again in stage B.
- Define the canvas storage now: 64×64 tiles, per-array precision, absent tiles. Define one access type that kernels iterate over: a tile view that today wraps the numpy arrays and in stage B is backed by real tiles.
- Port the first kernel (`paste_stroke`) against that access type, with the Python binding, the fallback and the test harness.
- Subagents start from this working pattern.

**Stage A: swap the kernels (no behaviour change).**
- Port the 62 numba kernels (about 3,000 lines: `_kernels.py` 8, `tools.py` 26, `watercolor.py` 10 dual-compiled, `painterly.py` 5, `pastel.py` 3, `spectral.py` 2) to Rust, one by one, behind the same Python call signatures.
- Replace the scipy calls (`gaussian_filter`, `sobel`, `zoom`, `maximum_filter`, `uniform_filter`, `cKDTree`; FFT for noise) with Rust functions in `pinsel_core`.
- The Python glue stays as it is. After each kernel, all examples still run.
- Also pull the two cheap memory wins forward (Derek): strip-wise PNG/TIFF output, and no full-size render temporaries.
- End state: no numba and no scipy, so no compile minute. Speed about the same or a bit better. Memory a little lower.

**Stage B: memory (the main reason for the port).**
- The canvas state moves into Rust, stored in tiles (64×64).
- **Half precision only when a tile is idle (Derek).** f16 has about three decimal digits, so a thin wash added to a loaded f16 tile can fall below the last place and vanish: twenty glazes would look like three. A tile stays f32 while it is wet or being written, and is packed to f16 only when it goes idle. `below_R` (reflectance 0..1, written once when a layer dries) is f16 at rest. The saving is nearly the same, because most of a big canvas is idle.
- Allocate lazily: wet-film, base-layer and watercolor tiles exist only where paint or water is. Most of a big canvas is dry or untouched most of the time.
- Fuse rendering per tile: no full-size temporaries. Write PNG/TIFF in strips.
- `Canvas` keeps compatible Python properties for the few internals that scripts touch (`stroke_id`, `tooth`, `mask`, `wc.bbox`, `film_vol`, `base_vol`, found by grepping studio, examples and private scripts). Array properties materialize a numpy copy and are documented as the slow path.
- Built-in checkpoints: save and restore tiles incrementally to disk. Nives had to build this herself.
- Copies returned by array properties are read-only, so a script that writes into one fails loudly instead of silently doing nothing (Derek).
- Target: peak memory 3–5× lower. Alpine under 1 GB. **Acceptance test (Derek): a full-canvas glaze at 4500×3000 under a 7 GB cgroup, no swap.** That's the worst case, because every tile is wet at once, and it is what killed Nives's print. Estimate: film K and S at f32, about 1.3 GB, plus the base layer, plus `below_R` in f16, about 0.3 GB, plus the fused render.

**Stage C: speed.**
- Move the per-stroke glue into Rust: path resampling, bristle setup, pressure and angle profiles, loading.
- Parallel render, bake and glaze over tiles. Strokes stay sequential, because order matters in paint.
- Previews and timelapse only when asked for, rendered from a downscaled tile pyramid.

**After parity: new features** (from the studio logs and the code read):
1. Print output: 16-bit TIFF, DPI tag, embedded sRGB ICC profile.
2. Pencils: graphite and coloured pencil (hard, thin, pressure-sensitive, tooth-catching, burnishing).
3. Drips and runs for thinned paste paint (gravity). Watercolor can tilt; oil can't yet.

## Verification

- **Reference set** from Python 0.1 on ML-WS: every example's image, run time and peak memory, saved once under `rust/reference/`.
- **Whole-image tests in stage A (Derek, tightened after stage 0):** every ported kernel must be **bit-identical to its numba kernel compiled without fastmath** over the whole example (per-stroke canvas trace via `PINSEL_TRACE` and the PNG checksum). Against default numba (fastmath on), the first differing stroke and the number of differing strokes must equal numba's own fastmath ON-vs-OFF numbers. Stage 0 met both for `paste_stroke`; the final PNGs were pixel-identical even against default numba (`rust/harness/STAGE0_RESULTS.md`). (A first reading that the engine was chaotic was wrong: the differences came from unseeded randomness in the Python engine, now fixed.)
- **Kernel tests:** a recorder captures real kernel inputs while the examples run (arrays and scalars at each call, sampled), **recorded single-threaded**. Each Rust kernel must reproduce the numba output with `|a − b| ≤ 1e-5 + 1e-4·|b|`: an absolute floor, because relative error means nothing where a result is the difference of two nearly equal sums. Almost all randomness happens in Python before the kernel call. Only `tools._nb_seed` seeds numba's internal RNG; that tool is tested at image level. Also:
  - kernels with thresholds (wet or not, tooth caught or not, mask on or off) report **flipped pixels as a separate count**, because fastmath can flip a comparison and then the difference isn't small;
  - every `prange` loop is checked for overlapping in-place writes, because there the numba result already depends on thread timing.
- **Tests with a right answer (Derek),** because ΔE and SSIM alone would pass an engine that deposits 20% too little or loses granulation:
  - **conservation:** paint loaded = paint deposited + paint left, per band; water in = water on paper + evaporated; old and new engine, every tool;
  - **single-mark scenes:** one stroke per tool on a blank ground with a fixed path; compare covered area, mean reflectance per band and the edge profile;
  - **seed distributions:** for RNG-dependent tools, 20 seeds per engine; compare the spread of those numbers, old against new.
- **Image tests** once stage C changes the RNG: per-example ΔE2000 distribution and SSIM, plus side-by-side images that Laura looks at. She stays the last check.
- **Performance table** per stage: on ML-WS and on a constrained box that looks like a cloud machine (4 cores via `taskset`, 7 GB memory cap via a cgroup).

## Who does what

- **Me (Tern):** the architecture and the `pinsel_core` skeleton; the 8 core kernels in `_kernels.py`; the stage B canvas state; every review and merge; running all comparisons myself. Subagent reports are not taken on trust.
- **Sonnet subagents** (each on its own branch in a worktree, with a written contract, the Python original and the comparison tests):
  - watercolor (10 dual kernels and the sim glue);
  - tools part 1 (knife, airbrush, splatter, sponge);
  - tools part 2 (oil crayon, pens, smudge and blender, stencil, stipple);
  - pastel, charcoal and kneaded eraser;
  - painterly kernels and the palette solver;
  - **the scipy replacements (Derek):** `gaussian_filter` (boundary modes, truncation), `zoom` (spline order, prefilter) and `sobel` have exact conventions, and a quiet mismatch shifts every picture. Established crates for FFT and the KD-tree; Sonnet ports the filters with scipy parity tests on awkward sizes.
- **Haiku subagents:** the kernel input recorder, the comparison harness and test scaffolding only.
- Subagents start only after stage 0, so they copy a working pattern instead of inventing one each.

## Build and distribution

- Rust (rustup, user-level) and maturin on ML-WS. Linux x86_64 manylinux wheels match the cloud sandboxes. The Windows wheel is cross-built there from stage A on (see above).
- Source of truth is GitHub. ML-WS gets the source by `git push` over SSH to a bare repo there, so no GitHub token is placed on ML-WS.
- **Open question for Laura:** how do the painters get Pinsel today? The repo is private, so a cloud sandbox can't download a release wheel without a token. Options: make the repo public; or serve the wheels from hurtig.ai as a tiny static pip index (`pip install --index-url https://hurtig.ai/pinsel/simple pinsel`); or keep handing them the files as now.

## Risks

- `fastmath` and parallel reductions change float order. Hence the tolerance-based kernel tests, not bit equality.
- f16 storage can band in very thin glazes. Measure ΔE on glaze-heavy scenes (alpine, Soglia passes) before committing to it; fall back to f32 per array.
- Painters' scripts poking internals beyond the six found. The compatibility properties cover those; anything else surfaces as an `AttributeError` with a clear message.
- Cost: several subagents at once spend tokens in bursts. Sonnet and Haiku are the cheap end, and they run only once the pattern exists.

## Rough time (agent time)

Stage 0 about half a day; stage A about 2–3 days; stage B about 2 days; stage C about 1–2 days; features after that.

## Changes after Derek's review (2026-11-27, one round, `RUST_PORT_REVIEW_DEREK.md`: "GO with changes")

1. Stage 0 fixes the tile layout and the kernel access type before any kernel is ported, so kernels are written once. Strip-wise output and fused render move into stage A.
2. Stage A gets whole-image same-seed tests (fraction of pixels with ΔE > 1, max), an absolute floor on kernel tolerances, single-threaded recording, threshold flips counted separately, and a `prange` overlap check.
3. f16 only for idle tiles (pack on idle), never for accumulating; the acceptance test is a full-canvas glaze at 4500×3000 under 7 GB, no swap.
4. Conservation tests, single-mark scenes per tool, and 20-seed distributions, because ΔE and SSIM alone would pass a wrong engine.
5. scipy replacements go to crates and Sonnet with parity tests, not Haiku. The Windows wheel comes from stage A on, with a numba fallback until then. Array-property copies are read-only.

Derek would like the oil crayon to survive the journey. It's on the single-mark list.
