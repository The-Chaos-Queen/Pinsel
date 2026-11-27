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
    pinsel-py/     PyO3 bindings -> Python extension module pinsel._core (abi3, Python >= 3.10)
  pinsel/          the Python package painters import; classes stay, heavy work calls pinsel._core
  pyproject.toml   build with maturin
```

The Python package stays the public face. A prebuilt wheel means `pip install` takes seconds: no compiler, no numba, no scipy. numpy stays a dependency because painters pass and get numpy arrays.

## Three stages, each one leaves a working engine

**Stage A: swap the kernels (no behaviour change).**
- Port the 62 numba kernels (about 3,000 lines: `_kernels.py` 8, `tools.py` 26, `watercolor.py` 10 dual-compiled, `painterly.py` 5, `pastel.py` 3, `spectral.py` 2) to Rust, one by one, behind the same Python call signatures.
- Replace the scipy calls (`gaussian_filter`, `sobel`, `zoom`, `maximum_filter`, `uniform_filter`, `cKDTree`; FFT for noise) with Rust functions in `_core`.
- The Python glue stays as it is. After each kernel, all examples still run.
- End state: no numba and no scipy, so no compile minute. Speed about the same or a bit better. Memory unchanged.

**Stage B: memory (the main reason for the port).**
- The canvas state moves into Rust, stored in tiles (64×64).
- Spectral bands are stored at half precision (f16) where it is safe: the dry stack `below_R` and the watercolor deposits. Wet films stay f32 while they change; check this against the image error budget.
- Allocate lazily: wet-film, base-layer and watercolor tiles exist only where paint or water is. Most of a big canvas is dry or untouched most of the time.
- Fuse rendering per tile: no full-size temporaries. Write PNG/TIFF in strips.
- `Canvas` keeps compatible Python properties for the few internals that scripts touch (`stroke_id`, `tooth`, `mask`, `wc.bbox`, `film_vol`, `base_vol`, found by grepping studio, examples and private scripts). Array properties materialize a numpy copy and are documented as the slow path.
- Built-in checkpoints: save and restore tiles incrementally to disk. Nives had to build this herself.
- Target: peak memory 3–5× lower. Concretely, the alpine scene under 1 GB, and a 4500×3000 oil canvas inside 7 GB without swap.

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
- **Kernel tests (exact):** a recorder captures real kernel inputs while the examples run (arrays and scalars at each call, sampled). Each Rust kernel must reproduce the numba output on those inputs within float tolerance: relative 1e-4, with `fastmath` reordering allowed. This works because almost all randomness happens in Python before the kernel call. Only `tools._nb_seed` seeds numba's internal RNG; that tool gets an image-level test instead.
- **Image tests (statistical)** wherever the RNG changes: per-example ΔE2000 distribution and SSIM against the reference; mean ΔE and a high percentile with fixed thresholds per example; and side-by-side images that a human (Laura) looks at.
- **Performance table** per stage: on ML-WS and on a constrained box that looks like a cloud machine (4 cores via `taskset`, 7 GB memory cap via a cgroup).

## Who does what

- **Me (Tern):** the architecture and `_core` skeleton; the 8 core kernels in `_kernels.py`; the stage B canvas state; every review and merge; running all comparisons myself. Subagent reports are not taken on trust.
- **Sonnet subagents** (each on its own branch in a worktree, with a written contract, the Python original and the comparison tests):
  - watercolor (10 dual kernels and the sim glue);
  - tools part 1 (knife, airbrush, splatter, sponge);
  - tools part 2 (oil crayon, pens, smudge and blender, stencil, stipple);
  - pastel, charcoal and kneaded eraser;
  - painterly kernels and the palette solver.
- **Haiku subagents:** the kernel input recorder and comparison harness; the scipy replacements (filters, zoom, KD-tree); pigment and path tables; test scaffolding.
- Subagents start only after the skeleton, the canvas buffer layout and the first ported kernel exist, so they copy a working pattern instead of inventing one each.

## Build and distribution

- Rust (rustup, user-level) and maturin on ML-WS. Linux x86_64 manylinux wheels match the cloud sandboxes. A Windows wheel for Laura's laptop comes later.
- Source of truth is GitHub. ML-WS gets the source by `git push` over SSH to a bare repo there, so no GitHub token is placed on ML-WS.
- **Open question for Laura:** how do the painters get Pinsel today? The repo is private, so a cloud sandbox can't download a release wheel without a token. Options: make the repo public; or serve the wheels from hurtig.ai as a tiny static pip index (`pip install --index-url https://hurtig.ai/pinsel/simple pinsel`); or keep handing them the files as now.

## Risks

- `fastmath` and parallel reductions change float order. Hence the tolerance-based kernel tests, not bit equality.
- f16 storage can band in very thin glazes. Measure ΔE on glaze-heavy scenes (alpine, Soglia passes) before committing to it; fall back to f32 per array.
- Painters' scripts poking internals beyond the six found. The compatibility properties cover those; anything else surfaces as an `AttributeError` with a clear message.
- Cost: several subagents at once spend tokens in bursts. Sonnet and Haiku are the cheap end, and they run only once the pattern exists.

## Rough time (agent time)

Stage A about 2–3 days; stage B about 2 days; stage C about 1–2 days; features after that.
