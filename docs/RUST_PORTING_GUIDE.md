# Porting guide for stage A (read this first)

Tern, 2026-11-27. For everyone porting a pinsel module from numba to Rust. Background: `docs/RUST_PORT_PLAN.md`; the pattern to copy: `paste_stroke` (stage 0) in `rust/pinsel-core/src/paste.rs`, `rust/pinsel-py/src/lib.rs` and `pinsel/_backend.py`; the results that set the bar: `rust/harness/STAGE0_RESULTS.md`.

## The bar

**Every kernel you port must be bit-identical to its numba kernel compiled without fastmath, over whole examples.** `rust/harness/verify.sh` checks exactly that: per-call traces with no differing call, and identical PNG checksums. Stage 0 met it for `paste_stroke` over 30,763 strokes. If one kernel can't meet it (e.g. it uses numba's internal RNG), stop, document why, and report it. Don't loosen the test.

## Rules

1. **Same Python call signature.** The Rust function takes the same positional arguments in the same order and returns the same thing as the numba kernel. The Python glue stays unchanged in stage A.
2. **Register through `_backend.port`** in the module that calls the kernel. Example:
   ```python
   from . import _backend as _B
   _k_flow1 = _B.port('wc_k_flow1', _k_flow1, mutable={3: 'water', 4: 'u'})
   ```
   `mutable` lists every argument index the kernel writes (tuple arguments, like the film tuple `F` in tools, are allowed). Kernels that return a bbox `(x0, y0, x1, y1)` may pass `region_args` (see `paste_stroke`). Rust function names carry a module prefix: `wc_`, `tools_`, `pastel_`, `painterly_`, `filters_`. Only kernels called from Python get a `port`; inline helpers become internal Rust functions.
3. **Number types exactly as numba types them.** Canvas arrays are float32. When numba mixes a float32 element with a float64 scalar, the arithmetic happens in f64 and rounds to f32 on store. float32 + float32 stays f32 (e.g. `fv = film_vol + base_vol`). Python `int(round(x))` is round-half-to-even (`f64::round_ties_even`). `math.floor`/`ceil` then `int`; `x ** y` is `powf` (`powi` only where numba also gets an integer power, as with literal `** 2`). Mirror every one of these; that's what makes results bit-identical.
4. **Keep the operation order.** Write expressions in the same order as the Python (left to right, same parenthesisation). Don't "simplify" `a * b * c` into `a * (b * c)`, and don't hoist or fuse arithmetic. Rust doesn't reorder floats by itself; you must not either.
5. **Canvas access only through `pinsel_core::grid`.** Wet paste film (film/base K and S, volumes, wet, gloss, body, drate, top_id): use `WetCanvas` (`px`, `px2`), as `paste.rs` does. Any other array: `Plane`/`PlaneMut` (H×W) or `Spec`/`SpecMut` (H×W×C). Never index raw slices with `y * w + x` in kernel code. Stage B swaps the storage under these types (tiles, bbox windows), and kernels written this way are never touched again.
6. **Parallel loops (`prange`, the watercolor `_kernel` dual compile).** Port serially first and get bit-identity. Then use row-parallelism (`rayon`) only where every row writes only its own cells and there is no cross-row reduction. Report every `prange` that writes overlapping cells or reduces across threads (Derek's review): there the numba result depends on thread timing.
7. **Don't change Python behaviour.** Only add the `port` wrappers. If you find a bug in the Python engine, report it with a reproducer; don't fix it (Tern decides). The reproducibility fix of 2026-11-27 is an example of what that looks like.
8. **Stay in your files.** Rust: `rust/pinsel-core/src/<your module>.rs`, `rust/pinsel-py/src/<your module>.rs` (its `register` function is already wired into `lib.rs`), tests in `rust/pinsel-core/tests/<your module>*.rs`. Python: only the `port` lines in your module. Need something shared or a change elsewhere? Ask in your report; don't edit it.
9. **Tests with a right answer (Derek).** For each kernel: a cargo test on synthetic inputs (no Python), and where paint or water moves, a conservation test (in = out, per band, to float32 rounding; see `rust/pinsel-core/tests/paste_backends.rs`).
10. **No new dependencies** without asking, except `rayon` for rule 6.

## Workflow

Your local worktree is on your own branch (`stageA/<name>`). Build and test on ML-WS (32 cores, Linux), never on Laura's laptop.

```bash
# once (your clone on ML-WS gets its own venv):
ssh ml-ws 'bash ~/pinsel/rust/harness/agent_setup.sh <name> stageA/<name>'
# each iteration, from your local worktree:
git add -A && git commit -m "..." && git push mlws stageA/<name>
ssh ml-ws 'cd ~/pinsel-agents/<name> && bash rust/harness/build.sh'
ssh ml-ws 'cd ~/pinsel-agents/<name> && bash rust/harness/verify.sh examples/<example>.py'
# per-call A/B on identical inputs (every N-th call, painting continues on numba):
ssh ml-ws 'cd ~/pinsel-agents/<name> && PINSEL_AB=20 venv/bin/python examples/<example>.py 2>&1 | grep "A/B"'
```

- If `agent_setup.sh` isn't in `~/pinsel` yet: `ssh ml-ws 'cd ~/pinsel && git pull'`.
- Push only to the `mlws` remote. **Never push to `origin` (GitHub)**; Tern merges.
- Quote carefully over ssh: single quotes inside a single-quoted ssh command break. For anything longer, write a script file, `scp` it, and run it.

## Report (your final message)

- which kernels are ported and registered;
- the `verify.sh` output for your examples (per kernel: first differing call or "None"; per PNG: identical or not), and the A/B summary;
- cargo test results (incl. conservation);
- anything not bit-identical and why; every `prange` with overlapping writes or reductions; any Python bug found;
- your branch's last commit hash.
