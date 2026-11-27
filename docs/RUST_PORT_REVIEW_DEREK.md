# Review of the Rust port plan (17b2fea), by Derek

2026-11-27. One round. I read RUST_PORT_PLAN.md in full. I did not read the engine source, the studio logs or anything private, so points about what the kernels do are questions to check, not findings.

## Verdict: GO with changes. Four of them matter.

### 1. Decide the tile buffer before stage A, or every kernel gets ported twice

Stage A ports 62 kernels against full-size numpy arrays. Stage B then moves the canvas into 64×64 tiles, some f16, some absent. Every kernel that reads or writes canvas state has to be touched again, and the subagents will have copied the full-array pattern five times over.

Do a short stage 0 first: fix the buffer layout and one access type (a view a kernel can iterate, whether it is backed by a numpy array today or by tiles tomorrow), and port the first kernel against that. Then stage A is still "no behaviour change", but the kernels are written once. This also answers your question 3: the order A, B, C can stay, as long as B's layout is designed before A starts. Memory is what breaks, so I would also pull the two cheap memory wins (strip-wise PNG/TIFF output, no full-size render temporaries) forward into A.

### 2. Stage A can be tested much harder than per kernel, and should be

Your own argument gives you more than you are taking. If the randomness lives in the Python glue and the glue is unchanged in stage A, then **the whole image is reproducible with the same seed**. So besides the per-kernel tests:

- render every example with the old and the new kernels, same seeds, and compare the images directly: fraction of pixels with ΔE > 1, and the maximum. That catches drift over 30,000 strokes, which a per-call tolerance of 1e-4 cannot see.
- for the per-kernel tests, use an absolute floor as well as the relative tolerance (something like |a − b| ≤ 1e-5 + 1e-4·|b|). Relative error is meaningless where a result is a difference of two near-equal sums.
- check which kernels have **thresholds** (wet or not, tooth caught or not, mask on or off). Fastmath reordering can flip a comparison, and then the difference is not small. Report the count of flipped pixels separately instead of letting it fail the tolerance.
- check which kernels use `prange` with in-place writes that overlap. If any do, the numba result already depends on thread timing, and "reproduce the numba output" has no single answer. Record with one thread.

### 3. f16 is fine for reading and dangerous for accumulating

- `below_R` (a reflectance in 0..1, written once when a layer dries): f16 is right.
- Watercolor deposits and anything else that **builds up in small steps**: half precision has about three decimal digits. Adding a thin wash to an already loaded tile can be below half a unit in the last place and simply vanish. Twenty glazes would then look like three. That is a real bug class, and it is exactly the glaze-heavy case you flag under risks.
- So: keep a tile in f32 while it is wet or being written, and pack it to f16 only when it goes idle. The saving is nearly the same, because most of a large canvas is idle.

And the target needs its worst case written down. A full-canvas glaze makes every tile wet at once; that is what killed Nives's print. Twelve bands of film K and S in f32 over 13.5 million pixels is about 1.3 GB, plus base, plus the dry stack in f16 at about 0.3 GB, plus whatever the render fuses. Inside 7 GB looks reachable, but only if "a full-canvas glaze at 4500×3000 under a 7 GB cgroup, no swap" is itself the acceptance test. "Alpine under 1 GB" I believe.

### 4. The image test for the random parts would pass a wrong engine

ΔE2000 distribution plus SSIM against one reference tells you the picture looks roughly alike. A tool that deposits 20% too little pigment, or loses granulation, or mixes two pigments slightly wrong, passes. Add tests that have a right answer:

- **conservation:** paint loaded on the brush equals paint deposited plus paint left, per band; water in equals water on paper plus evaporated. Old and new engine, every tool.
- **single-mark scenes:** one stroke per tool on a blank ground, fixed path. Compare covered area, mean reflectance per band, and the edge profile.
- **distributions, not single images:** for the RNG-dependent tools render 20 seeds with each engine and compare the spread of those numbers, old against new.
- Laura's eye stays the last check, as you have it.

## Smaller points

- **scipy replacements are not Haiku work.** `gaussian_filter` (boundary mode, truncation), `zoom` (spline order, prefilter) and `sobel` have exact conventions, and a quiet mismatch shifts every picture. Use established crates where they exist (FFT, KD-tree), give the filters to Sonnet with scipy-parity tests on awkward sizes, and leave Haiku the recorder and scaffolding.
- **The Windows wheel is not "later".** Laura's laptop is Windows, and I paint there with her anaconda Python. Until a Windows wheel exists, the pure-Python engine must keep importing on that branch, or two of the house's painters are locked out.
- **Array properties that return copies** (`stroke_id`, `mask`, `film_vol` …): scripts that write into them will silently do nothing. Make the copies read-only so that fails loudly.
- **Expect 2× on CPU**: honest, and good that it is said up front.
- **Distribution** is Laura's call; a static index on hurtig.ai keeps the repo private and needs no token in a sandbox.

## Your five questions, in one line each

1. Sound, with an absolute floor, single-threaded recording, threshold flips counted, and the whole-image same-seed test added.
2. Alpine yes. The 7 GB target only with pack-on-idle and the full-glaze case as the test. Do not accumulate in f16.
3. Keep the order, but fix the tile layout first (stage 0) so kernels are ported once.
4. As written it would pass a wrong engine. Add conservation, single-mark scenes and seed distributions.
5. Move the scipy replacements up to Sonnet or to crates; the rest is right, and "subagent reports are not taken on trust" is the right rule.

Derek (who would like the oil crayon to survive the journey)
