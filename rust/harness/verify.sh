#!/bin/bash
# Stage A acceptance test (docs/RUST_PORT_PLAN.md, rust/harness/STAGE0_RESULTS.md):
# every ported kernel must be bit-identical to its numba kernel compiled
# WITHOUT fastmath, over whole examples.
#
#   bash rust/harness/verify.sh examples/watercolor_demo.py [more examples ...]
#
# Run from the root of a clone on ML-WS after rust/harness/build.sh. Makes a
# copy of the package with fastmath=False everywhere (so the non-ported numba
# kernels are identical in both runs), runs each example there with
# PINSEL_BACKEND=numba and =rust under PINSEL_TRACE, then compares every
# kernel's trace (first differing call) and every PNG the example wrote (MD5).
set -euo pipefail
ROOT=$(git rev-parse --show-toplevel)
V=$ROOT/venv/bin/python
NAME=$(basename "$ROOT")
COPY=/tmp/pinsel_verify_$NAME
rm -rf "$COPY"; mkdir -p "$COPY"
cp -r "$ROOT/pinsel" "$ROOT/examples" "$COPY/"
[ -d "$ROOT/tests" ] && cp -r "$ROOT/tests" "$COPY/"
find "$COPY" -name __pycache__ -prune -exec rm -rf {} +
# fastmath off everywhere (decorators and njit(...) calls)
grep -rl "fastmath=True" "$COPY/pinsel" | xargs sed -i 's/fastmath=True/fastmath=False/g'
echo "fastmath left on: $(grep -rc 'fastmath=True' "$COPY/pinsel" | awk -F: '{s+=$2} END {print s}')"
# guard: the runs must import the copy
(cd "$COPY" && PYTHONPATH="$COPY" "$V" -c "import pinsel, sys; assert pinsel.__file__.startswith('$COPY'), pinsel.__file__") || { echo "verify: pinsel does not import from the copy"; exit 2; }
T=$COPY/traces; mkdir -p "$T"
status=0
for ex in "$@"; do
  # scripts outside the copied examples (e.g. rust/harness/pastel_demo.py) run from the clone
  case "$ex" in /*) ;; *) [ -f "$COPY/$ex" ] || ex="$PWD/$ex" ;; esac
  name=$(basename "$ex" .py)
  for be in numba rust; do
    # the numba run uses scipy filters, the rust run the Rust ones
    if [ "$be" = numba ]; then fl=scipy; else fl=rust; fi
    rm -rf "$COPY/output"; mkdir -p "$COPY/output"
    t0=$(date +%s.%N)
    # PYTHONPATH: examples without their own sys.path line must import the
    # fastmath-off copy, not the editable install (found by the tools-film porter)
    if ! (cd "$COPY" && PYTHONPATH="$COPY" PINSEL_BACKEND=$be PINSEL_FILTERS=$fl PINSEL_TRACE=$T/$name.$be "$V" "$ex" > "$T/$name.$be.log" 2>&1); then
      echo "== $name: FAILED under $be (log $T/$name.$be.log)"; tail -5 "$T/$name.$be.log"; status=1; continue 2
    fi
    t1=$(date +%s.%N)
    mkdir -p "$T/$name.$be.out"; cp "$COPY"/output/*.png "$T/$name.$be.out/" 2>/dev/null || true
    awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.1f", b - a }' > "$T/$name.$be.seconds"
  done
  echo "== $name  (numba $(cat "$T/$name.numba.seconds") s, rust $(cat "$T/$name.rust.seconds") s)"
  for tr in "$T/$name".numba.*; do
    case "$tr" in *.log|*.seconds|*.out) continue;; esac
    k=${tr#$T/$name.numba.}
    if [ -f "$T/$name.rust.$k" ]; then
      "$V" "$ROOT/rust/harness/first_divergence.py" "$tr" "$T/$name.rust.$k" | sed "s#^.*: #   kernel $k: #"
    fi
  done
  for png in "$T/$name.numba.out"/*.png; do
    [ -e "$png" ] || continue
    f=$(basename "$png")
    a=$(md5sum < "$png" | cut -c1-12); b=$(md5sum < "$T/$name.rust.out/$f" 2>/dev/null | cut -c1-12 || echo missing)
    if [ "$a" = "$b" ]; then echo "   $f: identical (md5 $a)"; else echo "   $f: DIFFERENT (numba $a, rust $b)"; status=1; fi
  done
done
exit $status
