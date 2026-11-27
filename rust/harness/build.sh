#!/bin/bash
# Pull, test and build inside a porter's clone on ML-WS, then install the
# wheel into the clone's venv.
#   bash rust/harness/build.sh            (run from the clone root)
set -euo pipefail
export PATH=$HOME/.cargo/bin:$PATH
ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"
git pull -q || true
V=$ROOT/venv/bin
( cd rust && cargo test -p pinsel-core --release 2>&1 | grep -E "^test |test result|error|warning: unused|panicked" )
( cd rust/pinsel-py && "$V/maturin" build --release -i "$V/python" 2>&1 | grep -E "Built wheel|error" )
WHL=$(ls -t rust/target/wheels/pinsel_core-*.whl | head -1)
"$V/pip" install -q --force-reinstall --no-deps "$WHL"
"$V/python" -c "import pinsel_core, pinsel._backend as B; print('pinsel_core', pinsel_core.__version__, '| functions:', sorted(n for n in dir(pinsel_core) if not n.startswith('_') and n != 'NB'), '| backend:', B.NAME)"
