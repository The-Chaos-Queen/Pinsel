"""Kernel backend for the Rust port (see docs/RUST_PORT_PLAN.md).

The Rust kernels live in the optional package ``pinsel_core``. When it is
installed pinsel uses it, otherwise the numba kernels in ``_kernels``, so
pinsel keeps working on machines without the wheel.

Environment:
  PINSEL_BACKEND=numba|rust  force a backend (rust fails loudly if missing)
  PINSEL_AB=N                every N-th call of a ported kernel runs through
                             BOTH backends on copies of the same inputs and the
                             differences are logged; the painting continues
                             with numba's result, so the run stays the old
                             engine's run (the stage A kernel test)
  PINSEL_AB_LOG=path         where to write the per-call JSON lines
"""
from __future__ import annotations

import atexit
import json
import os
import sys

import numpy as np

from . import _kernels as _nb

try:
    import pinsel_core as _rs
except ImportError:  # no wheel on this machine
    _rs = None

_choice = os.environ.get('PINSEL_BACKEND', '').strip().lower()
if _choice == 'rust' and _rs is None:
    raise ImportError('PINSEL_BACKEND=rust, but pinsel_core is not installed')
USE_RUST = _rs is not None and _choice != 'numba'
NAME = 'rust' if USE_RUST else 'numba'
AB_EVERY = int(os.environ.get('PINSEL_AB', '0') or 0)
AB_LOG = os.environ.get('PINSEL_AB_LOG', '')

# Absolute floor + relative tolerance (Derek): |a - b| <= ATOL + RTOL * |b|
ATOL, RTOL = 1e-5, 1e-4


class _AB:
    """Run sampled calls through both backends and compare."""

    def __init__(self, name, rust_fn, numba_fn, mutable):
        self.name, self.rust_fn, self.numba_fn = name, rust_fn, numba_fn
        self.mutable = mutable          # {arg index: arg name}
        self.calls = 0
        self.sampled = 0
        self.worst = {}                 # arg name -> worst stats seen
        self.bbox_mismatch = 0
        self.log = open(AB_LOG, 'a', encoding='utf-8') if AB_LOG else None
        atexit.register(self.report)

    def __call__(self, *args):
        self.calls += 1
        if self.calls % AB_EVERY:
            return self.numba_fn(*args)
        self.sampled += 1
        rargs = list(args)
        for i in self.mutable:
            rargs[i] = np.array(args[i], copy=True)
        rb = self.rust_fn(*rargs)
        nb = self.numba_fn(*args)
        rec = {'kernel': self.name, 'call': self.calls,
               'bbox_numba': list(map(int, nb)), 'bbox_rust': list(map(int, rb)), 'arrays': {}}
        if tuple(map(int, nb)) != tuple(map(int, rb)):
            self.bbox_mismatch += 1
        for i, an in self.mutable.items():
            a, b = np.asarray(rargs[i]), np.asarray(args[i])   # rust, numba
            if a.dtype.kind in 'iu':
                nbad = int(np.count_nonzero(a != b))
                st = {'n_bad': nbad, 'max_abs': float(np.abs(a.astype(np.int64) - b).max()) if nbad else 0.0}
            else:
                with np.errstate(invalid='ignore'):
                    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
                    both_nan = np.isnan(a) & np.isnan(b)
                    d[both_nan] = 0.0
                    bad = d > ATOL + RTOL * np.abs(b.astype(np.float64))
                st = {'n_bad': int(np.count_nonzero(bad)), 'max_abs': float(np.nanmax(d)) if d.size else 0.0,
                      'n_changed': int(np.count_nonzero(d))}
            rec['arrays'][an] = st
            w = self.worst.setdefault(an, {'n_bad': 0, 'max_abs': 0.0, 'calls_bad': 0})
            w['n_bad'] += st['n_bad']
            w['calls_bad'] += st['n_bad'] > 0
            w['max_abs'] = max(w['max_abs'], st['max_abs'])
        if self.log:
            self.log.write(json.dumps(rec) + '\n')
        return nb

    def report(self):
        if not self.sampled:
            return
        print(f'[pinsel A/B] {self.name}: {self.sampled} of {self.calls} calls compared, '
              f'bbox mismatches {self.bbox_mismatch}', file=sys.stderr)
        for an, w in self.worst.items():
            print(f'[pinsel A/B]   {an:9s} out-of-tolerance values {w["n_bad"]:7d} '
                  f'(in {w["calls_bad"]} calls), max |diff| {w["max_abs"]:.3g}', file=sys.stderr)
        if self.log:
            self.log.close()


def _pick(name, mutable):
    numba_fn = getattr(_nb, name)
    rust_fn = getattr(_rs, name, None) if _rs is not None else None
    if rust_fn is None:
        return numba_fn
    if AB_EVERY > 0:
        return _AB(name, rust_fn, numba_fn, mutable)
    return rust_fn if USE_RUST else numba_fn


# Ported kernels. Mutable argument positions follow _kernels.paste_stroke.
paste_stroke = _pick('paste_stroke', {
    12: 'tips', 13: 'res_K', 14: 'res_S', 15: 'res_vol', 17: 'tip_K', 18: 'tip_S',
    19: 'tip_vol', 35: 'film_K', 36: 'film_S', 37: 'film_vol', 38: 'wet', 39: 'gloss',
    40: 'body', 41: 'drate', 45: 'base_K', 46: 'base_S', 47: 'base_vol', 48: 'top_id'})
