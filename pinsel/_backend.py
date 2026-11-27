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
  PINSEL_TRACE=prefix        after every call of a ported kernel, write a hash
                             of the touched canvas region to <prefix>.<kernel>;
                             rust/harness/first_divergence.py compares runs
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


def _plain(v):
    """Return values as plain Python data, for comparing numba and Rust results."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (tuple, list)):
        return [_plain(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


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
            a = args[i]
            rargs[i] = (type(a)(np.array(x, copy=True) if isinstance(x, np.ndarray) else x for x in a)
                        if isinstance(a, (tuple, list)) else np.array(a, copy=True))
        rb = self.rust_fn(*rargs)
        nb = self.numba_fn(*args)
        rec = {'kernel': self.name, 'call': self.calls, 'ret_numba': _plain(nb), 'ret_rust': _plain(rb),
               'arrays': {}}
        if _plain(nb) != _plain(rb):
            self.bbox_mismatch += 1
        pairs = []
        for i, an in self.mutable.items():
            if isinstance(args[i], (tuple, list)):
                pairs += [(f'{an}[{k}]', r, n) for k, (r, n) in enumerate(zip(rargs[i], args[i]))
                          if isinstance(n, np.ndarray)]
            else:
                pairs.append((an, rargs[i], args[i]))
        for an, a, b in pairs:
            a, b = np.asarray(a), np.asarray(b)   # rust, numba
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
              f'return-value mismatches {self.bbox_mismatch}', file=sys.stderr)
        for an, w in self.worst.items():
            print(f'[pinsel A/B]   {an:9s} out-of-tolerance values {w["n_bad"]:7d} '
                  f'(in {w["calls_bad"]} calls), max |diff| {w["max_abs"]:.3g}', file=sys.stderr)
        if self.log:
            self.log.close()


TRACE = os.environ.get('PINSEL_TRACE', '')


class _Trace:
    """Write a hash of the touched canvas region after every call (Derek):
    two runs are bit-identical up to the first line where the hashes differ.
    Every 1000th call also hashes the whole canvas, which catches divergence
    caused outside this kernel."""

    def __init__(self, fn, path, canvas_args):
        import hashlib
        self._hl = hashlib
        self.fn, self.canvas_args = fn, canvas_args
        self.f = open(path, 'w', encoding='utf-8')
        self.n = 0
        atexit.register(self.f.close)

    def __call__(self, *args):
        bb = self.fn(*args)
        self.n += 1
        x0, y0, x1, y1 = map(int, bb)
        h = self._hl.blake2b(digest_size=10)
        h.update(np.array([x0, y0, x1, y1], np.int64).tobytes())
        if x1 >= 0:
            H, W = args[self.canvas_args[0]].shape[:2]
            ya, yb, xa, xb = max(0, y0 - 2), min(H, y1 + 3), max(0, x0 - 2), min(W, x1 + 3)
            for i in self.canvas_args:
                h.update(np.ascontiguousarray(args[i][ya:yb, xa:xb]).tobytes())
        full = '-'
        if self.n % 1000 == 0:
            g = self._hl.blake2b(digest_size=10)
            for i in self.canvas_args:
                g.update(np.ascontiguousarray(args[i]).tobytes())
            full = g.hexdigest()
        self.f.write(f'{self.n} {h.hexdigest()} {full}\n')
        return bb


class _TraceAny:
    """Trace for kernels without a bbox return: per call, a hash of a strided
    sample of every mutable argument (cheap), and every 1000th call a hash of
    the full arrays."""

    def __init__(self, fn, path, mutable):
        import hashlib
        self._hl = hashlib
        self.fn, self.mutable = fn, mutable
        self.f = open(path, 'w', encoding='utf-8')
        self.n = 0
        atexit.register(self.f.close)

    def __call__(self, *args):
        out = self.fn(*args)
        self.n += 1
        h = self._hl.blake2b(digest_size=10)
        g = self._hl.blake2b(digest_size=10) if self.n % 1000 == 0 else None
        if out is not None:
            h.update(repr(out if not isinstance(out, np.ndarray) else out.tobytes()).encode())
        for a in _arrays(args, self.mutable):
            a = np.ascontiguousarray(a)
            h.update(a.ravel()[::97].tobytes())
            if g is not None:
                g.update(a.tobytes())
        full = g.hexdigest() if g else '-'
        self.f.write(f'{self.n} {h.hexdigest()} {full}\n')
        return out


def _arrays(args, mutable):
    """The mutable arrays of a call, flattening tuple arguments (e.g. the film
    tuple F that tools kernels take)."""
    for i in mutable:
        a = args[i]
        if isinstance(a, (tuple, list)):
            yield from (x for x in a if isinstance(x, np.ndarray))
        else:
            yield a


def port(name, numba_fn, mutable, region_args=(), rust_name=None):
    """Dispatch one kernel: the Rust function ``pinsel_core.<rust_name or name>``
    when the wheel provides it, else ``numba_fn``.

    mutable     : {argument index: name} of every argument the kernel writes
                  (A/B mode copies and compares them; tuple arguments allowed)
    region_args : for kernels that return a bbox (x0, y0, x1, y1): the indices
                  of the canvas arrays to hash in that region (trace mode);
                  the first one must be 2-D (gives H, W)
    """
    rust_fn = getattr(_rs, rust_name or name, None) if _rs is not None else None
    if rust_fn is None:
        fn = numba_fn
    elif AB_EVERY > 0:
        return _AB(name, rust_fn, numba_fn, mutable)
    else:
        fn = rust_fn if USE_RUST else numba_fn
    if TRACE:
        if region_args:
            return _Trace(fn, f'{TRACE}.{name}', region_args)
        return _TraceAny(fn, f'{TRACE}.{name}', mutable)
    return fn


# Ported kernels. Mutable argument positions follow _kernels.paste_stroke.
paste_stroke = port('paste_stroke', _nb.paste_stroke, {
    12: 'tips', 13: 'res_K', 14: 'res_S', 15: 'res_vol', 17: 'tip_K', 18: 'tip_S',
    19: 'tip_vol', 35: 'film_K', 36: 'film_S', 37: 'film_vol', 38: 'wet', 39: 'gloss',
    40: 'body', 41: 'drate', 45: 'base_K', 46: 'base_S', 47: 'base_vol', 48: 'top_id'},
    region_args=(37, 35, 36, 38, 39, 40, 41, 45, 46, 47, 48))   # film_vol first (gives H, W)

fluid_stroke = port('fluid_stroke', _nb.fluid_stroke, {
    12: 'tips', 13: 'res_K', 14: 'res_S', 15: 'res_vol', 16: 'res_water',
    30: 'wc_water', 31: 'wc_K', 32: 'wc_S', 33: 'wc_m', 34: 'wc_g', 35: 'wc_st'},
    region_args=(30, 31, 32, 33, 34, 35))

render_reflectance = port('render_reflectance', _nb.render_reflectance, {13: 'out'})

bake_film = port('bake_film', _nb.bake_film, {
    1: 'below_R', 2: 'film_K', 3: 'film_S', 4: 'film_vol', 5: 'wet', 6: 'gloss', 7: 'body',
    8: 'drate', 9: 'h_dry', 10: 'gloss_dry', 12: 'base_K', 13: 'base_S', 14: 'base_vol'})

bake_deposit = port('bake_deposit', _nb.bake_deposit, {1: 'below_R', 2: 'dK', 3: 'dS'})
