"""numba when it is installed, plain Python otherwise.

pinsel's kernels are written for numba. With the Rust wheel (``pinsel_core``)
installed, ``_backend.port`` replaces every kernel that is called from Python,
so numba is not needed at all. Without numba the decorators below leave the
functions as plain Python: still correct, but far too slow for real canvases,
so ``_backend`` warns when neither Rust nor numba is there.
"""
from __future__ import annotations

try:
    from numba import get_num_threads, njit, prange
    HAVE_NUMBA = True
except ImportError:  # no numba on this machine
    HAVE_NUMBA = False
    prange = range

    def get_num_threads():
        return 1

    def njit(*args, **kwargs):
        """No-op stand-in for ``numba.njit`` (both ``@njit`` and ``@njit(...)``)."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f

__all__ = ['HAVE_NUMBA', 'get_num_threads', 'njit', 'prange']
