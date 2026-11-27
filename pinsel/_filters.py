"""Image filters, FFT and nearest-neighbour search for pinsel.

Same call signatures as the scipy / numpy functions pinsel uses (for the
arguments pinsel passes). With the optional ``pinsel_core`` wheel installed
they run in Rust, bit-identical to scipy 1.18 / numpy; without it (or for
arguments the Rust side does not cover: other dtypes, other boundary modes,
``order`` != the used ones) they fall back to scipy / numpy, so pinsel works
without the wheel on machines that have scipy.

Environment:
  PINSEL_FILTERS=scipy|rust  force a side (rust fails loudly if the wheel is
                             missing); default: rust when pinsel_core is
                             installed.

Functions (scipy name -> here, same name):
  gaussian_filter, sobel, uniform_filter, maximum_filter, zoom,
  binary_erosion, binary_dilation, binary_opening, binary_closing,
  KDTree (cKDTree), fftfreq, rfftfreq, rfft2, irfft2.
"""
from __future__ import annotations

import os

import numpy as np

try:
    import pinsel_core as _rs
except ImportError:  # no wheel on this machine
    _rs = None

_choice = os.environ.get('PINSEL_FILTERS', '').strip().lower()
if _choice not in ('', 'scipy', 'rust'):
    raise ValueError("PINSEL_FILTERS must be 'scipy' or 'rust'")
if _choice == 'rust' and (_rs is None or not hasattr(_rs, 'filters_gaussian_filter')):
    raise ImportError('PINSEL_FILTERS=rust, but pinsel_core with the filters is not installed')
USE_RUST = _rs is not None and hasattr(_rs, 'filters_gaussian_filter') and _choice != 'scipy'
NAME = 'rust' if USE_RUST else 'scipy'

_FLOATS = (np.dtype(np.float32), np.dtype(np.float64))


def _nd():
    from scipy import ndimage
    return ndimage


def _seq(v, n):
    """scipy's _normalize_sequence for numbers."""
    if np.isscalar(v) or np.ndim(v) == 0:
        return [v] * n
    v = list(v)
    if len(v) != n:
        raise RuntimeError('sequence argument must have length equal to input rank')
    return v


def _rust_ok(a):
    return USE_RUST and isinstance(a, np.ndarray) and a.dtype in _FLOATS and a.ndim >= 1 and a.size > 0


# ---------------------------------------------------------------------------
# gaussian_filter
# ---------------------------------------------------------------------------
def _gaussian_kernel(sd, truncate):
    """scipy.ndimage._filters._gaussian_kernel1d(sd, 0, radius) with numpy.
    Computed with numpy (not in Rust) because numpy's SIMD ``exp`` differs
    from libm in the last bit for some arguments on AVX-512 machines."""
    lw = int(truncate * sd + 0.5)
    sigma2 = sd * sd
    x = np.arange(-lw, lw + 1)
    phi = np.exp(-0.5 / sigma2 * x ** 2)
    return phi / phi.sum()


def gaussian_filter(input, sigma, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0):
    a = np.asarray(input)
    if (_rust_ok(a) and order == 0 and output is None and mode in ('reflect', 'nearest', 'wrap')):
        sig = [float(s) for s in _seq(sigma, a.ndim)]
        kernels = [(_gaussian_kernel(s, truncate) if s > 1e-15 else []) for s in sig]
        return _rs.filters_gaussian_filter(np.ascontiguousarray(a), sig, mode, float(truncate), kernels)
    return _nd().gaussian_filter(input, sigma, order=order, output=output, mode=mode, cval=cval,
                                 truncate=truncate)


# ---------------------------------------------------------------------------
# sobel, uniform_filter, maximum_filter
# ---------------------------------------------------------------------------
def sobel(input, axis=-1, output=None, mode='reflect', cval=0.0):
    a = np.asarray(input)
    if _rust_ok(a) and output is None and mode in ('reflect', 'nearest'):
        ax = int(axis) + a.ndim if axis < 0 else int(axis)
        if 0 <= ax < a.ndim:
            return _rs.filters_sobel(np.ascontiguousarray(a), ax, mode)
    return _nd().sobel(input, axis=axis, output=output, mode=mode, cval=cval)


def uniform_filter(input, size=3, output=None, mode='reflect', cval=0.0, origin=0):
    a = np.asarray(input)
    if (_rust_ok(a) and output is None and origin == 0 and np.ndim(size) == 0
            and mode in ('reflect', 'nearest', 'wrap')):
        return _rs.filters_uniform_filter(np.ascontiguousarray(a), int(size), mode)
    return _nd().uniform_filter(input, size, output=output, mode=mode, cval=cval, origin=origin)


def maximum_filter(input, size=None, footprint=None, output=None, mode='reflect', cval=0.0, origin=0):
    a = np.asarray(input)
    if (_rust_ok(a) and footprint is None and size is not None and output is None and origin == 0
            and np.ndim(size) == 0 and mode in ('reflect', 'nearest', 'wrap')):
        return _rs.filters_maximum_filter(np.ascontiguousarray(a), int(size), mode)
    return _nd().maximum_filter(input, size=size, footprint=footprint, output=output, mode=mode,
                                cval=cval, origin=origin)


# ---------------------------------------------------------------------------
# zoom
# ---------------------------------------------------------------------------
def zoom(input, zoom, output=None, order=3, mode='constant', cval=0.0, prefilter=True, *, grid_mode=False):
    a = np.asarray(input)
    if (_rust_ok(a) and output is None and not grid_mode and prefilter
            and ((order == 1 and mode == 'constant') or (order == 3 and mode == 'grid-wrap'))):
        z = [float(v) for v in _seq(zoom, a.ndim)]
        return _rs.filters_zoom(np.ascontiguousarray(a), z, int(order), mode, float(cval))
    return _nd().zoom(input, zoom, output=output, order=order, mode=mode, cval=cval,
                      prefilter=prefilter, grid_mode=grid_mode)


# ---------------------------------------------------------------------------
# binary morphology (default cross structure, border_value=0)
# ---------------------------------------------------------------------------
def _binary(name, input, iterations, kwargs):
    a = np.asarray(input)
    if (USE_RUST and a.dtype == np.bool_ and a.ndim == 2 and a.size > 0 and not kwargs
            and isinstance(iterations, (int, np.integer)) and iterations >= 1):
        return getattr(_rs, 'filters_' + name)(np.ascontiguousarray(a), int(iterations))
    return getattr(_nd(), name)(input, iterations=iterations, **kwargs)


def binary_erosion(input, iterations=1, **kwargs):
    return _binary('binary_erosion', input, iterations, kwargs)


def binary_dilation(input, iterations=1, **kwargs):
    return _binary('binary_dilation', input, iterations, kwargs)


def binary_opening(input, iterations=1, **kwargs):
    return _binary('binary_opening', input, iterations, kwargs)


def binary_closing(input, iterations=1, **kwargs):
    return _binary('binary_closing', input, iterations, kwargs)


# ---------------------------------------------------------------------------
# FFT (noise synthesis)
# ---------------------------------------------------------------------------
def fftfreq(n, d=1.0):
    if USE_RUST and d == 1.0 and n > 0:
        return _rs.filters_fftfreq(int(n))
    return np.fft.fftfreq(n, d)


def rfftfreq(n, d=1.0):
    if USE_RUST and d == 1.0 and n > 0:
        return _rs.filters_rfftfreq(int(n))
    return np.fft.rfftfreq(n, d)


def rfft2(a):
    a = np.asarray(a)
    if USE_RUST and a.ndim == 2 and a.size > 0 and a.dtype == np.float64:
        return _rs.filters_rfft2(np.ascontiguousarray(a))
    return np.fft.rfft2(a)


def irfft2(a, s=None):
    a = np.asarray(a)
    if USE_RUST and a.ndim == 2 and a.dtype == np.complex128 and s is not None and len(s) == 2 \
            and s[0] == a.shape[0] and s[1] // 2 + 1 == a.shape[1] and s[0] > 0 and s[1] > 0:
        return _rs.filters_irfft2(np.ascontiguousarray(a), int(s[1]))
    return np.fft.irfft2(a, s=s)


# ---------------------------------------------------------------------------
# KDTree
# ---------------------------------------------------------------------------
class KDTree:
    """scipy.spatial.cKDTree(data) with ``query(x, k=1, workers=1)``.

    Neighbours come back ordered by (distance, index); exact ties are
    resolved towards the lower index (cKDTree's tie order depends on its
    tree layout; see the report). Only the arguments pinsel passes are
    covered natively (``p=2``, ``eps=0``, no ``distance_upper_bound``); the
    scipy tree is built lazily as a fallback for anything else.
    """

    def __init__(self, data, *args, **kwargs):
        self._data = np.asarray(data, dtype=np.float64)
        self._args, self._kwargs = args, kwargs
        self._scipy = None
        self._rust = None
        if (USE_RUST and not args and not kwargs and self._data.ndim == 2 and self._data.shape[0] > 0
                and self._data.shape[1] > 0):
            self._rust = _rs.filters_KDTree(np.ascontiguousarray(self._data))
        else:
            self._scipy = self._build_scipy()

    def _build_scipy(self):
        from scipy.spatial import cKDTree
        return cKDTree(self._data, *self._args, **self._kwargs)

    @property
    def n(self):
        return self._data.shape[0]

    @property
    def data(self):
        return self._data

    def query(self, x, k=1, eps=0, p=2, distance_upper_bound=np.inf, workers=1):
        if self._rust is None or eps != 0 or p != 2 or distance_upper_bound != np.inf \
                or not isinstance(k, (int, np.integer)) or k < 1:
            if self._scipy is None:
                self._scipy = self._build_scipy()
            return self._scipy.query(x, k=k, eps=eps, p=p, distance_upper_bound=distance_upper_bound,
                                     workers=workers)
        x = np.asarray(x, dtype=np.float64)
        d = self._data.shape[1]
        if x.shape[-1] != d:
            raise ValueError('x must consist of vectors of length %d but has shape %s' % (d, x.shape))
        lead = x.shape[:-1]
        dist, idx = self._rust.query(np.ascontiguousarray(x.reshape(-1, d)), int(k), int(workers))
        if k == 1:
            return dist.reshape(lead), idx.reshape(lead)
        return dist.reshape(lead + (int(k),)), idx.reshape(lead + (int(k),))
