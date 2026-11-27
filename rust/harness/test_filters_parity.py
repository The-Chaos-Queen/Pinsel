"""Parity of pinsel._filters (Rust) against scipy / numpy.

Run on ML-WS in the clone:  venv/bin/python -m pytest -q rust/harness/test_filters_parity.py

Claims (all asserted with exact equality unless stated):
  * gaussian_filter, sobel, uniform_filter, maximum_filter, zoom (order 1
    with mode constant, order 3 with grid-wrap), binary_* : bit-identical to
    scipy.ndimage in float32 and float64.
  * KDTree.query: distances bit-identical; indices identical whenever the
    k nearest distances are distinct (ties: lower index first, scipy's tie
    order depends on its tree layout).
  * fftfreq / rfftfreq: bit-identical. rfft2 / irfft2: NOT bit-identical
    (rustfft vs pocketfft); asserted to 1e-13 relative to the max.
  * the noise fields pinsel generates (spectral_noise, fbm, worley, blur):
    bit-identical except for rare float32 elements, bounded below.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip('pinsel_core')
from scipy import ndimage as nd  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import pinsel._filters as F  # noqa: E402

assert F.USE_RUST, 'these tests are about the Rust side (PINSEL_FILTERS must not be scipy)'

DTYPES = [np.float32, np.float64]
SHAPES = [(17, 23), (1, 9), (9, 1), (1, 1), (2, 2), (5, 5), (64, 48), (7,), (3, 40, 3), (1, 5, 3)]


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (a.shape, b.shape)
    assert a.dtype == b.dtype, (a.dtype, b.dtype)
    assert np.array_equal(a, b), 'differs at %d of %d, max abs %g' % (
        (a != b).sum(), a.size, np.abs(a.astype(float) - b.astype(float)).max())


def data(shape, dt, seed=0):
    return np.random.default_rng(seed).standard_normal(shape).astype(dt)


# ---------------------------------------------------------------------------
@pytest.mark.parametrize('dt', DTYPES)
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('mode', ['reflect', 'nearest', 'wrap'])
@pytest.mark.parametrize('sigma', [0.35, 1.0, 2.5, 7.0, 40.0])
def test_gaussian_scalar_sigma(dt, shape, mode, sigma):
    a = data(shape, dt)
    same(nd.gaussian_filter(a, sigma, mode=mode), F.gaussian_filter(a, sigma, mode=mode))


@pytest.mark.parametrize('dt', DTYPES)
@pytest.mark.parametrize('mode', ['reflect', 'nearest', 'wrap'])
@pytest.mark.parametrize('sigma', [(2.0, 0.0), (0.0, 3.5), (0.0, 0.0), (1e-16, 2.0), (30.0, 0.5), (0.2, 0.2)])
def test_gaussian_per_axis_sigma_with_zeros(dt, mode, sigma):
    a = data((13, 29), dt, 1)
    same(nd.gaussian_filter(a, sigma, mode=mode), F.gaussian_filter(a, sigma, mode=mode))


@pytest.mark.parametrize('dt', DTYPES)
@pytest.mark.parametrize('sigma', [1.2, 6.0])
def test_gaussian_colour_image_as_painterly_calls_it(dt, sigma):
    a = data((31, 27, 3), dt, 2)
    same(nd.gaussian_filter(a, (sigma, sigma, 0), mode='nearest'),
         F.gaussian_filter(a, (sigma, sigma, 0), mode='nearest'))


@pytest.mark.parametrize('trunc', [2.0, 3.0, 4.0, 5.5])
def test_gaussian_truncate(trunc):
    a = data((20, 20), np.float64, 3)
    same(nd.gaussian_filter(a, 2.0, truncate=trunc), F.gaussian_filter(a, 2.0, truncate=trunc))


def test_gaussian_noncontiguous_and_fortran_input():
    a = data((40, 50), np.float64, 4)
    for v in (a[3:30, 5:41], a.T, a[::2, ::3], np.asfortranarray(a)):
        same(nd.gaussian_filter(v, 1.5, mode='reflect'), F.gaussian_filter(v, 1.5, mode='reflect'))


def test_gaussian_other_dtypes_fall_back_to_scipy():
    a = (data((10, 10), np.float64) * 50).astype(np.uint8)
    same(nd.gaussian_filter(a, 1.0), F.gaussian_filter(a, 1.0))


def test_gaussian_big_canvas_both_dtypes():
    for dt in DTYPES:
        a = data((300, 410), dt, 5)
        for mode, sg in (('reflect', 3.0), ('nearest', 0.7), ('wrap', 12.0)):
            same(nd.gaussian_filter(a, sg, mode=mode), F.gaussian_filter(a, sg, mode=mode))


# ---------------------------------------------------------------------------
@pytest.mark.parametrize('dt', DTYPES)
@pytest.mark.parametrize('shape', [s for s in SHAPES if len(s) >= 2])
@pytest.mark.parametrize('mode', ['nearest', 'reflect'])
def test_sobel(dt, shape, mode):
    a = data(shape, dt, 6)
    for ax in range(a.ndim):
        same(nd.sobel(a, axis=ax, mode=mode), F.sobel(a, axis=ax, mode=mode))
    same(nd.sobel(a, axis=-1, mode=mode), F.sobel(a, axis=-1, mode=mode))


@pytest.mark.parametrize('dt', DTYPES)
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('mode', ['reflect', 'nearest', 'wrap'])
def test_uniform_and_maximum(dt, shape, mode):
    a = data(shape, dt, 7)
    same(nd.uniform_filter(a, 3, mode=mode), F.uniform_filter(a, 3, mode=mode))
    same(nd.uniform_filter(a, 5, mode=mode), F.uniform_filter(a, 5, mode=mode))
    same(nd.maximum_filter(a, size=5, mode=mode), F.maximum_filter(a, size=5, mode=mode))
    same(nd.maximum_filter(a, size=3, mode=mode), F.maximum_filter(a, size=3, mode=mode))


def test_uniform_default_arguments():
    a = data((30, 31), np.float64, 8)
    same(nd.uniform_filter(a, 3), F.uniform_filter(a, 3))
    same(nd.maximum_filter(a, size=5), F.maximum_filter(a, size=5))


# ---------------------------------------------------------------------------
ZOOM1 = [2.0, 0.5, 1.0, 1.37, 3.0, 7.9, 0.1, 0.9, 1.04]


@pytest.mark.parametrize('dt', DTYPES)
@pytest.mark.parametrize('shape', [(10, 12), (1, 5), (4, 4), (30, 20), (2, 2), (3, 1), (1, 1), (9,), (7, 9, 3)])
@pytest.mark.parametrize('z', ZOOM1)
def test_zoom_order1_scalar_factor(dt, shape, z):
    a = data(shape, dt, 9)
    same(nd.zoom(a, z, order=1), F.zoom(a, z, order=1))


@pytest.mark.parametrize('dt', DTYPES)
def test_zoom_order1_as_painterly_calls_it(dt):
    rng = np.random.default_rng(10)
    for (h, w, H, W) in [(30, 40, 217, 301), (17, 23, 100, 100), (50, 50, 49, 51), (8, 8, 1000, 700)]:
        m = rng.random((h, w)).astype(dt)
        same(nd.zoom(m, (H / h, W / w), order=1), F.zoom(m, (H / h, W / w), order=1))
        m3 = rng.random((h, w, 3)).astype(dt)
        z = (H / h, W / w, 1)
        same(nd.zoom(m3, z, order=1), F.zoom(m3, z, order=1))


@pytest.mark.parametrize('dt', DTYPES)
@pytest.mark.parametrize('shape', [(10, 12), (7, 9), (5, 5), (4, 40), (4, 4), (6, 5)])
@pytest.mark.parametrize('s', [64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 3.5, 1.5])
def test_zoom_order3_gridwrap_as_fbm_calls_it(dt, shape, s):
    a = data(shape, dt, 11)
    same(nd.zoom(a, s, order=3, mode='grid-wrap'), F.zoom(a, s, order=3, mode='grid-wrap'))


def test_zoom_order3_fbm_grids():
    rng = np.random.default_rng(12)
    for (h, w, scale) in [(600, 800, 160.0), (721, 1013, 64.0), (257, 513, 51.4), (100, 100, 20.0)]:
        s = float(scale)
        for _ in range(5):
            gh, gw = int(np.ceil(h / s)) + 3, int(np.ceil(w / s)) + 3
            g = rng.standard_normal((gh, gw))
            same(nd.zoom(g, s, order=3, mode='grid-wrap'), F.zoom(g, s, order=3, mode='grid-wrap'))
            s = max(1.0, s / 2.0)


def test_zoom_unsupported_combinations_fall_back():
    a = data((10, 10), np.float64)
    same(nd.zoom(a, 2.0, order=3), F.zoom(a, 2.0, order=3))                # mode constant, order 3
    same(nd.zoom(a, 2.0, order=1, mode='nearest'), F.zoom(a, 2.0, order=1, mode='nearest'))


# ---------------------------------------------------------------------------
@pytest.mark.parametrize('shape', [(20, 30), (1, 7), (7, 1), (1, 1), (2, 2), (50, 41)])
@pytest.mark.parametrize('density', [0.2, 0.5, 0.8])
@pytest.mark.parametrize('iters', [1, 2, 6])
def test_binary_morphology(shape, density, iters):
    a = np.random.default_rng(13).random(shape) < density
    for name in ('binary_erosion', 'binary_dilation', 'binary_opening', 'binary_closing'):
        ref = getattr(nd, name)(a, iterations=iters)
        got = getattr(F, name)(a, iterations=iters)
        same(ref, got)
    same(nd.binary_opening(a, iterations=1) | (a & False), F.binary_opening(a, iterations=1))


def test_binary_as_watercolor_calls_it():
    rng = np.random.default_rng(14)
    ms = nd.gaussian_filter(rng.random((120, 160)), 2.0)
    hard = ms > 0.5
    ref = nd.binary_closing(nd.binary_opening(hard, iterations=1) | (ms > 0.97), iterations=1)
    got = F.binary_closing(F.binary_opening(hard, iterations=1) | (ms > 0.97), iterations=1)
    same(ref, got)


# ---------------------------------------------------------------------------
def brute(pts, q, k):
    d2 = ((q[:, None, :] - pts[None, :, :]) ** 2)
    # accumulate in dimension order like cKDTree
    s = np.zeros(d2.shape[:2])
    for j in range(pts.shape[1]):
        s = s + d2[:, :, j]
    order = np.lexsort((np.broadcast_to(np.arange(len(pts)), s.shape), s), axis=1)[:, :k]
    return np.sqrt(np.take_along_axis(s, order, 1)), order


@pytest.mark.parametrize('d', [1, 2, 3, 5])
@pytest.mark.parametrize('k', [1, 2, 7, 32])
def test_kdtree_random_points_match_scipy(d, k):
    rng = np.random.default_rng(15)
    pts = rng.uniform(-50, 50, (2000, d))
    q = rng.uniform(-60, 60, (500, d))
    a, b = cKDTree(pts), F.KDTree(pts)
    da, ia = a.query(q, k=k)
    db, ib = b.query(q, k=k)
    same(da, db)
    same(ia, ib)


def test_kdtree_worley_and_palette_shapes():
    rng = np.random.default_rng(16)
    h, w, cell = 90, 130, 12.0
    n = max(4, int(h * w / (cell * cell)))
    pts = np.column_stack([rng.uniform(-cell, h + cell, n), rng.uniform(-cell, w + cell, n)])
    yy, xx = np.mgrid[0:h, 0:w]
    q = np.column_stack([yy.ravel(), xx.ravel()])
    da, ia = cKDTree(pts).query(q, k=1, workers=-1)
    db, ib = F.KDTree(pts).query(q, k=1, workers=-1)
    same(da, db)
    same(ia, ib)
    lab = rng.uniform(-100, 100, (5000, 3))
    t = rng.uniform(-100, 100, (300, 3))
    same(cKDTree(lab).query(t, k=32, workers=-1)[1], F.KDTree(lab).query(t, k=32, workers=-1)[1])


def test_kdtree_k_larger_than_n_and_single_point_query():
    pts = np.random.default_rng(17).random((5, 2))
    a, b = cKDTree(pts), F.KDTree(pts)
    da, ia = a.query(pts[:2] + 0.01, k=8)
    db, ib = b.query(pts[:2] + 0.01, k=8)
    same(da, db)
    same(ia, ib)
    da, ia = a.query([0.3, 0.4], k=3)
    db, ib = b.query([0.3, 0.4], k=3)
    same(da, db)
    same(ia, ib)
    da, ia = a.query([0.3, 0.4])
    db, ib = b.query([0.3, 0.4])
    assert np.shape(da) == np.shape(db) == () and da == db and ia == ib


def test_kdtree_ties_resolve_to_lower_index_and_distances_match_scipy():
    rng = np.random.default_rng(18)
    pts = rng.integers(0, 12, (1500, 3)).astype(float)       # lattice: many exact ties, duplicates
    q = rng.integers(0, 12, (400, 3)).astype(float)
    for k in (1, 5):
        da, _ = cKDTree(pts).query(q, k=k)
        db, ib = F.KDTree(pts).query(q, k=k)
        same(da, db)                                           # distances always identical
        ed, ei = brute(pts, q, k)
        ib2 = ib.reshape(len(q), -1)
        assert np.array_equal(ib2, ei)                         # documented tie rule: (distance, index)


def test_kdtree_fallbacks():
    pts = np.random.default_rng(19).random((300, 2))
    q = np.random.default_rng(20).random((20, 2))
    same(cKDTree(pts).query(q, k=3, p=1)[0], F.KDTree(pts).query(q, k=3, p=1)[0])
    da, _ = cKDTree(pts).query(q, k=2, distance_upper_bound=0.05)
    db, _ = F.KDTree(pts).query(q, k=2, distance_upper_bound=0.05)
    same(da, db)


# ---------------------------------------------------------------------------
@pytest.mark.parametrize('n', [1, 2, 3, 7, 8, 100, 101, 1023, 1024])
def test_fftfreq_exact(n):
    same(np.fft.fftfreq(n), F.fftfreq(n))
    same(np.fft.rfftfreq(n), F.rfftfreq(n))


@pytest.mark.parametrize('shape', [(8, 8), (9, 7), (1, 16), (16, 1), (6, 10), (31, 33), (100, 101), (257, 513), (721, 1013)])
def test_irfft2_close(shape):
    h, w = shape
    g = np.random.default_rng(21)
    spec = g.standard_normal((h, w // 2 + 1)) + 1j * g.standard_normal((h, w // 2 + 1))
    ref = np.fft.irfft2(spec, s=shape)
    got = F.irfft2(spec, s=shape)
    assert ref.shape == got.shape and got.dtype == ref.dtype
    assert np.abs(ref - got).max() <= 1e-13 * np.abs(ref).max()


@pytest.mark.parametrize('shape', [(8, 8), (9, 7), (31, 33), (100, 101)])
def test_rfft2_close(shape):
    a = np.random.default_rng(22).standard_normal(shape)
    ref, got = np.fft.rfft2(a), F.rfft2(a)
    assert ref.shape == got.shape
    assert np.abs(ref - got).max() <= 1e-13 * np.abs(ref).max()


# ---------------------------------------------------------------------------
# The noise fields pinsel actually generates (pinsel.noise, unchanged), run once
# with scipy/numpy and once with the shim patched in.
class _Nd:
    zoom = staticmethod(F.zoom)
    gaussian_filter = staticmethod(F.gaussian_filter)


@pytest.fixture
def noise_pair():
    import scipy.spatial
    import pinsel.noise as nz
    orig_fft = (np.fft.irfft2, np.fft.fftfreq, np.fft.rfftfreq)
    orig = (nz.ndimage, nz.cKDTree)

    def run(fn, *a, **k):
        np.fft.irfft2, np.fft.fftfreq, np.fft.rfftfreq = orig_fft
        nz.ndimage, nz.cKDTree = orig
        x = fn(*a, **k)
        np.fft.irfft2 = lambda s_, s=None: F.irfft2(s_, s=s)
        np.fft.fftfreq, np.fft.rfftfreq = F.fftfreq, F.rfftfreq
        nz.ndimage, nz.cKDTree = _Nd, F.KDTree
        try:
            y = fn(*a, **k)
        finally:
            np.fft.irfft2, np.fft.fftfreq, np.fft.rfftfreq = orig_fft
            nz.ndimage, nz.cKDTree = orig
        return x, y
    return nz, run


@pytest.mark.parametrize('shape', [(600, 800), (257, 513), (721, 1013), (100, 1), (1, 100), (301, 300)])
@pytest.mark.parametrize('kw', [dict(beta=2.0), dict(beta=1.2, lowcut=0.02, aniso=(1.0, 8.0)),
                                dict(beta=3.0, highcut=0.3)])
def test_spectral_noise_field(noise_pair, shape, kw):
    nz, run = noise_pair
    x, y = run(nz.spectral_noise, shape, seed=7, **kw)
    assert x.dtype == y.dtype == np.float32 and x.shape == y.shape
    # float32 fields of unit std: FFT rounding (1e-16) only shows where a value
    # sits on a float32 rounding boundary
    assert (x != y).mean() <= 1e-5
    assert np.abs(x.astype(np.float64) - y).max() <= 1e-6


@pytest.mark.parametrize('shape', [(600, 800), (721, 1013), (101, 97)])
def test_fbm_worley_blur_fibers_bit_identical(noise_pair, shape):
    nz, run = noise_pair
    same(*run(nz.fbm, shape, scale=max(shape) / 5.0, octaves=5, seed=3))
    same(*run(nz.fbm, shape, scale=64.0, octaves=7, seed=3))        # reaches s = 1
    same(*run(nz.worley, shape, cell=12.0, seed=3))
    same(*run(nz.blur, np.random.default_rng(1).random(shape).astype(np.float32), 1.7))
    same(*run(nz.fibers, (200, 260), count=2000, seed=2))


# ---------------------------------------------------------------------------
def test_shim_falls_back_to_scipy_when_forced():
    code = ("import numpy as np, pinsel._filters as F;"
            "assert not F.USE_RUST and F.NAME=='scipy';"
            "a=np.random.default_rng(0).random((20,20));"
            "from scipy import ndimage as nd;"
            "assert np.array_equal(F.gaussian_filter(a,2.0),nd.gaussian_filter(a,2.0));"
            "assert np.array_equal(F.zoom(a,2.0,order=1),nd.zoom(a,2.0,order=1));"
            "d,i=F.KDTree(a).query(a[:3],k=2);print('ok')")
    env = dict(os.environ, PINSEL_FILTERS='scipy')
    root = os.path.join(os.path.dirname(__file__), '..', '..')
    r = subprocess.run([sys.executable, '-c', code], env=env, cwd=root, capture_output=True, text=True)
    assert r.returncode == 0 and 'ok' in r.stdout, r.stderr
