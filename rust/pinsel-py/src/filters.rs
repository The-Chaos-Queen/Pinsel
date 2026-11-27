//! Bindings for pinsel_core::filters. Register each function in `register`.
//! Python names carry the module prefix (see docs/RUST_PORTING_GUIDE.md):
//! `filters_gaussian_filter`, `filters_sobel`, ...  The Python shim
//! `pinsel/_filters.py` is the only caller; it passes C-contiguous float32
//! or float64 arrays.

use numpy::{Complex64, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pcore::filters as f;
use pcore::filters::Mode;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use rustfft::num_complex::Complex64 as FComplex;

use crate::util::ro;

fn mode_of(s: &str) -> PyResult<Mode> {
    Mode::parse(s).ok_or_else(|| PyValueError::new_err(format!("pinsel_core filters: unsupported mode '{s}'")))
}

/// Run `$body` on the f64 or f32 contents of `$a`: binds `$x` (slice),
/// `$shape` (Vec<usize>) and the type alias `$T`.
macro_rules! with_float {
    ($py:expr, $a:expr, $x:ident, $shape:ident, $T:ident, $body:block) => {{
        if let Ok(arr) = $a.extract::<PyReadonlyArrayDyn<'_, f64>>() {
            type $T = f64;
            let $shape: Vec<usize> = arr.shape().to_vec();
            let $x: &[f64] = ro(&arr, "a")?;
            $body
        } else if let Ok(arr) = $a.extract::<PyReadonlyArrayDyn<'_, f32>>() {
            type $T = f32;
            let $shape: Vec<usize> = arr.shape().to_vec();
            let $x: &[f32] = ro(&arr, "a")?;
            $body
        } else {
            Err(PyValueError::new_err("pinsel_core filters: expected a C-contiguous float32 or float64 array"))
        }
    }};
}

fn to_array<'py, T: numpy::Element>(py: Python<'py>, data: Vec<T>, shape: &[usize]) -> PyResult<Bound<'py, PyAny>> {
    let flat = numpy::PyArray1::from_vec(py, data);
    let arr: Bound<'py, PyArrayDyn<T>> = flat.reshape(shape.to_vec())?;
    Ok(arr.into_any())
}

#[pyfunction]
#[pyo3(signature = (a, sigmas, mode, truncate=4.0))]
fn filters_gaussian_filter<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    sigmas: Vec<f64>,
    mode: &str,
    truncate: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let m = mode_of(mode)?;
    with_float!(py, a, x, shape, T, {
        if sigmas.len() != shape.len() {
            return Err(PyValueError::new_err("sigma must have one entry per axis"));
        }
        let out: Vec<T> = py.detach(|| f::gaussian_filter::<T>(x, &shape, &sigmas, m, truncate));
        to_array(py, out, &shape)
    })
}

#[pyfunction]
fn filters_sobel<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, axis: usize, mode: &str) -> PyResult<Bound<'py, PyAny>> {
    let m = mode_of(mode)?;
    with_float!(py, a, x, shape, T, {
        if axis >= shape.len() {
            return Err(PyValueError::new_err("axis out of range"));
        }
        let out: Vec<T> = py.detach(|| f::sobel::<T>(x, &shape, axis, m));
        to_array(py, out, &shape)
    })
}

#[pyfunction]
#[pyo3(signature = (a, size, mode, cval=0.0))]
fn filters_uniform_filter<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    size: usize,
    mode: &str,
    cval: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let m = mode_of(mode)?;
    with_float!(py, a, x, shape, T, {
        let out: Vec<T> = py.detach(|| f::uniform_filter::<T>(x, &shape, size, m, cval));
        to_array(py, out, &shape)
    })
}

#[pyfunction]
#[pyo3(signature = (a, size, mode, cval=0.0))]
fn filters_maximum_filter<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    size: usize,
    mode: &str,
    cval: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let m = mode_of(mode)?;
    with_float!(py, a, x, shape, T, {
        let out: Vec<T> = py.detach(|| f::maximum_filter::<T>(x, &shape, size, m, cval));
        to_array(py, out, &shape)
    })
}

#[pyfunction]
#[pyo3(signature = (a, zoom, order, mode, cval=0.0))]
fn filters_zoom<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    zoom: Vec<f64>,
    order: usize,
    mode: &str,
    cval: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let m = mode_of(mode)?;
    if !(order == 1 || (order == 3 && m == Mode::GridWrap)) {
        return Err(PyValueError::new_err("filters_zoom: only order=1 (any of constant/nearest) and order=3 with grid-wrap"));
    }
    if order == 1 && !matches!(m, Mode::Constant | Mode::Nearest | Mode::GridWrap) {
        return Err(PyValueError::new_err("filters_zoom: unsupported mode"));
    }
    with_float!(py, a, x, shape, T, {
        if zoom.len() != shape.len() {
            return Err(PyValueError::new_err("zoom must have one entry per axis"));
        }
        let (out, oshape): (Vec<T>, Vec<usize>) = py.detach(|| f::zoom::<T>(x, &shape, &zoom, order, m, cval));
        to_array(py, out, &oshape)
    })
}

fn bool_in<'a>(arr: &'a PyReadonlyArrayDyn<'_, bool>) -> PyResult<(&'a [bool], usize, usize)> {
    let s = arr.shape();
    if s.len() != 2 {
        return Err(PyValueError::new_err("binary morphology: expected a 2-D boolean array"));
    }
    Ok((ro(arr, "a")?, s[0], s[1]))
}

fn binary_op(py: Python<'_>, a: &Bound<'_, PyAny>, iterations: usize, which: u8) -> PyResult<Py<PyAny>> {
    let arr: PyReadonlyArrayDyn<'_, bool> = a.extract()?;
    let (x, h, w) = bool_in(&arr)?;
    let bytes: Vec<u8> = x.iter().map(|&b| b as u8).collect();
    let out = py.detach(|| match which {
        0 => f::binary_morph(&bytes, h, w, false, iterations),
        1 => f::binary_morph(&bytes, h, w, true, iterations),
        2 => f::binary_opening(&bytes, h, w, iterations),
        _ => f::binary_closing(&bytes, h, w, iterations),
    });
    let b: Vec<bool> = out.into_iter().map(|v| v != 0).collect();
    Ok(to_array(py, b, &[h, w])?.unbind())
}

#[pyfunction]
#[pyo3(signature = (a, iterations=1))]
fn filters_binary_erosion(py: Python<'_>, a: &Bound<'_, PyAny>, iterations: usize) -> PyResult<Py<PyAny>> {
    binary_op(py, a, iterations, 0)
}
#[pyfunction]
#[pyo3(signature = (a, iterations=1))]
fn filters_binary_dilation(py: Python<'_>, a: &Bound<'_, PyAny>, iterations: usize) -> PyResult<Py<PyAny>> {
    binary_op(py, a, iterations, 1)
}
#[pyfunction]
#[pyo3(signature = (a, iterations=1))]
fn filters_binary_opening(py: Python<'_>, a: &Bound<'_, PyAny>, iterations: usize) -> PyResult<Py<PyAny>> {
    binary_op(py, a, iterations, 2)
}
#[pyfunction]
#[pyo3(signature = (a, iterations=1))]
fn filters_binary_closing(py: Python<'_>, a: &Bound<'_, PyAny>, iterations: usize) -> PyResult<Py<PyAny>> {
    binary_op(py, a, iterations, 3)
}

#[pyfunction]
fn filters_fftfreq<'py>(py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyAny>> {
    if n == 0 {
        return Err(PyValueError::new_err("n must be positive"));
    }
    to_array(py, f::fftfreq(n), &[n])
}

#[pyfunction]
fn filters_rfftfreq<'py>(py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyAny>> {
    if n == 0 {
        return Err(PyValueError::new_err("n must be positive"));
    }
    to_array(py, f::rfftfreq(n), &[n / 2 + 1])
}

/// `np.fft.irfft2(spec, s=(h, w))` for a complex128 (h, w//2+1) array.
#[pyfunction]
fn filters_irfft2<'py>(py: Python<'py>, spec: &Bound<'py, PyAny>, w: usize) -> PyResult<Bound<'py, PyAny>> {
    let arr: PyReadonlyArrayDyn<'_, Complex64> = spec.extract()?;
    let s = arr.shape().to_vec();
    if s.len() != 2 || s[1] != w / 2 + 1 || s[0] == 0 || w == 0 {
        return Err(PyValueError::new_err("irfft2: spec must be (h, w // 2 + 1) complex128"));
    }
    let h = s[0];
    let data: Vec<FComplex> = ro(&arr, "spec")?.iter().map(|c| FComplex::new(c.re, c.im)).collect();
    let out = py.detach(|| f::irfft2(&data, h, w));
    to_array(py, out, &[h, w])
}

/// `np.fft.rfft2(a)` for a float64 (h, w) array: complex128 (h, w//2+1).
#[pyfunction]
fn filters_rfft2<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr: PyReadonlyArrayDyn<'_, f64> = a.extract()?;
    let s = arr.shape().to_vec();
    if s.len() != 2 || s[0] == 0 || s[1] == 0 {
        return Err(PyValueError::new_err("rfft2: expected a non-empty 2-D float64 array"));
    }
    let x = ro(&arr, "a")?;
    let out = py.detach(|| f::rfft2(x, s[0], s[1]));
    let v: Vec<Complex64> = out.into_iter().map(|c| Complex64::new(c.re, c.im)).collect();
    to_array(py, v, &[s[0], s[1] / 2 + 1])
}

/// Exact k-d tree with scipy's `cKDTree(pts).query(q, k)` results.
#[pyclass(name = "filters_KDTree")]
struct KdTree {
    tree: f::KdTree,
    d: usize,
}

#[pymethods]
impl KdTree {
    #[new]
    fn new(pts: PyReadonlyArrayDyn<'_, f64>) -> PyResult<Self> {
        let s = pts.shape().to_vec();
        if s.len() != 2 || s[1] == 0 {
            return Err(PyValueError::new_err("pts must be (n, d) float64"));
        }
        Ok(KdTree { tree: f::KdTree::new(ro(&pts, "pts")?, s[1]), d: s[1] })
    }

    /// Returns (distances, indices), each (m, k) (the shim squeezes k = 1).
    #[pyo3(signature = (q, k, workers=-1))]
    fn query<'py>(
        &self,
        py: Python<'py>,
        q: PyReadonlyArrayDyn<'py, f64>,
        k: usize,
        workers: i64,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let s = q.shape().to_vec();
        if s.len() != 2 || s[1] != self.d || k == 0 {
            return Err(PyValueError::new_err("q must be (m, d) float64 and k >= 1"));
        }
        let m = s[0];
        let qs = ro(&q, "q")?;
        let nthreads = if workers < 0 {
            std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
        } else {
            (workers as usize).max(1)
        };
        let (dist, idx) = py.detach(|| {
            let mut dist = vec![0.0f64; m * k];
            let mut idx = vec![0i64; m * k];
            let chunk = m.div_ceil(nthreads).max(1);
            let d = self.d;
            let tree = &self.tree;
            std::thread::scope(|sc| {
                for ((dc, ic), qc) in dist.chunks_mut(chunk * k).zip(idx.chunks_mut(chunk * k)).zip(qs.chunks(chunk * d)) {
                    sc.spawn(move || {
                        for (r, qrow) in qc.chunks(d).enumerate() {
                            for (j, (dd, ii)) in tree.query(qrow, k).into_iter().enumerate() {
                                dc[r * k + j] = dd;
                                ic[r * k + j] = ii as i64;
                            }
                        }
                    });
                }
            });
            (dist, idx)
        });
        Ok((to_array(py, dist, &[m, k])?, to_array(py, idx, &[m, k])?))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(filters_gaussian_filter, m)?)?;
    m.add_function(wrap_pyfunction!(filters_sobel, m)?)?;
    m.add_function(wrap_pyfunction!(filters_uniform_filter, m)?)?;
    m.add_function(wrap_pyfunction!(filters_maximum_filter, m)?)?;
    m.add_function(wrap_pyfunction!(filters_zoom, m)?)?;
    m.add_function(wrap_pyfunction!(filters_binary_erosion, m)?)?;
    m.add_function(wrap_pyfunction!(filters_binary_dilation, m)?)?;
    m.add_function(wrap_pyfunction!(filters_binary_opening, m)?)?;
    m.add_function(wrap_pyfunction!(filters_binary_closing, m)?)?;
    m.add_function(wrap_pyfunction!(filters_fftfreq, m)?)?;
    m.add_function(wrap_pyfunction!(filters_rfftfreq, m)?)?;
    m.add_function(wrap_pyfunction!(filters_irfft2, m)?)?;
    m.add_function(wrap_pyfunction!(filters_rfft2, m)?)?;
    m.add_class::<KdTree>()?;
    Ok(())
}
