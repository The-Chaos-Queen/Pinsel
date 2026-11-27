//! Python module `pinsel_core`: the Rust kernels behind the same call
//! signatures as the numba kernels in `pinsel._kernels`, so the Python glue
//! stays unchanged in stage A. `pinsel._backend` picks this module when it
//! is installed and falls back to numba otherwise.

use numpy::{PyReadonlyArrayDyn, PyReadwriteArrayDyn, PyUntypedArrayMethods};
use pcore::grid::{DenseWet, WetArrays};
use pcore::paste::{Bristles, PasteParams};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn ro<'a, T: numpy::Element>(a: &'a PyReadonlyArrayDyn<'_, T>, name: &str) -> PyResult<&'a [T]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err(format!("pinsel_core: {name} must be a C-contiguous array")))
}

fn rw<'a, T: numpy::Element>(a: &'a mut PyReadwriteArrayDyn<'_, T>, name: &str) -> PyResult<&'a mut [T]> {
    a.as_slice_mut()
        .map_err(|_| PyValueError::new_err(format!("pinsel_core: {name} must be a C-contiguous array")))
}

/// Same arguments, in the same order, as `pinsel._kernels.paste_stroke`.
/// Returns the touched bounding box `(x0, y0, x1, y1)`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn paste_stroke(
    px: PyReadonlyArrayDyn<'_, f64>,
    py_: PyReadonlyArrayDyn<'_, f64>,
    pp: PyReadonlyArrayDyn<'_, f64>,
    pa: PyReadonlyArrayDyn<'_, f64>,
    ds: f64,
    speed: f64,
    boff: PyReadonlyArrayDyn<'_, f64>,
    bthr: PyReadonlyArrayDyn<'_, f64>,
    bfollow: PyReadonlyArrayDyn<'_, f64>,
    brad: PyReadonlyArrayDyn<'_, f64>,
    bsplay: PyReadonlyArrayDyn<'_, f64>,
    bdep: PyReadonlyArrayDyn<'_, f64>,
    mut tips: PyReadwriteArrayDyn<'_, f64>,
    mut res_k: PyReadwriteArrayDyn<'_, f64>,
    mut res_s: PyReadwriteArrayDyn<'_, f64>,
    mut res_vol: PyReadwriteArrayDyn<'_, f64>,
    cap: PyReadonlyArrayDyn<'_, f64>,
    mut tip_k: PyReadwriteArrayDyn<'_, f64>,
    mut tip_s: PyReadwriteArrayDyn<'_, f64>,
    mut tip_vol: PyReadwriteArrayDyn<'_, f64>,
    alpha_tip: f64,
    size: f64,
    sx0: f64,
    sx1: f64,
    sy0: f64,
    sy1: f64,
    sgamma: f64,
    splay: f64,
    alpha: f64,
    pickup: f64,
    plough: f64,
    contact_soft: f64,
    p_gloss: f64,
    p_body: f64,
    p_drate: f64,
    mut film_k: PyReadwriteArrayDyn<'_, f32>,
    mut film_s: PyReadwriteArrayDyn<'_, f32>,
    mut film_vol: PyReadwriteArrayDyn<'_, f32>,
    mut wet: PyReadwriteArrayDyn<'_, f32>,
    mut gloss: PyReadwriteArrayDyn<'_, f32>,
    mut body: PyReadwriteArrayDyn<'_, f32>,
    mut drate: PyReadwriteArrayDyn<'_, f32>,
    tooth: PyReadonlyArrayDyn<'_, f32>,
    mask: PyReadonlyArrayDyn<'_, f32>,
    tooth_fill: f64,
    mut base_k: PyReadwriteArrayDyn<'_, f32>,
    mut base_s: PyReadwriteArrayDyn<'_, f32>,
    mut base_vol: PyReadwriteArrayDyn<'_, f32>,
    mut top_id: PyReadwriteArrayDyn<'_, i32>,
    sid: i64,
) -> PyResult<(i64, i64, i64, i64)> {
    let shape = film_vol.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("pinsel_core: film_vol must be (H, W)"));
    }
    let (h, w) = (shape[0], shape[1]);
    let prm = PasteParams {
        ds,
        speed,
        alpha_tip,
        size,
        sx0,
        sx1,
        sy0,
        sy1,
        sgamma,
        splay,
        alpha,
        pickup,
        plough,
        contact_soft,
        p_gloss,
        p_body,
        p_drate,
        tooth_fill,
        sid: sid as i32,
    };
    let mut bristles = Bristles {
        boff: ro(&boff, "boff")?,
        bthr: ro(&bthr, "bthr")?,
        bfollow: ro(&bfollow, "bfollow")?,
        brad: ro(&brad, "brad")?,
        bsplay: ro(&bsplay, "bsplay")?,
        bdep: ro(&bdep, "bdep")?,
        tips: rw(&mut tips, "tips")?,
        res_k: rw(&mut res_k, "res_K")?,
        res_s: rw(&mut res_s, "res_S")?,
        res_vol: rw(&mut res_vol, "res_vol")?,
        cap: ro(&cap, "cap")?,
        tip_k: rw(&mut tip_k, "tip_K")?,
        tip_s: rw(&mut tip_s, "tip_S")?,
        tip_vol: rw(&mut tip_vol, "tip_vol")?,
    };
    let arrays = WetArrays {
        film_k: rw(&mut film_k, "film_K")?,
        film_s: rw(&mut film_s, "film_S")?,
        base_k: rw(&mut base_k, "base_K")?,
        base_s: rw(&mut base_s, "base_S")?,
        film_vol: rw(&mut film_vol, "film_vol")?,
        base_vol: rw(&mut base_vol, "base_vol")?,
        wet: rw(&mut wet, "wet")?,
        gloss: rw(&mut gloss, "gloss")?,
        body: rw(&mut body, "body")?,
        drate: rw(&mut drate, "drate")?,
        top_id: rw(&mut top_id, "top_id")?,
    };
    let mut cv = DenseWet::new(h, w, arrays, ro(&tooth, "tooth")?, ro(&mask, "mask")?);
    Ok(pcore::paste::paste_stroke(
        ro(&px, "px")?,
        ro(&py_, "py")?,
        ro(&pp, "pp")?,
        ro(&pa, "pa")?,
        &mut bristles,
        &prm,
        &mut cv,
    ))
}

#[pymodule]
fn pinsel_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(paste_stroke, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("NB", pcore::NB)?;
    Ok(())
}
