//! Bindings for pinsel_core::pastel. Python names carry the module prefix
//! (see docs/RUST_PORTING_GUIDE.md); arguments follow `pinsel.pastel`.

use crate::util::{ro, rw};
use numpy::{PyReadonlyArrayDyn, PyReadwriteArrayDyn, PyUntypedArrayMethods};
use pcore::grid::{DenseWet, Plane, PlaneMut, SpecMut, WetArrays};
use pcore::pastel::PastelParams;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Same arguments, in the same order, as `pinsel.pastel._pastel_stroke`.
/// Returns the touched bounding box `(x0, y0, x1, y1)`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn pastel_pastel_stroke(
    px: PyReadonlyArrayDyn<'_, f64>,
    py_: PyReadonlyArrayDyn<'_, f64>,
    pp: PyReadonlyArrayDyn<'_, f64>,
    pa: PyReadonlyArrayDyn<'_, f64>,
    rx_base: f64,
    ry_base: f64,
    hardness: f64,
    dust: f64,
    amount: f64,
    kpv: PyReadonlyArrayDyn<'_, f64>,
    spv: PyReadonlyArrayDyn<'_, f64>,
    mut film_k: PyReadwriteArrayDyn<'_, f32>,
    mut film_s: PyReadwriteArrayDyn<'_, f32>,
    mut film_vol: PyReadwriteArrayDyn<'_, f32>,
    mut wet: PyReadwriteArrayDyn<'_, f32>,
    mut gloss: PyReadwriteArrayDyn<'_, f32>,
    mut body: PyReadwriteArrayDyn<'_, f32>,
    mut drate: PyReadwriteArrayDyn<'_, f32>,
    tooth: PyReadonlyArrayDyn<'_, f32>,
    mask: PyReadonlyArrayDyn<'_, f32>,
    mut base_k: PyReadwriteArrayDyn<'_, f32>,
    mut base_s: PyReadwriteArrayDyn<'_, f32>,
    mut base_vol: PyReadwriteArrayDyn<'_, f32>,
    mut top_id: PyReadwriteArrayDyn<'_, i32>,
    sid: i64,
    seed: i64,
) -> PyResult<(i64, i64, i64, i64)> {
    let shape = film_vol.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("pinsel_core: film_vol must be (H, W)"));
    }
    let (h, w) = (shape[0], shape[1]);
    let prm = PastelParams { rx_base, ry_base, hardness, dust, amount, sid: sid as i32, seed };
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
    Ok(pcore::pastel::pastel_stroke(
        ro(&px, "px")?,
        ro(&py_, "py")?,
        ro(&pp, "pp")?,
        ro(&pa, "pa")?,
        ro(&kpv, "Kpv")?,
        ro(&spv, "Spv")?,
        &prm,
        &mut cv,
    ))
}

/// Same arguments, in the same order, as `pinsel.pastel._eraser_stroke`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn pastel_eraser_stroke(
    px: PyReadonlyArrayDyn<'_, f64>,
    py_: PyReadonlyArrayDyn<'_, f64>,
    pp: PyReadonlyArrayDyn<'_, f64>,
    r_base: f64,
    strength: f64,
    mut film_k: PyReadwriteArrayDyn<'_, f32>,
    mut film_s: PyReadwriteArrayDyn<'_, f32>,
    mut film_vol: PyReadwriteArrayDyn<'_, f32>,
    mut base_k: PyReadwriteArrayDyn<'_, f32>,
    mut base_s: PyReadwriteArrayDyn<'_, f32>,
    mut base_vol: PyReadwriteArrayDyn<'_, f32>,
    mask: PyReadonlyArrayDyn<'_, f32>,
) -> PyResult<(i64, i64, i64, i64)> {
    let shape = film_vol.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("pinsel_core: film_vol must be (H, W)"));
    }
    let (h, w) = (shape[0], shape[1]);
    let nb = pcore::NB;
    let mut fk = SpecMut::new(h, w, nb, rw(&mut film_k, "film_K")?);
    let mut fs = SpecMut::new(h, w, nb, rw(&mut film_s, "film_S")?);
    let mut bk = SpecMut::new(h, w, nb, rw(&mut base_k, "base_K")?);
    let mut bs = SpecMut::new(h, w, nb, rw(&mut base_s, "base_S")?);
    let mut fv = PlaneMut::new(h, w, rw(&mut film_vol, "film_vol")?);
    let mut bv = PlaneMut::new(h, w, rw(&mut base_vol, "base_vol")?);
    let mk = Plane::new(h, w, ro(&mask, "mask")?);
    Ok(pcore::pastel::eraser_stroke(
        ro(&px, "px")?,
        ro(&py_, "py")?,
        ro(&pp, "pp")?,
        r_base,
        strength,
        &mut fk,
        &mut fs,
        &mut fv,
        &mut bk,
        &mut bs,
        &mut bv,
        &mk,
    ))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pastel_pastel_stroke, m)?)?;
    m.add_function(wrap_pyfunction!(pastel_eraser_stroke, m)?)?;
    Ok(())
}
