//! Bindings for pinsel_core::painterly. Python names carry the module prefix
//! (see docs/RUST_PORTING_GUIDE.md); arguments follow `pinsel.painterly`.

use crate::util::{ro, rw};
use numpy::{PyReadonlyArrayDyn, PyReadwriteArrayDyn, PyUntypedArrayMethods};
use pcore::grid::{Plane, Spec, SpecMut};
use pcore::painterly::GrowParams;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// (A, B) of a 3-D array whose last axis has length `c`.
fn dims3(shape: &[usize], c: usize, name: &str) -> PyResult<(usize, usize)> {
    if shape.len() != 3 || shape[2] != c {
        return Err(PyValueError::new_err(format!("pinsel_core: {name} has the wrong shape {shape:?}")));
    }
    Ok((shape[0], shape[1]))
}

/// Same arguments, in the same order, as `pinsel.painterly._grow_strokes`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn painterly_grow_strokes(
    sx: PyReadonlyArrayDyn<'_, f64>,
    sy: PyReadonlyArrayDyn<'_, f64>,
    rots: PyReadonlyArrayDyn<'_, f64>,
    rf: PyReadonlyArrayDyn<'_, f32>,
    cvl: PyReadonlyArrayDyn<'_, f32>,
    v: PyReadonlyArrayDyn<'_, f32>,
    region: PyReadonlyArrayDyn<'_, f32>,
    step: f64,
    min_steps: i64,
    max_steps: i64,
    back_steps: i64,
    fc: f64,
    edge_stop: f64,
    turn_cos: f64,
    dab_len: f64,
    margin: f64,
    color_mix: f64,
    mut ctrl: PyReadwriteArrayDyn<'_, f64>,
    mut nctrl: PyReadwriteArrayDyn<'_, i64>,
    mut cols: PyReadwriteArrayDyn<'_, f64>,
) -> PyResult<()> {
    let (h, w) = dims3(rf.shape(), 3, "ref")?;
    if dims3(cvl.shape(), 3, "cvl")? != (h, w) || dims3(v.shape(), 2, "V")? != (h, w) {
        return Err(PyValueError::new_err("pinsel_core: ref, cvl, V and region must share (H, W)"));
    }
    let rs = region.shape();
    if rs.len() != 2 || (rs[0], rs[1]) != (h, w) {
        return Err(PyValueError::new_err("pinsel_core: ref, cvl, V and region must share (H, W)"));
    }
    let n = sx.len();
    let (nc, mc) = dims3(ctrl.shape(), 2, "ctrl")?;
    if nc != n || max_steps < 0 || (mc as i64) < max_steps + 2 {
        return Err(PyValueError::new_err("pinsel_core: ctrl must be (n, max_steps + 2, 2)"));
    }
    let cs = cols.shape();
    if cs.len() != 2 || cs[0] != n || cs[1] != 3 {
        return Err(PyValueError::new_err("pinsel_core: cols must be (n, 3)"));
    }
    let prm = GrowParams {
        step,
        min_steps,
        max_steps,
        back_steps,
        fc,
        edge_stop,
        turn_cos,
        dab_len,
        margin,
        color_mix,
    };
    let rf_s = Spec::new(h, w, 3, ro(&rf, "ref")?);
    let cvl_s = Spec::new(h, w, 3, ro(&cvl, "cvl")?);
    let v_s = Spec::new(h, w, 2, ro(&v, "V")?);
    let reg = Plane::new(h, w, ro(&region, "region")?);
    let mut ctrl_s = SpecMut::new(n, mc, 2, rw(&mut ctrl, "ctrl")?);
    // cols is (n, 3): one "row" of 3 channels per stroke
    let mut cols_s = SpecMut::new(n, 1, 3, rw(&mut cols, "cols")?);
    pcore::painterly::grow_strokes(
        ro(&sx, "sx")?,
        ro(&sy, "sy")?,
        ro(&rots, "rots")?,
        &rf_s,
        &cvl_s,
        &v_s,
        &reg,
        &prm,
        &mut ctrl_s,
        rw(&mut nctrl, "nctrl")?,
        &mut cols_s,
    );
    Ok(())
}

/// Same arguments, in the same order, as `pinsel.painterly._densify`.
#[pyfunction]
fn painterly_densify(
    ctrl: PyReadonlyArrayDyn<'_, f64>,
    nctrl: PyReadonlyArrayDyn<'_, i64>,
    sub: i64,
    mut out: PyReadwriteArrayDyn<'_, f64>,
    mut nout: PyReadwriteArrayDyn<'_, i64>,
) -> PyResult<()> {
    let (n, mc) = dims3(ctrl.shape(), 2, "ctrl")?;
    let (no, mo) = dims3(out.shape(), 2, "out")?;
    if no != n || sub < 1 {
        return Err(PyValueError::new_err("pinsel_core: ctrl/out shape mismatch or sub < 1"));
    }
    let nc = ro(&nctrl, "nctrl")?;
    let k_max = nc.iter().copied().max().unwrap_or(0).max(0) as usize;
    if k_max > mc || mo < (k_max.max(2) - 1) * sub as usize + 1 {
        return Err(PyValueError::new_err("pinsel_core: out is too small for the strokes"));
    }
    let c = Spec::new(n, mc, 2, ro(&ctrl, "ctrl")?);
    let mut o = SpecMut::new(n, mo, 2, rw(&mut out, "out")?);
    pcore::painterly::densify(&c, nc, sub, &mut o, rw(&mut nout, "nout")?);
    Ok(())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(painterly_grow_strokes, m)?)?;
    m.add_function(wrap_pyfunction!(painterly_densify, m)?)?;
    Ok(())
}
