//! Bindings for pinsel_core::tools_common. Register each function in `register`.
//! Python names carry the module prefix (see docs/RUST_PORTING_GUIDE.md).
//!
//! Called from Python: `_apply_volume_map` and `_apply_fluid_map` (through
//! `_deposit_map`). The other helpers are internal Rust functions of the
//! kernels; `_gauss_dep_bin` is only called from the knife kernel, so it has
//! no binding.
//!
//! This file also holds the glue the other tool bindings share: the film
//! tuple `F` -> `WetArrays` ([`Film`]) and read-only arrays that may be
//! strided views ([`ro_c`]).

use std::borrow::Cow;

use crate::util::{hw, hw_mut, ro, rw};
use numpy::{PyReadonlyArrayDyn, PyReadwriteArrayDyn, PyUntypedArrayMethods};
use pcore::grid::{DenseWet, Plane, PlaneMut, SpecMut, WetArrays};
use pcore::tools_common as tc;
use pcore::NB;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Read-only array as a row-major slice: borrowed when the numpy array is
/// C-contiguous, copied when it is a strided view (e.g. a region of the mask).
pub fn ro_c<'a, T: numpy::Element + Copy>(a: &'a PyReadonlyArrayDyn<'_, T>, _name: &str) -> PyResult<Cow<'a, [T]>> {
    Ok(match a.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(a.as_array().iter().copied().collect()),
    })
}

/// The film tuple `F` of `tools._film(canvas)`:
/// (film_K, film_S, film_vol, base_K, base_S, base_vol, top_id, wet, gloss, body, drate).
pub type FilmT<'py> = (
    PyReadwriteArrayDyn<'py, f32>,
    PyReadwriteArrayDyn<'py, f32>,
    PyReadwriteArrayDyn<'py, f32>,
    PyReadwriteArrayDyn<'py, f32>,
    PyReadwriteArrayDyn<'py, f32>,
    PyReadwriteArrayDyn<'py, f32>,
    PyReadwriteArrayDyn<'py, i32>,
    PyReadwriteArrayDyn<'py, f32>,
    PyReadwriteArrayDyn<'py, f32>,
    PyReadwriteArrayDyn<'py, f32>,
    PyReadwriteArrayDyn<'py, f32>,
);

/// Holds the borrows of the film arrays for the duration of a call.
pub struct Film<'py> {
    fk: PyReadwriteArrayDyn<'py, f32>,
    fs: PyReadwriteArrayDyn<'py, f32>,
    fv: PyReadwriteArrayDyn<'py, f32>,
    bk: PyReadwriteArrayDyn<'py, f32>,
    bs: PyReadwriteArrayDyn<'py, f32>,
    bv: PyReadwriteArrayDyn<'py, f32>,
    tid: PyReadwriteArrayDyn<'py, i32>,
    wet: PyReadwriteArrayDyn<'py, f32>,
    gloss: PyReadwriteArrayDyn<'py, f32>,
    body: PyReadwriteArrayDyn<'py, f32>,
    drate: PyReadwriteArrayDyn<'py, f32>,
}

impl<'py> Film<'py> {
    pub fn new(t: FilmT<'py>) -> Self {
        let (fk, fs, fv, bk, bs, bv, tid, wet, gloss, body, drate) = t;
        Film { fk, fs, fv, bk, bs, bv, tid, wet, gloss, body, drate }
    }

    /// (H, W) and the dense arrays for `DenseWet`.
    pub fn arrays(&mut self) -> PyResult<(usize, usize, WetArrays<'_>)> {
        let shape = self.fv.shape();
        if shape.len() != 2 {
            return Err(PyValueError::new_err("pinsel_core: film_vol must be (H, W)"));
        }
        let (h, w) = (shape[0], shape[1]);
        Ok((
            h,
            w,
            WetArrays {
                film_k: rw(&mut self.fk, "film_K")?,
                film_s: rw(&mut self.fs, "film_S")?,
                base_k: rw(&mut self.bk, "base_K")?,
                base_s: rw(&mut self.bs, "base_S")?,
                film_vol: rw(&mut self.fv, "film_vol")?,
                base_vol: rw(&mut self.bv, "base_vol")?,
                wet: rw(&mut self.wet, "wet")?,
                gloss: rw(&mut self.gloss, "gloss")?,
                body: rw(&mut self.body, "body")?,
                drate: rw(&mut self.drate, "drate")?,
                top_id: rw(&mut self.tid, "top_id")?,
            },
        ))
    }
}

/// `_apply_volume_map(vol, ox, oy, Kpv, Spv, p_gloss, p_body, p_drate, F, sid, mask)`
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn tools_apply_volume_map(
    vol: PyReadonlyArrayDyn<'_, f64>,
    ox: i64,
    oy: i64,
    kpv: PyReadonlyArrayDyn<'_, f64>,
    spv: PyReadonlyArrayDyn<'_, f64>,
    p_gloss: f64,
    p_body: f64,
    p_drate: f64,
    f: FilmT<'_>,
    sid: i64,
    mask: PyReadonlyArrayDyn<'_, f32>,
) -> PyResult<()> {
    let (vh, vw) = hw(&vol, "vol")?;
    let vol_s = ro_c(&vol, "vol")?;
    let vol_p = Plane::new(vh, vw, &vol_s);
    let kpv_s = ro(&kpv, "Kpv")?;
    let spv_s = ro(&spv, "Spv")?;
    let mask_s = ro_c(&mask, "mask")?;
    let mut film = Film::new(f);
    let (h, w, arrays) = film.arrays()?;
    // the tooth plane is never read here; the mask stands in for it
    let mut cv = DenseWet::new(h, w, arrays, &mask_s, &mask_s);
    tc::apply_volume_map(&vol_p, ox, oy, kpv_s, spv_s, p_gloss, p_body, p_drate, &mut cv, sid);
    Ok(())
}

/// `_apply_fluid_map(vol, ox, oy, water_ratio, settle_frac, Kpv, Spv, gran, stain,
/// wK, wS, wwat, wm, wg, wst, dK, dS, dm, dg, dst, mask)`
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn tools_apply_fluid_map(
    vol: PyReadonlyArrayDyn<'_, f64>,
    ox: i64,
    oy: i64,
    water_ratio: f64,
    settle_frac: f64,
    kpv: PyReadonlyArrayDyn<'_, f64>,
    spv: PyReadonlyArrayDyn<'_, f64>,
    gran: f64,
    stain: f64,
    mut wk: PyReadwriteArrayDyn<'_, f32>,
    mut ws: PyReadwriteArrayDyn<'_, f32>,
    mut wwat: PyReadwriteArrayDyn<'_, f32>,
    mut wm: PyReadwriteArrayDyn<'_, f32>,
    mut wg: PyReadwriteArrayDyn<'_, f32>,
    mut wst: PyReadwriteArrayDyn<'_, f32>,
    mut dk: PyReadwriteArrayDyn<'_, f32>,
    mut ds: PyReadwriteArrayDyn<'_, f32>,
    mut dm: PyReadwriteArrayDyn<'_, f32>,
    mut dg: PyReadwriteArrayDyn<'_, f32>,
    mut dst: PyReadwriteArrayDyn<'_, f32>,
    mask: PyReadonlyArrayDyn<'_, f32>,
) -> PyResult<()> {
    let (vh, vw) = hw(&vol, "vol")?;
    let vol_s = ro_c(&vol, "vol")?;
    let vol_p = Plane::new(vh, vw, &vol_s);
    let kpv_s = ro(&kpv, "Kpv")?;
    let spv_s = ro(&spv, "Spv")?;
    let (h, w) = hw_mut(&wwat, "water")?;
    let (mh, mw) = hw(&mask, "mask")?;
    let mask_s = ro_c(&mask, "mask")?;
    let mask_p = Plane::new(mh, mw, &mask_s);
    let mut wc = tc::WcLayer {
        k: SpecMut::new(h, w, NB, rw(&mut wk, "K")?),
        s: SpecMut::new(h, w, NB, rw(&mut ws, "S")?),
        water: PlaneMut::new(h, w, rw(&mut wwat, "water")?),
        m: PlaneMut::new(h, w, rw(&mut wm, "m")?),
        g: PlaneMut::new(h, w, rw(&mut wg, "g")?),
        st: PlaneMut::new(h, w, rw(&mut wst, "st")?),
        dk: SpecMut::new(h, w, NB, rw(&mut dk, "dK")?),
        ds: SpecMut::new(h, w, NB, rw(&mut ds, "dS")?),
        dm: PlaneMut::new(h, w, rw(&mut dm, "dm")?),
        dg: PlaneMut::new(h, w, rw(&mut dg, "dg")?),
        dst: PlaneMut::new(h, w, rw(&mut dst, "dst")?),
    };
    tc::apply_fluid_map(&vol_p, ox, oy, water_ratio, settle_frac, kpv_s, spv_s, gran, stain, &mut wc, &mask_p);
    Ok(())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tools_apply_volume_map, m)?)?;
    m.add_function(wrap_pyfunction!(tools_apply_fluid_map, m)?)?;
    Ok(())
}
