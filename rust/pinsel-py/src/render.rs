//! Bindings for pinsel_core::render: render_reflectance, bake_film,
//! bake_deposit, with the signatures of pinsel._kernels.

use crate::util::{ro, ro_cow, rw};
use numpy::{PyReadonlyArrayDyn, PyReadwriteArrayDyn, PyUntypedArrayMethods};
use pcore::grid::{DenseWet, Plane, PlaneMut, Spec, SpecMut, WetArrays};
use pcore::render::WcLayers;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn shape3(s: &[usize], name: &str) -> PyResult<(usize, usize, usize)> {
    if s.len() != 3 {
        return Err(PyValueError::new_err(format!("pinsel_core: {name} must be (H, W, C)")));
    }
    Ok((s[0], s[1], s[2]))
}

/// Same arguments as `pinsel._kernels.render_reflectance`. Inputs may be
/// non-contiguous region views; `out` must be a contiguous (h, w, 3) float32.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn render_reflectance(
    below_r: PyReadonlyArrayDyn<'_, f32>,
    base_k: PyReadonlyArrayDyn<'_, f32>,
    base_s: PyReadonlyArrayDyn<'_, f32>,
    film_k: PyReadonlyArrayDyn<'_, f32>,
    film_s: PyReadonlyArrayDyn<'_, f32>,
    has_wc: bool,
    wc_k: PyReadonlyArrayDyn<'_, f32>,
    wc_s: PyReadonlyArrayDyn<'_, f32>,
    wc_dk: PyReadonlyArrayDyn<'_, f32>,
    wc_ds: PyReadonlyArrayDyn<'_, f32>,
    wc_water: PyReadonlyArrayDyn<'_, f32>,
    wc_damp: PyReadonlyArrayDyn<'_, f32>,
    m: PyReadonlyArrayDyn<'_, f64>,
    mut out: PyReadwriteArrayDyn<'_, f32>,
) -> PyResult<()> {
    let (h, w, c) = shape3(below_r.shape(), "below_R")?;
    let (bl, bk, bs, fk, fs) = (ro_cow(&below_r), ro_cow(&base_k), ro_cow(&base_s), ro_cow(&film_k), ro_cow(&film_s));
    let (wk, wsg, wdk, wds, wwat, wdmp);
    let wc = if has_wc {
        wk = ro_cow(&wc_k);
        wsg = ro_cow(&wc_s);
        wdk = ro_cow(&wc_dk);
        wds = ro_cow(&wc_ds);
        wwat = ro_cow(&wc_water);
        wdmp = ro_cow(&wc_damp);
        Some(WcLayers {
            k: Spec::new(h, w, c, &wk),
            s: Spec::new(h, w, c, &wsg),
            dk: Spec::new(h, w, c, &wdk),
            ds: Spec::new(h, w, c, &wds),
            water: Plane::new(h, w, &wwat),
            damp: Plane::new(h, w, &wdmp),
        })
    } else {
        None
    };
    let mm = ro_cow(&m);
    let o = rw(&mut out, "out")?;
    let mut ov = SpecMut::new(h, w, 3, o);
    pcore::render::render_reflectance(
        Spec::new(h, w, c, &bl),
        Spec::new(h, w, c, &bk),
        Spec::new(h, w, c, &bs),
        Spec::new(h, w, c, &fk),
        Spec::new(h, w, c, &fs),
        wc,
        &mm,
        &mut ov,
    );
    Ok(())
}

/// Same arguments as `pinsel._kernels.bake_film`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn bake_film(
    sel: PyReadonlyArrayDyn<'_, bool>,
    mut below_r: PyReadwriteArrayDyn<'_, f32>,
    mut film_k: PyReadwriteArrayDyn<'_, f32>,
    mut film_s: PyReadwriteArrayDyn<'_, f32>,
    mut film_vol: PyReadwriteArrayDyn<'_, f32>,
    mut wet: PyReadwriteArrayDyn<'_, f32>,
    mut gloss: PyReadwriteArrayDyn<'_, f32>,
    mut body: PyReadwriteArrayDyn<'_, f32>,
    mut drate: PyReadwriteArrayDyn<'_, f32>,
    mut h_dry: PyReadwriteArrayDyn<'_, f32>,
    mut gloss_dry: PyReadwriteArrayDyn<'_, f32>,
    gloss_dry_factor: f64,
    mut base_k: PyReadwriteArrayDyn<'_, f32>,
    mut base_s: PyReadwriteArrayDyn<'_, f32>,
    mut base_vol: PyReadwriteArrayDyn<'_, f32>,
) -> PyResult<()> {
    let (h, w, c) = shape3(below_r.shape(), "below_R")?;
    // bake_film never reads or writes top_id; the wet-film access type wants one.
    let mut top_id = vec![0i32; h * w];
    let zero_plane = vec![0f32; h * w];
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
        top_id: &mut top_id,
    };
    let mut cv = DenseWet::new(h, w, arrays, &zero_plane, &zero_plane);
    let sel_s = ro(&sel, "sel")?;
    let mut br = SpecMut::new(h, w, c, rw(&mut below_r, "below_R")?);
    let mut hd = PlaneMut::new(h, w, rw(&mut h_dry, "h_dry")?);
    let mut gd = PlaneMut::new(h, w, rw(&mut gloss_dry, "gloss_dry")?);
    pcore::render::bake_film(Plane::new(h, w, sel_s), &mut br, &mut cv, &mut hd, &mut gd, gloss_dry_factor);
    Ok(())
}

/// Same arguments as `pinsel._kernels.bake_deposit`.
#[pyfunction]
fn bake_deposit(
    sel: PyReadonlyArrayDyn<'_, bool>,
    mut below_r: PyReadwriteArrayDyn<'_, f32>,
    mut dk: PyReadwriteArrayDyn<'_, f32>,
    mut ds: PyReadwriteArrayDyn<'_, f32>,
) -> PyResult<()> {
    let (h, w, c) = shape3(below_r.shape(), "below_R")?;
    let sel_s = ro(&sel, "sel")?;
    let mut br = SpecMut::new(h, w, c, rw(&mut below_r, "below_R")?);
    let mut k = SpecMut::new(h, w, c, rw(&mut dk, "dK")?);
    let mut s = SpecMut::new(h, w, c, rw(&mut ds, "dS")?);
    pcore::render::bake_deposit(Plane::new(h, w, sel_s), &mut br, &mut k, &mut s);
    Ok(())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(render_reflectance, m)?)?;
    m.add_function(wrap_pyfunction!(bake_film, m)?)?;
    m.add_function(wrap_pyfunction!(bake_deposit, m)?)?;
    Ok(())
}
