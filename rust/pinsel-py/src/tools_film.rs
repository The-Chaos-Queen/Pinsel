//! Bindings for pinsel_core::tools_film. Register each function in `register`.
//! Python names carry the module prefix (see docs/RUST_PORTING_GUIDE.md).
//!
//! Every function takes the arguments of its numba kernel in the same order.
//! The film tuple `F` is mapped onto `DenseWet` (see `tools_common::Film`).
//! The kernels read the canvas mask through `WetCanvas::mask`; where a kernel
//! has no tooth plane, the mask stands in for it (never read).

use crate::tools_common::{ro_c, Film, FilmT};
use crate::util::{hw, hw_mut, ro, rw};
use numpy::{PyReadonlyArrayDyn, PyReadwriteArrayDyn, PyUntypedArrayMethods};
use pcore::grid::{DenseWet, Plane, PlaneMut};
use pcore::tools_film as tf;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// `_knife_kernel(px, py, pp, pa, ps, mode, half, nk, res_K, res_S, res_vol, cap, prm,
/// lump_tex, stri, chat, flex_n, p_gloss, p_body, p_drate, F, sid, h_ground, h_dry,
/// mask, smax, sox, soy)` -> bbox
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn tools_knife_kernel(
    px: PyReadonlyArrayDyn<'_, f64>,
    py_: PyReadonlyArrayDyn<'_, f64>,
    pp: PyReadonlyArrayDyn<'_, f64>,
    pa: PyReadonlyArrayDyn<'_, f64>,
    ps: PyReadonlyArrayDyn<'_, f64>,
    mode: i64,
    half: f64,
    nk: i64,
    mut res_k: PyReadwriteArrayDyn<'_, f64>,
    mut res_s: PyReadwriteArrayDyn<'_, f64>,
    mut res_vol: PyReadwriteArrayDyn<'_, f64>,
    cap: PyReadonlyArrayDyn<'_, f64>,
    prm: PyReadonlyArrayDyn<'_, f64>,
    lump_tex: PyReadonlyArrayDyn<'_, f64>,
    stri: PyReadonlyArrayDyn<'_, f64>,
    chat: PyReadonlyArrayDyn<'_, f64>,
    flex_n: i64,
    p_gloss: f64,
    p_body: f64,
    p_drate: f64,
    f: FilmT<'_>,
    sid: i64,
    h_ground: PyReadonlyArrayDyn<'_, f32>,
    h_dry: PyReadonlyArrayDyn<'_, f32>,
    mask: PyReadonlyArrayDyn<'_, f32>,
    smax: PyReadonlyArrayDyn<'_, f32>,
    sox: i64,
    soy: i64,
) -> PyResult<(i64, i64, i64, i64)> {
    if nk < 2 {
        return Err(PyValueError::new_err("pinsel_core: nk must be >= 2"));
    }
    if prm.len() < 21 {
        return Err(PyValueError::new_err("pinsel_core: prm needs 21 values"));
    }
    let (lh, lw) = hw(&lump_tex, "lump_tex")?;
    let (gh, gw) = hw(&h_ground, "h_ground")?;
    let (dh, dw) = hw(&h_dry, "h_dry")?;
    let (sh, sw) = hw(&smax, "smax")?;
    let lump_s = ro_c(&lump_tex, "lump_tex")?;
    let hg_s = ro_c(&h_ground, "h_ground")?;
    let hd_s = ro_c(&h_dry, "h_dry")?;
    let sm_s = ro_c(&smax, "smax")?;
    let mask_s = ro_c(&mask, "mask")?;
    let a = tf::KnifeIn {
        px: ro(&px, "px")?,
        py: ro(&py_, "py")?,
        pp: ro(&pp, "pp")?,
        pa: ro(&pa, "pa")?,
        ps: ro(&ps, "ps")?,
        mode,
        half,
        nk,
        cap: ro(&cap, "cap")?,
        prm: ro(&prm, "prm")?,
        lump_tex: Plane::new(lh, lw, &lump_s),
        stri: ro(&stri, "stri")?,
        chat: ro(&chat, "chat")?,
        flex_n,
        p_gloss,
        p_body,
        p_drate,
        sid,
        h_ground: Plane::new(gh, gw, &hg_s),
        h_dry: Plane::new(dh, dw, &hd_s),
        smax: Plane::new(sh, sw, &sm_s),
        sox,
        soy,
    };
    let mut film = Film::new(f);
    let (h, w, arrays) = film.arrays()?;
    let mut cv = DenseWet::new(h, w, arrays, &mask_s, &mask_s);
    Ok(tf::knife_kernel(
        &a,
        rw(&mut res_k, "res_K")?,
        rw(&mut res_s, "res_S")?,
        rw(&mut res_vol, "res_vol")?,
        &mut cv,
    ))
}

/// `_sgraffito_kernel(px, py, pp, r0, r1, valley, burr, carry, ragged, noise, nox, noy,
/// F, sid, h_ground, h_dry, mask)` -> bbox
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn tools_sgraffito_kernel(
    px: PyReadonlyArrayDyn<'_, f64>,
    py_: PyReadonlyArrayDyn<'_, f64>,
    pp: PyReadonlyArrayDyn<'_, f64>,
    r0: f64,
    r1: f64,
    valley: f64,
    burr: f64,
    carry: f64,
    ragged: f64,
    noise: PyReadonlyArrayDyn<'_, f64>,
    nox: i64,
    noy: i64,
    f: FilmT<'_>,
    sid: i64,
    h_ground: PyReadonlyArrayDyn<'_, f32>,
    h_dry: PyReadonlyArrayDyn<'_, f32>,
    mask: PyReadonlyArrayDyn<'_, f32>,
) -> PyResult<(i64, i64, i64, i64)> {
    let (nh, nw) = hw(&noise, "noise")?;
    let (gh, gw) = hw(&h_ground, "h_ground")?;
    let (dh, dw) = hw(&h_dry, "h_dry")?;
    let noise_s = ro_c(&noise, "noise")?;
    let hg_s = ro_c(&h_ground, "h_ground")?;
    let hd_s = ro_c(&h_dry, "h_dry")?;
    let mask_s = ro_c(&mask, "mask")?;
    let prm = tf::SgraffitoParams { r0, r1, valley, burr, carry, ragged, nox, noy, sid };
    let mut film = Film::new(f);
    let (h, w, arrays) = film.arrays()?;
    let mut cv = DenseWet::new(h, w, arrays, &mask_s, &mask_s);
    Ok(tf::sgraffito_kernel(
        ro(&px, "px")?,
        ro(&py_, "py")?,
        ro(&pp, "pp")?,
        &prm,
        &Plane::new(nh, nw, &noise_s),
        &Plane::new(gh, gw, &hg_s),
        &Plane::new(dh, dw, &hd_s),
        &mut cv,
    ))
}

/// `_smudge_kernel(px, py, pp, R, rate, spread, F, sid, mask, lK, lS, lV, lP)` -> bbox
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn tools_smudge_kernel(
    px: PyReadonlyArrayDyn<'_, f64>,
    py_: PyReadonlyArrayDyn<'_, f64>,
    pp: PyReadonlyArrayDyn<'_, f64>,
    r: i64,
    rate: f64,
    spread: f64,
    f: FilmT<'_>,
    sid: i64,
    mask: PyReadonlyArrayDyn<'_, f32>,
    mut l_k: PyReadwriteArrayDyn<'_, f64>,
    mut l_s: PyReadwriteArrayDyn<'_, f64>,
    mut l_v: PyReadwriteArrayDyn<'_, f64>,
    mut l_p: PyReadwriteArrayDyn<'_, f64>,
) -> PyResult<(i64, i64, i64, i64)> {
    if r < 1 {
        return Err(PyValueError::new_err("pinsel_core: R must be >= 1"));
    }
    let nl = (2 * r + 1) as usize;
    if l_k.len() != nl * pcore::NB || l_p.len() != nl * 4 || l_v.len() != nl {
        return Err(PyValueError::new_err("pinsel_core: lane arrays do not match R"));
    }
    let mask_s = ro_c(&mask, "mask")?;
    let mut film = Film::new(f);
    let (h, w, arrays) = film.arrays()?;
    let mut cv = DenseWet::new(h, w, arrays, &mask_s, &mask_s);
    Ok(tf::smudge_kernel(
        ro(&px, "px")?,
        ro(&py_, "py")?,
        ro(&pp, "pp")?,
        r,
        rate,
        spread,
        &mut cv,
        sid,
        rw(&mut l_k, "lK")?,
        rw(&mut l_s, "lS")?,
        rw(&mut l_v, "lV")?,
        rw(&mut l_p, "lP")?,
    ))
}

/// `_crayon_stroke(film, sid, tooth, relief, mask, wax, xy, pressures, angles,
/// size, side, K, S)` -> (left, top, right, bottom)
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn tools_crayon_stroke(
    film: FilmT<'_>,
    sid: i64,
    tooth: PyReadonlyArrayDyn<'_, f32>,
    relief: f64,
    mask: PyReadonlyArrayDyn<'_, f32>,
    mut wax: PyReadwriteArrayDyn<'_, f32>,
    xy: PyReadonlyArrayDyn<'_, f64>,
    pressures: PyReadonlyArrayDyn<'_, f64>,
    angles: PyReadonlyArrayDyn<'_, f64>,
    size: f64,
    side: f64,
    k: PyReadonlyArrayDyn<'_, f64>,
    s: PyReadonlyArrayDyn<'_, f64>,
) -> PyResult<(i64, i64, i64, i64)> {
    let tooth_s = ro_c(&tooth, "tooth")?;
    let mask_s = ro_c(&mask, "mask")?;
    let xy_s = ro_c(&xy, "xy")?;
    let k_s = ro_c(&k, "K")?;
    let s_s = ro_c(&s, "S")?;
    let (wh, ww) = hw_mut(&wax, "wax")?;
    let n = xy_s.len() / 2;
    if pressures.len() < n || angles.len() < n || k_s.len() != pcore::NB || s_s.len() != pcore::NB {
        return Err(PyValueError::new_err("pinsel_core: crayon argument lengths do not match"));
    }
    let mut f = Film::new(film);
    let (h, w, arrays) = f.arrays()?;
    let mut cv = DenseWet::new(h, w, arrays, &tooth_s, &mask_s);
    let mut wax_p = PlaneMut::new(wh, ww, rw(&mut wax, "wax")?);
    Ok(tf::crayon_stroke(
        &mut cv,
        sid,
        relief,
        &mut wax_p,
        &xy_s,
        ro(&pressures, "pressures")?,
        ro(&angles, "angles")?,
        size,
        side,
        &k_s,
        &s_s,
    ))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tools_knife_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(tools_sgraffito_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(tools_smudge_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(tools_crayon_stroke, m)?)?;
    Ok(())
}
