//! Bindings for pinsel_core::tools_spray. Register each function in `register`.
//! Python names carry the module prefix (see docs/RUST_PORTING_GUIDE.md).
//!
//! Every function takes the arguments of its numba kernel in the same order.
//! Read-only maps go through `ro_c` (copied when they are strided views);
//! the maps a kernel writes (`acc`, `cov`, `amt`) must be C-contiguous, as
//! `np.zeros` makes them.

use crate::tools_common::{ro_c, Film, FilmT};
use crate::util::{hw, hw_mut, rw};
use numpy::{PyReadonlyArrayDyn, PyReadwriteArrayDyn, PyUntypedArrayMethods};
use pcore::grid::{DenseWet, Plane, PlaneMut, SpecMut};
use pcore::tools_common::WcLayer;
use pcore::tools_spray as ts;
use pcore::NB;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// `_spray_accumulate(px, py, amt, sig, tail_frac, tail_mult, ox, oy, acc)`
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn tools_spray_accumulate(
    px: PyReadonlyArrayDyn<'_, f64>,
    py_: PyReadonlyArrayDyn<'_, f64>,
    amt: PyReadonlyArrayDyn<'_, f64>,
    sig: PyReadonlyArrayDyn<'_, f64>,
    tail_frac: f64,
    tail_mult: f64,
    ox: f64,
    oy: f64,
    mut acc: PyReadwriteArrayDyn<'_, f64>,
) -> PyResult<()> {
    let (h, w) = hw_mut(&acc, "acc")?;
    let (px_s, py_s, amt_s, sig_s) = (ro_c(&px, "px")?, ro_c(&py_, "py")?, ro_c(&amt, "amt")?, ro_c(&sig, "sig")?);
    if py_s.len() < px_s.len() || amt_s.len() < px_s.len() || sig_s.len() < px_s.len() {
        return Err(PyValueError::new_err("pinsel_core: px, py, amt, sig must have the same length"));
    }
    let mut a = PlaneMut::new(h, w, rw(&mut acc, "acc")?);
    ts::spray_accumulate(&px_s, &py_s, &amt_s, &sig_s, tail_frac, tail_mult, ox, oy, &mut a);
    Ok(())
}

/// `_splat_discs(xs, ys, rs, vs, ox, oy, acc, tooth, rough)`
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn tools_splat_discs(
    xs: PyReadonlyArrayDyn<'_, f64>,
    ys: PyReadonlyArrayDyn<'_, f64>,
    rs: PyReadonlyArrayDyn<'_, f64>,
    vs: PyReadonlyArrayDyn<'_, f64>,
    ox: i64,
    oy: i64,
    mut acc: PyReadwriteArrayDyn<'_, f64>,
    tooth: PyReadonlyArrayDyn<'_, f32>,
    rough: f64,
) -> PyResult<()> {
    let (h, w) = hw_mut(&acc, "acc")?;
    let (th, tw) = hw(&tooth, "tooth")?;
    let tooth_s = ro_c(&tooth, "tooth")?;
    let (xs_s, ys_s, rs_s, vs_s) = (ro_c(&xs, "xs")?, ro_c(&ys, "ys")?, ro_c(&rs, "rs")?, ro_c(&vs, "vs")?);
    if ys_s.len() < xs_s.len() || rs_s.len() < xs_s.len() || vs_s.len() < xs_s.len() {
        return Err(PyValueError::new_err("pinsel_core: xs, ys, rs, vs must have the same length"));
    }
    let mut a = PlaneMut::new(h, w, rw(&mut acc, "acc")?);
    ts::splat_discs(&xs_s, &ys_s, &rs_s, &vs_s, ox, oy, &mut a, &Plane::new(th, tw, &tooth_s), rough);
    Ok(())
}

/// `_splat_drops(xs, ys, rs, es, th, vs, ph, ox, oy, acc, tooth, rough, dome)`
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn tools_splat_drops(
    xs: PyReadonlyArrayDyn<'_, f64>,
    ys: PyReadonlyArrayDyn<'_, f64>,
    rs: PyReadonlyArrayDyn<'_, f64>,
    es: PyReadonlyArrayDyn<'_, f64>,
    th: PyReadonlyArrayDyn<'_, f64>,
    vs: PyReadonlyArrayDyn<'_, f64>,
    ph: PyReadonlyArrayDyn<'_, f64>,
    ox: i64,
    oy: i64,
    mut acc: PyReadwriteArrayDyn<'_, f64>,
    tooth: PyReadonlyArrayDyn<'_, f32>,
    rough: f64,
    dome: f64,
) -> PyResult<()> {
    let (h, w) = hw_mut(&acc, "acc")?;
    let (tgh, tgw) = hw(&tooth, "tooth")?;
    let tooth_s = ro_c(&tooth, "tooth")?;
    let (xs_s, ys_s, rs_s, es_s) = (ro_c(&xs, "xs")?, ro_c(&ys, "ys")?, ro_c(&rs, "rs")?, ro_c(&es, "es")?);
    let (th_s, vs_s, ph_s) = (ro_c(&th, "th")?, ro_c(&vs, "vs")?, ro_c(&ph, "ph")?);
    let n = xs_s.len();
    if [ys_s.len(), rs_s.len(), es_s.len(), th_s.len(), vs_s.len(), ph_s.len()].iter().any(|&l| l < n) {
        return Err(PyValueError::new_err("pinsel_core: drop arrays must have the same length"));
    }
    let mut a = PlaneMut::new(h, w, rw(&mut acc, "acc")?);
    ts::splat_drops(
        &xs_s,
        &ys_s,
        &rs_s,
        &es_s,
        &th_s,
        &vs_s,
        &ph_s,
        ox,
        oy,
        &mut a,
        &Plane::new(tgh, tgw, &tooth_s),
        rough,
        dome,
    );
    Ok(())
}

/// `_sponge_dab(cx, cy, rot, scale, tex, depth, soft, micro, alpha, pickup, rK, rS, rV, cap,
/// p_gloss, p_body, p_drate, fluid, water_lvl, wat_ratio, gran, stain, F, sid, mask, x0, y0,
/// tooth, wK, wS, wwat, wm, wg, wst, dK, dS, dm, dg, dst, wdamp)` -> bbox
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn tools_sponge_dab(
    cx: f64,
    cy: f64,
    rot: f64,
    scale: f64,
    tex: PyReadonlyArrayDyn<'_, f64>,
    depth: f64,
    soft: f64,
    micro: f64,
    alpha: f64,
    pickup: f64,
    mut r_k: PyReadwriteArrayDyn<'_, f64>,
    mut r_s: PyReadwriteArrayDyn<'_, f64>,
    mut r_v: PyReadwriteArrayDyn<'_, f64>,
    cap: f64,
    p_gloss: f64,
    p_body: f64,
    p_drate: f64,
    fluid: bool,
    water_lvl: f64,
    wat_ratio: f64,
    gran: f64,
    stain: f64,
    f: FilmT<'_>,
    sid: i64,
    mask: PyReadonlyArrayDyn<'_, f32>,
    x0: i64,
    y0: i64,
    tooth: PyReadonlyArrayDyn<'_, f32>,
    mut w_k: PyReadwriteArrayDyn<'_, f32>,
    mut w_s: PyReadwriteArrayDyn<'_, f32>,
    mut wwat: PyReadwriteArrayDyn<'_, f32>,
    mut wm: PyReadwriteArrayDyn<'_, f32>,
    mut wg: PyReadwriteArrayDyn<'_, f32>,
    mut wst: PyReadwriteArrayDyn<'_, f32>,
    mut d_k: PyReadwriteArrayDyn<'_, f32>,
    mut d_s: PyReadwriteArrayDyn<'_, f32>,
    mut dm: PyReadwriteArrayDyn<'_, f32>,
    mut dg: PyReadwriteArrayDyn<'_, f32>,
    mut dst: PyReadwriteArrayDyn<'_, f32>,
    mut wdamp: PyReadwriteArrayDyn<'_, f32>,
) -> PyResult<(i64, i64, i64, i64)> {
    let (t_h, t_w) = hw(&tex, "tex")?;
    if t_h < 2 || t_w < 2 || t_h != t_w {
        return Err(PyValueError::new_err("pinsel_core: tex must be square, at least 2 x 2"));
    }
    if r_k.shape()[1..] != [NB] || r_s.shape()[1..] != [NB] || r_v.len() < 1 {
        return Err(PyValueError::new_err("pinsel_core: reservoir arrays must be (n, NB) and (n,)"));
    }
    let tex_s = ro_c(&tex, "tex")?;
    let (mh, mw) = hw(&mask, "mask")?;
    let mask_s = ro_c(&mask, "mask")?;
    let (th, tw) = hw(&tooth, "tooth")?;
    let tooth_s = ro_c(&tooth, "tooth")?;
    // the water layer (real arrays, or 1 x 1 dummies when the paint is not fluid)
    let (wh, ww) = hw_mut(&wwat, "water")?;
    let (dh, dw) = hw_mut(&wdamp, "damp")?;
    let mut wc = WcLayer {
        k: SpecMut::new(wh, ww, NB, rw(&mut w_k, "wK")?),
        s: SpecMut::new(wh, ww, NB, rw(&mut w_s, "wS")?),
        water: PlaneMut::new(wh, ww, rw(&mut wwat, "water")?),
        m: PlaneMut::new(wh, ww, rw(&mut wm, "wm")?),
        g: PlaneMut::new(wh, ww, rw(&mut wg, "wg")?),
        st: PlaneMut::new(wh, ww, rw(&mut wst, "wst")?),
        dk: SpecMut::new(wh, ww, NB, rw(&mut d_k, "dK")?),
        ds: SpecMut::new(wh, ww, NB, rw(&mut d_s, "dS")?),
        dm: PlaneMut::new(wh, ww, rw(&mut dm, "dm")?),
        dg: PlaneMut::new(wh, ww, rw(&mut dg, "dg")?),
        dst: PlaneMut::new(wh, ww, rw(&mut dst, "dst")?),
    };
    let mut damp = PlaneMut::new(dh, dw, rw(&mut wdamp, "damp")?);
    let p = ts::SpongeParams {
        cx,
        cy,
        rot,
        scale,
        depth,
        soft,
        micro,
        alpha,
        pickup,
        cap,
        p_gloss,
        p_body,
        p_drate,
        fluid,
        water_lvl,
        wat_ratio,
        gran,
        stain,
        sid,
        x0,
        y0,
    };
    let mut film = Film::new(f);
    let (h, w, arrays) = film.arrays()?;
    if (h, w) != (th, tw) {
        return Err(PyValueError::new_err("pinsel_core: tooth must match the film size"));
    }
    // the film mask is never read here (the region mask comes in as `mask`); tooth stands in
    let mut cv = DenseWet::new(h, w, arrays, &tooth_s, &tooth_s);
    Ok(ts::sponge_dab(
        &p,
        &Plane::new(t_h, t_w, &tex_s),
        rw(&mut r_k, "rK")?,
        rw(&mut r_s, "rS")?,
        rw(&mut r_v, "rV")?,
        &mut cv,
        &Plane::new(mh, mw, &mask_s),
        &Plane::new(th, tw, &tooth_s),
        &mut wc,
        &mut damp,
    ))
}

/// `_pen_raster(px, py, rad, ink, skip, gap, fill, kind, nca, nsa, nib_hw, nib_ht,
/// ox, oy, cov, amt, tooth)`
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn tools_pen_raster(
    px: PyReadonlyArrayDyn<'_, f64>,
    py_: PyReadonlyArrayDyn<'_, f64>,
    rad: PyReadonlyArrayDyn<'_, f64>,
    ink: PyReadonlyArrayDyn<'_, f64>,
    skip: PyReadonlyArrayDyn<'_, f64>,
    gap: PyReadonlyArrayDyn<'_, f64>,
    fill: PyReadonlyArrayDyn<'_, f64>,
    kind: i64,
    nca: f64,
    nsa: f64,
    nib_hw: f64,
    nib_ht: f64,
    ox: i64,
    oy: i64,
    mut cov: PyReadwriteArrayDyn<'_, f64>,
    mut amt: PyReadwriteArrayDyn<'_, f64>,
    tooth: PyReadonlyArrayDyn<'_, f32>,
) -> PyResult<()> {
    let (h, w) = hw_mut(&cov, "cov")?;
    if hw_mut(&amt, "amt")? != (h, w) {
        return Err(PyValueError::new_err("pinsel_core: cov and amt must have the same shape"));
    }
    let (th, tw) = hw(&tooth, "tooth")?;
    let tooth_s = ro_c(&tooth, "tooth")?;
    let (px_s, py_s, rad_s, ink_s) = (ro_c(&px, "px")?, ro_c(&py_, "py")?, ro_c(&rad, "rad")?, ro_c(&ink, "ink")?);
    let (skip_s, gap_s, fill_s) = (ro_c(&skip, "skip")?, ro_c(&gap, "gap")?, ro_c(&fill, "fill")?);
    let n = px_s.len();
    if n == 0 {
        return Ok(());
    }
    if [py_s.len(), rad_s.len(), ink_s.len(), skip_s.len(), gap_s.len(), fill_s.len()].iter().any(|&l| l < n) {
        return Err(PyValueError::new_err("pinsel_core: pen arrays must have the same length"));
    }
    let mut c = PlaneMut::new(h, w, rw(&mut cov, "cov")?);
    let mut a = PlaneMut::new(h, w, rw(&mut amt, "amt")?);
    ts::pen_raster(
        &px_s,
        &py_s,
        &rad_s,
        &ink_s,
        &skip_s,
        &gap_s,
        &fill_s,
        kind,
        nca,
        nsa,
        nib_hw,
        nib_ht,
        ox,
        oy,
        &mut c,
        &mut a,
        &Plane::new(th, tw, &tooth_s),
    );
    Ok(())
}

/// `_feather(cov, amt, perm, iters, decay)`
#[pyfunction]
fn tools_feather(
    mut cov: PyReadwriteArrayDyn<'_, f64>,
    mut amt: PyReadwriteArrayDyn<'_, f64>,
    perm: PyReadonlyArrayDyn<'_, f64>,
    iters: i64,
    decay: f64,
) -> PyResult<()> {
    let (h, w) = hw_mut(&cov, "cov")?;
    if hw_mut(&amt, "amt")? != (h, w) || hw(&perm, "perm")? != (h, w) {
        return Err(PyValueError::new_err("pinsel_core: cov, amt and perm must have the same shape"));
    }
    let perm_s = ro_c(&perm, "perm")?;
    let mut c = PlaneMut::new(h, w, rw(&mut cov, "cov")?);
    let mut a = PlaneMut::new(h, w, rw(&mut amt, "amt")?);
    ts::feather(&mut c, &mut a, &Plane::new(h, w, &perm_s), iters, decay);
    Ok(())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tools_spray_accumulate, m)?)?;
    m.add_function(wrap_pyfunction!(tools_splat_discs, m)?)?;
    m.add_function(wrap_pyfunction!(tools_splat_drops, m)?)?;
    m.add_function(wrap_pyfunction!(tools_sponge_dab, m)?)?;
    m.add_function(wrap_pyfunction!(tools_pen_raster, m)?)?;
    m.add_function(wrap_pyfunction!(tools_feather, m)?)?;
    Ok(())
}
