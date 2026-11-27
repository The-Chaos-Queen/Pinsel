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
    w_k: &Bound<'_, PyAny>,
    w_s: &Bound<'_, PyAny>,
    wwat: &Bound<'_, PyAny>,
    wm: &Bound<'_, PyAny>,
    wg: &Bound<'_, PyAny>,
    wst: &Bound<'_, PyAny>,
    d_k: &Bound<'_, PyAny>,
    d_s: &Bound<'_, PyAny>,
    dm: &Bound<'_, PyAny>,
    dg: &Bound<'_, PyAny>,
    dst: &Bound<'_, PyAny>,
    wdamp: &Bound<'_, PyAny>,
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
    // The water layer: real canvas arrays when fluid; otherwise Python passes the
    // same 1 x 1 dummy for several of them (aliased, never touched), which cannot be
    // borrowed mutably together, so they are ignored and local dummies stand in.
    let mut fl: Vec<PyReadwriteArrayDyn<'_, f32>> = Vec::new();
    if fluid {
        for (a, n) in [
            (w_k, "wK"),
            (w_s, "wS"),
            (wwat, "water"),
            (wm, "wm"),
            (wg, "wg"),
            (wst, "wst"),
            (d_k, "dK"),
            (d_s, "dS"),
            (dm, "dm"),
            (dg, "dg"),
            (dst, "dst"),
            (wdamp, "damp"),
        ] {
            fl.push(a.extract().map_err(|_| PyValueError::new_err(format!("pinsel_core: {n} must be a float32 array")))?);
        }
    }
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
    let tex_p = Plane::new(t_h, t_w, &tex_s);
    let mask_p = Plane::new(mh, mw, &mask_s);
    let tooth_p = Plane::new(th, tw, &tooth_s);
    let (rk_s, rs_s, rv_s) = (rw(&mut r_k, "rK")?, rw(&mut r_s, "rS")?, rw(&mut r_v, "rV")?);
    if fluid {
        let mut it = fl.iter_mut();
        let mut next = || it.next().unwrap();
        let (a_k, a_s, a_w, a_m, a_g, a_st, a_dk, a_ds, a_dm, a_dg, a_dst, a_damp) = (
            next(),
            next(),
            next(),
            next(),
            next(),
            next(),
            next(),
            next(),
            next(),
            next(),
            next(),
            next(),
        );
        let (wh, ww) = hw_mut(a_w, "water")?;
        let (dh, dw) = hw_mut(a_damp, "damp")?;
        let mut wc = WcLayer {
            k: SpecMut::new(wh, ww, NB, rw(a_k, "wK")?),
            s: SpecMut::new(wh, ww, NB, rw(a_s, "wS")?),
            water: PlaneMut::new(wh, ww, rw(a_w, "water")?),
            m: PlaneMut::new(wh, ww, rw(a_m, "wm")?),
            g: PlaneMut::new(wh, ww, rw(a_g, "wg")?),
            st: PlaneMut::new(wh, ww, rw(a_st, "wst")?),
            dk: SpecMut::new(wh, ww, NB, rw(a_dk, "dK")?),
            ds: SpecMut::new(wh, ww, NB, rw(a_ds, "dS")?),
            dm: PlaneMut::new(wh, ww, rw(a_dm, "dm")?),
            dg: PlaneMut::new(wh, ww, rw(a_dg, "dg")?),
            dst: PlaneMut::new(wh, ww, rw(a_dst, "dst")?),
        };
        let mut damp = PlaneMut::new(dh, dw, rw(a_damp, "damp")?);
        Ok(ts::sponge_dab(&p, &tex_p, rk_s, rs_s, rv_s, &mut cv, &mask_p, &tooth_p, &mut wc, &mut damp))
    } else {
        let mut d3: Vec<Vec<f32>> = vec![vec![0.0; NB]; 4];
        let mut d2: Vec<Vec<f32>> = vec![vec![0.0; 1]; 9];
        let (d3a, d3b) = d3.split_at_mut(2);
        let (d2a, d2b) = d2.split_at_mut(4);
        let (d2b, d2c) = d2b.split_at_mut(2);
        let mut i3a = d3a.iter_mut();
        let mut i3b = d3b.iter_mut();
        let mut i2a = d2a.iter_mut();
        let mut i2b = d2b.iter_mut();
        let mut i2c = d2c.iter_mut();
        let mut wc = WcLayer {
            k: SpecMut::new(1, 1, NB, i3a.next().unwrap()),
            s: SpecMut::new(1, 1, NB, i3a.next().unwrap()),
            water: PlaneMut::new(1, 1, i2a.next().unwrap()),
            m: PlaneMut::new(1, 1, i2a.next().unwrap()),
            g: PlaneMut::new(1, 1, i2a.next().unwrap()),
            st: PlaneMut::new(1, 1, i2a.next().unwrap()),
            dk: SpecMut::new(1, 1, NB, i3b.next().unwrap()),
            ds: SpecMut::new(1, 1, NB, i3b.next().unwrap()),
            dm: PlaneMut::new(1, 1, i2b.next().unwrap()),
            dg: PlaneMut::new(1, 1, i2b.next().unwrap()),
            dst: PlaneMut::new(1, 1, i2c.next().unwrap()),
        };
        let mut damp_v = vec![0.0f32; 1];
        let mut damp = PlaneMut::new(1, 1, &mut damp_v);
        Ok(ts::sponge_dab(&p, &tex_p, rk_s, rs_s, rv_s, &mut cv, &mask_p, &tooth_p, &mut wc, &mut damp))
    }
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
