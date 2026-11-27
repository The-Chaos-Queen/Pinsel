//! Bindings for pinsel_core::watercolor. Register each function in `register`.
//! Python names carry the module prefix (see docs/RUST_PORTING_GUIDE.md):
//! `wc_k_flow1`, ... one per kernel of `pinsel/watercolor.py`, with the same
//! arguments in the same order.
//!
//! The simulation kernels are called with windows of the canvas arrays
//! (`wc.water[y0:y1, x0:x1]`, ...): rows are strided, columns are not. The
//! helpers below turn such a numpy view into a stride-aware `Pl` / `Sp`
//! without copying, and reject anything else with a clear error.

use crate::util::ro;
use numpy::{Element, PyReadonlyArrayDyn, PyReadwriteArrayDyn, PyUntypedArrayMethods};
use pcore::watercolor as wc;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn bad(name: &str, why: &str) -> PyErr {
    PyValueError::new_err(format!("pinsel_core: {name} {why}"))
}

/// (h, w, row stride) of a 2-D view whose columns are contiguous.
fn dims2(shape: &[usize], strides: &[isize], name: &str) -> PyResult<(usize, usize, usize)> {
    if shape.len() != 2 {
        return Err(bad(name, "must be 2-D"));
    }
    let (h, w) = (shape[0], shape[1]);
    if (w > 1 && strides[1] != 1) || (h > 1 && strides[0] < w as isize) {
        return Err(bad(name, "needs contiguous rows (a window of a C-contiguous array)"));
    }
    Ok((h, w, if h > 1 { strides[0] as usize } else { w }))
}

/// (h, w, c, row stride) of a 3-D view whose (x, channel) axes are contiguous.
fn dims3(shape: &[usize], strides: &[isize], name: &str) -> PyResult<(usize, usize, usize, usize)> {
    if shape.len() != 3 {
        return Err(bad(name, "must be 3-D"));
    }
    let (h, w, c) = (shape[0], shape[1], shape[2]);
    if (c > 1 && strides[2] != 1) || (w > 1 && strides[1] != c as isize) || (h > 1 && strides[0] < (w * c) as isize) {
        return Err(bad(name, "needs contiguous rows (a window of a C-contiguous array)"));
    }
    Ok((h, w, c, if h > 1 { strides[0] as usize } else { w * c }))
}

fn span(h: usize, w: usize, c: usize, rs: usize) -> usize {
    if h == 0 || w == 0 || c == 0 {
        0
    } else {
        (h - 1) * rs + w * c
    }
}

fn pl<'a, T: Element + Copy>(a: &'a PyReadonlyArrayDyn<'_, T>, name: &str) -> PyResult<wc::Pl<'a, T>> {
    let v = a.as_array();
    let (h, w, rs) = dims2(v.shape(), v.strides(), name)?;
    // SAFETY: the view covers exactly these elements; the borrow guard lives as long as `a`.
    let d = unsafe { std::slice::from_raw_parts(v.as_ptr(), span(h, w, 1, rs)) };
    Ok(wc::Pl::new(h, w, rs, d))
}

fn pl_mut<'a, T: Element + Copy>(a: &'a mut PyReadwriteArrayDyn<'_, T>, name: &str) -> PyResult<wc::PlM<'a, T>> {
    let mut v = a.as_array_mut();
    let (h, w, rs) = dims2(v.shape(), v.strides(), name)?;
    // SAFETY: as above; the exclusive borrow guard lives as long as `a`.
    let d = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr(), span(h, w, 1, rs)) };
    Ok(wc::PlM::new(h, w, rs, d))
}

/// A contiguous 1-D array as an (n, 1) view.
fn col_mut<'a, T: Element + Copy>(a: &'a mut PyReadwriteArrayDyn<'_, T>, name: &str) -> PyResult<wc::PlM<'a, T>> {
    let mut v = a.as_array_mut();
    if v.ndim() != 1 || (v.len() > 1 && v.strides()[0] != 1) {
        return Err(bad(name, "must be a contiguous 1-D array"));
    }
    let n = v.len();
    // SAFETY: n contiguous elements.
    let d = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr(), n) };
    Ok(wc::PlM::new(n, 1, 1, d))
}

fn sp<'a>(a: &'a PyReadonlyArrayDyn<'_, f32>, name: &str) -> PyResult<wc::Sp<'a, f32>> {
    let v = a.as_array();
    let (h, w, c, rs) = dims3(v.shape(), v.strides(), name)?;
    // SAFETY: as in `pl`.
    let d = unsafe { std::slice::from_raw_parts(v.as_ptr(), span(h, w, c, rs)) };
    Ok(wc::Sp::new(h, w, c, rs, d))
}

fn sp_mut<'a>(a: &'a mut PyReadwriteArrayDyn<'_, f32>, name: &str) -> PyResult<wc::SpM<'a, f32>> {
    let mut v = a.as_array_mut();
    let (h, w, c, rs) = dims3(v.shape(), v.strides(), name)?;
    // SAFETY: as in `pl_mut`.
    let d = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr(), span(h, w, c, rs)) };
    Ok(wc::SpM::new(h, w, c, rs, d))
}

/// All views of one call must describe the same (h, w) window; the kernels
/// index them with the same (y, x).
fn same(name: &str, got: (usize, usize), want: (usize, usize)) -> PyResult<()> {
    if got == want {
        Ok(())
    } else {
        Err(bad(name, &format!("has window {got:?}, expected {want:?}")))
    }
}

/// Faces arrays: (h, w + 1) and (h + 1, w).
fn faces(name: &str, got: (usize, usize), h: usize, w: usize, vertical: bool) -> PyResult<()> {
    let want = if vertical { (h + 1, w) } else { (h, w + 1) };
    same(name, got, want)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn wc_k_flow1(
    h: PyReadonlyArrayDyn<'_, f32>,
    s: PyReadonlyArrayDyn<'_, f32>,
    hp: PyReadonlyArrayDyn<'_, f32>,
    cxf: PyReadonlyArrayDyn<'_, f32>,
    cyf: PyReadonlyArrayDyn<'_, f32>,
    th: PyReadonlyArrayDyn<'_, f32>,
    rho: PyReadonlyArrayDyn<'_, f32>,
    mask: PyReadonlyArrayDyn<'_, f32>,
    mut u: PyReadwriteArrayDyn<'_, f32>,
    mut v: PyReadwriteArrayDyn<'_, f32>,
    mut ur: PyReadwriteArrayDyn<'_, f32>,
    mut vr: PyReadwriteArrayDyn<'_, f32>,
    mut r: PyReadwriteArrayDyn<'_, f32>,
    prm: PyReadonlyArrayDyn<'_, f64>,
    tx: f64,
    ty: f64,
    wall: bool,
    use_tiles: bool,
    tiles: PyReadonlyArrayDyn<'_, bool>,
    mut rowtile: PyReadwriteArrayDyn<'_, f32>,
) -> PyResult<()> {
    let (h, s, hp, cxf, cyf, th, rho, mask) = (
        pl(&h, "h")?,
        pl(&s, "s")?,
        pl(&hp, "hp")?,
        pl(&cxf, "cxf")?,
        pl(&cyf, "cyf")?,
        pl(&th, "th")?,
        pl(&rho, "rho")?,
        pl(&mask, "mask")?,
    );
    let (n, w) = (h.height(), h.width());
    for (nm, p) in [("s", &s), ("hp", &hp), ("cxf", &cxf), ("cyf", &cyf), ("th", &th), ("rho", &rho), ("mask", &mask)] {
        same(nm, (p.height(), p.width()), (n, w))?;
    }
    let (u, v, ur, vr, r) = (
        pl_mut(&mut u, "u")?,
        pl_mut(&mut v, "v")?,
        pl_mut(&mut ur, "ur")?,
        pl_mut(&mut vr, "vr")?,
        pl_mut(&mut r, "r")?,
    );
    faces("u", (u.height(), u.width()), n, w, false)?;
    faces("ur", (ur.height(), ur.width()), n, w, false)?;
    faces("v", (v.height(), v.width()), n, w, true)?;
    faces("vr", (vr.height(), vr.width()), n, w, true)?;
    same("r", (r.height(), r.width()), (n, w))?;
    let tiles = pl(&tiles, "tiles")?;
    let rowtile = pl_mut(&mut rowtile, "rowtile")?;
    if rowtile.height() != n {
        return Err(bad("rowtile", "needs one row per window row"));
    }
    let prm = ro(&prm, "prm")?;
    if prm.len() < 25 {
        return Err(bad("prm", "needs 25 entries"));
    }
    if use_tiles && (tiles.height() * wc::TS < n || tiles.width() * wc::TS < w) {
        return Err(bad("tiles", "too small for the window"));
    }
    if rowtile.width() * wc::TS < w {
        return Err(bad("rowtile", "too narrow for the window"));
    }
    wc::flow1(&h, &s, &hp, &cxf, &cyf, &th, &rho, &mask, &u, &v, &ur, &vr, &r, prm, tx, ty, wall, use_tiles, &tiles, &rowtile);
    Ok(())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn wc_k_flow2(
    mut h: PyReadwriteArrayDyn<'_, f32>,
    mut u: PyReadwriteArrayDyn<'_, f32>,
    mut v: PyReadwriteArrayDyn<'_, f32>,
    ur: PyReadonlyArrayDyn<'_, f32>,
    vr: PyReadonlyArrayDyn<'_, f32>,
    r: PyReadonlyArrayDyn<'_, f32>,
    mut ax: PyReadwriteArrayDyn<'_, f32>,
    mut ay: PyReadwriteArrayDyn<'_, f32>,
    use_tiles: bool,
    tiles: PyReadonlyArrayDyn<'_, bool>,
) -> PyResult<()> {
    let h = pl_mut(&mut h, "h")?;
    let (n, w) = (h.height(), h.width());
    let (u, v, ax, ay) = (pl_mut(&mut u, "u")?, pl_mut(&mut v, "v")?, pl_mut(&mut ax, "ax")?, pl_mut(&mut ay, "ay")?);
    let (ur, vr, r) = (pl(&ur, "ur")?, pl(&vr, "vr")?, pl(&r, "r")?);
    for (nm, p) in [("u", &u), ("ax", &ax)] {
        faces(nm, (p.height(), p.width()), n, w, false)?;
    }
    for (nm, p) in [("v", &v), ("ay", &ay)] {
        faces(nm, (p.height(), p.width()), n, w, true)?;
    }
    faces("ur", (ur.height(), ur.width()), n, w, false)?;
    faces("vr", (vr.height(), vr.width()), n, w, true)?;
    same("r", (r.height(), r.width()), (n, w))?;
    let tiles = pl(&tiles, "tiles")?;
    if use_tiles && (tiles.height() * wc::TS < n || tiles.width() * wc::TS < w) {
        return Err(bad("tiles", "too small for the window"));
    }
    wc::flow2(&h, &u, &v, &ur, &vr, &r, &ax, &ay, use_tiles, &tiles);
    Ok(())
}

#[pyfunction]
fn wc_k_outflow(
    h: PyReadonlyArrayDyn<'_, f32>,
    ax: PyReadonlyArrayDyn<'_, f32>,
    ay: PyReadonlyArrayDyn<'_, f32>,
    mut den: PyReadwriteArrayDyn<'_, f32>,
) -> PyResult<()> {
    let (h, ax, ay, den) = (pl(&h, "h")?, pl(&ax, "ax")?, pl(&ay, "ay")?, pl_mut(&mut den, "den")?);
    let (n, w) = (h.height(), h.width());
    faces("ax", (ax.height(), ax.width()), n, w, false)?;
    faces("ay", (ay.height(), ay.width()), n, w, true)?;
    same("den", (den.height(), den.width()), (n, w))?;
    wc::outflow(&h, &ax, &ay, &den);
    Ok(())
}

/// Shared argument handling of the two pigment passes.
#[allow(clippy::too_many_arguments)]
fn pass(
    horizontal: bool,
    h: &PyReadonlyArrayDyn<'_, f32>,
    den: &PyReadonlyArrayDyn<'_, f32>,
    k: &mut PyReadwriteArrayDyn<'_, f32>,
    s: &mut PyReadwriteArrayDyn<'_, f32>,
    m: &mut PyReadwriteArrayDyn<'_, f32>,
    g: &mut PyReadwriteArrayDyn<'_, f32>,
    st: &mut PyReadwriteArrayDyn<'_, f32>,
    acc: &mut PyReadwriteArrayDyn<'_, f32>,
    salt: &PyReadonlyArrayDyn<'_, f32>,
    cf: &PyReadonlyArrayDyn<'_, f32>,
    cmix: f64,
    csalt: f64,
    hwet: f64,
    salt_on: bool,
) -> PyResult<()> {
    let (h, den, salt, cf) = (pl(h, "h")?, pl(den, "den")?, pl(salt, "salt")?, pl(cf, "cf")?);
    let (n, w) = (h.height(), h.width());
    for (nm, p) in [("den", &den), ("salt", &salt), ("cf", &cf)] {
        same(nm, (p.height(), p.width()), (n, w))?;
    }
    let (k, s) = (sp_mut(k, "K")?, sp_mut(s, "S")?);
    let (m, g, st) = (pl_mut(m, "m")?, pl_mut(g, "g")?, pl_mut(st, "st")?);
    same("K", (k.height(), k.width()), (n, w))?;
    same("S", (s.height(), s.width()), (n, w))?;
    if s.channels() != k.channels() {
        return Err(bad("S", "needs the channels of K"));
    }
    for (nm, p) in [("m", &m), ("g", &g), ("st", &st)] {
        same(nm, (p.height(), p.width()), (n, w))?;
    }
    let acc = pl_mut(acc, if horizontal { "ax" } else { "ay" })?;
    faces("acc", (acc.height(), acc.width()), n, w, !horizontal)?;
    let pg = wc::Pigment { k: &k, s: &s, m: &m, g: &g, st: &st };
    if horizontal {
        wc::xpass(&h, &den, &pg, &acc, &salt, &cf, cmix, csalt, hwet, salt_on);
    } else {
        wc::ypass(&h, &den, &pg, &acc, &salt, &cf, cmix, csalt, hwet, salt_on);
    }
    Ok(())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn wc_k_xpass(
    h: PyReadonlyArrayDyn<'_, f32>,
    den: PyReadonlyArrayDyn<'_, f32>,
    mut k: PyReadwriteArrayDyn<'_, f32>,
    mut s: PyReadwriteArrayDyn<'_, f32>,
    mut m: PyReadwriteArrayDyn<'_, f32>,
    mut g: PyReadwriteArrayDyn<'_, f32>,
    mut st: PyReadwriteArrayDyn<'_, f32>,
    mut ax: PyReadwriteArrayDyn<'_, f32>,
    salt: PyReadonlyArrayDyn<'_, f32>,
    cxf: PyReadonlyArrayDyn<'_, f32>,
    cmix: f64,
    csalt: f64,
    hwet: f64,
    salt_on: bool,
) -> PyResult<()> {
    pass(true, &h, &den, &mut k, &mut s, &mut m, &mut g, &mut st, &mut ax, &salt, &cxf, cmix, csalt, hwet, salt_on)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn wc_k_ypass(
    h: PyReadonlyArrayDyn<'_, f32>,
    den: PyReadonlyArrayDyn<'_, f32>,
    mut k: PyReadwriteArrayDyn<'_, f32>,
    mut s: PyReadwriteArrayDyn<'_, f32>,
    mut m: PyReadwriteArrayDyn<'_, f32>,
    mut g: PyReadwriteArrayDyn<'_, f32>,
    mut st: PyReadwriteArrayDyn<'_, f32>,
    mut ay: PyReadwriteArrayDyn<'_, f32>,
    salt: PyReadonlyArrayDyn<'_, f32>,
    cyf: PyReadonlyArrayDyn<'_, f32>,
    cmix: f64,
    csalt: f64,
    hwet: f64,
    salt_on: bool,
) -> PyResult<()> {
    pass(false, &h, &den, &mut k, &mut s, &mut m, &mut g, &mut st, &mut ay, &salt, &cyf, cmix, csalt, hwet, salt_on)
}

#[pyfunction]
fn wc_k_edge(
    h: PyReadonlyArrayDyn<'_, f32>,
    mask: PyReadonlyArrayDyn<'_, f32>,
    hwet: f64,
    r: i64,
    mut tmp: PyReadwriteArrayDyn<'_, f32>,
    mut e: PyReadwriteArrayDyn<'_, f32>,
) -> PyResult<()> {
    let (h, mask, tmp, e) = (pl(&h, "h")?, pl(&mask, "mask")?, pl_mut(&mut tmp, "tmp")?, pl_mut(&mut e, "E")?);
    let (n, w) = (h.height(), h.width());
    same("mask", (mask.height(), mask.width()), (n, w))?;
    same("tmp", (tmp.height(), tmp.width()), (n, w))?;
    same("E", (e.height(), e.width()), (n, w))?;
    if r < 0 {
        return Err(bad("R", "must be >= 0"));
    }
    wc::edge(&h, &mask, hwet, r, &tmp, &e);
    Ok(())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn wc_k_local(
    mut h: PyReadwriteArrayDyn<'_, f32>,
    s: PyReadonlyArrayDyn<'_, f32>,
    mut s2: PyReadwriteArrayDyn<'_, f32>,
    mut k: PyReadwriteArrayDyn<'_, f32>,
    mut sg: PyReadwriteArrayDyn<'_, f32>,
    mut m: PyReadwriteArrayDyn<'_, f32>,
    mut g: PyReadwriteArrayDyn<'_, f32>,
    mut st: PyReadwriteArrayDyn<'_, f32>,
    mut dk: PyReadwriteArrayDyn<'_, f32>,
    mut ds: PyReadwriteArrayDyn<'_, f32>,
    mut dm: PyReadwriteArrayDyn<'_, f32>,
    mut dg: PyReadwriteArrayDyn<'_, f32>,
    mut dst: PyReadwriteArrayDyn<'_, f32>,
    mut bond: PyReadwriteArrayDyn<'_, f32>,
    salt: PyReadonlyArrayDyn<'_, f32>,
    e: PyReadonlyArrayDyn<'_, f32>,
    gr: PyReadonlyArrayDyn<'_, f32>,
    mask: PyReadonlyArrayDyn<'_, f32>,
    cxc: PyReadonlyArrayDyn<'_, f32>,
    cyc: PyReadonlyArrayDyn<'_, f32>,
    u: PyReadonlyArrayDyn<'_, f32>,
    v: PyReadonlyArrayDyn<'_, f32>,
    evm: PyReadonlyArrayDyn<'_, f32>,
    prm: PyReadonlyArrayDyn<'_, f64>,
    dt: f64,
    dcap: f64,
    pdt: f64,
) -> PyResult<()> {
    let h = pl_mut(&mut h, "h")?;
    let (n, w) = (h.height(), h.width());
    let (s, salt, e, gr, mask, cxc, cyc, evm) = (
        pl(&s, "s")?,
        pl(&salt, "salt")?,
        pl(&e, "E")?,
        pl(&gr, "gr")?,
        pl(&mask, "mask")?,
        pl(&cxc, "cxc")?,
        pl(&cyc, "cyc")?,
        pl(&evm, "evm")?,
    );
    for (nm, p) in [("s", &s), ("salt", &salt), ("E", &e), ("gr", &gr), ("mask", &mask), ("cxc", &cxc), ("cyc", &cyc), ("evm", &evm)] {
        same(nm, (p.height(), p.width()), (n, w))?;
    }
    let s2 = pl_mut(&mut s2, "s2")?;
    same("s2", (s2.height(), s2.width()), (n, w))?;
    let (k, sg) = (sp_mut(&mut k, "K")?, sp_mut(&mut sg, "S")?);
    let (dk, ds) = (sp_mut(&mut dk, "dK")?, sp_mut(&mut ds, "dS")?);
    for (nm, p) in [("K", &k), ("S", &sg), ("dK", &dk), ("dS", &ds)] {
        same(nm, (p.height(), p.width()), (n, w))?;
        if p.channels() != k.channels() {
            return Err(bad(nm, "needs the channels of K"));
        }
    }
    let (m, g, st) = (pl_mut(&mut m, "m")?, pl_mut(&mut g, "g")?, pl_mut(&mut st, "st")?);
    let (dm, dg, dst, bond) = (pl_mut(&mut dm, "dm")?, pl_mut(&mut dg, "dg")?, pl_mut(&mut dst, "dst")?, pl_mut(&mut bond, "bond")?);
    for (nm, p) in [("m", &m), ("g", &g), ("st", &st), ("dm", &dm), ("dg", &dg), ("dst", &dst), ("bond", &bond)] {
        same(nm, (p.height(), p.width()), (n, w))?;
    }
    let (u, v) = (pl(&u, "u")?, pl(&v, "v")?);
    faces("u", (u.height(), u.width()), n, w, false)?;
    faces("v", (v.height(), v.width()), n, w, true)?;
    let prm = ro(&prm, "prm")?;
    if prm.len() < 25 {
        return Err(bad("prm", "needs 25 entries"));
    }
    let pg = wc::Pigment { k: &k, s: &sg, m: &m, g: &g, st: &st };
    let dp = wc::Deposit { dk: &dk, ds: &ds, dm: &dm, dg: &dg, dst: &dst, bond: &bond };
    wc::local(&h, &s, &s2, &pg, &dp, &salt, &e, &gr, &mask, &cxc, &cyc, &u, &v, &evm, prm, dt, dcap, pdt);
    Ok(())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn wc_k_salt(
    salt: PyReadonlyArrayDyn<'_, f32>,
    mut s2: PyReadwriteArrayDyn<'_, f32>,
    h: PyReadonlyArrayDyn<'_, f32>,
    s: PyReadonlyArrayDyn<'_, f32>,
    cxc: PyReadonlyArrayDyn<'_, f32>,
    cyc: PyReadonlyArrayDyn<'_, f32>,
    d: f64,
    hwet: f64,
) -> PyResult<()> {
    let (salt, h, s, cxc, cyc) = (pl(&salt, "salt")?, pl(&h, "h")?, pl(&s, "s")?, pl(&cxc, "cxc")?, pl(&cyc, "cyc")?);
    let (n, w) = (salt.height(), salt.width());
    for (nm, p) in [("h", &h), ("s", &s), ("cxc", &cxc), ("cyc", &cyc)] {
        same(nm, (p.height(), p.width()), (n, w))?;
    }
    let s2 = pl_mut(&mut s2, "s2")?;
    same("s2", (s2.height(), s2.width()), (n, w))?;
    wc::salt(&salt, &s2, &h, &s, &cxc, &cyc, d, hwet);
    Ok(())
}

#[pyfunction]
fn wc_k_active(
    h: PyReadonlyArrayDyn<'_, f32>,
    s: PyReadonlyArrayDyn<'_, f32>,
    m: PyReadonlyArrayDyn<'_, f32>,
    mut rowx0: PyReadwriteArrayDyn<'_, i64>,
    mut rowx1: PyReadwriteArrayDyn<'_, i64>,
) -> PyResult<()> {
    let (h, s, m) = (pl(&h, "h")?, pl(&s, "s")?, pl(&m, "m")?);
    let (n, w) = (h.height(), h.width());
    same("s", (s.height(), s.width()), (n, w))?;
    same("m", (m.height(), m.width()), (n, w))?;
    let (r0, r1) = (col_mut(&mut rowx0, "rowx0")?, col_mut(&mut rowx1, "rowx1")?);
    if r0.height() < n || r1.height() < n {
        return Err(bad("rowx0/rowx1", "need one entry per row"));
    }
    wc::active(&h, &s, &m, &r0, &r1);
    Ok(())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn wc_k_settle_all(
    mut k: PyReadwriteArrayDyn<'_, f32>,
    mut sg: PyReadwriteArrayDyn<'_, f32>,
    mut m: PyReadwriteArrayDyn<'_, f32>,
    mut g: PyReadwriteArrayDyn<'_, f32>,
    mut st: PyReadwriteArrayDyn<'_, f32>,
    mut dk: PyReadwriteArrayDyn<'_, f32>,
    mut ds: PyReadwriteArrayDyn<'_, f32>,
    mut dm: PyReadwriteArrayDyn<'_, f32>,
    mut dg: PyReadwriteArrayDyn<'_, f32>,
    mut dst: PyReadwriteArrayDyn<'_, f32>,
    mut h: PyReadwriteArrayDyn<'_, f32>,
    mut s: PyReadwriteArrayDyn<'_, f32>,
) -> PyResult<()> {
    let h = pl_mut(&mut h, "h")?;
    let (n, w) = (h.height(), h.width());
    let s = pl_mut(&mut s, "s")?;
    same("s", (s.height(), s.width()), (n, w))?;
    let (k, sg, dk, ds) = (sp_mut(&mut k, "K")?, sp_mut(&mut sg, "S")?, sp_mut(&mut dk, "dK")?, sp_mut(&mut ds, "dS")?);
    for (nm, p) in [("K", &k), ("S", &sg), ("dK", &dk), ("dS", &ds)] {
        same(nm, (p.height(), p.width()), (n, w))?;
        if p.channels() != k.channels() {
            return Err(bad(nm, "needs the channels of K"));
        }
    }
    let (m, g, st) = (pl_mut(&mut m, "m")?, pl_mut(&mut g, "g")?, pl_mut(&mut st, "st")?);
    let (dm, dg, dst) = (pl_mut(&mut dm, "dm")?, pl_mut(&mut dg, "dg")?, pl_mut(&mut dst, "dst")?);
    for (nm, p) in [("m", &m), ("g", &g), ("st", &st), ("dm", &dm), ("dg", &dg), ("dst", &dst)] {
        same(nm, (p.height(), p.width()), (n, w))?;
    }
    let pg = wc::Pigment { k: &k, s: &sg, m: &m, g: &g, st: &st };
    wc::settle_all(&pg, &dk, &ds, &dm, &dg, &dst, &h, &s);
    Ok(())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn wc_k_fibres(
    hh: i64,
    ww: i64,
    xs: PyReadonlyArrayDyn<'_, f64>,
    ys: PyReadonlyArrayDyn<'_, f64>,
    ang: PyReadonlyArrayDyn<'_, f64>,
    lens: PyReadonlyArrayDyn<'_, f64>,
    curv: PyReadonlyArrayDyn<'_, f64>,
    wts: PyReadonlyArrayDyn<'_, f64>,
    mut cx: PyReadwriteArrayDyn<'_, f64>,
    mut cy: PyReadwriteArrayDyn<'_, f64>,
    mut cov: PyReadwriteArrayDyn<'_, f64>,
) -> PyResult<()> {
    let (xs, ys, ang, lens, curv, wts) =
        (ro(&xs, "xs")?, ro(&ys, "ys")?, ro(&ang, "ang")?, ro(&lens, "lens")?, ro(&curv, "curv")?, ro(&wts, "wts")?);
    let nf = xs.len();
    if [ys.len(), ang.len(), lens.len(), curv.len(), wts.len()].iter().any(|&l| l != nf) {
        return Err(bad("xs, ys, ang, lens, curv, wts", "must have the same length"));
    }
    let (cx, cy, cov) = (pl_mut(&mut cx, "cx")?, pl_mut(&mut cy, "cy")?, pl_mut(&mut cov, "cov")?);
    for (nm, p) in [("cx", &cx), ("cy", &cy), ("cov", &cov)] {
        same(nm, (p.height(), p.width()), (hh.max(0) as usize, ww.max(0) as usize))?;
    }
    wc::fibres(hh, ww, xs, ys, ang, lens, curv, wts, &cx, &cy, &cov);
    Ok(())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(wc_k_flow1, m)?)?;
    m.add_function(wrap_pyfunction!(wc_k_flow2, m)?)?;
    m.add_function(wrap_pyfunction!(wc_k_outflow, m)?)?;
    m.add_function(wrap_pyfunction!(wc_k_xpass, m)?)?;
    m.add_function(wrap_pyfunction!(wc_k_ypass, m)?)?;
    m.add_function(wrap_pyfunction!(wc_k_edge, m)?)?;
    m.add_function(wrap_pyfunction!(wc_k_local, m)?)?;
    m.add_function(wrap_pyfunction!(wc_k_salt, m)?)?;
    m.add_function(wrap_pyfunction!(wc_k_active, m)?)?;
    m.add_function(wrap_pyfunction!(wc_k_settle_all, m)?)?;
    m.add_function(wrap_pyfunction!(wc_k_fibres, m)?)?;
    Ok(())
}
