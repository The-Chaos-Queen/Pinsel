//! Port of pinsel (Python 0.1) kernels: tools_common. See docs/RUST_PORTING_GUIDE.md.
//!
//! Shared paint helpers of `pinsel/tools.py`: the numba functions that every
//! tool kernel (knife, sgraffito, smudge, crayon, and later spray, splatter,
//! sponge and pen) builds on. They are plain `pub` functions here so that the
//! other tool modules in this crate can call them.
//!
//! Number types mirror numba (docs/RUST_PORTING_GUIDE.md rule 3). The easy
//! mistakes, all handled below:
//! * the wet film arrays are f32. `f32 op f32` stays f32 (`fv = film + base`,
//!   `wet * fv`), `f32 op f64` is f64; every store back rounds to f32;
//! * a comparison of an f32 value with a float literal happens in f64, so f32
//!   values are widened (`as f64`) before they meet a constant;
//! * `min(a, b)` / `max(a, b)` follow Python (first argument unless the second
//!   is strictly smaller / larger);
//! * Python `round` is half-to-even.
//!
//! Film access goes through [`WetPx`] (one pixel from a `WetCanvas`); the
//! watercolour layer, which is not part of the paste film, goes through
//! `Plane`/`Spec` views.

use crate::grid::{PlaneMut, Plane, SpecMut, WetCanvas, WetPx};
use crate::NB;

/// `canvas.LEVELING` (keep in sync with `Canvas.surface_height`).
pub const LEVELING: f64 = 2.5;

/// Python `min(a, b)` on floats.
#[inline(always)]
pub fn pymin(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// Python `max(a, b)` on floats.
#[inline(always)]
pub fn pymax(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// Python `int(round(v))`: half to even.
#[inline(always)]
pub fn py_round(v: f64) -> i64 {
    v.round_ties_even() as i64
}

/// `_cl01`: clamp to 0..1.
#[inline(always)]
pub fn cl01(v: f64) -> f64 {
    if v < 0.0 {
        return 0.0;
    }
    if v > 1.0 {
        return 1.0;
    }
    v
}

/// `_sstep`: smoothstep of the clamped value.
#[inline(always)]
pub fn sstep(v: f64) -> f64 {
    let v = cl01(v);
    v * v * (3.0 - 2.0 * v)
}

/// `_surface_top`: visible surface height (`Canvas.surface_height`).
/// `fv` is the total wet volume (top + base, summed in f32 by the caller).
/// All four heights are float32 canvas values: `hd + fv * bd` is f32 arithmetic.
#[inline]
pub fn surface_top(hg: f32, hd: f32, fv: f32, bd: f32) -> f64 {
    let ph: f32 = hd + fv * bd;
    let phd = ph as f64;
    hg as f64 * (-LEVELING * phd).exp() + phd
}

/// `_solid_top`: height of ground + dried paint (what a rigid tool rides on).
#[inline]
pub fn solid_top(hg: f32, hd: f32) -> f64 {
    let hdd = hd as f64;
    hg as f64 * (-LEVELING * hdd).exp() + hdd
}

/// `_solve_paint_height`: paint height `ph >= lo` with
/// `hg * exp(-LEVELING * ph) + ph == b`. (Numba types `pm`, `ph` as f64
/// because `lo` is widened when it meets f64 values; so everything is f64.)
pub fn solve_paint_height(hg: f64, b: f64, lo: f64) -> f64 {
    let mut pm = lo;
    let khg = LEVELING * hg;
    if khg > 1.0 {
        let pmin = khg.ln() / LEVELING;
        if pmin > pm {
            pm = pmin;
        }
    }
    if pm + hg * (-LEVELING * pm).exp() >= b {
        return lo;
    }
    let mut ph = if b > pm { b } else { pm };
    for _ in 0..10 {
        let e = hg * (-LEVELING * ph).exp();
        let fp = 1.0 - LEVELING * e;
        if fp < 1e-6 {
            break;
        }
        ph -= (ph + e - b) / fp;
        if ph < pm {
            ph = pm;
            break;
        }
    }
    if ph > lo {
        ph
    } else {
        lo
    }
}

/// `_push`: a new mark (id `sid`) lies on top of older wet paint: push the
/// pixel's top layer down into the base layer.
pub fn push(q: &mut WetPx<'_>, sid: i64) {
    if *q.top_id as i64 != sid {
        if (*q.film_vol as f64) > 0.0 || (q.film_k[0] as f64) > 0.0 {
            for b in 0..NB {
                q.base_k[b] += q.film_k[b];
                q.base_s[b] += q.film_s[b];
                q.film_k[b] = 0.0;
                q.film_s[b] = 0.0;
            }
            *q.base_vol += *q.film_vol;
            *q.film_vol = 0.0;
        }
        *q.top_id = sid as i32;
    }
}

/// `_dep`: deposit `dv` volume of fresh paint (per-volume K/S) on top of the pixel.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dep(
    q: &mut WetPx<'_>,
    sid: i64,
    dv: f64,
    kpv: &[f64],
    spv: &[f64],
    p_gloss: f64,
    p_body: f64,
    p_drate: f64,
) {
    dep_w(q, sid, dv, kpv, spv, 1.0, p_gloss, p_body, p_drate);
}

/// `_dep_w`: deposit `dv` volume of paint with wetness `p_wet` on top of the pixel.
#[allow(clippy::too_many_arguments)]
pub fn dep_w(
    q: &mut WetPx<'_>,
    sid: i64,
    dv: f64,
    kpv: &[f64],
    spv: &[f64],
    p_wet: f64,
    p_gloss: f64,
    p_body: f64,
    p_drate: f64,
) {
    if dv <= 0.0 {
        return;
    }
    push(q, sid);
    let fv: f32 = *q.film_vol + *q.base_vol; // f32 + f32, as in numba
    let nv = fv as f64 + dv;
    for b in 0..NB {
        q.film_k[b] = (q.film_k[b] as f64 + kpv[b] * dv) as f32;
        q.film_s[b] = (q.film_s[b] as f64 + spv[b] * dv) as f32;
    }
    // `wet[y, x] * fv` is f32 * f32 (rounded to f32) before it meets f64
    *q.wet = (((*q.wet * fv) as f64 + p_wet * dv) / nv) as f32;
    *q.gloss = (((*q.gloss * fv) as f64 + p_gloss * dv) / nv) as f32;
    *q.body = (((*q.body * fv) as f64 + p_body * dv) / nv) as f32;
    *q.drate = (((*q.drate * fv) as f64 + p_drate * dv) / nv) as f32;
    *q.film_vol = (*q.film_vol as f64 + dv) as f32;
}

/// `_take`: remove volume `dv` of wet paint (top layer first, then the older
/// base layer) and add its K/S to the row `rk` / `rs` (NB values each).
/// Returns the volume actually removed.
pub fn take(q: &mut WetPx<'_>, dv: f64, rk: &mut [f64], rs: &mut [f64]) -> f64 {
    if dv <= 0.0 {
        return 0.0;
    }
    let ft = *q.film_vol;
    let fb = *q.base_vol;
    let (ftd, fbd) = (ft as f64, fb as f64);
    let dt = if dv < ftd { dv } else { ftd };
    let mut db = dv - dt;
    if db > fbd {
        db = fbd;
    }
    let frt = if ftd > 1e-12 { dt / ftd } else { 0.0 };
    let frb = if fbd > 1e-12 { db / fbd } else { 0.0 };
    for b in 0..NB {
        let k1 = q.film_k[b] as f64 * frt;
        let s1 = q.film_s[b] as f64 * frt;
        let k2 = q.base_k[b] as f64 * frb;
        let s2 = q.base_s[b] as f64 * frb;
        q.film_k[b] = (q.film_k[b] as f64 - k1) as f32;
        q.film_s[b] = (q.film_s[b] as f64 - s1) as f32;
        q.base_k[b] = (q.base_k[b] as f64 - k2) as f32;
        q.base_s[b] = (q.base_s[b] as f64 - s2) as f32;
        rk[b] += k1 + k2;
        rs[b] += s1 + s2;
    }
    *q.film_vol = (ftd - dt) as f32;
    *q.base_vol = (fbd - db) as f32;
    dt + db
}

/// `_gauss_dep_bin`: deposit `dvt` volume from reservoir row `ri` as a
/// gaussian blob (ridge segment) centred at (`ex`, `ey`). `rk`, `rs` are the
/// reservoir's K and S (rows of NB values), `rv` its volumes. Grows the bbox
/// `bb = [x0, y0, x1, y1]`; returns the volume used. Respects the canvas mask.
#[allow(clippy::too_many_arguments)]
pub fn gauss_dep_bin<C: WetCanvas>(
    cv: &mut C,
    sid: i64,
    ex: f64,
    ey: f64,
    sig: f64,
    dvt: f64,
    rk: &mut [f64],
    rs: &mut [f64],
    rv: &mut [f64],
    ri: usize,
    p_gloss: f64,
    p_body: f64,
    p_drate: f64,
    bb: &mut [i64; 4],
) -> f64 {
    let vb = rv[ri];
    if vb <= 1e-9 || dvt <= 1e-10 {
        return 0.0;
    }
    let h = cv.height() as i64;
    let w = cv.width() as i64;
    let dvt = if dvt > vb { vb } else { dvt };
    let rr = (2.5 * sig).ceil() as i64;
    let ix = py_round(ex);
    let iy = py_round(ey);
    let i2 = 0.5 / (sig * sig);
    let mut wsum = 0.0f64;
    for yy in (iy - rr)..=(iy + rr) {
        for xx in (ix - rr)..=(ix + rr) {
            let dx = xx as f64 - ex;
            let dy = yy as f64 - ey;
            wsum += (-(dx * dx + dy * dy) * i2).exp();
        }
    }
    if wsum <= 1e-12 {
        return 0.0;
    }
    let mut kk = [0.0f64; NB];
    let mut ks = [0.0f64; NB];
    for b in 0..NB {
        kk[b] = rk[ri * NB + b] / vb;
        ks[b] = rs[ri * NB + b] / vb;
    }
    let mut used = 0.0f64;
    for yy in (iy - rr)..=(iy + rr) {
        for xx in (ix - rr)..=(ix + rr) {
            if xx < 0 || yy < 0 || xx >= w || yy >= h {
                continue;
            }
            let dx = xx as f64 - ex;
            let dy = yy as f64 - ey;
            let dv = dvt * (-(dx * dx + dy * dy) * i2).exp() / wsum
                * (1.0 - cv.mask(yy as usize, xx as usize) as f64);
            if dv <= 1e-10 {
                continue;
            }
            used += dv;
            dep(&mut cv.px(yy as usize, xx as usize), sid, dv, &kk, &ks, p_gloss, p_body, p_drate);
            if xx < bb[0] {
                bb[0] = xx;
            }
            if yy < bb[1] {
                bb[1] = yy;
            }
            if xx > bb[2] {
                bb[2] = xx;
            }
            if yy > bb[3] {
                bb[3] = yy;
            }
        }
    }
    let fu = used / vb;
    for b in 0..NB {
        rk[ri * NB + b] *= 1.0 - fu;
        rs[ri * NB + b] *= 1.0 - fu;
    }
    rv[ri] = vb - used;
    used
}

/// `_bins_shift`: move fraction `fr` of knife bin `src` into bin `dst`.
pub fn bins_shift(res_k: &mut [f64], res_s: &mut [f64], res_vol: &mut [f64], src: usize, dst: usize, fr: f64) {
    if fr <= 0.0 {
        return;
    }
    for b in 0..NB {
        let dk = res_k[src * NB + b] * fr;
        let ds = res_s[src * NB + b] * fr;
        res_k[src * NB + b] -= dk;
        res_s[src * NB + b] -= ds;
        res_k[dst * NB + b] += dk;
        res_s[dst * NB + b] += ds;
    }
    let dv = res_vol[src] * fr;
    res_vol[src] -= dv;
    res_vol[dst] += dv;
}

/// The watercolour layer's arrays used by the tools (`canvas.wc`): suspended
/// pigment `K`/`S`, water, pigment mass `m` with granulation `g` and staining
/// `st` channels, and the settled (deposited) layer `dK`/`dS`/`dm`/`dg`/`dst`.
pub struct WcLayer<'a> {
    pub k: SpecMut<'a, f32>,
    pub s: SpecMut<'a, f32>,
    pub water: PlaneMut<'a, f32>,
    pub m: PlaneMut<'a, f32>,
    pub g: PlaneMut<'a, f32>,
    pub st: PlaneMut<'a, f32>,
    pub dk: SpecMut<'a, f32>,
    pub ds: SpecMut<'a, f32>,
    pub dm: PlaneMut<'a, f32>,
    pub dg: PlaneMut<'a, f32>,
    pub dst: PlaneMut<'a, f32>,
}

/// `_wc_add`: add `water` and pigment mass `m` (suspended) to the watercolour layer.
#[allow(clippy::too_many_arguments)]
pub fn wc_add(
    wc: &mut WcLayer<'_>,
    y: usize,
    x: usize,
    water: f64,
    m: f64,
    kpv: &[f64],
    spv: &[f64],
    gran: f64,
    stain: f64,
) {
    if m > 0.0 {
        let wk = wc.k.px_mut(y, x);
        let ws = wc.s.px_mut(y, x);
        for b in 0..NB {
            wk[b] = (wk[b] as f64 + kpv[b] * m) as f32;
            ws[b] = (ws[b] as f64 + spv[b] * m) as f32;
        }
        let v = wc.m.get_mut(y, x);
        *v = (*v as f64 + m) as f32;
        let v = wc.g.get_mut(y, x);
        *v = (*v as f64 + m * gran) as f32;
        let v = wc.st.get_mut(y, x);
        *v = (*v as f64 + m * stain) as f32;
    }
    if water > 0.0 {
        let v = wc.water.get_mut(y, x);
        *v = (*v as f64 + water) as f32;
    }
}

/// `_wc_settle_add`: add pigment mass `m` directly to the deposited (settled) layer.
#[allow(clippy::too_many_arguments)]
pub fn wc_settle_add(
    wc: &mut WcLayer<'_>,
    y: usize,
    x: usize,
    m: f64,
    kpv: &[f64],
    spv: &[f64],
    gran: f64,
    stain: f64,
) {
    if m <= 0.0 {
        return;
    }
    let dk = wc.dk.px_mut(y, x);
    let ds = wc.ds.px_mut(y, x);
    for b in 0..NB {
        dk[b] = (dk[b] as f64 + kpv[b] * m) as f32;
        ds[b] = (ds[b] as f64 + spv[b] * m) as f32;
    }
    let v = wc.dm.get_mut(y, x);
    *v = (*v as f64 + m) as f32;
    let v = wc.dg.get_mut(y, x);
    *v = (*v as f64 + m * gran) as f32;
    let v = wc.dst.get_mut(y, x);
    *v = (*v as f64 + m * stain) as f32;
}

/// `_apply_volume_map`: deposit a bbox-local paint volume map (`vol`, `h` x `w`,
/// row-major, origin (`ox`, `oy`) on the canvas) on top of the wet film.
/// Respects the canvas mask (`cv.mask`).
#[allow(clippy::too_many_arguments)]
pub fn apply_volume_map<C: WetCanvas>(
    vol: &Plane<'_, f64>,
    ox: i64,
    oy: i64,
    kpv: &[f64],
    spv: &[f64],
    p_gloss: f64,
    p_body: f64,
    p_drate: f64,
    cv: &mut C,
    sid: i64,
) {
    let (h, w) = (vol.height(), vol.width());
    let hh = cv.height() as i64;
    let ww = cv.width() as i64;
    for i in 0..h {
        let yy = oy + i as i64;
        if yy < 0 || yy >= hh {
            continue;
        }
        for j in 0..w {
            let xx = ox + j as i64;
            if xx < 0 || xx >= ww {
                continue;
            }
            let (yu, xu) = (yy as usize, xx as usize);
            let dv = vol.at(i, j) * (1.0 - cv.mask(yu, xu) as f64);
            if dv > 1e-9 {
                dep(&mut cv.px(yu, xu), sid, dv, kpv, spv, p_gloss, p_body, p_drate);
            }
        }
    }
}

/// `_apply_fluid_map`: deposit a bbox-local pigment-mass map into the
/// watercolour layer: `settle_frac` of it lands as dried-on-contact pigment,
/// the rest is suspended in `water_ratio * mass` of water. `mask` is the
/// watercolour mask of the (clipped) region starting at (max(0, ox), max(0, oy)).
#[allow(clippy::too_many_arguments)]
pub fn apply_fluid_map(
    vol: &Plane<'_, f64>,
    ox: i64,
    oy: i64,
    water_ratio: f64,
    settle_frac: f64,
    kpv: &[f64],
    spv: &[f64],
    gran: f64,
    stain: f64,
    wc: &mut WcLayer<'_>,
    mask: &Plane<'_, f32>,
) {
    let (h, w) = (vol.height(), vol.width());
    let hh = wc.water.height() as i64;
    let ww = wc.water.width() as i64;
    let mx = ox.max(0);
    let my = oy.max(0);
    for i in 0..h {
        let yy = oy + i as i64;
        if yy < 0 || yy >= hh {
            continue;
        }
        for j in 0..w {
            let xx = ox + j as i64;
            if xx < 0 || xx >= ww {
                continue;
            }
            let m = vol.at(i, j) * (1.0 - mask.at((yy - my) as usize, (xx - mx) as usize) as f64);
            if m <= 1e-10 {
                continue;
            }
            let ms = m * settle_frac;
            let (yu, xu) = (yy as usize, xx as usize);
            wc_settle_add(wc, yu, xu, ms, kpv, spv, gran, stain);
            wc_add(wc, yu, xu, m * water_ratio, m - ms, kpv, spv, gran, stain);
        }
    }
}
