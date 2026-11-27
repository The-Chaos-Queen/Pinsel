//! Port of pinsel (Python 0.1) kernels: tools_film. See docs/RUST_PORTING_GUIDE.md.
//!
//! The film-based tools of `pinsel/tools.py`: palette knife (spread, scrape),
//! sgraffito, smudge (and Blender's finger work) and the oil crayon. They move
//! paste paint in the wet film, so they take a [`WetCanvas`]; other arrays
//! (`h_ground`, `h_dry`, `smax`, noise maps, `wax`) come in as `Plane` views.
//! The shared helpers live in [`crate::tools_common`].
//!
//! The kernels read the film-tuple `F` of the numba version as a `WetCanvas`.
//! `WetCanvas::mask` is the canvas mask; `WetCanvas::tooth` is only used by the
//! crayon (the other kernels never call it).
//!
//! None of these kernels draws random numbers (the numba originals do not call
//! `np.random`), so all four are expected to be bit-identical to numba.
//!
//! Number types and operation order mirror the Python line by line; see the
//! notes at the top of `tools_common.rs`.

use crate::grid::{Plane, PlaneMut, WetCanvas};
use crate::tools_common::{
    bins_shift, cl01, dep, dep_w, gauss_dep_bin, push, py_round, pymax, pymin, solid_top, solve_paint_height, sstep,
    surface_top, take,
};
use crate::NB;

#[inline(always)]
fn grow(bb: &mut [i64; 4], x: i64, y: i64) {
    if x < bb[0] {
        bb[0] = x;
    }
    if y < bb[1] {
        bb[1] = y;
    }
    if x > bb[2] {
        bb[2] = x;
    }
    if y > bb[3] {
        bb[3] = y;
    }
}

/// (top + base wet volume as f32, body) of a pixel.
#[inline(always)]
fn vol_body<C: WetCanvas>(cv: &mut C, y: usize, x: usize) -> (f32, f32) {
    let q = cv.px(y, x);
    (*q.film_vol + *q.base_vol, *q.body)
}

#[inline(always)]
fn wet_at<C: WetCanvas>(cv: &mut C, y: usize, x: usize) -> f32 {
    *cv.px(y, x).wet
}

/// Rows of a (rows, NB) reservoir.
fn row_mut(a: &mut [f64], i: usize) -> &mut [f64] {
    &mut a[i * NB..(i + 1) * NB]
}

// ==========================================================================
// Palette knife
// ==========================================================================

/// Inputs of `_knife_kernel` that are not paint state (names as in Python).
pub struct KnifeIn<'a> {
    pub px: &'a [f64],
    pub py: &'a [f64],
    pub pp: &'a [f64],
    pub pa: &'a [f64],
    /// Arclength of every path sample.
    pub ps: &'a [f64],
    /// 0 = spread (lay paint), 1 = scrape.
    pub mode: i64,
    pub half: f64,
    /// Samples along the blade for the blade-level profile.
    pub nk: i64,
    /// Capacity of each bin of the knife's load.
    pub cap: &'a [f64],
    /// The 21 tuning parameters (`prm[0]` thickness ... `prm[20]` bite).
    pub prm: &'a [f64],
    /// Paint lumpiness texture, (NU, NS).
    pub lump_tex: Plane<'a, f64>,
    pub stri: &'a [f64],
    pub chat: &'a [f64],
    pub flex_n: i64,
    pub p_gloss: f64,
    pub p_body: f64,
    pub p_drate: f64,
    pub sid: i64,
    pub h_ground: Plane<'a, f32>,
    pub h_dry: Plane<'a, f32>,
    /// Local maximum of the solid relief for the region starting at (`sox`, `soy`).
    pub smax: Plane<'a, f32>,
    pub sox: i64,
    pub soy: i64,
}

/// Sweep a rigid blade along the path (`_knife_kernel`). `res_k`, `res_s`
/// (nbin x NB) and `res_vol` (nbin) are the blade's paint load and are updated.
/// Returns the touched bbox `(x0, y0, x1, y1)`.
pub fn knife_kernel<C: WetCanvas>(
    a: &KnifeIn<'_>,
    res_k: &mut [f64],
    res_s: &mut [f64],
    res_vol: &mut [f64],
    cv: &mut C,
) -> (i64, i64, i64, i64) {
    let (px, py, pp, pa, ps) = (a.px, a.py, a.pp, a.pa, a.ps);
    let (mode, half, nk) = (a.mode, a.half, a.nk);
    let (cap, prm) = (a.cap, a.prm);
    let (stri, chat, flex_n) = (a.stri, a.chat, a.flex_n);
    let (p_gloss, p_body, p_drate, sid) = (a.p_gloss, a.p_body, a.p_drate, a.sid);
    let thickness = prm[0];
    let float_k = prm[1];
    let reach_base = prm[2];
    let reach_press = prm[3];
    let lumpiness = prm[4];
    let stri_amp = prm[5];
    let ride = prm[6];
    let edge_w = prm[7];
    let transfer = prm[8];
    let pickup = prm[9];
    let squeeze = prm[10];
    let plow = prm[11];
    let ridge_w = prm[12];
    let leave = prm[13];
    let scrape_eff = prm[14];
    let lift_frac = prm[15];
    let touch_frac = prm[16];
    let lift_noise = prm[17];
    let spill_cap = prm[18];
    let diffuse = prm[19];
    let bite = prm[20];
    let h = cv.height() as i64;
    let w = cv.width() as i64;
    let m_n = px.len();
    let nbin = res_vol.len() as i64;
    let l = 2.0 * half;
    let nu = a.lump_tex.height() as i64;
    let ns = a.lump_tex.width() as i64;
    let nku = nk as usize;
    let mut bb: [i64; 4] = [w, h, -1, -1];
    let mut lev = vec![0.0f64; nku];
    let mut wlev = vec![0.0f64; nku];
    let mut cc = vec![0.0f64; nku];
    let mut tmp = vec![0.0f64; nku];
    let mut kk = [0.0f64; NB];
    let mut ks = [0.0f64; NB];
    let (mut pnx, mut pny, mut pcx, mut pcy) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let soft = 0.12 * thickness + 0.03;
    for j in 0..m_n {
        let cx = px[j];
        let cy = py[j];
        let p = pp[j];
        let ang = pa[j];
        let bxd = ang.cos();
        let byd = ang.sin();
        let j0 = if j > 0 { j - 1 } else { 0 };
        let j1 = if j > 0 {
            j
        } else if m_n > 1 {
            1
        } else {
            0
        };
        let mut mx = px[j1] - px[j0];
        let mut my = py[j1] - py[j0];
        let step = (mx * mx + my * my).sqrt();
        if step > 1e-9 {
            mx /= step;
            my /= step;
        } else {
            mx = -byd;
            my = bxd;
        }
        let mut nx = -byd;
        let mut ny = bxd;
        if nx * mx + ny * my < 0.0 {
            nx = -nx;
            ny = -ny;
        }
        let bm = bxd * mx + byd * my; // cos(angle between blade and motion)
        // rigid blade: one float height from the blade-averaged load
        let mut lsum = 0.0f64;
        let mut csum = 0.0f64;
        for ib in 0..nbin as usize {
            lsum += res_vol[ib];
            csum += cap[ib];
        }
        let lmean = pymin(lsum / (csum + 1e-12), 1.5);
        let tkm = if lmean > 0.0 { thickness * lmean.powf(0.7) } else { 0.0 };
        if j == 0 {
            pnx = nx;
            pny = ny;
            pcx = cx;
            pcy = cy;
            // touch down: a little paint squeezed out behind the blade
            if mode == 0 && touch_frac > 0.0 {
                for k in 0..nku {
                    let u = -half + l * (k as f64 + 0.5) / nk as f64;
                    let ib = ((((u + half) / l * nbin as f64) as i64).max(0)).min(nbin - 1) as usize;
                    let vb = res_vol[ib];
                    if vb <= 1e-9 {
                        continue;
                    }
                    let ldt = pymin(vb / (cap[ib] + 1e-12), 1.5);
                    let xx = py_round(cx + bxd * u - nx * 0.7);
                    let yy = py_round(cy + byd * u - ny * 0.7);
                    if xx < 0 || yy < 0 || xx >= w || yy >= h {
                        continue;
                    }
                    let (yu, xu) = (yy as usize, xx as usize);
                    let mut dv = touch_frac * thickness * ldt.powf(0.7) * (0.5 + 0.5 * p) * (1.0 - cv.mask(yu, xu) as f64);
                    if dv <= 0.0 {
                        continue;
                    }
                    let fd = pymin(dv / vb, 0.5);
                    dv = vb * fd;
                    for b in 0..NB {
                        kk[b] = res_k[ib * NB + b] / vb;
                        ks[b] = res_s[ib * NB + b] / vb;
                        res_k[ib * NB + b] -= res_k[ib * NB + b] * fd;
                        res_s[ib * NB + b] -= res_s[ib * NB + b] * fd;
                    }
                    res_vol[ib] = vb - dv;
                    dep(&mut cv.px(yu, xu), sid, dv, &kk, &ks, p_gloss, p_body, p_drate);
                }
            }
            continue;
        }
        if pnx * nx + pny * ny < 0.0 {
            pnx = -pnx;
            pny = -pny;
        }
        // scraping: the sharp, pressed edge bites a little below the ridges
        let f = if mode == 0 { tkm * float_k * (1.0 - p) + 0.02 } else { leave - bite * p };
        // ---- blade level along the blade: rides on the highest solid relief;
        //      a lightly held blade also rides on soft wet paint
        let rf = if mode == 0 { pymin(ride * pymax(1.0 - p, 0.0).sqrt(), 1.0) } else { 0.0 };
        for k in 0..nku {
            let u = -half + l * k as f64 / (nk - 1) as f64;
            // the blade face rests on what lies just ahead of its trailing edge
            let mut lv = -1e9f64;
            wlev[k] = 0.0;
            for ahead in [1.5f64, 4.0f64] {
                let xx = py_round(cx + bxd * u + nx * ahead);
                let yy = py_round(cy + byd * u + ny * ahead);
                if xx < 0 || yy < 0 || xx >= w || yy >= h {
                    continue;
                }
                let (yu, xu) = (yy as usize, xx as usize);
                let hd = a.h_dry.at(yu, xu);
                let hg = a.h_ground.at(yu, xu);
                let sol = solid_top(hg, hd);
                let (fvv, bdd) = vol_body(cv, yu, xu);
                let top = surface_top(hg, hd, fvv, bdd);
                if sol > lv {
                    lv = sol;
                }
                let wv = (top - sol) * rf;
                if wv > wlev[k] {
                    wlev[k] = wv;
                }
            }
            lev[k] = lv;
        }
        // rigid-ish blade rests on the highest solid points within its flex
        // window; soft wet paint only supports it locally
        for k in 0..nk {
            let mut mxv = -1e9f64;
            for q in 0i64.max(k - flex_n)..nk.min(k + flex_n + 1) {
                if lev[q as usize] > mxv {
                    mxv = lev[q as usize];
                }
            }
            tmp[k as usize] = mxv;
        }
        for k in 0..nk {
            let mut sm = 0.0f64;
            let mut cnt = 0i64;
            for q in 0i64.max(k - flex_n)..nk.min(k + flex_n + 1) {
                if tmp[q as usize] > -1e8 {
                    sm += tmp[q as usize];
                    cnt += 1;
                }
            }
            let mut sw = 0.0f64;
            let mut cw = 0i64;
            for q in 0i64.max(k - 3)..nk.min(k + 4) {
                sw += wlev[q as usize];
                cw += 1;
            }
            cc[k as usize] = (if cnt > 0 { sm / cnt as f64 } else { 0.0 }) + (if cw > 0 { sw / cw as f64 } else { 0.0 });
        }
        // ---- bbox of the swept quad
        let xa = pymin(pcx, cx) - bxd.abs() * half - 2.0;
        let xb = pymax(pcx, cx) + bxd.abs() * half + 2.0;
        let ya = pymin(pcy, cy) - byd.abs() * half - 2.0;
        let yb = pymax(pcy, cy) + byd.abs() * half + 2.0;
        let x0 = (xa.floor() as i64).max(0);
        let x1 = (xb.ceil() as i64).min(w - 1);
        let y0 = (ya.floor() as i64).max(0);
        let y1 = (yb.ceil() as i64).min(h - 1);
        let isj = (ps[j] as i64).min(ns - 1);
        let chv = chat[isj.min(chat.len() as i64 - 1) as usize];
        for yy in y0..=y1 {
            for xx in x0..=x1 {
                let dc = (xx as f64 - cx) * nx + (yy as f64 - cy) * ny;
                if dc > 0.0 {
                    continue;
                }
                let dp = (xx as f64 - pcx) * pnx + (yy as f64 - pcy) * pny;
                if dp <= 0.0 || dc < -(step + 2.5) {
                    continue;
                }
                let u = (xx as f64 - cx) * bxd + (yy as f64 - cy) * byd;
                let au = u.abs();
                if au > half + edge_w {
                    continue;
                }
                let cov = cl01((half - au) / edge_w + 0.5);
                let (yu, xu) = (yy as usize, xx as usize);
                let m = (1.0 - cv.mask(yu, xu) as f64) * cov;
                if m <= 0.0 {
                    continue;
                }
                grow(&mut bb, xx, yy);
                let kf = pymin(pymax((u + half) / l * (nk - 1) as f64, 0.0), nk as f64 - 1.0);
                let k0 = kf as i64;
                let k1 = (k0 + 1).min(nk - 1);
                let fk = kf - k0 as f64;
                let cu = cc[k0 as usize] * (1.0 - fk) + cc[k1 as usize] * fk;
                let ib = (((((u + half) / l * nbin as f64) as i64).max(0)).min(nbin - 1)) as usize;
                let load = pymin(res_vol[ib] / (cap[ib] + 1e-12), 1.5);
                let tk = if load > 0.0 { thickness * load.powf(0.7) } else { 0.0 };
                let iu = (((u + half) as i64).max(0)).min(nu - 1) as usize;
                let bb_h = cu + f + stri_amp * stri[iu] + chv;
                let hd = a.h_dry.at(yu, xu);
                let hg = a.h_ground.at(yu, xu);
                if mode == 0 {
                    // fresh knife paint lies on top of older wet paint
                    push(&mut cv.px(yu, xu), sid);
                }
                let (mut fv, bd) = vol_body(cv, yu, xu);
                let mut top = surface_top(hg, hd, fv, bd);
                let mut gap = bb_h - top;
                if gap < 0.0 {
                    // -- wet paint above the blade is scraped onto the knife
                    let wh: f32 = fv * bd;
                    if (fv as f64) > 1e-7 && (wh as f64) > 1e-9 {
                        let pht = solve_paint_height(hg as f64, bb_h, hd as f64);
                        let nwh = pymax(pht - hd as f64, 0.0);
                        let whd = wh as f64;
                        let wetv = wet_at(cv, yu, xu);
                        let mut fr = (whd - nwh) / whd * scrape_eff * m * cl01(1.6 * wetv as f64);
                        if fr > 0.98 {
                            fr = 0.98;
                        }
                        if fr > 0.0 {
                            let t = take(
                                &mut cv.px(yu, xu),
                                fv as f64 * fr,
                                row_mut(res_k, ib),
                                row_mut(res_s, ib),
                            );
                            res_vol[ib] += t;
                        }
                    }
                    continue;
                }
                if mode != 0 {
                    continue;
                }
                // -- spread: knife paint bridges the gap where its lumpy load reaches
                let lf = 1.6 - pymin(load, 1.0);
                let mut tl = tk * (1.0 + lumpiness * lf * a.lump_tex.at(iu, isj as usize));
                if tl < 0.0 {
                    tl = 0.0;
                }
                let wetv = wet_at(cv, yu, xu) as f64;
                let wetc = if (fv as f64) > 0.05 { wetv } else { 0.0 };
                let r = f + tl * (reach_base + reach_press * p * p) * (1.0 + 0.4 * wetc);
                // viscous paint bridges the small valleys of the weave: contact
                // is decided against the local maximum of the solid relief
                let mut topc = top;
                let iy2 = yy - a.soy;
                let ix2 = xx - a.sox;
                if 0 <= iy2 && iy2 < a.smax.height() as i64 && 0 <= ix2 && ix2 < a.smax.width() as i64 {
                    let sv = a.smax.at(iy2 as usize, ix2 as usize) as f64;
                    if sv > topc {
                        topc = sv;
                    }
                }
                let c = cl01((r - (bb_h - topc)) / soft + 0.5);
                if c <= 0.0 {
                    continue;
                }
                // pick-up of underlying wet paint (smearing / dirty knife)
                if pickup > 0.0 && (fv as f64) > 1e-6 {
                    let pf = pymin(pickup * c * m * wetv, 0.5);
                    let t = take(&mut cv.px(yu, xu), fv as f64 * pf, row_mut(res_k, ib), row_mut(res_s, ib));
                    res_vol[ib] += t;
                    fv = vol_body(cv, yu, xu).0;
                    top = surface_top(hg, hd, fv, bd);
                    gap = bb_h - top;
                    if gap <= 0.0 {
                        continue;
                    }
                }
                let vb = res_vol[ib];
                if vb <= 1e-9 {
                    continue;
                }
                let dv = pymin(gap / pymax(p_body, 0.05) * transfer * c * m, 0.35 * vb);
                if dv <= 1e-9 {
                    continue;
                }
                let fd = dv / vb;
                for b in 0..NB {
                    kk[b] = res_k[ib * NB + b] / vb;
                    ks[b] = res_s[ib * NB + b] / vb;
                    res_k[ib * NB + b] -= res_k[ib * NB + b] * fd;
                    res_s[ib * NB + b] -= res_s[ib * NB + b] * fd;
                }
                res_vol[ib] = vb - dv;
                dep(&mut cv.px(yu, xu), sid, dv, &kk, &ks, p_gloss, p_body, p_drate);
            }
        }
        // ---- pressing spreads the load along the blade a little
        if nbin > 1 && diffuse > 0.0 {
            let dq = pymin(diffuse * step * (0.3 + p), 0.24);
            for ib in 0..(nbin - 1) as usize {
                for b in 0..NB {
                    let dk = (res_k[ib * NB + b] - res_k[(ib + 1) * NB + b]) * dq;
                    let ds = (res_s[ib * NB + b] - res_s[(ib + 1) * NB + b]) * dq;
                    res_k[ib * NB + b] -= dk;
                    res_k[(ib + 1) * NB + b] += dk;
                    res_s[ib * NB + b] -= ds;
                    res_s[(ib + 1) * NB + b] += ds;
                }
                let dv = (res_vol[ib] - res_vol[ib + 1]) * dq;
                res_vol[ib] -= dv;
                res_vol[ib + 1] += dv;
            }
        }
        // ---- scraping: the pressed bead squeezes outwards along the blade
        if mode == 1 && nbin > 2 {
            let qo = pymin(0.06 * squeeze * step * p, 0.3);
            let hb = nbin / 2;
            for ib in 1..hb {
                bins_shift(res_k, res_s, res_vol, ib as usize, (ib - 1) as usize, qo);
            }
            let mut ib = nbin - 2;
            while ib > nbin - 1 - hb {
                bins_shift(res_k, res_s, res_vol, ib as usize, (ib + 1) as usize, qo);
                ib -= 1;
            }
        }
        // ---- paint migrates along an angled blade towards its trailing end
        if plow > 0.0 && nbin > 1 {
            let q = pymin(plow * bm.abs() * 0.16 * step / (l / nbin as f64), 0.45);
            if bm > 0.0 {
                // end at u<0 trails: move towards lower bins
                for ib in 1..nbin {
                    bins_shift(res_k, res_s, res_vol, ib as usize, (ib - 1) as usize, q);
                }
            } else {
                let mut ib = nbin - 2;
                while ib >= 0 {
                    bins_shift(res_k, res_s, res_vol, ib as usize, (ib + 1) as usize, q);
                    ib -= 1;
                }
            }
        }
        // ---- squeeze-out ridges at both blade ends
        for e in [-1i64, 1i64] {
            let ib = if e < 0 { 0 } else { (nbin - 1) as usize };
            let vb = res_vol[ib];
            if vb <= 1e-9 {
                continue;
            }
            let trail = pymax(-e as f64 * bm, 0.0);
            let ld = vb / (cap[ib] + 1e-12);
            let mut dvt;
            if mode == 0 {
                let tke = pymin(thickness * pymin(ld, 1.5).powf(0.7), 1.5 * tkm);
                dvt = squeeze * (0.3 + p) * step * tke * 0.5 * (1.0 + 3.0 * trail);
                if ld > spill_cap {
                    dvt += 0.02 * (ld - spill_cap) * cap[ib] * step;
                }
            } else {
                // the scraped bead piles up at the blade ends
                dvt = vb * pymin(0.5, 0.05 * squeeze * step * (0.3 + p) * (1.0 + 2.0 * trail));
            }
            dvt = pymin(dvt, 0.5 * vb);
            let ex = cx + bxd * e as f64 * (half + 0.6 + ridge_w) - nx * 0.4;
            let ey = cy + byd * e as f64 * (half + 0.6 + ridge_w) - ny * 0.4;
            gauss_dep_bin(cv, sid, ex, ey, ridge_w, dvt, res_k, res_s, res_vol, ib, p_gloss, p_body, p_drate, &mut bb);
        }
        pnx = nx;
        pny = ny;
        pcx = cx;
        pcy = cy;
    }
    // ---- lift-off: the paint bead under the blade stays behind as a ridge
    if lift_frac > 0.0 && m_n > 1 {
        let cx = px[m_n - 1];
        let cy = py[m_n - 1];
        let ang = pa[m_n - 1];
        let bxd = ang.cos();
        let byd = ang.sin();
        let nsm = ((l * 2.0) as i64).max(4);
        let rw = ridge_w * 1.2;
        for k in 0..nsm {
            let u = -half + l * (k as f64 + 0.5) / nsm as f64;
            let ib = ((((u + half) / l * nbin as f64) as i64).max(0)).min(nbin - 1) as usize;
            let vb = res_vol[ib];
            if vb <= 1e-9 {
                continue;
            }
            let iu = (((u + half) as i64).max(0)).min(nu - 1) as usize;
            let nzv = pymax(1.0 + lift_noise * a.lump_tex.at(iu, (ns - 1) as usize), 0.1);
            let ldl = pymin(vb / (cap[ib] + 1e-12), 3.0);
            let dvt = pymin(lift_frac * thickness * ldl.powf(0.7) * 5.0 * (l / nsm as f64) * nzv, 0.6 * vb);
            gauss_dep_bin(
                cv,
                sid,
                cx + bxd * u + pnx * 0.9,
                cy + byd * u + pny * 0.9,
                rw,
                dvt,
                res_k,
                res_s,
                res_vol,
                ib,
                p_gloss,
                p_body,
                p_drate,
                &mut bb,
            );
        }
    }
    (bb[0], bb[1], bb[2], bb[3])
}

// ==========================================================================
// Sgraffito
// ==========================================================================

/// Scalar parameters of `_sgraffito_kernel`.
pub struct SgraffitoParams {
    pub r0: f64,
    pub r1: f64,
    pub valley: f64,
    pub burr: f64,
    pub carry: f64,
    pub ragged: f64,
    /// Origin of the bbox-local `noise` map on the canvas.
    pub nox: i64,
    pub noy: i64,
    pub sid: i64,
}

/// Scratch a line through the wet film (both wet sub-layers) with a tip
/// (`_sgraffito_kernel`). Removed paint is pushed to both sides of the groove
/// (burr ridges); a little is carried along and dropped where the tip lifts.
/// Returns the touched bbox.
#[allow(clippy::too_many_arguments)]
pub fn sgraffito_kernel<C: WetCanvas>(
    px: &[f64],
    py: &[f64],
    pp: &[f64],
    prm: &SgraffitoParams,
    noise: &Plane<'_, f64>,
    h_ground: &Plane<'_, f32>,
    h_dry: &Plane<'_, f32>,
    cv: &mut C,
) -> (i64, i64, i64, i64) {
    let (r0, r1, valley, burr, carry, ragged) = (prm.r0, prm.r1, prm.valley, prm.burr, prm.carry, prm.ragged);
    let (nox, noy, sid) = (prm.nox, prm.noy, prm.sid);
    let h = cv.height() as i64;
    let w = cv.width() as i64;
    let m_n = px.len();
    let mut bb: [i64; 4] = [w, h, -1, -1];
    let mut cr_k = [0.0f64; NB]; // carried crumb on the tip
    let mut cr_s = [0.0f64; NB];
    let mut kk = [0.0f64; NB];
    let mut ks = [0.0f64; NB];
    let mut cv_ = 0.0f64;
    let mut cg = 0.0f64;
    let mut cb = 0.0f64;
    let mut cd = 0.0f64;
    for j in 0..m_n {
        let cx = px[j];
        let cy = py[j];
        let p = pp[j];
        if p <= 0.0 {
            continue;
        }
        let r = r0 + (r1 - r0) * p;
        let j0 = if j > 0 { j - 1 } else { 0 };
        let j1 = if j < m_n - 1 { j + 1 } else { m_n - 1 };
        let mut mx = px[j1] - px[j0];
        let mut my = py[j1] - py[j0];
        let ml = (mx * mx + my * my).sqrt();
        if ml > 1e-9 {
            mx /= ml;
            my /= ml;
        } else {
            mx = 1.0;
            my = 0.0;
        }
        let nx = -my;
        let ny = mx;
        let x0 = ((cx - r - 1.0).floor() as i64).max(0);
        let x1 = ((cx + r + 1.0).ceil() as i64).min(w - 1);
        let y0 = ((cy - r - 1.0).floor() as i64).max(0);
        let y1 = ((cy + r + 1.0).ceil() as i64).min(h - 1);
        if x1 < x0 || y1 < y0 {
            continue;
        }
        // contact level of the tip: highest solid point under it
        let mut ctop = -1e9f64;
        for yy in y0..=y1 {
            for xx in x0..=x1 {
                let dx = xx as f64 - cx;
                let dy = yy as f64 - cy;
                if dx * dx + dy * dy <= r * r {
                    let s = solid_top(h_ground.at(yy as usize, xx as usize), h_dry.at(yy as usize, xx as usize));
                    if s > ctop {
                        ctop = s;
                    }
                }
            }
        }
        for yy in y0..=y1 {
            for xx in x0..=x1 {
                let dx = xx as f64 - cx;
                let dy = yy as f64 - cy;
                let d = (dx * dx + dy * dy).sqrt();
                if d > r + 0.7 {
                    continue;
                }
                let (yu, xu) = (yy as usize, xx as usize);
                let m = 1.0 - cv.mask(yu, xu) as f64;
                let (fv, _) = vol_body(cv, yu, xu);
                if m <= 0.0 || (fv as f64) <= 1e-7 {
                    continue;
                }
                let cov = cl01(r - d + 0.5);
                let deep = ctop - solid_top(h_ground.at(yu, xu), h_dry.at(yu, xu));
                let keep = cl01(deep / (valley + 1e-6) - 0.3) * 0.85;
                let (wv, pg, pb, pd) = {
                    let q = cv.px(yu, xu);
                    (*q.wet, *q.gloss, *q.body, *q.drate)
                };
                let mut nzv = 0.0f64;
                let iy = yy - noy;
                let ix = xx - nox;
                if 0 <= iy && iy < noise.height() as i64 && 0 <= ix && ix < noise.width() as i64 {
                    nzv = noise.at(iy as usize, ix as usize);
                }
                // drier paint tears raggedly instead of parting cleanly
                let tear = cl01((1.0 - wv as f64) * ragged * 2.0);
                let thr = cl01(1.0 - tear * (0.5 + 0.5 * nzv));
                let fr = pymin(cov * m * (1.0 - keep) * thr * (0.4 + 0.6 * p), 0.995);
                if fr <= 0.0 {
                    continue;
                }
                let (pg, pb, pd) = (pg as f64, pb as f64, pd as f64);
                let mut rem = [0.0f64; NB]; // scratch row for this pixel's removal
                let mut rem_s = [0.0f64; NB];
                let dv = take(&mut cv.px(yu, xu), fv as f64 * fr, &mut rem, &mut rem_s);
                if dv <= 0.0 {
                    continue;
                }
                // burr on the side of the groove, the rest is carried
                let lat = dx * nx + dy * ny;
                let side = if lat >= 0.0 { 1.0 } else { -1.0 };
                let x2 = py_round(cx + nx * side * (r + 1.0));
                let y2 = py_round(cy + ny * side * (r + 1.0));
                let mut fb = burr;
                if x2 < 0 || y2 < 0 || x2 >= w || y2 >= h || (cv.mask(y2 as usize, x2 as usize) as f64) >= 1.0 {
                    fb = 0.0;
                }
                if fb > 0.0 {
                    for b in 0..NB {
                        kk[b] = rem[b] / dv;
                        ks[b] = rem_s[b] / dv;
                    }
                    let (y2u, x2u) = (y2 as usize, x2 as usize);
                    dep(&mut cv.px(y2u, x2u), sid, dv * fb, &kk, &ks, pg, pb, pd);
                    {
                        let q = cv.px(y2u, x2u);
                        if *q.wet < wv {
                            *q.wet = wv;
                        }
                    }
                    grow(&mut bb, x2, y2);
                }
                let cvn = dv * (1.0 - fb);
                if cvn > 0.0 {
                    for b in 0..NB {
                        cr_k[b] += rem[b] * (1.0 - fb);
                        cr_s[b] += rem_s[b] * (1.0 - fb);
                    }
                    let tw = cv_ + cvn;
                    cg = (cg * cv_ + pg * cvn) / tw;
                    cb = (cb * cv_ + pb * cvn) / tw;
                    cd = (cd * cv_ + pd * cvn) / tw;
                    cv_ = tw;
                }
                grow(&mut bb, xx, yy);
            }
        }
        // the carried crumb leaks back a little behind the tip, at both sides
        if cv_ > 1e-7 && carry < 1.0 {
            for e in [-1.0f64, 1.0f64] {
                let xx = py_round(cx - mx * (r + 1.0) + e * nx * (r + 0.8));
                let yy = py_round(cy - my * (r + 1.0) + e * ny * (r + 0.8));
                if xx < 0 || yy < 0 || xx >= w || yy >= h || (cv.mask(yy as usize, xx as usize) as f64) >= 1.0 {
                    continue;
                }
                let dv = cv_ * (1.0 - carry) * 0.5;
                for b in 0..NB {
                    kk[b] = cr_k[b] / cv_;
                    ks[b] = cr_s[b] / cv_;
                }
                dep(&mut cv.px(yy as usize, xx as usize), sid, dv, &kk, &ks, cg, cb, cd);
                for b in 0..NB {
                    cr_k[b] -= kk[b] * dv;
                    cr_s[b] -= ks[b] * dv;
                }
                cv_ -= dv;
            }
        }
    }
    // the crumb is dropped where the tip lifts
    if cv_ > 1e-7 && m_n > 0 {
        let cx = px[m_n - 1];
        let cy = py[m_n - 1];
        let rr = 2i64;
        let mut wsum = 0.0f64;
        let (icx, icy) = (cx as i64, cy as i64);
        for yy in (icy - rr)..(icy + rr + 2) {
            for xx in (icx - rr)..(icx + rr + 2) {
                let dx = xx as f64 - cx;
                let dy = yy as f64 - cy;
                wsum += (-(dx * dx + dy * dy) / 2.0).exp();
            }
        }
        for b in 0..NB {
            kk[b] = cr_k[b] / cv_;
            ks[b] = cr_s[b] / cv_;
        }
        for yy in (icy - rr)..(icy + rr + 2) {
            for xx in (icx - rr)..(icx + rr + 2) {
                if xx < 0 || yy < 0 || xx >= w || yy >= h {
                    continue;
                }
                let dx = xx as f64 - cx;
                let dy = yy as f64 - cy;
                let (yu, xu) = (yy as usize, xx as usize);
                let dv = cv_ * (-(dx * dx + dy * dy) / 2.0).exp() / wsum * (1.0 - cv.mask(yu, xu) as f64);
                dep(&mut cv.px(yu, xu), sid, dv, &kk, &ks, cg, cb, cd);
            }
        }
    }
    (bb[0], bb[1], bb[2], bb[3])
}

// ==========================================================================
// Smudge / blender
// ==========================================================================

/// Finger smudge along the path (`_smudge_kernel`). The finger's lanes carry
/// paint: `l_k`, `l_s` (nl x NB), `l_v` (nl) and `l_p` (nl x 4: wet, gloss,
/// body, drate, each weighted by volume) are updated. `r` is the finger radius
/// in px, `nl = 2 * r + 1`. Volume is conserved; only wet paint moves.
#[allow(clippy::too_many_arguments)]
pub fn smudge_kernel<C: WetCanvas>(
    px: &[f64],
    py: &[f64],
    pp: &[f64],
    r: i64,
    rate: f64,
    spread: f64,
    cv: &mut C,
    sid: i64,
    l_k: &mut [f64],
    l_s: &mut [f64],
    l_v: &mut [f64],
    l_p: &mut [f64],
) -> (i64, i64, i64, i64) {
    let h = cv.height() as i64;
    let w = cv.width() as i64;
    let nl = (2 * r + 1) as usize;
    let mut prof = vec![0.0f64; nl];
    for i in 0..nl {
        let t = (i as i64 - r) as f64 / (r as f64 + 0.5);
        let d2 = t * t;
        prof[i] = if d2 < 1.0 { (1.0 - d2).powf(0.6) } else { 0.0 };
    }
    let mut t_k = [0.0f64; NB];
    let mut t_s = [0.0f64; NB];
    let mut kk = [0.0f64; NB];
    let mut ks = [0.0f64; NB];
    let mut bb: [i64; 4] = [w, h, -1, -1];
    let m_n = px.len();
    let mut last_x = vec![-1i64; nl];
    let mut last_y = vec![-1i64; nl];
    for j in 0..m_n {
        let p = pp[j];
        if p <= 0.0 {
            continue;
        }
        let j0 = if j > 0 { j - 1 } else { 0 };
        let j1 = if j < m_n - 1 { j + 1 } else { m_n - 1 };
        let mut mx = px[j1] - px[j0];
        let mut my = py[j1] - py[j0];
        let ml = (mx * mx + my * my).sqrt();
        if ml < 1e-9 {
            continue;
        }
        mx /= ml;
        my /= ml;
        let nx = -my;
        let ny = mx;
        for i in 0..nl {
            let wgt = prof[i] * p;
            if wgt <= 0.0 {
                continue;
            }
            let lane = (i as i64 - r) as f64;
            let xx = py_round(px[j] + nx * lane);
            let yy = py_round(py[j] + ny * lane);
            if xx < 0 || yy < 0 || xx >= w || yy >= h {
                continue;
            }
            if xx == last_x[i] && yy == last_y[i] {
                continue; // the lane has not moved onto a new pixel yet
            }
            last_x[i] = xx;
            last_y[i] = yy;
            let (yu, xu) = (yy as usize, xx as usize);
            let m = 1.0 - cv.mask(yu, xu) as f64;
            if m <= 0.0 {
                continue;
            }
            let f = pymin(rate * wgt * m, 0.6);
            // lay down part of the carried paint
            let v = l_v[i];
            let out = v * f;
            // take part of the paint under the lane
            let (fv, _) = vol_body(cv, yu, xu);
            let (pw, pg, pb, pd) = {
                let q = cv.px(yu, xu);
                (*q.wet, *q.gloss, *q.body, *q.drate)
            };
            let mob = cl01(pw as f64 * 1.4);
            let mut dvt = 0.0f64;
            if (fv as f64) > 1e-7 && mob > 0.0 {
                t_k = [0.0; NB];
                t_s = [0.0; NB];
                dvt = take(&mut cv.px(yu, xu), fv as f64 * f * mob, &mut t_k, &mut t_s);
            }
            if out > 1e-9 {
                for c in 0..NB {
                    kk[c] = l_k[i * NB + c] / v;
                    ks[c] = l_s[i * NB + c] / v;
                }
                dep_w(
                    &mut cv.px(yu, xu),
                    sid,
                    out,
                    &kk,
                    &ks,
                    l_p[i * 4] / v,
                    l_p[i * 4 + 1] / v,
                    l_p[i * 4 + 2] / v,
                    l_p[i * 4 + 3] / v,
                );
                for c in 0..NB {
                    l_k[i * NB + c] -= kk[c] * out;
                    l_s[i * NB + c] -= ks[c] * out;
                }
                for c in 0..4 {
                    l_p[i * 4 + c] *= 1.0 - f;
                }
                l_v[i] = v - out;
            }
            if dvt > 0.0 {
                for c in 0..NB {
                    l_k[i * NB + c] += t_k[c];
                    l_s[i * NB + c] += t_s[c];
                }
                l_v[i] += dvt;
                l_p[i * 4] += pw as f64 * dvt;
                l_p[i * 4 + 1] += pg as f64 * dvt;
                l_p[i * 4 + 2] += pb as f64 * dvt;
                l_p[i * 4 + 3] += pd as f64 * dvt;
            }
            grow(&mut bb, xx, yy);
        }
        // neighbouring lanes share a little paint (soft, smeared edges)
        if spread > 0.0 {
            for i in 0..nl - 1 {
                let q = spread;
                let dv = (l_v[i] - l_v[i + 1]) * q;
                for c in 0..NB {
                    let dk = (l_k[i * NB + c] - l_k[(i + 1) * NB + c]) * q;
                    let ds = (l_s[i * NB + c] - l_s[(i + 1) * NB + c]) * q;
                    l_k[i * NB + c] -= dk;
                    l_k[(i + 1) * NB + c] += dk;
                    l_s[i * NB + c] -= ds;
                    l_s[(i + 1) * NB + c] += ds;
                }
                for c in 0..4 {
                    let dq = (l_p[i * 4 + c] - l_p[(i + 1) * 4 + c]) * q;
                    l_p[i * 4 + c] -= dq;
                    l_p[(i + 1) * 4 + c] += dq;
                }
                l_v[i] -= dv;
                l_v[i + 1] += dv;
            }
        }
    }
    (bb[0], bb[1], bb[2], bb[3])
}

// ==========================================================================
// Oil crayon
// ==========================================================================

/// One oil-crayon stroke (`_crayon_stroke`): `xy` (M x 2, row-major) positions,
/// `pressures`, `angles` per sample, `size` tip diameter, `side` 0 (round tip)
/// to 1 (broad side), pigment optical depth per volume `k`, `s` (NB each).
/// `wax` records the coverage. Returns `(left, top, right, bottom)`, or
/// `right < left` if nothing was touched.
#[allow(clippy::too_many_arguments)]
pub fn crayon_stroke<C: WetCanvas>(
    cv: &mut C,
    sid: i64,
    relief: f64,
    wax: &mut PlaneMut<'_, f32>,
    xy: &[f64],
    pressures: &[f64],
    angles: &[f64],
    size: f64,
    side: f64,
    k: &[f64],
    s: &[f64],
) -> (i64, i64, i64, i64) {
    let height = cv.height() as i64;
    let width = cv.width() as i64;
    let (mut left, mut top, mut right, mut bottom) = (width, height, -1i64, -1i64);
    for i in 0..xy.len() / 2 {
        let p = pressures[i];
        if p <= 0.0 {
            continue;
        }
        let cx = xy[2 * i];
        let cy = xy[2 * i + 1];
        let radius = 0.5 * size * (0.48 + 0.52 * p.sqrt());
        let major = radius * (1.0 + 1.8 * side);
        let minor = radius * (1.0 - 0.45 * side);
        let c = angles[i].cos();
        let sn = angles[i].sin();
        let dx = (major * c).hypot(minor * sn) + 1.0;
        let dy = (major * sn).hypot(minor * c) + 1.0;
        let ylo = 0i64.max((cy - dy).floor() as i64);
        let yhi = height.min((cy + dy).ceil() as i64 + 1);
        let xlo = 0i64.max((cx - dx).floor() as i64);
        let xhi = width.min((cx + dx).ceil() as i64 + 1);
        for y in ylo..yhi {
            for x in xlo..xhi {
                let (yu, xu) = (y as usize, x as usize);
                let mask = cv.mask(yu, xu) as f64;
                if mask >= 1.0 {
                    continue;
                }
                let u = ((x as f64 - cx) * c + (y as f64 - cy) * sn) / major;
                let v = ((y as f64 - cy) * c - (x as f64 - cx) * sn) / minor;
                let edge = cl01(0.5 + (1.0 - (u * u + v * v).sqrt()) * minor);
                if edge == 0.0 {
                    continue;
                }
                // The paper pattern stays put; the crumbling stick leaves a
                // slightly different scatter of pigment each time it passes.
                let hash = x.wrapping_mul(73856093) ^ y.wrapping_mul(19349663) ^ sid.wrapping_mul(83492791);
                let grain = hash.rem_euclid(1021) as f64 / 1021.0;
                let (this_mark, under, fvt) = {
                    let q = cv.px(yu, xu);
                    let this_mark = *q.top_id as i64 == sid;
                    let under = *q.base_vol as f64 + (if this_mark { 0.0 } else { *q.film_vol as f64 });
                    (this_mark, under, *q.film_vol as f64)
                };
                let bite = relief * cv.tooth(yu, xu) as f64 / (1.0 + under);
                let contact = sstep(1.65 * p + bite + 0.65 * (grain - 0.5) - 0.12);
                let target = edge * contact * p * (0.9 + p) * (1.0 - mask);
                // A single sweep coats each pixel once even as adjacent
                // footprints overlap. A fresh stroke can build a new layer.
                let dv = target - (if this_mark { fvt } else { 0.0 });
                if dv > 1e-6 {
                    dep_w(&mut cv.px(yu, xu), sid, dv, k, s, 0.12, 0.26, 0.16, 0.0);
                    // Count only newly transferred wax. Overlapping dabs in
                    // this pass fill the same footprint; later passes build it up.
                    let nw = wax.at(yu, xu) as f64 + dv;
                    wax.set(yu, xu, (if nw < 1.0 { nw } else { 1.0 }) as f32);
                    left = left.min(x);
                    top = top.min(y);
                    right = right.max(x);
                    bottom = bottom.max(y);
                }
            }
        }
    }
    (left, top, right, bottom)
}
