//! Paste media (oil, acrylic, gouache): the bristle brush stroke.
//!
//! A line-by-line port of `pinsel._kernels.paste_stroke` (Python 0.1). Every
//! bristle carries its own reservoir (`res_*`) and a loosely held tip
//! reservoir (`tip_*`) for paint picked up from the canvas. Along the path,
//! each bristle deposits through a soft round footprint, picks wet paint up,
//! pushes older wet paint into the base layer when a new stroke arrives, and
//! stiff bristles plough paint aside into ridges.
//!
//! Number types mirror numba: canvas fields are f32; arithmetic that mixes
//! them with f64 scalars happens in f64 and rounds on store; f32 + f32 stays
//! f32 (the base-layer push and the volume sum `fv`).

use crate::grid::{WetCanvas, WetPx};
use crate::NB;

/// Per-bristle brush state. Arrays have N (or N*2, N*NB) elements, row-major.
pub struct Bristles<'a> {
    /// Root offsets in brush units, N*2 (x, y).
    pub boff: &'a [f64],
    /// Pressure at which a bristle starts touching.
    pub bthr: &'a [f64],
    /// How quickly a bristle tip follows its root (per resample step).
    pub bfollow: &'a [f64],
    /// Contact radius.
    pub brad: &'a [f64],
    /// Splay factor under pressure.
    pub bsplay: &'a [f64],
    /// Deposit multiplier.
    pub bdep: &'a [f64],
    /// Current tip positions, N*2; NaN = not placed yet.
    pub tips: &'a mut [f64],
    /// Reservoir optical depths, N*NB, and volume, N.
    pub res_k: &'a mut [f64],
    pub res_s: &'a mut [f64],
    pub res_vol: &'a mut [f64],
    /// Reservoir capacity.
    pub cap: &'a [f64],
    /// Picked-up paint held loosely at the tip, N*NB and N.
    pub tip_k: &'a mut [f64],
    pub tip_s: &'a mut [f64],
    pub tip_vol: &'a mut [f64],
}

/// Scalar parameters of one stroke (names as in the Python kernel).
#[derive(Clone, Copy, Debug)]
pub struct PasteParams {
    pub ds: f64,
    pub speed: f64,
    pub alpha_tip: f64,
    pub size: f64,
    pub sx0: f64,
    pub sx1: f64,
    pub sy0: f64,
    pub sy1: f64,
    pub sgamma: f64,
    pub splay: f64,
    pub alpha: f64,
    pub pickup: f64,
    pub plough: f64,
    pub contact_soft: f64,
    pub p_gloss: f64,
    pub p_body: f64,
    pub p_drate: f64,
    pub tooth_fill: f64,
    pub sid: i32,
}

#[inline(always)]
fn clamp01(v: f64) -> f64 {
    if v < 0.0 {
        0.0
    } else if v > 1.0 {
        1.0
    } else {
        v
    }
}

/// Python 3 `round()` (half to even), then `int`.
#[inline(always)]
fn py_round(v: f64) -> i64 {
    v.round_ties_even() as i64
}

/// Drag the brush along the resampled path (`px`, `py` positions, `pp`
/// pressure 0..1, `pa` angle in radians). Returns the touched bounding box
/// `(x0, y0, x1, y1)`, or `(W, H, -1, -1)` if nothing was touched.
pub fn paste_stroke<C: WetCanvas>(
    px: &[f64],
    py: &[f64],
    pp: &[f64],
    pa: &[f64],
    b: &mut Bristles<'_>,
    prm: &PasteParams,
    cv: &mut C,
) -> (i64, i64, i64, i64) {
    let h = cv.height() as i64;
    let w = cv.width() as i64;
    let n = b.bthr.len();
    let m = px.len();
    assert!(py.len() == m && pp.len() == m && pa.len() == m, "path arrays differ in length");
    assert!(b.boff.len() == 2 * n && b.tips.len() == 2 * n, "bristle arrays differ in length");
    assert!(b.res_k.len() == n * NB && b.tip_k.len() == n * NB, "reservoir arrays differ in length");
    let (mut bx0, mut by0, mut bx1, mut by1) = (w, h, -1i64, -1i64);
    let spd_dep = 1.0 / prm.speed.max(0.05).sqrt();
    let spd_depth = 1.0 / prm.speed.max(0.05).powf(0.3);

    for j in 0..m {
        let cx = px[j];
        let cy = py[j];
        let p = pp[j];
        let a = pa[j];
        let ca = a.cos();
        let sa = a.sin();
        // direction of motion
        let j0 = if j > 0 { j - 1 } else { 0 };
        let j1 = if j + 1 < m { j + 1 } else { m - 1 };
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
        let pg = if p > 0.0 { p.powf(prm.sgamma) } else { 0.0 };
        let sx = prm.size * (prm.sx0 + (prm.sx1 - prm.sx0) * pg);
        let sy = prm.size * (prm.sy0 + (prm.sy1 - prm.sy0) * pg);
        // footprint width across the motion direction (pickup normalisation)
        let across = 2.0
            * ((sx * (ca * nx + sa * ny)).powi(2) + (sy * (-sa * nx + ca * ny)).powi(2)).sqrt()
            + 1.0;
        let pick_base = 2.0 * prm.pickup * across / n as f64 * prm.ds;
        let dep_base = prm.alpha * prm.ds * spd_dep;
        let dep_tip = prm.alpha_tip * prm.ds;
        let mig = 0.25 * prm.alpha_tip * prm.ds;

        for i in 0..n {
            let kb = i * NB;
            // loosely held (picked-up) paint slowly works into the belly
            if b.tip_vol[i] > 1e-9 {
                for k in 0..NB {
                    let mk = b.tip_k[kb + k] * mig;
                    let ms = b.tip_s[kb + k] * mig;
                    b.tip_k[kb + k] -= mk;
                    b.tip_s[kb + k] -= ms;
                    b.res_k[kb + k] += mk;
                    b.res_s[kb + k] += ms;
                }
                let mv = b.tip_vol[i] * mig;
                b.tip_vol[i] -= mv;
                b.res_vol[i] += mv;
            }
            let spl = 1.0 + prm.splay * b.bsplay[i] * p * p;
            let lx = b.boff[2 * i] * sx * spl;
            let ly = b.boff[2 * i + 1] * sy * spl;
            let gx = cx + lx * ca - ly * sa;
            let gy = cy + lx * sa + ly * ca;
            if j == 0 || b.tips[2 * i].is_nan() {
                b.tips[2 * i] = gx;
                b.tips[2 * i + 1] = gy;
            } else {
                let f = b.bfollow[i];
                b.tips[2 * i] += (gx - b.tips[2 * i]) * f;
                b.tips[2 * i + 1] += (gy - b.tips[2 * i + 1]) * f;
            }
            let thr = b.bthr[i];
            if p <= thr {
                continue;
            }
            let cp = clamp01((p - thr) / (1.0 - thr + 1e-6));
            let tx = b.tips[2 * i];
            let ty = b.tips[2 * i + 1];
            let r = b.brad[i] * (0.55 + 0.45 * cp);
            let r2 = r * r;
            let mut x0 = (tx - r).floor() as i64;
            let mut x1 = (tx + r).ceil() as i64;
            let mut y0 = (ty - r).floor() as i64;
            let mut y1 = (ty + r).ceil() as i64;
            if x1 < 0 || y1 < 0 || x0 >= w || y0 >= h {
                continue;
            }
            x0 = x0.max(0);
            y0 = y0.max(0);
            x1 = x1.min(w - 1);
            y1 = y1.min(h - 1);
            let mut wsum = 0.0f64;
            for yy in y0..=y1 {
                let dy = yy as f64 - ty;
                for xx in x0..=x1 {
                    let dx = xx as f64 - tx;
                    let d2 = dx * dx + dy * dy;
                    if d2 < r2 {
                        let u = 1.0 - d2 / r2;
                        wsum += u * u;
                    }
                }
            }
            if wsum <= 1e-9 {
                continue;
            }
            let inv = 1.0 / wsum;
            let mut v = b.res_vol[i];
            let load = clamp01(v / (b.cap[i] + 1e-12));
            let depth = 2.6 * (0.42 * cp + 0.58 * load.sqrt()) * spd_depth;
            let level = 1.0 - depth;
            let press = 0.3 + 0.7 * cp;
            bx0 = bx0.min(x0);
            by0 = by0.min(y0);
            bx1 = bx1.max(x1);
            by1 = by1.max(y1);

            for yy in y0..=y1 {
                let dy = yy as f64 - ty;
                for xx in x0..=x1 {
                    let dx = xx as f64 - tx;
                    let d2 = dx * dx + dy * dy;
                    if d2 >= r2 {
                        continue;
                    }
                    let u = 1.0 - d2 / r2;
                    let wgt = u * u * inv;
                    let (yu, xu) = (yy as usize, xx as usize);
                    let msk = 1.0 - cv.mask(yu, xu) as f64;
                    if msk <= 0.0 {
                        continue;
                    }
                    let tooth = cv.tooth(yu, xu);
                    let q = cv.px(yu, xu);
                    // a new stroke lies on top of older wet paint: push the
                    // previous stroke's paint down into the base layer
                    if *q.top_id != prm.sid {
                        if *q.film_vol > 0.0 || q.film_k[0] > 0.0 {
                            for k in 0..NB {
                                q.base_k[k] += q.film_k[k];
                                q.base_s[k] += q.film_s[k];
                                q.film_k[k] = 0.0;
                                q.film_s[k] = 0.0;
                            }
                            *q.base_vol += *q.film_vol;
                            *q.film_vol = 0.0;
                        }
                        *q.top_id = prm.sid;
                    }
                    let fvt = *q.film_vol;
                    let fvb = *q.base_vol;
                    let fv = fvt + fvb; // f32 + f32, as in numba
                    let t = tooth as f64 * (-(fv as f64) * prm.tooth_fill).exp();
                    let contact = clamp01((t - level) / prm.contact_soft + 0.5);
                    let wc = wgt * contact * msk * press;
                    if wc <= 0.0 {
                        continue;
                    }
                    let tv = b.tip_vol[i];
                    let mut dvb = v * dep_base * wc * b.bdep[i];
                    if dvb > 0.5 * v {
                        dvb = 0.5 * v;
                    }
                    let mut dvt = tv * dep_tip * wc;
                    if dvt > 0.5 * tv {
                        dvt = 0.5 * tv;
                    }
                    let dv = dvb + dvt;
                    let mut pf = 0.0f64;
                    if fv > 1e-7 && prm.pickup > 0.0 {
                        let room = 1.0 - (v + tv) / (1.6 * b.cap[i] + 1e-12);
                        if room > 0.0 {
                            pf = pick_base * wc * *q.wet as f64 * room;
                            if pf > 0.6 {
                                pf = 0.6;
                            }
                        }
                    }
                    let pfb = pf * (-(fvt as f64) * 2.5).exp();
                    let pv = fvt as f64 * pf + fvb as f64 * pfb;
                    if dv <= 0.0 && pv <= 0.0 {
                        continue;
                    }
                    let fdep = if v > 1e-12 { dvb / v } else { 0.0 };
                    let ftip = if tv > 1e-12 { dvt / tv } else { 0.0 };
                    for k in 0..NB {
                        let dk = b.res_k[kb + k] * fdep;
                        let dsg = b.res_s[kb + k] * fdep;
                        let tk = b.tip_k[kb + k] * ftip;
                        let tsg = b.tip_s[kb + k] * ftip;
                        let qk = q.film_k[k] as f64 * pf;
                        let qs = q.film_s[k] as f64 * pf;
                        let bk = q.base_k[k] as f64 * pfb;
                        let bs = q.base_s[k] as f64 * pfb;
                        q.film_k[k] = (q.film_k[k] as f64 + (dk + tk - qk)) as f32;
                        q.film_s[k] = (q.film_s[k] as f64 + (dsg + tsg - qs)) as f32;
                        q.base_k[k] = (q.base_k[k] as f64 - bk) as f32;
                        q.base_s[k] = (q.base_s[k] as f64 - bs) as f32;
                        b.res_k[kb + k] -= dk;
                        b.res_s[kb + k] -= dsg;
                        b.tip_k[kb + k] += qk + bk - tk;
                        b.tip_s[kb + k] += qs + bs - tsg;
                    }
                    b.tip_vol[i] = tv + pv - dvt;
                    *q.base_vol = (fvb as f64 * (1.0 - pfb)) as f32;
                    let keep = fv as f64 - pv;
                    let nv = keep + dv;
                    if dv > 0.0 && nv > 1e-12 {
                        *q.wet = ((*q.wet as f64 * keep + dv) / nv) as f32;
                        *q.gloss = ((*q.gloss as f64 * keep + prm.p_gloss * dv) / nv) as f32;
                        *q.body = ((*q.body as f64 * keep + prm.p_body * dv) / nv) as f32;
                        *q.drate = ((*q.drate as f64 * keep + prm.p_drate * dv) / nv) as f32;
                    }
                    *q.film_vol = (fvt as f64 * (1.0 - pf) + dv) as f32;
                    v -= dvb;
                    b.res_vol[i] = v;
                    let wet_here = *q.wet;
                    // ploughing: stiff bristles push wet paint aside -> ridges
                    if prm.plough > 0.0 && nv > 0.05 {
                        let lat = dx * nx + dy * ny;
                        let side = if lat >= 0.0 { 1.0 } else { -1.0 };
                        let x2 = py_round(tx + nx * side * (r + 0.9));
                        let y2 = py_round(ty + ny * side * (r + 0.9));
                        if (0..w).contains(&x2) && (0..h).contains(&y2) && (x2 != xx || y2 != yy) {
                            let (y2u, x2u) = (y2 as usize, x2 as usize);
                            let m2 = cv.mask(y2u, x2u);
                            if m2 < 0.999 {
                                let mut fr =
                                    prm.plough * contact * u * 0.25 * wet_here as f64 * (1.0 - m2 as f64);
                                if fr > 0.5 {
                                    fr = 0.5;
                                }
                                let (mut src, mut dst) = cv.px2((yu, xu), (y2u, x2u));
                                move_film(&mut src, &mut dst, fr);
                                if *dst.wet < *src.wet {
                                    *dst.wet = *src.wet;
                                    *dst.gloss = *src.gloss;
                                    *dst.body = *src.body;
                                    *dst.drate = *src.drate;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    (bx0, by0, bx1, by1)
}

/// Move fraction `frac` of the wet top film from `src` to `dst`
/// (`pinsel._kernels.move_film`). Also used by tools in later stages.
#[inline]
pub fn move_film(src: &mut WetPx<'_>, dst: &mut WetPx<'_>, frac: f64) {
    if frac <= 0.0 {
        return;
    }
    for k in 0..NB {
        let dk = src.film_k[k] as f64 * frac;
        let ds = src.film_s[k] as f64 * frac;
        src.film_k[k] = (src.film_k[k] as f64 - dk) as f32;
        src.film_s[k] = (src.film_s[k] as f64 - ds) as f32;
        dst.film_k[k] = (dst.film_k[k] as f64 + dk) as f32;
        dst.film_s[k] = (dst.film_s[k] as f64 + ds) as f32;
    }
    let dv = *src.film_vol as f64 * frac;
    *src.film_vol = (*src.film_vol as f64 - dv) as f32;
    *dst.film_vol = (*dst.film_vol as f64 + dv) as f32;
}
