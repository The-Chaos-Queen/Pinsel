//! Fluid media (watercolour, ink): the bristle brush exchanging water and
//! pigment with the paper. A line-by-line port of
//! `pinsel._kernels.fluid_stroke` (Python 0.1).
//!
//! The brush (`Bristles`, shared with paste) carries water `res_water` next to
//! its pigment reservoir. Where the brush is wetter than the paper it lays
//! water and pigment down; where it is thirstier it lifts water and suspended
//! pigment back up. Number types mirror numba (see `paste.rs`).

use crate::grid::{Plane, PlaneMut, SpecMut};
use crate::NB;

/// Per-bristle state of a watercolour brush (N, N*2, N*NB elements).
pub struct FluidBristles<'a> {
    pub boff: &'a [f64],
    pub bthr: &'a [f64],
    pub bfollow: &'a [f64],
    pub brad: &'a [f64],
    pub bsplay: &'a [f64],
    pub bdep: &'a [f64],
    pub tips: &'a mut [f64],
    pub res_k: &'a mut [f64],
    pub res_s: &'a mut [f64],
    pub res_vol: &'a mut [f64],
    pub res_water: &'a mut [f64],
    /// Water capacity per bristle.
    pub wcap: &'a [f64],
}

/// Scalar parameters of one fluid stroke (names as in the Python kernel).
#[derive(Clone, Copy, Debug)]
pub struct FluidParams {
    pub ds: f64,
    pub speed: f64,
    pub size: f64,
    pub sx0: f64,
    pub sx1: f64,
    pub sy0: f64,
    pub sy1: f64,
    pub sgamma: f64,
    pub splay: f64,
    pub alpha: f64,
    pub lift: f64,
    pub contact_soft: f64,
    pub gran: f64,
    pub stain: f64,
    pub water_ref: f64,
}

/// The watercolour state a fluid stroke writes: surface water, suspended
/// pigment K and S, suspended mass m, granulation- and staining-weighted mass.
pub struct WcSurface<'a> {
    pub water: PlaneMut<'a, f32>,
    pub k: SpecMut<'a, f32>,
    pub s: SpecMut<'a, f32>,
    pub m: PlaneMut<'a, f32>,
    pub g: PlaneMut<'a, f32>,
    pub st: PlaneMut<'a, f32>,
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

/// Drag a watercolour brush along the resampled path. Returns the touched
/// bounding box `(x0, y0, x1, y1)`, or `(W, H, -1, -1)` if nothing was touched.
#[allow(clippy::too_many_arguments)]
pub fn fluid_stroke(
    px: &[f64],
    py: &[f64],
    pp: &[f64],
    pa: &[f64],
    b: &mut FluidBristles<'_>,
    prm: &FluidParams,
    wc: &mut WcSurface<'_>,
    tooth: Plane<'_, f32>,
    mask: Plane<'_, f32>,
) -> (i64, i64, i64, i64) {
    let h = wc.water.height() as i64;
    let w = wc.water.width() as i64;
    let n = b.bthr.len();
    let m = px.len();
    assert!(py.len() == m && pp.len() == m && pa.len() == m, "path arrays differ in length");
    assert!(b.boff.len() == 2 * n && b.tips.len() == 2 * n, "bristle arrays differ in length");
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
        let across = 2.0
            * ((sx * (ca * nx + sa * ny)).powi(2) + (sy * (-sa * nx + ca * ny)).powi(2)).sqrt()
            + 1.0;
        let lift_base = prm.lift * across / n as f64 * prm.ds;
        let dep_base = prm.alpha * prm.ds * spd_dep;

        for i in 0..n {
            let kb = i * NB;
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
            let x0 = (tx - r).floor() as i64;
            let x1 = (tx + r).ceil() as i64;
            let y0 = (ty - r).floor() as i64;
            let y1 = (ty + r).ceil() as i64;
            if x1 < 0 || y1 < 0 || x0 >= w || y0 >= h {
                continue;
            }
            let x0 = x0.max(0);
            let y0 = y0.max(0);
            let x1 = x1.min(w - 1);
            let y1 = y1.min(h - 1);
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
            bx0 = bx0.min(x0);
            by0 = by0.min(y0);
            bx1 = bx1.max(x1);
            by1 = by1.max(y1);
            let mut wv = b.res_water[i];
            let mut sat_b = wv / (b.wcap[i] + 1e-12);
            let depth = 2.6 * (0.35 * cp + 0.65 * clamp01(sat_b).sqrt()) * spd_depth;
            let level = 1.0 - depth;
            let press = 0.35 + 0.65 * cp;
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
                    let msk = 1.0 - mask.at(yu, xu) as f64;
                    if msk <= 0.0 {
                        continue;
                    }
                    let cw = wc.water.at(yu, xu);
                    let t = tooth.at(yu, xu) as f64 * (-(cw as f64) * 2.0).exp();
                    let contact = clamp01((t - level) / prm.contact_soft + 0.5);
                    let wcf = wgt * contact * msk * press;
                    if wcf <= 0.0 {
                        continue;
                    }
                    let sat_c = cw as f64 / prm.water_ref;
                    let diff = sat_b - 0.8 * sat_c;
                    if diff > 0.0 && wv > 1e-9 {
                        // brush -> paper: water carries pigment along
                        let mut f = dep_base * wcf * (0.4 + diff) * b.bdep[i];
                        if f > 0.5 {
                            f = 0.5;
                        }
                        let dw = wv * f;
                        let dm = b.res_vol[i] * f;
                        let pk = wc.k.px_mut(yu, xu);
                        for k in 0..NB {
                            let dk = b.res_k[kb + k] * f;
                            pk[k] = (pk[k] as f64 + dk) as f32;
                            b.res_k[kb + k] -= dk;
                        }
                        let ps = wc.s.px_mut(yu, xu);
                        for k in 0..NB {
                            let dsg = b.res_s[kb + k] * f;
                            ps[k] = (ps[k] as f64 + dsg) as f32;
                            b.res_s[kb + k] -= dsg;
                        }
                        wc.water.set(yu, xu, (cw as f64 + dw) as f32);
                        let mm = wc.m.at(yu, xu);
                        wc.m.set(yu, xu, (mm as f64 + dm) as f32);
                        let gg = wc.g.at(yu, xu);
                        wc.g.set(yu, xu, (gg as f64 + dm * prm.gran) as f32);
                        let ss = wc.st.at(yu, xu);
                        wc.st.set(yu, xu, (ss as f64 + dm * prm.stain) as f32);
                        b.res_vol[i] -= dm;
                        wv -= dw;
                        b.res_water[i] = wv;
                    } else if diff < 0.0 && prm.lift > 0.0 {
                        // thirsty brush lifts water and suspended pigment
                        let room = 1.0 - wv / (1.3 * b.wcap[i] + 1e-12);
                        if room > 0.0 {
                            let mut f = lift_base * wcf * (-diff) * room;
                            if f > 0.5 {
                                f = 0.5;
                            }
                            let dw = cw as f64 * f;
                            let mm = wc.m.at(yu, xu);
                            let pk = wc.k.px_mut(yu, xu);
                            for k in 0..NB {
                                let dk = pk[k] as f64 * f;
                                pk[k] = (pk[k] as f64 - dk) as f32;
                                b.res_k[kb + k] += dk;
                            }
                            let ps = wc.s.px_mut(yu, xu);
                            for k in 0..NB {
                                let dsg = ps[k] as f64 * f;
                                ps[k] = (ps[k] as f64 - dsg) as f32;
                                b.res_s[kb + k] += dsg;
                            }
                            wc.water.set(yu, xu, (cw as f64 - dw) as f32);
                            wc.m.set(yu, xu, (mm as f64 * (1.0 - f)) as f32);
                            let gg = wc.g.at(yu, xu);
                            wc.g.set(yu, xu, (gg as f64 * (1.0 - f)) as f32);
                            let ss = wc.st.at(yu, xu);
                            wc.st.set(yu, xu, (ss as f64 * (1.0 - f)) as f32);
                            b.res_vol[i] += mm as f64 * f;
                            wv += dw;
                            b.res_water[i] = wv;
                        }
                    }
                    sat_b = wv / (b.wcap[i] + 1e-12);
                }
            }
        }
    }
    (bx0, by0, bx1, by1)
}
