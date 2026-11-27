//! Port of pinsel (Python 0.1) kernels: tools_spray. See docs/RUST_PORTING_GUIDE.md.
//!
//! The airbrush, splatter, sponge and pen kernels of `pinsel/tools.py`:
//! `_spray_accumulate`, `_splat_discs`, `_splat_drops`, `_sponge_dab`,
//! `_pen_raster` and `_feather`. Each mirrors its numba kernel (compiled
//! without fastmath) operation by operation: same typing (f32 canvas values
//! widen to f64 when they meet an f64 scalar, f32 op f32 stays f32, every store
//! to an f32 array rounds), same order of arithmetic, `x ** 2` as `x * x`,
//! other powers as `powf`, `int(math.floor(..))` as `floor() as i64`.
//!
//! Nothing here is parallel: the raster kernels walk overlapping footprints
//! (`max` / `+=` into shared cells) and the sponge dab reads and writes a shared
//! reservoir, so they stay serial like the numba kernels. `_feather` could run
//! row-parallel, but its maps are a few hundred pixels across; not worth it.
//!
//! The random numbers (Poisson counts, drop positions) are drawn in Python;
//! none of these kernels uses `np.random`, so they can be bit-identical.

use crate::grid::{Plane, PlaneMut, WetCanvas};
use crate::tools_common::{cl01, dep, pymax, pymin, take, wc_add, WcLayer};
use crate::NB;
use std::f64::consts::PI;

#[inline(always)]
fn ifloor(v: f64) -> i64 {
    v.floor() as i64
}

#[inline(always)]
fn iceil(v: f64) -> i64 {
    v.ceil() as i64
}

/// `_spray_accumulate`: add gaussian spray cones (integral `amt[j]`) into the
/// bbox map `acc`.
#[allow(clippy::too_many_arguments)]
pub fn spray_accumulate(
    px: &[f64],
    py: &[f64],
    amt: &[f64],
    sig: &[f64],
    tail_frac: f64,
    tail_mult: f64,
    ox: f64,
    oy: f64,
    acc: &mut PlaneMut<'_, f64>,
) {
    let h = acc.height() as i64;
    let w = acc.width() as i64;
    for j in 0..px.len() {
        let a = amt[j];
        if a <= 0.0 {
            continue;
        }
        let s = sig[j];
        let st = s * tail_mult;
        let r = 3.2 * (if tail_frac > 0.0 { st } else { s });
        let cx = px[j] - ox;
        let cy = py[j] - oy;
        let x0 = ifloor(cx - r).max(0);
        let x1 = iceil(cx + r).min(w - 1);
        let y0 = ifloor(cy - r).max(0);
        let y1 = iceil(cy + r).min(h - 1);
        if x1 < x0 || y1 < y0 {
            continue;
        }
        let nx = (x1 - x0 + 1) as usize;
        let mut gxs = vec![0.0f64; nx];
        let mut gxt = vec![0.0f64; nx];
        let i2s = 0.5 / (s * s);
        let i2t = 0.5 / (st * st);
        for i in 0..nx {
            let dx = (x0 + i as i64) as f64 - cx;
            gxs[i] = (-dx * dx * i2s).exp();
            gxt[i] = (-dx * dx * i2t).exp();
        }
        let ns = a * (1.0 - tail_frac) / (2.0 * PI * s * s);
        let nt = a * tail_frac / (2.0 * PI * st * st);
        for yy in y0..=y1 {
            let dy = yy as f64 - cy;
            let gys = (-dy * dy * i2s).exp() * ns;
            let gyt = (-dy * dy * i2t).exp() * nt;
            for i in 0..nx {
                let c = acc.get_mut(yy as usize, (x0 + i as i64) as usize);
                *c += gys * gxs[i] + gyt * gxt[i];
            }
        }
    }
}

/// Tooth value at bbox cell (`yy`, `xx`) shifted by the map origin, 0 outside
/// the canvas (`tf = 0.0; if 0 <= tY < H and 0 <= tX < W: tf = tooth[tY, tX]`).
#[inline(always)]
fn tooth_at(tooth: &Plane<'_, f32>, yy: i64, xx: i64, oy: i64, ox: i64) -> f64 {
    let ty = yy + oy;
    let tx = xx + ox;
    if ty >= 0 && ty < tooth.height() as i64 && tx >= 0 && tx < tooth.width() as i64 {
        tooth.at(ty as usize, tx as usize) as f64
    } else {
        0.0
    }
}

/// `_splat_discs`: add droplets (dome-shaped discs of volume `vs`) into `acc`.
/// `rough` lets the paper tooth fray the droplet outline.
#[allow(clippy::too_many_arguments)]
pub fn splat_discs(
    xs: &[f64],
    ys: &[f64],
    rs: &[f64],
    vs: &[f64],
    ox: i64,
    oy: i64,
    acc: &mut PlaneMut<'_, f64>,
    tooth: &Plane<'_, f32>,
    rough: f64,
) {
    let h = acc.height() as i64;
    let w = acc.width() as i64;
    for k in 0..xs.len() {
        let r = rs[k];
        let cx = xs[k] - ox as f64;
        let cy = ys[k] - oy as f64;
        let rr = r + 1.5;
        let x0 = ifloor(cx - rr).max(0);
        let x1 = iceil(cx + rr).min(w - 1);
        let y0 = ifloor(cy - rr).max(0);
        let y1 = iceil(cy + rr).min(h - 1);
        if x1 < x0 || y1 < y0 {
            continue;
        }
        // sub-pixel droplets: bilinear splat keeps the volume
        if r < 0.6 {
            let fx = cx - cx.floor();
            let fy = cy - cy.floor();
            let ix = ifloor(cx);
            let iy = ifloor(cy);
            for dy in 0..2i64 {
                for dx in 0..2i64 {
                    let xx = ix + dx;
                    let yy = iy + dy;
                    if 0 <= xx && xx < w && 0 <= yy && yy < h {
                        let wx = if dx == 1 { fx } else { 1.0 - fx };
                        let wy = if dy == 1 { fy } else { 1.0 - fy };
                        *acc.get_mut(yy as usize, xx as usize) += vs[k] * wx * wy;
                    }
                }
            }
            continue;
        }
        let mut wsum = 0.0f64;
        for yy in y0..=y1 {
            for xx in x0..=x1 {
                let ddx = xx as f64 - cx;
                let ddy = yy as f64 - cy;
                let d = (ddx * ddx + ddy * ddy).sqrt();
                let tf = tooth_at(tooth, yy, xx, oy, ox);
                let re = r * (1.0 + rough * 0.12 * tf);
                let q = d / re;
                if q < 1.0 + 0.5 / re {
                    let cov = cl01((re - d) + 0.5);
                    wsum += cov * pymax(1.0 - q * q, 0.05).sqrt();
                }
            }
        }
        if wsum <= 0.0 {
            continue;
        }
        for yy in y0..=y1 {
            for xx in x0..=x1 {
                let ddx = xx as f64 - cx;
                let ddy = yy as f64 - cy;
                let d = (ddx * ddx + ddy * ddy).sqrt();
                let tf = tooth_at(tooth, yy, xx, oy, ox);
                let re = r * (1.0 + rough * 0.12 * tf);
                let q = d / re;
                if q < 1.0 + 0.5 / re {
                    let cov = cl01((re - d) + 0.5);
                    *acc.get_mut(yy as usize, xx as usize) +=
                        vs[k] * cov * pymax(1.0 - q * q, 0.05).sqrt() / wsum;
                }
            }
        }
    }
}

/// `_splat_drops`: rasterise impact drops into the bbox map `acc`: wobbly
/// elliptic outline, frayed by the tooth, with a tapering tail for oblique
/// impacts. `dome` 1 = rounded blob profile, 0 = flat.
#[allow(clippy::too_many_arguments)]
pub fn splat_drops(
    xs: &[f64],
    ys: &[f64],
    rs: &[f64],
    es: &[f64],
    th: &[f64],
    vs: &[f64],
    ph: &[f64],
    ox: i64,
    oy: i64,
    acc: &mut PlaneMut<'_, f64>,
    tooth: &Plane<'_, f32>,
    rough: f64,
    dome: f64,
) {
    let h = acc.height() as i64;
    let w = acc.width() as i64;
    for k in 0..xs.len() {
        let r = rs[k];
        let e = es[k];
        let a = r * e.sqrt();
        let b = r / e.sqrt();
        let ct = th[k].cos();
        let st = th[k].sin();
        let mut lt = 0.0f64;
        if e > 1.5 {
            lt = (e - 1.0) * r * 1.3;
        }
        let cx = xs[k] - ox as f64;
        let cy = ys[k] - oy as f64;
        let rr = a + lt + 2.0;
        let x0 = ifloor(cx - rr).max(0);
        let x1 = iceil(cx + rr).min(w - 1);
        let y0 = ifloor(cy - rr).max(0);
        let y1 = iceil(cy + rr).min(h - 1);
        if x1 < x0 || y1 < y0 {
            continue;
        }
        let p0 = ph[k];
        if r < 0.55 {
            // sub-pixel droplet: bilinear splat
            let fx = cx - cx.floor();
            let fy = cy - cy.floor();
            let ix = ifloor(cx);
            let iy = ifloor(cy);
            for dy in 0..2i64 {
                for dx in 0..2i64 {
                    let xx = ix + dx;
                    let yy = iy + dy;
                    if 0 <= xx && xx < w && 0 <= yy && yy < h {
                        *acc.get_mut(yy as usize, xx as usize) += vs[k]
                            * (if dx == 1 { fx } else { 1.0 - fx })
                            * (if dy == 1 { fy } else { 1.0 - fy });
                    }
                }
            }
            continue;
        }
        let mut wsum = 0.0f64;
        for pas in 0..2 {
            for yy in y0..=y1 {
                for xx in x0..=x1 {
                    let dx = xx as f64 - cx;
                    let dy = yy as f64 - cy;
                    let u = dx * ct + dy * st;
                    let v = -dx * st + dy * ct;
                    let phi = (v / b).atan2(u / a);
                    let mut wob = 1.0
                        + 0.07 * (3.0 * phi + p0).sin()
                        + 0.05 * (5.0 * phi + 2.3 * p0).sin()
                        + 0.03 * (8.0 * phi + 4.1 * p0).sin();
                    let tf = tooth_at(tooth, yy, xx, oy, ox);
                    wob *= 1.0 + rough * 0.10 * tf;
                    let ua = u / a;
                    let vb = v / b;
                    let q = (ua * ua + vb * vb).sqrt() / wob;
                    let mut val = 0.0f64;
                    let edge = (1.0 - q) * pymin(a, b) + 0.5;
                    if edge > 0.0 {
                        let prof = if dome > 0.5 { pymax(1.0 - q * q, 0.0).sqrt() } else { 1.0 };
                        val = cl01(edge) * (0.15 + 0.85 * prof);
                    }
                    if lt > 0.0 && u > 0.3 * a {
                        let tu = (u - 0.3 * a) / (lt + 0.7 * a);
                        if tu < 1.0 {
                            let hwd = b * 0.6 * (1.0 - tu).powf(0.8) * wob;
                            let te = hwd - v.abs() + 0.5;
                            if te > 0.0 {
                                let tv = cl01(te) * (if dome > 0.5 { 0.6 } else { 1.0 }) * (1.0 - 0.5 * tu);
                                if tv > val {
                                    val = tv;
                                }
                            }
                        }
                    }
                    if val <= 0.0 {
                        continue;
                    }
                    if pas == 0 {
                        wsum += val;
                    } else {
                        *acc.get_mut(yy as usize, xx as usize) += vs[k] * val / wsum;
                    }
                }
            }
            if wsum <= 1e-9 {
                break;
            }
        }
    }
}

/// Scalar inputs of `_sponge_dab` (everything but the arrays).
pub struct SpongeParams {
    pub cx: f64,
    pub cy: f64,
    pub rot: f64,
    pub scale: f64,
    pub depth: f64,
    pub soft: f64,
    pub micro: f64,
    pub alpha: f64,
    pub pickup: f64,
    pub cap: f64,
    pub p_gloss: f64,
    pub p_body: f64,
    pub p_drate: f64,
    pub fluid: bool,
    pub water_lvl: f64,
    pub wat_ratio: f64,
    pub gran: f64,
    pub stain: f64,
    pub sid: i64,
    pub x0: i64,
    pub y0: i64,
}

/// `_sponge_dab`: press the sponge texture once at (cx, cy) rotated by `rot`.
/// `tex` is the T x T height texture, `mask` the (clipped) region mask whose
/// top left cell is canvas pixel (x0, y0); `rk`, `rs` the reservoir's first
/// row (NB values each), `rv` its volume array (`rV`, only element 0 is used);
/// `wc` the watercolour layer (1 x 1 dummies when not fluid) with its `damp`
/// map `wdamp`. Returns the bbox (x0, y0, x1, y1) of the touched pixels,
/// (W, H, -1, -1) when nothing was touched.
#[allow(clippy::too_many_arguments)]
pub fn sponge_dab<C: WetCanvas>(
    p: &SpongeParams,
    tex: &Plane<'_, f64>,
    rk: &mut [f64],
    rs: &mut [f64],
    rv: &mut [f64],
    cv: &mut C,
    mask: &Plane<'_, f32>,
    tooth: &Plane<'_, f32>,
    wc: &mut WcLayer<'_>,
    wdamp: &mut PlaneMut<'_, f32>,
) -> (i64, i64, i64, i64) {
    let t = tex.height() as i64;
    let c = (t - 1) as f64 / 2.0;
    let hh = tooth.height() as i64;
    let ww = tooth.width() as i64;
    let (x0, y0) = (p.x0, p.y0);
    let x1 = x0 + mask.width() as i64 - 1;
    let y1 = y0 + mask.height() as i64 - 1;
    let mut bb = [ww, hh, -1i64, -1i64];
    if x1 < x0 || y1 < y0 {
        return (bb[0], bb[1], bb[2], bb[3]);
    }
    let cr = p.rot.cos();
    let sr = p.rot.sin();
    let level = 1.0 - p.depth;
    let mut kk = [0.0f64; NB];
    let mut ks = [0.0f64; NB];
    let tm1 = t as f64 - 1.0;
    for yy in y0..=y1 {
        for xx in x0..=x1 {
            let m = 1.0 - mask.at((yy - y0) as usize, (xx - x0) as usize) as f64;
            if m <= 0.0 {
                continue;
            }
            let dx = (xx as f64 - p.cx) / p.scale;
            let dy = (yy as f64 - p.cy) / p.scale;
            let u = cr * dx + sr * dy + c;
            let v = -sr * dx + cr * dy + c;
            if u < 0.0 || v < 0.0 || u >= tm1 || v >= tm1 {
                continue;
            }
            let iu = u as i64;
            let iv = v as i64;
            let fu = u - iu as f64;
            let fv_ = v - iv as f64;
            let (ivu, iuu) = (iv as usize, iu as usize);
            let mut hgt = tex.at(ivu, iuu) * (1.0 - fu) * (1.0 - fv_)
                + tex.at(ivu, iuu + 1) * fu * (1.0 - fv_)
                + tex.at(ivu + 1, iuu) * (1.0 - fu) * fv_
                + tex.at(ivu + 1, iuu + 1) * fu * fv_;
            // a soft sponge conforms to the tooth only a little
            hgt += 0.06 * tooth.at(yy as usize, xx as usize) as f64;
            let cnt = cl01((hgt - level) / p.soft + 0.5) * m;
            if cnt <= 0.0 {
                continue;
            }
            // pressed deeper -> more paint transferred
            let pen = 0.35 + 0.65 * cl01((hgt - level) / 0.3);
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
            let mut mic = 1.0 + p.micro * (tex.at(ivu, iuu) - tex.at(ivu + 1, iuu + 1)) * 4.0;
            if mic < 0.2 {
                mic = 0.2;
            }
            let (yu, xu) = (yy as usize, xx as usize);
            if p.fluid {
                // water + pigment from the sponge; a damp sponge blots
                let wsat = wc.water.at(yu, xu);
                if p.water_lvl > 0.25 * (1.0 + wsat as f64) {
                    let dw = p.alpha * cnt * pen * p.water_lvl * mic * 0.9;
                    let vb = rv[0];
                    let mut dmass = dw * p.wat_ratio;
                    if dmass > 0.02 * vb {
                        dmass = 0.02 * vb;
                    }
                    if vb > 1e-12 {
                        for b in 0..NB {
                            kk[b] = rk[b] / vb;
                            ks[b] = rs[b] / vb;
                        }
                        let fr = dmass / vb;
                        for b in 0..NB {
                            rk[b] -= rk[b] * fr;
                            rs[b] -= rs[b] * fr;
                        }
                        rv[0] = vb - dmass;
                        wc_add(wc, yu, xu, dw, dmass, &kk, &ks, p.gran, p.stain);
                    } else {
                        let q = wc.water.get_mut(yu, xu);
                        *q = (*q as f64 + dw) as f32;
                    }
                } else if p.pickup > 0.0 {
                    let thirst = 1.0 - p.water_lvl;
                    // a thirsty sponge also draws moisture out of the paper
                    {
                        let q = wdamp.get_mut(yu, xu);
                        *q = (*q as f64 * (1.0 - pymin(0.6 * cnt * thirst, 0.9))) as f32;
                    }
                    // it re-wets settled pigment and lifts it (staining pigments resist)
                    let dmm = wc.dm.at(yu, xu);
                    if dmm as f64 > 1e-6 {
                        let stn: f32 = wc.dst.at(yu, xu) / dmm; // f32 / f32
                        let fl = pymin(p.pickup * cnt * pen * 2.6 * thirst * (1.0 - pymin(stn as f64, 1.0)), 0.85);
                        for b in 0..NB {
                            let dk = wc.dk.px(yu, xu)[b] as f64 * fl;
                            let dsg = wc.ds.px(yu, xu)[b] as f64 * fl;
                            let q = &mut wc.dk.px_mut(yu, xu)[b];
                            *q = (*q as f64 - dk) as f32;
                            let q = &mut wc.ds.px_mut(yu, xu)[b];
                            *q = (*q as f64 - dsg) as f32;
                            rk[b] += dk;
                            rs[b] += dsg;
                        }
                        *wc.dm.get_mut(yu, xu) = (dmm as f64 * (1.0 - fl)) as f32;
                        let q = wc.dg.get_mut(yu, xu);
                        *q = (*q as f64 * (1.0 - fl)) as f32;
                        let q = wc.dst.get_mut(yu, xu);
                        *q = (*q as f64 * (1.0 - fl)) as f32;
                        rv[0] += dmm as f64 * fl;
                    }
                    if wsat as f64 <= 1e-4 {
                        continue;
                    }
                    let fr = pymin(p.pickup * cnt * pen * 2.6 * thirst, 0.92);
                    let mm = wc.m.at(yu, xu) as f64 * fr;
                    for b in 0..NB {
                        let dk = wc.k.px(yu, xu)[b] as f64 * fr;
                        let dsg = wc.s.px(yu, xu)[b] as f64 * fr;
                        let q = &mut wc.k.px_mut(yu, xu)[b];
                        *q = (*q as f64 - dk) as f32;
                        let q = &mut wc.s.px_mut(yu, xu)[b];
                        *q = (*q as f64 - dsg) as f32;
                        rk[b] += dk;
                        rs[b] += dsg;
                    }
                    let q = wc.m.get_mut(yu, xu);
                    *q = (*q as f64 - mm) as f32;
                    let q = wc.g.get_mut(yu, xu);
                    *q = (*q as f64 * (1.0 - fr)) as f32;
                    let q = wc.st.get_mut(yu, xu);
                    *q = (*q as f64 * (1.0 - fr)) as f32;
                    let q = wc.water.get_mut(yu, xu);
                    *q = (*q as f64 * (1.0 - fr)) as f32;
                    rv[0] += mm;
                }
                continue;
            }
            // paste: pick up wet paint (sponge gets dirty / lifts texture) ...
            let (fvt, wetv): (f32, f32) = {
                let q = cv.px(yu, xu);
                (*q.film_vol + *q.base_vol, *q.wet) // f32 + f32
            };
            if p.pickup > 0.0 && fvt as f64 > 1e-6 {
                let room = 1.0 - rv[0] / (1.5 * p.cap + 1e-12);
                if room > 0.0 {
                    let pf = pymin(p.pickup * cnt * wetv as f64 * room, 0.6);
                    let tk = take(&mut cv.px(yu, xu), fvt as f64 * pf, &mut rk[..NB], &mut rs[..NB]);
                    rv[0] += tk;
                }
            }
            // ... and deposit from its load, textured by the sponge surface
            let vb = rv[0];
            if vb <= 1e-9 {
                continue;
            }
            let ld = vb / (p.cap + 1e-12);
            let mut dv = p.alpha * cnt * pen * mic * pymin(ld, 1.5).powf(1.3);
            if dv > 0.05 * vb {
                dv = 0.05 * vb;
            }
            let fr = dv / vb;
            for b in 0..NB {
                kk[b] = rk[b] / vb;
                ks[b] = rs[b] / vb;
                rk[b] -= rk[b] * fr;
                rs[b] -= rs[b] * fr;
            }
            rv[0] = vb - dv;
            dep(&mut cv.px(yu, xu), p.sid, dv, &kk, &ks, p.p_gloss, p.p_body, p.p_drate);
        }
    }
    (bb[0], bb[1], bb[2], bb[3])
}

/// `_pen_raster`: rasterise a pen line into the bbox maps `cov` (max coverage)
/// and `amt` (max ink density). kind 0 = round tip, 1 = broad-edge nib
/// (rectangle of half sizes `nib_hw` x `nib_ht` along (`nca`, `nsa`)),
/// 2 = pointed flexible nib (two tines `gap` apart, inside filled by `fill`).
#[allow(clippy::too_many_arguments)]
pub fn pen_raster(
    px: &[f64],
    py: &[f64],
    rad: &[f64],
    ink: &[f64],
    skip: &[f64],
    gap: &[f64],
    fill: &[f64],
    kind: i64,
    nca: f64,
    nsa: f64,
    nib_hw: f64,
    nib_ht: f64,
    ox: i64,
    oy: i64,
    cov: &mut PlaneMut<'_, f64>,
    amt: &mut PlaneMut<'_, f64>,
    tooth: &Plane<'_, f32>,
) {
    let h = cov.height() as i64;
    let w = cov.width() as i64;
    let m_ = px.len();
    for j in 0..m_ {
        let cx = px[j] - ox as f64;
        let cy = py[j] - oy as f64;
        let r = rad[j];
        let rr = if kind == 1 {
            nib_hw + nib_ht + 1.5
        } else if kind == 2 {
            0.5 * gap[j] + r + 1.5
        } else {
            r + 1.5
        };
        let x0 = ifloor(cx - rr).max(0);
        let x1 = iceil(cx + rr).min(w - 1);
        let y0 = ifloor(cy - rr).max(0);
        let y1 = iceil(cy + rr).min(h - 1);
        if x1 < x0 || y1 < y0 {
            continue;
        }
        let j0 = if j > 0 { j - 1 } else { 0 };
        let j1 = if j < m_ - 1 { j + 1 } else { m_ - 1 };
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
        for yy in y0..=y1 {
            for xx in x0..=x1 {
                let dx = xx as f64 - cx;
                let dy = yy as f64 - cy;
                let mut c: f64;
                if kind == 1 {
                    let u = dx * nca + dy * nsa;
                    let v = -dx * nsa + dy * nca;
                    c = cl01(nib_hw - u.abs() + 0.5) * cl01(nib_ht - v.abs() + 0.5);
                } else if kind == 2 {
                    let lat = -dx * my + dy * mx;
                    let alo = dx * mx + dy * my;
                    let hg = 0.5 * gap[j];
                    // distance to the nearer tine, and to the inner band
                    let t = lat.abs() - hg;
                    let dt = (alo * alo + t * t).sqrt();
                    c = cl01(r - dt + 0.5);
                    if lat.abs() < hg {
                        let inner = cl01(r - alo.abs() + 0.5) * fill[j];
                        if inner > c {
                            c = inner;
                        }
                    }
                } else {
                    let d = (dx * dx + dy * dy).sqrt();
                    c = cl01(r - d + 0.5);
                }
                if c <= 0.0 {
                    continue;
                }
                if skip[j] > -2.5 {
                    let ty = yy + oy;
                    let tx = xx + ox;
                    if ty >= 0 && ty < tooth.height() as i64 && tx >= 0 && tx < tooth.width() as i64 {
                        c *= cl01((tooth.at(ty as usize, tx as usize) as f64 - skip[j]) / 0.45 + 0.5);
                    }
                }
                let (yu, xu) = (yy as usize, xx as usize);
                if c > cov.at(yu, xu) {
                    cov.set(yu, xu, c);
                }
                let a = c * ink[j];
                if a > amt.at(yu, xu) {
                    amt.set(yu, xu, a);
                }
            }
        }
    }
}

/// `_feather`: ink wicking into paper fibres: coverage creeps outward along
/// high-permeability paths (hairy, feathered line edges).
pub fn feather(
    cov: &mut PlaneMut<'_, f64>,
    amt: &mut PlaneMut<'_, f64>,
    perm: &Plane<'_, f64>,
    iters: i64,
    decay: f64,
) {
    let h = cov.height();
    let w = cov.width();
    // c2 / a2: the copies numba makes (`cov.copy()`, `amt.copy()`)
    let mut c2 = vec![0.0f64; h * w];
    let mut a2 = vec![0.0f64; h * w];
    let mut c2p = PlaneMut::new(h, w, &mut c2);
    let mut a2p = PlaneMut::new(h, w, &mut a2);
    for y in 0..h {
        for x in 0..w {
            c2p.set(y, x, cov.at(y, x));
            a2p.set(y, x, amt.at(y, x));
        }
    }
    for _ in 0..iters {
        for y in 1..h.saturating_sub(1) {
            for x in 1..w.saturating_sub(1) {
                let pm = perm.at(y, x);
                if pm <= 0.0 {
                    continue;
                }
                let mut best = 0.0f64;
                let mut ba = 0.0f64;
                for dy in -1i64..2 {
                    for dx in -1i64..2 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let (yy, xx) = ((y as i64 + dy) as usize, (x as i64 + dx) as usize);
                        let v = cov.at(yy, xx);
                        if v > best {
                            best = v;
                            ba = amt.at(yy, xx);
                        }
                    }
                }
                let nv = best * pm * decay;
                if nv > c2p.at(y, x) {
                    c2p.set(y, x, nv);
                    let cur = a2p.at(y, x);
                    a2p.set(y, x, pymax(cur, ba * pm * decay * 0.8));
                }
            }
        }
        for y in 0..h {
            for x in 0..w {
                cov.set(y, x, c2p.at(y, x));
                amt.set(y, x, a2p.at(y, x));
            }
        }
    }
}
