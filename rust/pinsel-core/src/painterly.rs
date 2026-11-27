//! Flow-field stroke growing of the AutoPainter.
//!
//! Line-by-line ports of `pinsel.painterly._bil`, `_field_dir`, `_walk`,
//! `_grow_strokes` and `_densify` (Python 0.1). No canvas state: the kernels
//! read float32 images (reference, rendered canvas, direction field, region)
//! and write float64 stroke arrays. See docs/RUST_PORTING_GUIDE.md.
//!
//! Number types mirror numba: image samples are f32 and are widened to f64 by
//! the interpolation weights; everything else is f64 / i64. `x ** 2` is a
//! plain product. Only `grow_strokes` and `densify` are called from Python;
//! `bil`, `field_dir` and `walk` are internal.

use crate::grid::{Plane, Spec, SpecMut};

/// Bilinear sample of channel `c`, coordinates clamped to the image.
#[inline(always)]
pub fn bil(img: &Spec<'_, f32>, x: f64, y: f64, c: usize) -> f64 {
    let hh = img.height() as i64;
    let ww = img.width() as i64;
    let mut x = x;
    let mut y = y;
    if x < 0.0 {
        x = 0.0;
    } else if x > ww as f64 - 1.0 {
        x = ww as f64 - 1.0;
    }
    if y < 0.0 {
        y = 0.0;
    } else if y > hh as f64 - 1.0 {
        y = hh as f64 - 1.0;
    }
    let x0 = x as i64;
    let y0 = y as i64;
    let x1 = if x0 < ww - 1 { x0 + 1 } else { x0 };
    let y1 = if y0 < hh - 1 { y0 + 1 } else { y0 };
    let fx = x - x0 as f64;
    let fy = y - y0 as f64;
    let (y0, x0, y1, x1) = (y0 as usize, x0 as usize, y1 as usize, x1 as usize);
    (img.px(y0, x0)[c] as f64 * (1.0 - fx) + img.px(y0, x1)[c] as f64 * fx) * (1.0 - fy)
        + (img.px(y1, x0)[c] as f64 * (1.0 - fx) + img.px(y1, x1)[c] as f64 * fx) * fy
}

/// Local stroke direction from the doubled-angle field `v` (H x W x 2):
/// `(dx, dy, magnitude)`; `(0, 0, m)` where the field is too weak.
#[inline]
pub fn field_dir(v: &Spec<'_, f32>, x: f64, y: f64, rot: f64) -> (f64, f64, f64) {
    let vx = bil(v, x, y, 0);
    let vy = bil(v, x, y, 1);
    let m = (vx * vx + vy * vy).sqrt();
    if m < 0.04 {
        return (0.0, 0.0, m);
    }
    let th = 0.5 * vy.atan2(vx) + rot;
    (th.cos(), th.sin(), m)
}

/// Fixed arguments of one stroke walk.
pub struct WalkParams {
    pub step: f64,
    pub min_steps: i64,
    pub fc: f64,
    pub edge_stop: f64,
    pub turn_cos: f64,
    pub margin: f64,
}

/// Grow one stroke branch from `(x, y)` heading `(px, py)`; writes at most
/// `nmax` points into `outx`/`outy` and returns how many.
#[allow(clippy::too_many_arguments)]
pub fn walk(
    x: f64,
    y: f64,
    px: f64,
    py: f64,
    nmax: i64,
    count0: i64,
    c0: f64,
    c1: f64,
    c2: f64,
    rf: &Spec<'_, f32>,
    cvl: &Spec<'_, f32>,
    v: &Spec<'_, f32>,
    region: &Plane<'_, f32>,
    prm: &WalkParams,
    outx: &mut [f64],
    outy: &mut [f64],
    use_hz: bool,
    rot: f64,
) -> i64 {
    let hh = region.height() as i64;
    let ww = region.width() as i64;
    let (mut x, mut y, mut px, mut py) = (x, y, px, py);
    let mut n: i64 = 0;
    for _ in 0..nmax {
        let (mut dx, mut dy, m) = field_dir(v, x, y, rot);
        if m < 0.04 {
            dx = px;
            dy = py;
        } else if dx * px + dy * py < 0.0 {
            dx = -dx;
            dy = -dy;
        }
        dx = prm.fc * dx + (1.0 - prm.fc) * px;
        dy = prm.fc * dy + (1.0 - prm.fc) * py;
        let l = (dx * dx + dy * dy).sqrt();
        if l < 1e-9 {
            break;
        }
        dx /= l;
        dy /= l;
        if dx * px + dy * py < prm.turn_cos {
            break;
        }
        let nx = x + prm.step * dx;
        let ny = y + prm.step * dy;
        if nx < -prm.margin
            || ny < -prm.margin
            || nx > ww as f64 - 1.0 + prm.margin
            || ny > hh as f64 - 1.0 + prm.margin
        {
            break;
        }
        let mut xi = (nx + 0.5) as i64;
        let mut yi = (ny + 0.5) as i64;
        if xi < 0 {
            xi = 0;
        } else if xi > ww - 1 {
            xi = ww - 1;
        }
        if yi < 0 {
            yi = 0;
        } else if yi > hh - 1 {
            yi = hh - 1;
        }
        if region.at(yi as usize, xi as usize) < 0.5 {
            break;
        }
        let r0 = bil(rf, nx, ny, 0);
        let r1 = bil(rf, nx, ny, 1);
        let r2 = bil(rf, nx, ny, 2);
        let dc = ((r0 - c0) * (r0 - c0) + (r1 - c1) * (r1 - c1) + (r2 - c2) * (r2 - c2)).sqrt();
        if dc > prm.edge_stop {
            break;
        }
        if use_hz && count0 + n >= prm.min_steps {
            let k0 = bil(cvl, nx, ny, 0);
            let k1 = bil(cvl, nx, ny, 1);
            let k2 = bil(cvl, nx, ny, 2);
            let dk = ((r0 - k0) * (r0 - k0) + (r1 - k1) * (r1 - k1) + (r2 - k2) * (r2 - k2)).sqrt();
            if dk < dc {
                break;
            }
        }
        outx[n as usize] = nx;
        outy[n as usize] = ny;
        n += 1;
        x = nx;
        y = ny;
        px = dx;
        py = dy;
    }
    n
}

/// Scalar parameters of `grow_strokes` (names as in the Python kernel).
#[derive(Clone, Copy, Debug)]
pub struct GrowParams {
    pub step: f64,
    pub min_steps: i64,
    pub max_steps: i64,
    pub back_steps: i64,
    pub fc: f64,
    pub edge_stop: f64,
    pub turn_cos: f64,
    pub dab_len: f64,
    pub margin: f64,
    pub color_mix: f64,
}

/// Grow Hertzmann-style curved strokes from seed points `(sx, sy)`.
/// `ctrl` is (n, max_steps + 2, 2) out: control points; `nctrl` (n) out:
/// their count; `cols` (n, 3) out: stroke colour (Lab).
#[allow(clippy::too_many_arguments)]
pub fn grow_strokes(
    sx: &[f64],
    sy: &[f64],
    rots: &[f64],
    rf: &Spec<'_, f32>,
    cvl: &Spec<'_, f32>,
    v: &Spec<'_, f32>,
    region: &Plane<'_, f32>,
    prm: &GrowParams,
    ctrl: &mut SpecMut<'_, f64>,
    nctrl: &mut [i64],
    cols: &mut SpecMut<'_, f64>,
) {
    let n = sx.len();
    assert!(sy.len() == n && rots.len() == n && nctrl.len() == n, "seed arrays differ in length");
    let max_steps = prm.max_steps;
    let buf = (max_steps + 2).max(0) as usize;
    let mut bx = vec![0.0f64; buf];
    let mut by = vec![0.0f64; buf];
    let mut fx = vec![0.0f64; buf];
    let mut fy = vec![0.0f64; buf];
    let wp = WalkParams {
        step: prm.step,
        min_steps: prm.min_steps,
        fc: prm.fc,
        edge_stop: prm.edge_stop,
        turn_cos: prm.turn_cos,
        margin: prm.margin,
    };
    for s in 0..n {
        let x0 = sx[s];
        let y0 = sy[s];
        let c0 = bil(rf, x0, y0, 0);
        let c1 = bil(rf, x0, y0, 1);
        let c2 = bil(rf, x0, y0, 2);
        let rot = rots[s];
        let (mut d0x, mut d0y, m) = field_dir(v, x0, y0, rot);
        if m < 0.04 {
            d0x = rot.cos();
            d0y = rot.sin();
        }
        let mut nb: i64 = 0;
        if prm.back_steps > 0 && max_steps > 0 {
            nb = walk(
                x0, y0, -d0x, -d0y, prm.back_steps.min(max_steps), 0, c0, c1, c2, rf, cvl, v, region,
                &wp, &mut bx, &mut by, false, rot,
            );
        }
        let nf = walk(
            x0, y0, d0x, d0y, max_steps - nb, nb, c0, c1, c2, rf, cvl, v, region, &wp, &mut fx,
            &mut fy, true, rot,
        );
        let mut k: usize = 0;
        for i in (0..nb as usize).rev() {
            let c = ctrl.px_mut(s, k);
            c[0] = bx[i];
            c[1] = by[i];
            k += 1;
        }
        {
            let c = ctrl.px_mut(s, k);
            c[0] = x0;
            c[1] = y0;
        }
        k += 1;
        for i in 0..nf as usize {
            let c = ctrl.px_mut(s, k);
            c[0] = fx[i];
            c[1] = fy[i];
            k += 1;
        }
        if k == 1 {
            // a dab: short press along the local direction
            let c = ctrl.px_mut(s, 0);
            c[0] = x0 - 0.5 * prm.dab_len * d0x;
            c[1] = y0 - 0.5 * prm.dab_len * d0y;
            let c = ctrl.px_mut(s, 1);
            c[0] = x0 + 0.5 * prm.dab_len * d0x;
            c[1] = y0 + 0.5 * prm.dab_len * d0y;
            k = 2;
        }
        nctrl[s] = k as i64;
        // stroke colour: seed colour blended with the mean along the stroke
        let mut m0 = 0.0f64;
        let mut m1 = 0.0f64;
        let mut m2 = 0.0f64;
        for i in 0..k {
            let (cx, cy) = {
                let c = ctrl.px(s, i);
                (c[0], c[1])
            };
            m0 += bil(rf, cx, cy, 0);
            m1 += bil(rf, cx, cy, 1);
            m2 += bil(rf, cx, cy, 2);
        }
        let kf = k as f64;
        let cm = prm.color_mix;
        let o = cols.px_mut(s, 0);
        o[0] = (1.0 - cm) * c0 + cm * m0 / kf;
        o[1] = (1.0 - cm) * c1 + cm * m1 / kf;
        o[2] = (1.0 - cm) * c2 + cm * m2 / kf;
    }
}

/// Uniform Catmull-Rom through the control points -> dense polylines.
/// `out` is (n, >= (max_steps + 1) * sub + 2, 2); `nout` (n) gets the point
/// counts.
pub fn densify(
    ctrl: &Spec<'_, f64>,
    nctrl: &[i64],
    sub: i64,
    out: &mut SpecMut<'_, f64>,
    nout: &mut [i64],
) {
    let n = ctrl.height();
    assert!(nctrl.len() == n && nout.len() == n, "stroke arrays differ in length");
    let subf = sub as f64;
    for s in 0..n {
        let k = nctrl[s] as usize;
        if k < 3 {
            // straight segment
            for j in 0..=(sub as usize) {
                let t = j as f64 / subf;
                let (a0, a1) = (ctrl.px(s, 0)[0], ctrl.px(s, 0)[1]);
                let (b0, b1) = (ctrl.px(s, 1)[0], ctrl.px(s, 1)[1]);
                let o = out.px_mut(s, j);
                o[0] = a0 + (b0 - a0) * t;
                o[1] = a1 + (b1 - a1) * t;
            }
            nout[s] = sub + 1;
            continue;
        }
        let mut m: usize = 0;
        for i in 0..k - 1 {
            let i0 = if i > 0 { i - 1 } else { 0 };
            let i3 = if i + 2 < k { i + 2 } else { k - 1 };
            for c in 0..2 {
                let mut p0 = ctrl.px(s, i0)[c];
                let p1 = ctrl.px(s, i)[c];
                let p2 = ctrl.px(s, i + 1)[c];
                let mut p3 = ctrl.px(s, i3)[c];
                if i == 0 {
                    p0 = 2.0 * p1 - p2;
                }
                if i + 2 >= k {
                    p3 = 2.0 * p2 - p1;
                }
                for j in 0..sub as usize {
                    let t = j as f64 / subf;
                    let t2 = t * t;
                    let t3 = t2 * t;
                    out.px_mut(s, m + j)[c] = 0.5
                        * ((2.0 * p1)
                            + (-p0 + p2) * t
                            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3);
                }
            }
            m += sub as usize;
        }
        let (e0, e1) = (ctrl.px(s, k - 1)[0], ctrl.px(s, k - 1)[1]);
        let o = out.px_mut(s, m);
        o[0] = e0;
        o[1] = e1;
        nout[s] = m as i64 + 1;
    }
}
