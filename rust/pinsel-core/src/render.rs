//! Rendering and drying: `pinsel._kernels.render_reflectance`, `bake_film`,
//! `bake_deposit`.
//!
//! Number types mirror numba: canvas arrays are f32; the reflectance is
//! carried in f64 (km_over returns f64); film K and S at a pixel are summed
//! as f32 (film + suspended watercolour pigment) before km_over, as numba does.
//! Every pixel is independent, so rows run in parallel without changing any
//! result.

use crate::grid::{Plane, PlaneMut, Spec, SpecMut, WetCanvas};
use crate::spectral::km_over;

/// The watercolour layers render needs (present only when the canvas has
/// watercolour state).
#[derive(Clone, Copy)]
pub struct WcLayers<'a> {
    /// Suspended pigment.
    pub k: Spec<'a, f32>,
    pub s: Spec<'a, f32>,
    /// Deposited pigment.
    pub dk: Spec<'a, f32>,
    pub ds: Spec<'a, f32>,
    pub water: Plane<'a, f32>,
    pub damp: Plane<'a, f32>,
}

/// Composite all layers with Kubelka-Munk and project to linear sRGB with the
/// 3 x NB matrix `m` (row-major), writing (H, W, 3) into `out`.
#[allow(clippy::too_many_arguments)]
pub fn render_reflectance(
    below_r: Spec<'_, f32>,
    base_k: Spec<'_, f32>,
    base_s: Spec<'_, f32>,
    film_k: Spec<'_, f32>,
    film_s: Spec<'_, f32>,
    wc: Option<WcLayers<'_>>,
    m: &[f64],
    out: &mut SpecMut<'_, f32>,
) {
    let w = below_r.width();
    let nb = below_r.channels();
    assert_eq!(m.len(), 3 * nb, "matrix must be 3 x NB");
    assert_eq!(out.channels(), 3);
    out.par_rows(|y, row| {
        for x in 0..w {
            let mut r = 0.0f64;
            let mut g = 0.0f64;
            let mut bl = 0.0f64;
            let mut wetd = 0.0f64;
            if let Some(wc) = &wc {
                wetd = wc.water.at(y, x) as f64 + 0.5 * wc.damp.at(y, x) as f64;
                if wetd > 1.0 {
                    wetd = 1.0;
                }
            }
            let br = below_r.px(y, x);
            let bk = base_k.px(y, x);
            let bs = base_s.px(y, x);
            let fk = film_k.px(y, x);
            let fs = film_s.px(y, x);
            for k in 0..nb {
                let mut rr = br[k] as f64;
                if let Some(wc) = &wc {
                    let dk = wc.dk.px(y, x)[k];
                    let dsg = wc.ds.px(y, x)[k];
                    if (dk + dsg) as f64 > 1e-6 {
                        rr = km_over(dk as f64, dsg as f64, rr);
                    }
                }
                if (bk[k] + bs[k]) as f64 > 1e-6 {
                    rr = km_over(bk[k] as f64, bs[k] as f64, rr);
                }
                let mut kk = fk[k];
                let mut ss = fs[k];
                if let Some(wc) = &wc {
                    kk += wc.k.px(y, x)[k];
                    ss += wc.s.px(y, x)[k];
                }
                if (kk + ss) as f64 > 1e-6 {
                    rr = km_over(kk as f64, ss as f64, rr);
                }
                if wetd > 0.0 {
                    rr = rr.powf(1.0 + 0.45 * wetd) * (1.0 - 0.05 * wetd);
                }
                r += m[k] * rr;
                g += m[nb + k] * rr;
                bl += m[2 * nb + k] * rr;
            }
            row[3 * x] = r as f32;
            row[3 * x + 1] = g as f32;
            row[3 * x + 2] = bl as f32;
        }
    });
}

/// Dry the wet film where `sel` is true: fold base and film into `below_r`,
/// add the dried height to `h_dry`, blend `gloss_dry`, clear the film.
pub fn bake_film<C: WetCanvas>(
    sel: Plane<'_, bool>,
    below_r: &mut SpecMut<'_, f32>,
    cv: &mut C,
    h_dry: &mut PlaneMut<'_, f32>,
    gloss_dry: &mut PlaneMut<'_, f32>,
    gloss_dry_factor: f64,
) {
    let (h, w) = (cv.height(), cv.width());
    for y in 0..h {
        for x in 0..w {
            if !sel.at(y, x) {
                continue;
            }
            let q = cv.px(y, x);
            let br = below_r.px_mut(y, x);
            let nb = br.len();
            let mut btot = 0.0f64;
            for k in 0..nb {
                btot += (q.base_k[k] + q.base_s[k]) as f64;
            }
            if btot > 1e-7 {
                for k in 0..nb {
                    br[k] = km_over(q.base_k[k] as f64, q.base_s[k] as f64, br[k] as f64) as f32;
                }
            }
            let mut tot = 0.0f64;
            for k in 0..nb {
                tot += (q.film_k[k] + q.film_s[k]) as f64;
            }
            let fv = *q.film_vol + *q.base_vol; // f32 + f32, as in numba
            if tot > 1e-7 {
                for k in 0..nb {
                    br[k] = km_over(q.film_k[k] as f64, q.film_s[k] as f64, br[k] as f64) as f32;
                }
            }
            if fv as f64 > 1e-7 {
                let hd = fv * *q.body; // f32 * f32
                let old = h_dry.at(y, x);
                h_dry.set(y, x, old + hd);
                // dried gloss: thick fresh layers dominate
                let wgt = fv as f64 / (fv as f64 + 0.3);
                let gd = gloss_dry.at(y, x) as f64 * (1.0 - wgt) + *q.gloss as f64 * gloss_dry_factor * wgt;
                gloss_dry.set(y, x, gd as f32);
            }
            for k in 0..nb {
                q.film_k[k] = 0.0;
                q.film_s[k] = 0.0;
                q.base_k[k] = 0.0;
                q.base_s[k] = 0.0;
            }
            *q.film_vol = 0.0;
            *q.base_vol = 0.0;
            *q.wet = 0.0;
            *q.gloss = 0.0;
            *q.body = 0.0;
            *q.drate = 0.0;
        }
    }
}

/// Fold deposited watercolour pigment into `below_r` where `sel` is true,
/// and clear the deposit there.
pub fn bake_deposit(
    sel: Plane<'_, bool>,
    below_r: &mut SpecMut<'_, f32>,
    dk: &mut SpecMut<'_, f32>,
    ds: &mut SpecMut<'_, f32>,
) {
    let (h, w) = (below_r.height(), below_r.width());
    for y in 0..h {
        for x in 0..w {
            if !sel.at(y, x) {
                continue;
            }
            let br = below_r.px_mut(y, x);
            let pk = dk.px_mut(y, x);
            let ps = ds.px_mut(y, x);
            for k in 0..br.len() {
                if (pk[k] + ps[k]) as f64 > 1e-7 {
                    br[k] = km_over(pk[k] as f64, ps[k] as f64, br[k] as f64) as f32;
                }
                pk[k] = 0.0;
                ps[k] = 0.0;
            }
        }
    }
}
