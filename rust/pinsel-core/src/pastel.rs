//! Dry powdery media: soft pastel, charcoal, kneaded eraser.
//!
//! Line-by-line ports of `pinsel.pastel._pastel_stroke` and
//! `pinsel.pastel._eraser_stroke` (Python 0.1). See docs/RUST_PORTING_GUIDE.md.
//!
//! Number types mirror numba: canvas fields are f32; arithmetic that mixes an
//! f32 element with an f64 scalar happens in f64 and rounds on store; f32 op
//! f32 stays f32 (the base-layer push, `fv`, and the products `wet * fv`).
//! The pastel kernel carries its own 64-bit LCG for grain and dust; the
//! wrapping multiply/add and the bit-to-float conversion are exactly numba's.

use crate::grid::{Plane, PlaneMut, SpecMut, WetCanvas};

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

const LCG_A: u64 = 6364136223846793005;
const LCG_C: u64 = 1442695040888963407;

/// Initial generator state: numba computes `seed * A + C` in int64 (wrapping)
/// and reinterprets it as uint64.
#[inline(always)]
pub fn lcg_init(seed: i64) -> u64 {
    seed.wrapping_mul(LCG_A as i64).wrapping_add(LCG_C as i64) as u64
}

/// One generator step and the uniform value in 0..=1:
/// `float((state >> 33) & 0x7FFFFFFF) / 2147483647.0`.
#[inline(always)]
pub fn lcg_next(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(LCG_A).wrapping_add(LCG_C);
    ((*state >> 33) & 0x7FFF_FFFF) as f64 / 2147483647.0
}

/// Scalar parameters of one pastel stroke (names as in the Python kernel).
#[derive(Clone, Copy, Debug)]
pub struct PastelParams {
    pub rx_base: f64,
    pub ry_base: f64,
    pub hardness: f64,
    pub dust: f64,
    pub amount: f64,
    pub sid: i32,
    pub seed: i64,
}

/// Drag a pastel/charcoal stick along the resampled path. `kpv`/`spv` are the
/// stick's optical depth per unit volume, one value per band. Returns the
/// touched bounding box `(x0, y0, x1, y1)`, or `(W, H, -1, -1)`.
#[allow(clippy::too_many_arguments)]
pub fn pastel_stroke<C: WetCanvas>(
    px: &[f64],
    py: &[f64],
    pp: &[f64],
    pa: &[f64],
    kpv: &[f64],
    spv: &[f64],
    prm: &PastelParams,
    cv: &mut C,
) -> (i64, i64, i64, i64) {
    let h = cv.height() as i64;
    let w = cv.width() as i64;
    let m = px.len();
    assert!(py.len() == m && pp.len() == m && pa.len() == m, "path arrays differ in length");
    let nb = crate::NB;
    assert!(kpv.len() >= nb && spv.len() >= nb, "Kpv/Spv need NB bands");
    let (rx_base, ry_base, hardness, dust, amount) =
        (prm.rx_base, prm.ry_base, prm.hardness, prm.dust, prm.amount);
    let (mut bx0, mut by0, mut bx1, mut by1) = (w, h, -1i64, -1i64);
    let mut state = lcg_init(prm.seed);

    for j in 0..m {
        let cx = px[j];
        let cy = py[j];
        let p = pp[j];
        if p <= 1e-4 {
            continue;
        }
        let a = pa[j];
        let ca = a.cos();
        let sa = a.sin();
        let scale = 0.45 + 0.55 * p;
        let rx = rx_base * scale;
        let ry = ry_base * scale;
        let dust_pad = 1.0 + 0.55 * dust;
        let rmax = (if rx >= ry { rx } else { ry }) * dust_pad + 1.0;
        let x0 = i64::max(0, (cx - rmax).floor() as i64);
        let x1 = (w - 1).min((cx + rmax).ceil() as i64);
        let y0 = i64::max(0, (cy - rmax).floor() as i64);
        let y1 = (h - 1).min((cy + rmax).ceil() as i64);
        if x1 < x0 || y1 < y0 {
            continue;
        }
        bx0 = bx0.min(x0);
        by0 = by0.min(y0);
        bx1 = bx1.max(x1);
        by1 = by1.max(y1);

        // Light pressure catches only high tooth peaks; firm pressure reaches valleys
        let level = 0.95 - 1.95 * p - 0.35 * (1.0 - hardness);
        let soft = 0.35 + 0.35 * (1.0 - hardness);
        let inv_rx2 = 1.0 / (rx * rx + 1e-9);
        let inv_ry2 = 1.0 / (ry * ry + 1e-9);
        let rxd = rx * dust_pad;
        let ryd = ry * dust_pad;
        let inv_drx2 = 1.0 / (rxd * rxd + 1e-9);
        let inv_dry2 = 1.0 / (ryd * ryd + 1e-9);

        for yy in y0..=y1 {
            let dy = yy as f64 - cy;
            for xx in x0..=x1 {
                let (yu, xu) = (yy as usize, xx as usize);
                let msk = 1.0 - cv.mask(yu, xu) as f64;
                if msk <= 0.0 {
                    continue;
                }
                let dx = xx as f64 - cx;
                let lx = dx * ca + dy * sa;
                let ly = -dx * sa + dy * ca;
                let d2 = lx * lx * inv_rx2 + ly * ly * inv_ry2;
                let dv;
                if d2 < 1.0 {
                    let fall = (1.0 - d2).powf(1.3);
                    let t = cv.tooth(yu, xu);
                    let contact = clamp01((t as f64 - level) / soft + 0.5);
                    // crumbly grain variation
                    let rnd = lcg_next(&mut state);
                    let grain = 0.65 + 0.7 * rnd;
                    dv = 0.14 * amount * (0.25 + 0.75 * p) * fall * contact * grain * msk;
                } else if dust > 0.0 {
                    let dd2 = lx * lx * inv_drx2 + ly * ly * inv_dry2;
                    if dd2 >= 1.0 {
                        continue;
                    }
                    let rnd = lcg_next(&mut state);
                    if rnd > 0.28 * dust * (1.0 - dd2) {
                        continue;
                    }
                    dv = 0.035 * amount * dust * (1.0 - dd2) * p * msk;
                } else {
                    continue;
                }

                if dv <= 1e-6 {
                    continue;
                }
                let q = cv.px(yu, xu);
                if *q.top_id != prm.sid {
                    if *q.film_vol as f64 > 0.0 || q.film_k[0] as f64 > 0.0 {
                        for b in 0..nb {
                            q.base_k[b] += q.film_k[b];
                            q.base_s[b] += q.film_s[b];
                            q.film_k[b] = 0.0;
                            q.film_s[b] = 0.0;
                        }
                        *q.base_vol += *q.film_vol;
                        *q.film_vol = 0.0;
                    }
                    *q.top_id = prm.sid;
                }

                let fv = *q.film_vol; // f32
                let nv = fv as f64 + dv;
                for b in 0..nb {
                    q.film_k[b] = (q.film_k[b] as f64 + kpv[b] * dv) as f32;
                    q.film_s[b] = (q.film_s[b] as f64 + spv[b] * dv) as f32;
                }
                // Matte, unfixed powdery deposit; the products wet*fv etc. are f32*f32.
                *q.wet = (((*q.wet * fv) as f64 + 1.0 * dv) / nv) as f32;
                *q.gloss = (((*q.gloss * fv) as f64 + 0.0 * dv) / nv) as f32;
                *q.body = (((*q.body * fv) as f64 + 0.04 * dv) / nv) as f32;
                *q.drate = (((*q.drate * fv) as f64 + 0.0 * dv) / nv) as f32;
                *q.film_vol = nv as f32;
            }
        }
    }
    (bx0, by0, bx1, by1)
}

/// Kneaded eraser: lifts a fraction of the film and base pigment under a
/// round, soft footprint (respecting `mask`). Returns the touched box.
#[allow(clippy::too_many_arguments)]
pub fn eraser_stroke(
    px: &[f64],
    py: &[f64],
    pp: &[f64],
    r_base: f64,
    strength: f64,
    film_k: &mut SpecMut<'_, f32>,
    film_s: &mut SpecMut<'_, f32>,
    film_vol: &mut PlaneMut<'_, f32>,
    base_k: &mut SpecMut<'_, f32>,
    base_s: &mut SpecMut<'_, f32>,
    base_vol: &mut PlaneMut<'_, f32>,
    mask: &Plane<'_, f32>,
) -> (i64, i64, i64, i64) {
    let h = film_vol.height() as i64;
    let w = film_vol.width() as i64;
    let nb = film_k.channels();
    let m = px.len();
    assert!(py.len() == m && pp.len() == m, "path arrays differ in length");
    let (mut bx0, mut by0, mut bx1, mut by1) = (w, h, -1i64, -1i64);
    for j in 0..m {
        let cx = px[j];
        let cy = py[j];
        let p = pp[j];
        if p <= 1e-4 {
            continue;
        }
        let r = r_base * (0.5 + 0.5 * p);
        let r2 = r * r;
        let x0 = i64::max(0, (cx - r).floor() as i64);
        let x1 = (w - 1).min((cx + r).ceil() as i64);
        let y0 = i64::max(0, (cy - r).floor() as i64);
        let y1 = (h - 1).min((cy + r).ceil() as i64);
        if x1 < x0 || y1 < y0 {
            continue;
        }
        bx0 = bx0.min(x0);
        by0 = by0.min(y0);
        bx1 = bx1.max(x1);
        by1 = by1.max(y1);
        for yy in y0..=y1 {
            let dy = yy as f64 - cy;
            for xx in x0..=x1 {
                let dx = xx as f64 - cx;
                let d2 = dx * dx + dy * dy;
                if d2 >= r2 {
                    continue;
                }
                let (yu, xu) = (yy as usize, xx as usize);
                let msk = 1.0 - mask.at(yu, xu) as f64;
                if msk <= 0.0 {
                    continue;
                }
                let fall = 1.0 - d2 / r2;
                let lift = clamp01(strength * p * fall * fall * msk * 0.45);
                if lift <= 1e-6 {
                    continue;
                }
                let keep = 1.0 - lift;
                if film_vol.at(yu, xu) as f64 > 0.0 {
                    for b in 0..nb {
                        let k = &mut film_k.px_mut(yu, xu)[b];
                        *k = (*k as f64 * keep) as f32;
                        let s = &mut film_s.px_mut(yu, xu)[b];
                        *s = (*s as f64 * keep) as f32;
                    }
                    let v = film_vol.get_mut(yu, xu);
                    *v = (*v as f64 * keep) as f32;
                }
                if base_vol.at(yu, xu) as f64 > 0.0 {
                    for b in 0..nb {
                        let k = &mut base_k.px_mut(yu, xu)[b];
                        *k = (*k as f64 * keep) as f32;
                        let s = &mut base_s.px_mut(yu, xu)[b];
                        *s = (*s as f64 * keep) as f32;
                    }
                    let v = base_vol.get_mut(yu, xu);
                    *v = (*v as f64 * keep) as f32;
                }
            }
        }
    }
    (bx0, by0, bx1, by1)
}
