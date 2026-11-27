//! Synthetic-input checks for the pastel kernels (no Python):
//! * the generator reproduces the 64-bit LCG recurrence;
//! * pastel deposits conserve pigment (canvas gain = Kpv * volume gain, per
//!   band) including the push of an older stroke into the base layer;
//! * the same stroke on DenseWet and on TiledWet is bit-identical;
//! * masked pixels are untouched; seeds change the dust pattern;
//! * the eraser removes exactly what its footprint says (removed from the
//!   canvas = taken up), per band, and never touches masked pixels.

use pinsel_core::grid::{DenseWet, Plane, PlaneMut, SpecMut, TiledWet, WetArrays, WetCanvas};
use pinsel_core::pastel::{eraser_stroke, lcg_init, lcg_next, pastel_stroke, PastelParams};
use pinsel_core::NB;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

#[derive(Clone)]
struct Canvas {
    h: usize,
    w: usize,
    film_k: Vec<f32>,
    film_s: Vec<f32>,
    base_k: Vec<f32>,
    base_s: Vec<f32>,
    film_vol: Vec<f32>,
    base_vol: Vec<f32>,
    wet: Vec<f32>,
    gloss: Vec<f32>,
    body: Vec<f32>,
    drate: Vec<f32>,
    top_id: Vec<i32>,
    tooth: Vec<f32>,
    mask: Vec<f32>,
}

impl Canvas {
    fn new(h: usize, w: usize, g: &mut Lcg) -> Self {
        let n = h * w;
        Canvas {
            h,
            w,
            film_k: vec![0.0; n * NB],
            film_s: vec![0.0; n * NB],
            base_k: vec![0.0; n * NB],
            base_s: vec![0.0; n * NB],
            film_vol: vec![0.0; n],
            base_vol: vec![0.0; n],
            wet: vec![0.0; n],
            gloss: vec![0.0; n],
            body: vec![0.0; n],
            drate: vec![0.0; n],
            top_id: vec![0; n],
            tooth: (0..n).map(|_| (g.next() * 3.2 - 1.6) as f32).collect(),
            mask: vec![0.0; n],
        }
    }

    fn dense(&mut self) -> DenseWet<'_> {
        DenseWet::new(
            self.h,
            self.w,
            WetArrays {
                film_k: &mut self.film_k,
                film_s: &mut self.film_s,
                base_k: &mut self.base_k,
                base_s: &mut self.base_s,
                film_vol: &mut self.film_vol,
                base_vol: &mut self.base_vol,
                wet: &mut self.wet,
                gloss: &mut self.gloss,
                body: &mut self.body,
                drate: &mut self.drate,
                top_id: &mut self.top_id,
            },
            &self.tooth,
            &self.mask,
        )
    }

    /// Per-band totals of K and S over film and base, and total volume.
    fn totals(&self) -> ([f64; NB], [f64; NB], f64) {
        let (mut k, mut s) = ([0.0; NB], [0.0; NB]);
        for i in 0..self.h * self.w {
            for b in 0..NB {
                k[b] += self.film_k[i * NB + b] as f64 + self.base_k[i * NB + b] as f64;
                s[b] += self.film_s[i * NB + b] as f64 + self.base_s[i * NB + b] as f64;
            }
        }
        let v = self.film_vol.iter().chain(&self.base_vol).map(|v| *v as f64).sum();
        (k, s, v)
    }
}

fn path(n: usize, x0: f64, y0: f64, x1: f64, y1: f64, p: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let px: Vec<f64> = (0..n).map(|i| x0 + (x1 - x0) * i as f64 / (n - 1) as f64).collect();
    let py: Vec<f64> = (0..n).map(|i| y0 + (y1 - y0) * i as f64 / (n - 1) as f64).collect();
    let pp: Vec<f64> = (0..n).map(|i| p * (0.4 + 0.6 * (i as f64 / (n - 1) as f64))).collect();
    let ang = (y1 - y0).atan2(x1 - x0) + std::f64::consts::FRAC_PI_2;
    (px, py, pp, vec![ang; n])
}

fn kpv_spv() -> (Vec<f64>, Vec<f64>) {
    let k: Vec<f64> = (0..NB).map(|b| 0.5 + 0.13 * b as f64).collect();
    let s: Vec<f64> = (0..NB).map(|b| 2.0 - 0.07 * b as f64).collect();
    (k, s)
}

fn prm(seed: i64, sid: i32, dust: f64) -> PastelParams {
    PastelParams { rx_base: 5.0, ry_base: 5.0, hardness: 0.35, dust, amount: 1.0, sid, seed }
}

#[test]
fn lcg_matches_the_recurrence() {
    const A: u128 = 6364136223846793005;
    const C: u128 = 1442695040888963407;
    const M: u128 = 1u128 << 64;
    let seed: i64 = 1_234_567_891;
    let mut want = ((seed as u128 * A + C) % M) as u64;
    let mut st = lcg_init(seed);
    assert_eq!(st, want);
    for _ in 0..1000 {
        want = ((want as u128 * A + C) % M) as u64;
        let u = lcg_next(&mut st);
        assert_eq!(st, want);
        assert_eq!(u, (((want >> 33) & 0x7FFF_FFFF) as f64) / 2147483647.0);
        assert!((0.0..=1.0).contains(&u));
    }
    // a seed whose product overflows int64 still wraps like numba's int64
    assert_eq!(lcg_init(i64::MAX), ((i64::MAX as u128 * A + C) % M) as u64);
}

#[test]
fn pastel_conserves_pigment_and_pushes_older_stroke() {
    let mut g = Lcg(5);
    let mut c = Canvas::new(48, 64, &mut g);
    let (kpv, spv) = kpv_spv();
    // stroke 1, then stroke 2 across it (older film is pushed into the base)
    let (px, py, pp, pa) = path(20, 8.0, 14.0, 54.0, 30.0, 0.9);
    let bb1 = pastel_stroke(&px, &py, &pp, &pa, &kpv, &spv, &prm(11, 1, 0.5), &mut c.dense());
    let (k1, s1, v1) = c.totals();
    assert!(bb1.2 >= bb1.0 && bb1.3 >= bb1.1, "stroke 1 touched nothing");
    assert!(v1 > 0.0);
    for b in 0..NB {
        let (want_k, want_s) = (kpv[b] * v1, spv[b] * v1);
        assert!((k1[b] - want_k).abs() <= 2e-5 * want_k, "band {b}: K {} vs {}", k1[b], want_k);
        assert!((s1[b] - want_s).abs() <= 2e-5 * want_s, "band {b}: S {} vs {}", s1[b], want_s);
    }
    let (px, py, pp, pa) = path(20, 10.0, 34.0, 56.0, 12.0, 0.8);
    pastel_stroke(&px, &py, &pp, &pa, &kpv, &spv, &prm(12, 2, 0.5), &mut c.dense());
    let (k2, s2, v2) = c.totals();
    assert!(v2 > v1);
    assert!(c.base_vol.iter().any(|v| *v > 0.0), "older stroke was not pushed into the base layer");
    for b in 0..NB {
        let (dk, ds) = (k2[b] - k1[b], s2[b] - s1[b]);
        let (want_k, want_s) = (kpv[b] * (v2 - v1), spv[b] * (v2 - v1));
        assert!((dk - want_k).abs() <= 1e-4 * want_k, "band {b}: dK {dk} vs {want_k}");
        assert!((ds - want_s).abs() <= 1e-4 * want_s, "band {b}: dS {ds} vs {want_s}");
    }
    // every pixel holds the id of the last stroke that touched it
    assert!(c.top_id.iter().all(|t| *t == 0 || *t == 1 || *t == 2));
    // wet/gloss/body stay in their physical range for a matte deposit
    assert!(c.wet.iter().all(|v| (0.0..=1.0001).contains(v)));
    assert!(c.gloss.iter().all(|v| v.abs() < 1e-6));
}

#[test]
fn pastel_respects_mask_and_bbox() {
    let mut g = Lcg(9);
    let mut c = Canvas::new(40, 40, &mut g);
    for y in 0..40 {
        for x in 0..20 {
            c.mask[y * 40 + x] = 1.0; // left half fully protected
        }
    }
    let (kpv, spv) = kpv_spv();
    let (px, py, pp, pa) = path(12, 4.0, 20.0, 36.0, 20.0, 1.0);
    let bb = pastel_stroke(&px, &py, &pp, &pa, &kpv, &spv, &prm(3, 1, 0.8), &mut c.dense());
    assert!(bb.0 <= bb.2 && bb.1 <= bb.3);
    for y in 0..40usize {
        for x in 0..40usize {
            let i = y * 40 + x;
            let inside = (x as i64) >= bb.0 && (x as i64) <= bb.2 && (y as i64) >= bb.1 && (y as i64) <= bb.3;
            if x < 20 || !inside {
                assert_eq!(c.film_vol[i], 0.0, "painted outside bbox or under mask at ({x},{y})");
            }
        }
    }
    assert!(c.film_vol.iter().any(|v| *v > 0.0));
}

#[test]
fn pastel_empty_or_light_path_touches_nothing() {
    let mut g = Lcg(2);
    let mut c = Canvas::new(16, 16, &mut g);
    let (kpv, spv) = kpv_spv();
    let bb = pastel_stroke(&[8.0], &[8.0], &[0.0], &[0.0], &kpv, &spv, &prm(1, 1, 0.3), &mut c.dense());
    assert_eq!(bb, (16, 16, -1, -1));
    assert!(c.film_vol.iter().all(|v| *v == 0.0));
}

#[test]
fn pastel_seed_changes_dust_and_is_repeatable() {
    let (kpv, spv) = kpv_spv();
    let run = |seed: i64| {
        let mut g = Lcg(5);
        let mut c = Canvas::new(40, 40, &mut g);
        let (px, py, pp, pa) = path(10, 6.0, 20.0, 34.0, 20.0, 0.8);
        pastel_stroke(&px, &py, &pp, &pa, &kpv, &spv, &prm(seed, 1, 0.9), &mut c.dense());
        c.film_vol
    };
    let a = run(7);
    assert_eq!(a, run(7));
    assert_ne!(a, run(8));
}

#[test]
fn pastel_tiled_equals_dense() {
    let mut g = Lcg(5);
    let mut c = Canvas::new(70, 90, &mut g); // crosses tile edges
    let (kpv, spv) = kpv_spv();
    let paths = [(4.0, 10.0, 80.0, 40.0, 1i32), (10.0, 60.0, 85.0, 20.0, 2i32)];
    let mut t = TiledWet::new(c.h, c.w, c.tooth.clone(), c.mask.clone());
    for (x0, y0, x1, y1, sid) in paths {
        let (px, py, pp, pa) = path(30, x0, y0, x1, y1, 0.85);
        let p = prm(40 + sid as i64, sid, 0.6);
        let b1 = pastel_stroke(&px, &py, &pp, &pa, &kpv, &spv, &p, &mut c.dense());
        let b2 = pastel_stroke(&px, &py, &pp, &pa, &kpv, &spv, &p, &mut t);
        assert_eq!(b1, b2);
    }
    let mut d = Canvas::new(70, 90, &mut Lcg(1));
    let mut arrays = WetArrays {
        film_k: &mut d.film_k,
        film_s: &mut d.film_s,
        base_k: &mut d.base_k,
        base_s: &mut d.base_s,
        film_vol: &mut d.film_vol,
        base_vol: &mut d.base_vol,
        wet: &mut d.wet,
        gloss: &mut d.gloss,
        body: &mut d.body,
        drate: &mut d.drate,
        top_id: &mut d.top_id,
    };
    t.to_dense(&mut arrays);
    assert_eq!(c.film_k, d.film_k);
    assert_eq!(c.film_s, d.film_s);
    assert_eq!(c.base_k, d.base_k);
    assert_eq!(c.film_vol, d.film_vol);
    assert_eq!(c.base_vol, d.base_vol);
    assert_eq!(c.wet, d.wet);
    assert_eq!(c.top_id, d.top_id);
}

#[test]
fn eraser_removes_exactly_the_footprint() {
    let (h, w) = (40usize, 50usize);
    let n = h * w;
    // paint with a fixed K/S per unit volume, so removed K = Kc * removed vol
    let kc: Vec<f32> = (0..NB).map(|b| 0.3 + 0.1 * b as f32).collect();
    let sc: Vec<f32> = (0..NB).map(|b| 1.5 - 0.05 * b as f32).collect();
    let mut g = Lcg(77);
    let mut film_vol: Vec<f32> = (0..n).map(|_| (0.2 + g.next()) as f32).collect();
    let mut base_vol: Vec<f32> = (0..n).map(|i| if i % 3 == 0 { 0.0 } else { (0.1 + g.next()) as f32 }).collect();
    let spread = |vol: &[f32], c: &[f32]| -> Vec<f32> {
        (0..n * NB).map(|i| vol[i / NB] * c[i % NB]).collect()
    };
    let mut film_k = spread(&film_vol, &kc);
    let mut film_s = spread(&film_vol, &sc);
    let mut base_k = spread(&base_vol, &kc);
    let mut base_s = spread(&base_vol, &sc);
    let mut mask = vec![0.0f32; n];
    for y in 0..h {
        for x in 24..30 {
            mask[y * w + x] = 1.0; // a protected stripe across the stroke
        }
    }
    let before = (film_vol.clone(), base_vol.clone(), film_k.clone(), base_k.clone());
    let (r_base, strength) = (6.0, 0.8);
    let (px, py, pp) = (vec![12.0, 24.5, 33.0], vec![18.0, 20.0, 22.5], vec![0.9, 1.0, 0.6]);
    let bb = eraser_stroke(
        &px,
        &py,
        &pp,
        r_base,
        strength,
        &mut SpecMut::new(h, w, NB, &mut film_k),
        &mut SpecMut::new(h, w, NB, &mut film_s),
        &mut PlaneMut::new(h, w, &mut film_vol),
        &mut SpecMut::new(h, w, NB, &mut base_k),
        &mut SpecMut::new(h, w, NB, &mut base_s),
        &mut PlaneMut::new(h, w, &mut base_vol),
        &Plane::new(h, w, &mask),
    );
    assert!(bb.0 <= bb.2 && bb.1 <= bb.3);

    // independent expectation of the footprint, from the formula
    let mut expect_film_removed = 0.0f64;
    let mut expect_base_removed = 0.0f64;
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let mut keep = 1.0f64;
            for j in 0..3 {
                let r = r_base * (0.5 + 0.5 * pp[j]);
                let (dx, dy) = (x as f64 - px[j], y as f64 - py[j]);
                let d2 = dx * dx + dy * dy;
                if d2 >= r * r || mask[i] >= 1.0 {
                    continue;
                }
                let fall = 1.0 - d2 / (r * r);
                let lift = (strength * pp[j] * fall * fall * (1.0 - mask[i] as f64) * 0.45).clamp(0.0, 1.0);
                if lift > 1e-6 {
                    keep *= 1.0 - lift;
                }
            }
            expect_film_removed += before.0[i] as f64 * (1.0 - keep);
            expect_base_removed += before.1[i] as f64 * (1.0 - keep);
            // never adds paint, protected pixels untouched
            assert!(film_vol[i] <= before.0[i] && base_vol[i] <= before.1[i]);
            if mask[i] >= 1.0 {
                assert_eq!(film_vol[i], before.0[i]);
                assert_eq!(base_vol[i], before.1[i]);
            }
        }
    }
    let removed_film: f64 = (0..n).map(|i| (before.0[i] - film_vol[i]) as f64).sum();
    let removed_base: f64 = (0..n).map(|i| (before.1[i] - base_vol[i]) as f64).sum();
    assert!(removed_film > 0.0 && removed_base > 0.0);
    assert!((removed_film - expect_film_removed).abs() <= 1e-4 * expect_film_removed);
    assert!((removed_base - expect_base_removed).abs() <= 1e-4 * expect_base_removed);
    // per band: removed optical depth = Kc[b] (Sc[b]) times removed volume
    for b in 0..NB {
        let rk: f64 = (0..n)
            .map(|i| (before.2[i * NB + b] + before.3[i * NB + b] - film_k[i * NB + b] - base_k[i * NB + b]) as f64)
            .sum();
        let want = kc[b] as f64 * (removed_film + removed_base);
        assert!((rk - want).abs() <= 1e-4 * want, "band {b}: removed K {rk} vs {want}");
    }
    let rs: Vec<f64> = (0..NB)
        .map(|b| {
            (0..n)
                .map(|i| {
                    let (fs, bs) = (film_s[i * NB + b] as f64, base_s[i * NB + b] as f64);
                    let (vf, vb) = (film_vol[i] as f64, base_vol[i] as f64);
                    // remaining S of this band, relative to its volume, is unchanged
                    (fs - sc[b] as f64 * vf).abs() + (bs - sc[b] as f64 * vb).abs()
                })
                .sum()
        })
        .collect();
    assert!(rs.iter().all(|e| *e < 1e-2), "S per volume drifted: {rs:?}");
}

#[test]
fn eraser_light_path_touches_nothing() {
    let (h, w) = (8usize, 8usize);
    let mut a = vec![1.0f32; h * w * NB];
    let mut b = a.clone();
    let mut c = a.clone();
    let mut d = a.clone();
    let mut fv = vec![1.0f32; h * w];
    let mut bv = fv.clone();
    let mask = vec![0.0f32; h * w];
    let bb = eraser_stroke(
        &[4.0],
        &[4.0],
        &[0.0],
        3.0,
        1.0,
        &mut SpecMut::new(h, w, NB, &mut a),
        &mut SpecMut::new(h, w, NB, &mut b),
        &mut PlaneMut::new(h, w, &mut fv),
        &mut SpecMut::new(h, w, NB, &mut c),
        &mut SpecMut::new(h, w, NB, &mut d),
        &mut PlaneMut::new(h, w, &mut bv),
        &Plane::new(h, w, &mask),
    );
    assert_eq!(bb, (8, 8, -1, -1));
    assert!(fv.iter().all(|v| *v == 1.0));
}
