//! Stage 0 checks for the canvas access type:
//! 1. the same stroke on DenseWet and on TiledWet gives bit-identical canvases
//!    and brush state (the access type is the only thing that differs);
//! 2. paint is conserved: brush + canvas optical depth before = after,
//!    up to float32 rounding (Derek's "tests with a right answer");
//! 3. untouched canvas allocates no tiles.

use pinsel_core::grid::{DenseWet, TiledWet, WetArrays, WetCanvas, TILE};
use pinsel_core::paste::{paste_stroke, Bristles, PasteParams};
use pinsel_core::NB;

/// Tiny deterministic generator so the test needs no crates.
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
        let mut c = Canvas {
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
        };
        // an older wet stroke (id 3) in a band, so the new stroke pushes it
        // down and picks some up
        for y in 40..70 {
            for x in 10..w - 10 {
                let i = y * w + x;
                for k in 0..NB {
                    c.film_k[i * NB + k] = (0.2 + 0.05 * k as f64 * g.next()) as f32;
                    c.film_s[i * NB + k] = (0.1 + 0.02 * g.next()) as f32;
                }
                c.film_vol[i] = (0.6 + 0.3 * g.next()) as f32;
                c.wet[i] = 1.0;
                c.gloss[i] = 0.7;
                c.body[i] = 1.0;
                c.drate[i] = 1.0 / 3600.0;
                c.top_id[i] = 3;
            }
        }
        // masking tape across part of the path
        for y in 0..h {
            for x in 90..100 {
                c.mask[y * w + x] = 1.0;
            }
        }
        c
    }

    fn arrays(&mut self) -> WetArrays<'_> {
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
        }
    }

    fn total_k(&self) -> f64 {
        self.film_k.iter().chain(&self.base_k).map(|v| *v as f64).sum()
    }
}

#[derive(Clone)]
struct Brush {
    boff: Vec<f64>,
    bthr: Vec<f64>,
    bfollow: Vec<f64>,
    brad: Vec<f64>,
    bsplay: Vec<f64>,
    bdep: Vec<f64>,
    tips: Vec<f64>,
    res_k: Vec<f64>,
    res_s: Vec<f64>,
    res_vol: Vec<f64>,
    cap: Vec<f64>,
    tip_k: Vec<f64>,
    tip_s: Vec<f64>,
    tip_vol: Vec<f64>,
}

impl Brush {
    fn new(n: usize, g: &mut Lcg) -> Self {
        let mut b = Brush {
            boff: (0..2 * n).map(|_| g.next() * 2.0 - 1.0).collect(),
            bthr: (0..n).map(|_| g.next() * 0.4).collect(),
            bfollow: (0..n).map(|_| 0.3 + 0.6 * g.next()).collect(),
            brad: (0..n).map(|_| 1.5 + 2.0 * g.next()).collect(),
            bsplay: (0..n).map(|_| g.next()).collect(),
            bdep: (0..n).map(|_| 0.7 + 0.6 * g.next()).collect(),
            tips: vec![f64::NAN; 2 * n],
            res_k: vec![0.0; n * NB],
            res_s: vec![0.0; n * NB],
            res_vol: (0..n).map(|_| 0.8 + 0.4 * g.next()).collect(),
            cap: (0..n).map(|_| 0.8 + 0.4 * g.next()).collect(),
            tip_k: vec![0.0; n * NB],
            tip_s: vec![0.0; n * NB],
            tip_vol: vec![0.0; n],
        };
        for i in 0..n {
            for k in 0..NB {
                b.res_k[i * NB + k] = b.res_vol[i] * (0.5 + 2.0 * (k as f64 / NB as f64));
                b.res_s[i * NB + k] = b.res_vol[i] * 0.8;
            }
        }
        b
    }

    fn view(&mut self) -> Bristles<'_> {
        Bristles {
            boff: &self.boff,
            bthr: &self.bthr,
            bfollow: &self.bfollow,
            brad: &self.brad,
            bsplay: &self.bsplay,
            bdep: &self.bdep,
            tips: &mut self.tips,
            res_k: &mut self.res_k,
            res_s: &mut self.res_s,
            res_vol: &mut self.res_vol,
            cap: &self.cap,
            tip_k: &mut self.tip_k,
            tip_s: &mut self.tip_s,
            tip_vol: &mut self.tip_vol,
        }
    }

    fn total_k(&self) -> f64 {
        self.res_k.iter().chain(&self.tip_k).sum()
    }
}

fn params() -> PasteParams {
    PasteParams {
        ds: 0.5,
        speed: 1.0,
        alpha_tip: 0.4,
        size: 9.0,
        sx0: 0.6,
        sx1: 1.0,
        sy0: 0.25,
        sy1: 0.45,
        sgamma: 0.8,
        splay: 0.6,
        alpha: 0.05,
        pickup: 0.3,
        plough: 0.6,
        contact_soft: 0.35,
        p_gloss: 0.8,
        p_body: 1.0,
        p_drate: 1.0 / 7200.0,
        tooth_fill: 1.2,
        sid: 7,
    }
}

/// A wavy path across several tiles (and across the masked band).
fn path() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let m = 420;
    let mut px = Vec::new();
    let mut py = Vec::new();
    let mut pp = Vec::new();
    let mut pa = Vec::new();
    for j in 0..m {
        let t = j as f64 / (m - 1) as f64;
        px.push(8.0 + 175.0 * t);
        py.push(55.0 + 18.0 * (t * 9.0).sin());
        pp.push(0.15 + 0.85 * (std::f64::consts::PI * t).sin());
        pa.push(0.3 + 0.8 * t);
    }
    (px, py, pp, pa)
}

#[test]
fn dense_and_tiled_give_identical_results() {
    let (h, w) = (150, 190); // partial tiles at the right and bottom edges
    let mut g = Lcg(42);
    let mut dense = Canvas::new(h, w, &mut g);
    let mut tiled_src = dense.clone();
    let brush0 = Brush::new(60, &mut g);
    let (px, py, pp, pa) = path();
    let prm = params();

    // dense run
    let mut b1 = brush0.clone();
    let bbox1 = {
        let (h_, w_) = (dense.h, dense.w);
        let tooth = dense.tooth.clone();
        let mask = dense.mask.clone();
        let mut cv = DenseWet::new(h_, w_, dense.arrays(), &tooth, &mask);
        paste_stroke(&px, &py, &pp, &pa, &mut b1.view(), &prm, &mut cv)
    };

    // tiled run from the same starting canvas
    let mut b2 = brush0.clone();
    let tooth = tiled_src.tooth.clone();
    let mask = tiled_src.mask.clone();
    let mut tiled = TiledWet::from_dense(h, w, &tiled_src.arrays(), &tooth, &mask);
    let tiles_before = tiled.allocated_tiles();
    let bbox2 = paste_stroke(&px, &py, &pp, &pa, &mut b2.view(), &prm, &mut tiled);
    let tiles_after = tiled.allocated_tiles();
    tiled.to_dense(&mut tiled_src.arrays());

    assert_eq!(bbox1, bbox2);
    assert!(bbox1.2 >= 0, "stroke touched nothing");
    let same = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
    assert!(same(&dense.film_k, &tiled_src.film_k), "film_K differs");
    assert!(same(&dense.film_s, &tiled_src.film_s), "film_S differs");
    assert!(same(&dense.base_k, &tiled_src.base_k), "base_K differs");
    assert!(same(&dense.film_vol, &tiled_src.film_vol), "film_vol differs");
    assert!(same(&dense.base_vol, &tiled_src.base_vol), "base_vol differs");
    assert!(same(&dense.wet, &tiled_src.wet), "wet differs");
    assert!(same(&dense.gloss, &tiled_src.gloss), "gloss differs");
    assert_eq!(dense.top_id, tiled_src.top_id, "top_id differs");
    let same64 = |a: &[f64], b: &[f64]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
    assert!(same64(&b1.res_k, &b2.res_k) && same64(&b1.tip_k, &b2.tip_k), "brush state differs");

    // lazy tiles: the stroke spans ~4x2 tiles of a 3x3-tile canvas, plus
    // the pre-existing band; never all of them for this path
    let total_tiles = h.div_ceil(TILE) * w.div_ceil(TILE);
    assert!(tiles_before < total_tiles && tiles_after <= total_tiles);
    println!("tiles: {tiles_before} before, {tiles_after} after, of {total_tiles}");
}

#[test]
fn paint_is_conserved() {
    let (h, w) = (150, 190);
    let mut g = Lcg(7);
    let mut c = Canvas::new(h, w, &mut g);
    let mut b = Brush::new(60, &mut g);
    let (px, py, pp, pa) = path();
    let before = c.total_k() + b.total_k();
    {
        let tooth = c.tooth.clone();
        let mask = c.mask.clone();
        let mut cv = DenseWet::new(h, w, c.arrays(), &tooth, &mask);
        paste_stroke(&px, &py, &pp, &pa, &mut b.view(), &params(), &mut cv);
    }
    let after = c.total_k() + b.total_k();
    let rel = (after - before).abs() / before;
    println!("K before {before:.6}, after {after:.6}, rel {rel:.2e}");
    assert!(rel < 1e-5, "paint not conserved: rel error {rel:e}");
}

#[test]
fn untouched_canvas_allocates_nothing() {
    let (h, w) = (300, 400);
    let t = TiledWet::new(h, w, vec![0.0; h * w], vec![0.0; h * w]);
    assert_eq!(t.allocated_tiles(), 0);
    assert_eq!(t.height(), h);
}
