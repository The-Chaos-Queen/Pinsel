//! Tests with a right answer for the film tools (docs/RUST_PORTING_GUIDE.md rule 9):
//! synthetic inputs per kernel, no Python, and paint conservation (in = out per
//! band, to float32 rounding) for every kernel that moves paint: knife spread
//! and scrape, sgraffito (carry and burr), smudge, oil crayon, and the two
//! deposit maps. Bit-identity against numba is checked by rust/harness/verify.sh.

use pinsel_core::grid::{DenseWet, Plane, PlaneMut, Spec, SpecMut, WetArrays};
use pinsel_core::tools_common as tc;
use pinsel_core::tools_film as tf;
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
    h_ground: Vec<f32>,
    h_dry: Vec<f32>,
}

impl Canvas {
    /// A bare canvas with a woven ground (h_ground ~ 0.2..0.8) and nothing wet.
    fn bare(h: usize, w: usize, g: &mut Lcg) -> Self {
        let n = h * w;
        let mut hg = vec![0.0f32; n];
        let mut tooth = vec![0.0f32; n];
        for y in 0..h {
            for x in 0..w {
                let wave = 0.5 + 0.25 * ((x as f64 * 0.9).sin() * (y as f64 * 0.8).cos());
                let v = wave + 0.1 * g.next();
                hg[y * w + x] = v as f32;
                tooth[y * w + x] = ((v - 0.5) * 3.0).clamp(-1.6, 1.6) as f32;
            }
        }
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
            tooth,
            mask: vec![0.0; n],
            h_ground: hg,
            h_dry: vec![0.0; n],
        }
    }

    /// Wet paint (older stroke id 3) in a rectangle: thick top film plus a little base.
    fn with_wet_band(mut self, y0: usize, y1: usize, x0: usize, x1: usize, g: &mut Lcg, vol: f64) -> Self {
        for y in y0..y1 {
            for x in x0..x1 {
                let i = y * self.w + x;
                for k in 0..NB {
                    self.film_k[i * NB + k] = ((0.2 + 0.05 * k as f64) * vol * (0.8 + 0.4 * g.next())) as f32;
                    self.film_s[i * NB + k] = (0.1 * vol * (0.8 + 0.4 * g.next())) as f32;
                    self.base_k[i * NB + k] = (0.02 * k as f64 * g.next()) as f32;
                    self.base_s[i * NB + k] = (0.01 * g.next()) as f32;
                }
                self.film_vol[i] = (vol * (0.8 + 0.4 * g.next())) as f32;
                self.base_vol[i] = (0.1 * g.next()) as f32;
                self.wet[i] = (0.6 + 0.4 * g.next()) as f32;
                self.gloss[i] = 0.7;
                self.body[i] = 1.0;
                self.drate[i] = 1.0 / 3600.0;
                self.top_id[i] = 3;
            }
        }
        self
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

    /// Wet-film totals per band: (K, S) of top + base, and volume.
    fn totals(&self) -> ([f64; NB], [f64; NB], f64) {
        let mut k = [0.0; NB];
        let mut s = [0.0; NB];
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

fn assert_conserved(what: &str, before: &([f64; NB], [f64; NB], f64), after: &([f64; NB], [f64; NB], f64)) {
    for b in 0..NB {
        let rk = (after.0[b] - before.0[b]).abs() / before.0[b].max(1e-9);
        let rs = (after.1[b] - before.1[b]).abs() / before.1[b].max(1e-9);
        assert!(rk < 1e-5, "{what}: K band {b} not conserved, rel {rk:e}");
        assert!(rs < 1e-5, "{what}: S band {b} not conserved, rel {rs:e}");
    }
    let rv = (after.2 - before.2).abs() / before.2.max(1e-9);
    println!("{what}: volume {:.6} -> {:.6}, rel {rv:.2e}", before.2, after.2);
    assert!(rv < 1e-5, "{what}: volume not conserved, rel {rv:e}");
}

/// Add a reservoir's totals to the canvas totals.
fn plus(t: &([f64; NB], [f64; NB], f64), rk: &[f64], rs: &[f64], rv: &[f64]) -> ([f64; NB], [f64; NB], f64) {
    let (mut k, mut s, mut v) = *t;
    for (i, x) in rk.iter().enumerate() {
        k[i % NB] += x;
    }
    for (i, x) in rs.iter().enumerate() {
        s[i % NB] += x;
    }
    v += rv.iter().sum::<f64>();
    (k, s, v)
}

fn line(n: usize, a: (f64, f64), b: (f64, f64), p: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut px = vec![];
    let mut py = vec![];
    let mut pp = vec![];
    let mut ps = vec![];
    let mut ang = vec![];
    let phi = (b.1 - a.1).atan2(b.0 - a.0);
    let len = ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt();
    for j in 0..n {
        let t = j as f64 / (n - 1) as f64;
        px.push(a.0 + (b.0 - a.0) * t);
        py.push(a.1 + (b.1 - a.1) * t);
        pp.push(p);
        ps.push(len * t);
        ang.push(phi + 80f64.to_radians());
    }
    (px, py, pp, ps, ang)
}

// --------------------------------------------------------------------------
// shared helpers
// --------------------------------------------------------------------------

#[test]
fn helper_functions_have_their_defining_properties() {
    assert_eq!(tc::cl01(-0.3), 0.0);
    assert_eq!(tc::cl01(1.7), 1.0);
    assert_eq!(tc::cl01(0.25), 0.25);
    assert_eq!(tc::sstep(0.0), 0.0);
    assert_eq!(tc::sstep(1.0), 1.0);
    assert!((tc::sstep(0.5) - 0.5).abs() < 1e-15);
    // surface_top with no paint equals solid_top
    let (hg, hd) = (0.6f32, 0.2f32);
    assert_eq!(tc::surface_top(hg, hd, 0.0, 1.0), tc::solid_top(hg, hd));
    // the solved paint height reproduces the blade level B
    for &(hg, b, lo) in &[(0.5f64, 1.4f64, 0.1f64), (0.3, 0.9, 0.0), (0.8, 2.5, 0.4)] {
        let ph = tc::solve_paint_height(hg, b, lo);
        assert!(ph >= lo);
        let surface = hg * (-tc::LEVELING * ph).exp() + ph;
        assert!((surface - b).abs() < 1e-9, "root {ph}: surface {surface} vs B {b}");
    }
    // a blade below the ground: all wet paint goes
    assert_eq!(tc::solve_paint_height(0.5, 0.05, 0.1), 0.1);
}

#[test]
fn dep_and_take_round_trip() {
    let mut g = Lcg(5);
    let mut c = Canvas::bare(4, 4, &mut g);
    let mask = c.mask.clone();
    let (h, w) = (c.h, c.w);
    let mut cv = DenseWet::new(h, w, c.arrays(), &mask, &mask);
    let kpv: Vec<f64> = (0..NB).map(|b| 0.3 + 0.1 * b as f64).collect();
    let spv = vec![0.2f64; NB];
    use pinsel_core::grid::WetCanvas;
    tc::dep(&mut cv.px(1, 2), 9, 0.8, &kpv, &spv, 0.8, 1.0, 1e-4);
    {
        let q = cv.px(1, 2);
        assert_eq!(*q.top_id, 9);
        assert!((*q.film_vol as f64 - 0.8).abs() < 1e-6);
        assert!((*q.wet as f64 - 1.0).abs() < 1e-6);
        assert!((q.film_k[3] as f64 - 0.6 * 0.8).abs() < 1e-6);
    }
    // a second mark pushes the first down into the base layer
    tc::dep(&mut cv.px(1, 2), 10, 0.5, &kpv, &spv, 0.8, 1.0, 1e-4);
    {
        let q = cv.px(1, 2);
        assert!((*q.base_vol as f64 - 0.8).abs() < 1e-6);
        assert!((*q.film_vol as f64 - 0.5).abs() < 1e-6);
    }
    let mut rk = [0.0f64; NB];
    let mut rs = [0.0f64; NB];
    let got = tc::take(&mut cv.px(1, 2), 1.0, &mut rk, &mut rs);
    assert!((got - 1.0).abs() < 1e-6);
    let q = cv.px(1, 2);
    assert!((*q.film_vol as f64).abs() < 1e-6, "top layer goes first");
    assert!((*q.base_vol as f64 - 0.3).abs() < 1e-6);
    // K removed + K left = K deposited
    let k3 = rk[3] + q.film_k[3] as f64 + q.base_k[3] as f64;
    assert!((k3 - kpv[3] * 1.3).abs() < 1e-5);
}

#[test]
fn volume_map_deposits_everything_it_is_given() {
    let mut g = Lcg(11);
    let mut c = Canvas::bare(40, 50, &mut g);
    let (h, w) = (c.h, c.w);
    let vol: Vec<f64> = (0..12 * 15).map(|_| 0.1 + 0.5 * g.next()).collect();
    let kpv: Vec<f64> = (0..NB).map(|b| 0.3 + 0.1 * b as f64).collect();
    let spv = vec![0.2f64; NB];
    let mask = c.mask.clone();
    let before = c.totals();
    {
        let mut cv = DenseWet::new(h, w, c.arrays(), &mask, &mask);
        // partly off the canvas on the left: those cells are dropped
        tc::apply_volume_map(&Plane::new(12, 15, &vol), -3, 10, &kpv, &spv, 0.8, 1.0, 1e-4, &mut cv, 5);
    }
    let after = c.totals();
    let mut expect = 0.0;
    for i in 0..12 {
        for j in 3..15 {
            expect += vol[i * 15 + j];
        }
    }
    assert!((after.2 - before.2 - expect).abs() / expect < 1e-5);
    for b in 0..NB {
        assert!((after.0[b] - before.0[b] - expect * kpv[b]).abs() / (expect * kpv[b]) < 1e-5);
    }
}

#[test]
fn fluid_map_splits_mass_into_suspended_and_settled() {
    let (h, w) = (30usize, 40usize);
    let n = h * w;
    let mut wk = vec![0.0f32; n * NB];
    let mut ws = vec![0.0f32; n * NB];
    let mut dk = vec![0.0f32; n * NB];
    let mut ds = vec![0.0f32; n * NB];
    let mut water = vec![0.0f32; n];
    let (mut m, mut gg, mut st) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
    let (mut dm, mut dg, mut dst) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
    let vol: Vec<f64> = (0..10 * 10).map(|i| 0.05 + 0.01 * (i % 7) as f64).collect();
    let kpv: Vec<f64> = (0..NB).map(|b| 0.3 + 0.1 * b as f64).collect();
    let spv = vec![0.2f64; NB];
    let mask = vec![0.0f32; 10 * 10];
    {
        let mut wc = tc::WcLayer {
            k: SpecMut::new(h, w, NB, &mut wk),
            s: SpecMut::new(h, w, NB, &mut ws),
            water: PlaneMut::new(h, w, &mut water),
            m: PlaneMut::new(h, w, &mut m),
            g: PlaneMut::new(h, w, &mut gg),
            st: PlaneMut::new(h, w, &mut st),
            dk: SpecMut::new(h, w, NB, &mut dk),
            ds: SpecMut::new(h, w, NB, &mut ds),
            dm: PlaneMut::new(h, w, &mut dm),
            dg: PlaneMut::new(h, w, &mut dg),
            dst: PlaneMut::new(h, w, &mut dst),
        };
        tc::apply_fluid_map(
            &Plane::new(10, 10, &vol),
            5,
            8,
            3.0,
            0.25,
            &kpv,
            &spv,
            0.4,
            0.3,
            &mut wc,
            &Plane::new(10, 10, &mask),
        );
    }
    let total: f64 = vol.iter().sum();
    let sum = |a: &[f32]| a.iter().map(|v| *v as f64).sum::<f64>();
    assert!((sum(&dm) - 0.25 * total).abs() / total < 1e-5, "settled mass");
    assert!((sum(&m) - 0.75 * total).abs() / total < 1e-5, "suspended mass");
    assert!((sum(&water) - 3.0 * total).abs() / total < 1e-5, "water = water_ratio * mass");
    // pigment conservation per band: K in = K*mass out (suspended + settled)
    let spec = Spec::new(h, w, NB, &wk);
    let mut k5 = 0.0;
    for y in 0..h {
        for x in 0..w {
            k5 += spec.px(y, x)[5] as f64;
        }
    }
    let dk5: f64 = (0..h * w).map(|i| dk[i * NB + 5] as f64).sum();
    assert!((k5 + dk5 - kpv[5] * total).abs() / (kpv[5] * total) < 1e-5);
}

// --------------------------------------------------------------------------
// palette knife
// --------------------------------------------------------------------------

struct KnifeSetup {
    px: Vec<f64>,
    py: Vec<f64>,
    pp: Vec<f64>,
    pa: Vec<f64>,
    ps: Vec<f64>,
    cap: Vec<f64>,
    prm: Vec<f64>,
    lump: Vec<f64>,
    nu: usize,
    ns: usize,
    stri: Vec<f64>,
    chat: Vec<f64>,
    res_k: Vec<f64>,
    res_s: Vec<f64>,
    res_vol: Vec<f64>,
    half: f64,
    nk: i64,
    flex_n: i64,
}

fn knife_setup(mode: i64, pressure: f64, loaded: bool, g: &mut Lcg) -> KnifeSetup {
    let length = 40.0f64;
    let nbin = 20usize;
    let (px, py, pp, ps, pa) = line(240, (30.0, 60.0), (150.0, 66.0), pressure);
    let nu = length as usize + 4;
    let ns = ps[ps.len() - 1] as usize + 4;
    let lump: Vec<f64> = (0..nu * ns).map(|_| g.next() * 2.0 - 1.0).collect();
    let stri: Vec<f64> = (0..nu).map(|_| 0.2 * (g.next() - 0.5)).collect();
    let chat: Vec<f64> = (0..ns).map(|_| 0.05 * (g.next() - 0.5)).collect();
    let thickness = 3.0;
    let edge_w = 0.35 + 2.5 * (1.0 - 0.85);
    let prm = vec![
        thickness,
        0.9,
        0.55,
        1.4,
        0.55,
        0.025 * thickness,
        1.1,
        edge_w,
        0.95,
        if mode == 0 { 0.08 } else { 0.0 },
        0.45,
        0.6,
        1.1,
        if mode == 0 { 0.05 } else { 0.0 },
        0.97,
        if mode == 0 { 0.3 } else { 0.7 },
        0.15,
        0.9,
        1.6,
        0.012,
        if mode == 0 { 0.0 } else { 0.35 },
    ];
    let cap = vec![320.0 * (length / nbin as f64) * thickness * 0.5; nbin];
    let mut res_k = vec![0.0; nbin * NB];
    let mut res_s = vec![0.0; nbin * NB];
    let mut res_vol = vec![0.0; nbin];
    if loaded {
        for i in 0..nbin {
            let v = cap[i] * (0.85 + 0.3 * g.next());
            res_vol[i] = v;
            for b in 0..NB {
                res_k[i * NB + b] = v * (0.3 + 0.1 * b as f64);
                res_s[i * NB + b] = v * 0.4;
            }
        }
    }
    KnifeSetup {
        px,
        py,
        pp,
        pa,
        ps,
        cap,
        prm,
        lump,
        nu,
        ns,
        stri,
        chat,
        res_k,
        res_s,
        res_vol,
        half: length / 2.0,
        nk: length as i64 + 1,
        flex_n: 9,
    }
}

fn run_knife(c: &mut Canvas, ks: &mut KnifeSetup, mode: i64) -> (i64, i64, i64, i64) {
    let (h, w) = (c.h, c.w);
    let mask = c.mask.clone();
    let hg = c.h_ground.clone();
    let hd = c.h_dry.clone();
    let a = tf::KnifeIn {
        px: &ks.px,
        py: &ks.py,
        pp: &ks.pp,
        pa: &ks.pa,
        ps: &ks.ps,
        mode,
        half: ks.half,
        nk: ks.nk,
        cap: &ks.cap,
        prm: &ks.prm,
        lump_tex: Plane::new(ks.nu, ks.ns, &ks.lump),
        stri: &ks.stri,
        chat: &ks.chat,
        flex_n: ks.flex_n,
        p_gloss: 0.8,
        p_body: 1.0,
        p_drate: 1e-5,
        sid: 7,
        h_ground: Plane::new(h, w, &hg),
        h_dry: Plane::new(h, w, &hd),
        smax: Plane::new(h, w, &hg),
        sox: 0,
        soy: 0,
    };
    let mut cv = DenseWet::new(h, w, c.arrays(), &mask, &mask);
    tf::knife_kernel(&a, &mut ks.res_k, &mut ks.res_s, &mut ks.res_vol, &mut cv)
}

#[test]
fn knife_spread_lays_paint_and_conserves_it() {
    let mut g = Lcg(21);
    let mut c = Canvas::bare(140, 190, &mut g);
    let mut ks = knife_setup(0, 0.45, true, &mut g);
    let before = plus(&c.totals(), &ks.res_k, &ks.res_s, &ks.res_vol);
    let blade_before: f64 = ks.res_vol.iter().sum();
    let bbox = run_knife(&mut c, &mut ks, 0);
    let after = plus(&c.totals(), &ks.res_k, &ks.res_s, &ks.res_vol);
    assert_conserved("knife spread", &before, &after);
    assert!(bbox.2 >= bbox.0 && bbox.3 >= bbox.1, "knife touched nothing");
    assert!(bbox.0 >= 0 && bbox.2 < 190 && bbox.1 >= 0 && bbox.3 < 140);
    let on_canvas = c.totals().2;
    let blade_after: f64 = ks.res_vol.iter().sum();
    assert!(on_canvas > 0.05 * blade_before, "paint reached the canvas: {on_canvas}");
    assert!(blade_after < blade_before, "the blade gave paint up");
    // nothing is painted outside the reported bbox
    for y in 0..c.h {
        for x in 0..c.w {
            let i = y * c.w + x;
            if c.film_vol[i] > 0.0 {
                let (xi, yi) = (x as i64, y as i64);
                assert!(xi >= bbox.0 && xi <= bbox.2 && yi >= bbox.1 && yi <= bbox.3, "paint outside bbox at {x},{y}");
            }
        }
    }
}

#[test]
fn knife_scrape_collects_wet_paint_and_conserves_it() {
    let mut g = Lcg(33);
    let mut c = Canvas::bare(140, 190, &mut g).with_wet_band(30, 100, 20, 170, &mut g, 2.5);
    let mut ks = knife_setup(1, 0.9, false, &mut g);
    // a little paint on the blade so the scraped bead has somewhere to pile
    for i in 0..ks.res_vol.len() {
        ks.res_vol[i] = 0.5;
        for b in 0..NB {
            ks.res_k[i * NB + b] = 0.5 * 0.3;
            ks.res_s[i * NB + b] = 0.5 * 0.4;
        }
    }
    let before = plus(&c.totals(), &ks.res_k, &ks.res_s, &ks.res_vol);
    let film_before = c.totals().2;
    let blade_before: f64 = ks.res_vol.iter().sum();
    run_knife(&mut c, &mut ks, 1);
    let after = plus(&c.totals(), &ks.res_k, &ks.res_s, &ks.res_vol);
    assert_conserved("knife scrape", &before, &after);
    let blade_after: f64 = ks.res_vol.iter().sum();
    println!("scrape: blade {blade_before:.2} -> {blade_after:.2}, film {film_before:.2} -> {:.2}", c.totals().2);
    assert!(blade_after > blade_before + 1.0, "the blade collected paint");
}

#[test]
fn knife_respects_the_mask() {
    let mut g = Lcg(45);
    let mut c = Canvas::bare(140, 190, &mut g);
    for y in 0..c.h {
        for x in 90..110 {
            c.mask[y * c.w + x] = 1.0;
        }
    }
    let mut ks = knife_setup(0, 0.45, true, &mut g);
    let before = plus(&c.totals(), &ks.res_k, &ks.res_s, &ks.res_vol);
    run_knife(&mut c, &mut ks, 0);
    for y in 0..c.h {
        for x in 90..110 {
            assert_eq!(c.film_vol[y * c.w + x], 0.0, "paint on masked pixel {x},{y}");
        }
    }
    let after = plus(&c.totals(), &ks.res_k, &ks.res_s, &ks.res_vol);
    assert_conserved("knife spread behind tape", &before, &after);
}

// --------------------------------------------------------------------------
// sgraffito
// --------------------------------------------------------------------------

#[test]
fn sgraffito_scratches_a_groove_and_conserves_paint() {
    let mut g = Lcg(52);
    let mut c = Canvas::bare(90, 160, &mut g).with_wet_band(20, 70, 10, 150, &mut g, 1.5);
    // the tip is held clear of the canvas edges; the half-dry paint tears raggedly
    let (px, py, pp, _, _) = line(300, (30.0, 45.0), (130.0, 48.0), 0.85);
    let noise: Vec<f64> = (0..90 * 160).map(|_| (g.next() * 2.0 - 1.0).tanh()).collect();
    let (h, w) = (c.h, c.w);
    let (mask, hg, hd) = (c.mask.clone(), c.h_ground.clone(), c.h_dry.clone());
    let before = c.totals();
    let mid = (45 * c.w + 80) as usize;
    let vol_mid_before = c.film_vol[mid] + c.base_vol[mid];
    let prm = tf::SgraffitoParams { r0: 0.6, r1: 1.4, valley: 0.35, burr: 0.55, carry: 0.85, ragged: 0.8, nox: 0, noy: 0, sid: 8 };
    let bbox = {
        let mut cv = DenseWet::new(h, w, c.arrays(), &mask, &mask);
        tf::sgraffito_kernel(
            &px,
            &py,
            &pp,
            &prm,
            &Plane::new(90, 160, &noise),
            &Plane::new(h, w, &hg),
            &Plane::new(h, w, &hd),
            &mut cv,
        )
    };
    let after = c.totals();
    // the crumb that is still carried when the tip lifts is dropped (<=1e-7 left over)
    assert_conserved("sgraffito", &before, &after);
    assert!(bbox.2 >= bbox.0);
    let vol_mid_after = c.film_vol[mid] + c.base_vol[mid];
    assert!(vol_mid_after < vol_mid_before, "the tip removed paint on its line");
}

// --------------------------------------------------------------------------
// smudge
// --------------------------------------------------------------------------

#[test]
fn smudge_drags_paint_and_conserves_it() {
    let mut g = Lcg(61);
    let mut c = Canvas::bare(100, 180, &mut g).with_wet_band(30, 70, 20, 90, &mut g, 1.2);
    // red-ish paint left of x=55, other colour right of it
    for y in 30..70 {
        for x in 55..90 {
            let i = y * c.w + x;
            for b in 0..NB {
                c.film_k[i * NB + b] = 0.05 * (NB - b) as f32;
            }
        }
    }
    let r = 6i64;
    let nl = (2 * r + 1) as usize;
    let mut l_k = vec![0.0f64; nl * NB];
    let mut l_s = vec![0.0f64; nl * NB];
    let mut l_v = vec![0.0f64; nl];
    let mut l_p = vec![0.0f64; nl * 4];
    let (px, py, pp, _, _) = line(260, (25.0, 50.0), (150.0, 52.0), 0.7);
    let (h, w) = (c.h, c.w);
    let mask = c.mask.clone();
    let before = c.totals();
    let k_before = c.film_k[(50 * w + 30) * NB];
    let bbox = {
        let mut cv = DenseWet::new(h, w, c.arrays(), &mask, &mask);
        tf::smudge_kernel(&px, &py, &pp, r, 0.14, 0.15 * 0.5, &mut cv, 9, &mut l_k, &mut l_s, &mut l_v, &mut l_p)
    };
    let after = plus(&c.totals(), &l_k, &l_s, &l_v);
    assert_conserved("smudge", &before, &after);
    assert!(bbox.2 > bbox.0);
    // paint travelled: beyond the old edge (x > 90), where nothing was, there is paint now
    let beyond: f64 = (45..56).map(|y| c.film_vol[y * w + 95] as f64 + c.base_vol[y * w + 95] as f64).sum();
    assert!(beyond > 0.0, "the finger carried paint past the edge of the wet area");
    // and the carried wetness / gloss etc. stay sane
    assert!(l_p.iter().all(|v| v.is_finite()));
    let _ = k_before;
}

// --------------------------------------------------------------------------
// oil crayon
// --------------------------------------------------------------------------

#[test]
fn crayon_lays_pigment_in_proportion_and_counts_wax() {
    let mut g = Lcg(71);
    let mut c = Canvas::bare(64, 112, &mut g);
    let mut wax = vec![0.0f32; c.h * c.w];
    let kpv: Vec<f64> = (0..NB).map(|b| 0.4 + 0.2 * b as f64).collect();
    let spv: Vec<f64> = (0..NB).map(|b| 0.3 + 0.05 * b as f64).collect();
    let n = 160usize;
    let xy: Vec<f64> = (0..n).flat_map(|j| [10.0 + 92.0 * j as f64 / (n - 1) as f64, 32.0]).collect();
    let pressures = vec![0.24f64; n];
    let angles = vec![std::f64::consts::FRAC_PI_2; n];
    let (h, w) = (c.h, c.w);
    let (tooth, mask) = (c.tooth.clone(), c.mask.clone());
    let bbox = {
        let mut cv = DenseWet::new(h, w, c.arrays(), &tooth, &mask);
        let mut wax_p = PlaneMut::new(h, w, &mut wax);
        tf::crayon_stroke(&mut cv, 4, 1.0, &mut wax_p, &xy, &pressures, &angles, 18.0, 0.8, &kpv, &spv)
    };
    assert!(bbox.2 >= bbox.0 && bbox.3 >= bbox.1);
    let (k, s, v) = c.totals();
    assert!(v > 1.0, "the crayon laid paint: {v}");
    // everything laid down carries the stick's pigment: K and S per band = K*vol, S*vol
    for b in 0..NB {
        assert!((k[b] - kpv[b] * v).abs() / (kpv[b] * v) < 1e-5, "K band {b}");
        assert!((s[b] - spv[b] * v).abs() / (spv[b] * v) < 1e-5, "S band {b}");
    }
    // wax counts newly transferred paint (no pixel reached the cap of 1)
    let wmax = wax.iter().cloned().fold(0.0f32, f32::max);
    let wsum: f64 = wax.iter().map(|v| *v as f64).sum();
    println!("crayon: volume {v:.4}, wax sum {wsum:.4}, wax max {wmax}");
    assert!(wmax < 1.0);
    assert!((wsum - v).abs() / v < 1e-5, "wax {wsum} vs paint {v}");
    // light pressure catches the peaks of the paper more than the pits
    let (mut peak, mut np, mut pit, mut nq) = (0.0, 0, 0.0, 0);
    for y in 28..36 {
        for x in 16..96 {
            let i = y * w + x;
            if c.tooth[i] > 0.25 {
                peak += c.film_vol[i] as f64;
                np += 1;
            } else if c.tooth[i] < -0.25 {
                pit += c.film_vol[i] as f64;
                nq += 1;
            }
        }
    }
    assert!(np > 0 && nq > 0);
    assert!(peak / np as f64 > pit / nq as f64, "grain catches on the peaks");
}

#[test]
fn crayon_respects_the_mask_and_zero_pressure() {
    let mut g = Lcg(72);
    let mut c = Canvas::bare(64, 112, &mut g);
    for y in 40..64 {
        for x in 50..65 {
            c.mask[y * c.w + x] = 1.0;
        }
    }
    let mut wax = vec![0.0f32; c.h * c.w];
    let kpv = vec![0.5f64; NB];
    let spv = vec![0.3f64; NB];
    let xy = vec![-10.0, 54.0, 120.0, 54.0];
    let (h, w) = (c.h, c.w);
    let (tooth, mask) = (c.tooth.clone(), c.mask.clone());
    {
        let mut cv = DenseWet::new(h, w, c.arrays(), &tooth, &mask);
        let mut wax_p = PlaneMut::new(h, w, &mut wax);
        // zero pressure at the first sample touches nothing; full pressure at the second
        let b = tf::crayon_stroke(&mut cv, 4, 1.0, &mut wax_p, &xy, &[0.0, 1.0], &[0.0, 0.0], 12.0, 1.0, &kpv, &spv);
        assert!(b.2 >= b.0);
        let none = tf::crayon_stroke(&mut cv, 5, 1.0, &mut wax_p, &xy[..2], &[0.0], &[0.0], 12.0, 1.0, &kpv, &spv);
        assert_eq!(none, (w as i64, h as i64, -1, -1));
    }
    for y in 40..64 {
        for x in 50..65 {
            assert_eq!(c.film_vol[y * w + x], 0.0);
            assert_eq!(wax[y * w + x], 0.0);
        }
    }
}
