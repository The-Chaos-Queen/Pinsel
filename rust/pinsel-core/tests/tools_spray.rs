//! Tests with a right answer for the spray / splatter / sponge / pen kernels
//! (docs/RUST_PORTING_GUIDE.md rule 9): synthetic inputs, no Python.
//! Volume maps conserve what they are given (spray cones, discs, drops), the
//! pen raster obeys its coverage rules, feathering only grows coverage, and
//! the sponge moves paint (or pigment and water) between its reservoir and the
//! canvas without losing any. Bit-identity against numba is checked by
//! rust/harness/verify.sh.

use pinsel_core::grid::{DenseWet, Plane, PlaneMut, SpecMut, WetArrays};
use pinsel_core::tools_common::WcLayer;
use pinsel_core::tools_spray as ts;
use pinsel_core::NB;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

fn tooth(h: usize, w: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; h * w];
    for y in 0..h {
        for x in 0..w {
            t[y * w + x] = (((x as f64 * 0.9).sin() * (y as f64 * 0.8).cos()) * 1.2) as f32;
        }
    }
    t
}

fn sum(v: &[f64]) -> f64 {
    v.iter().sum()
}

#[test]
fn spray_cones_keep_their_amount() {
    let (h, w) = (80usize, 100usize);
    let mut acc = vec![0.0f64; h * w];
    let px = [30.0, 50.5, 70.2];
    let py = [30.0, 40.3, 45.0];
    let amt = [2.0, 0.5, 1.25];
    let sig = [3.0, 4.0, 2.5];
    {
        let mut a = PlaneMut::new(h, w, &mut acc);
        ts::spray_accumulate(&px, &py, &amt, &sig, 0.2, 3.0, 0.0, 0.0, &mut a);
    }
    // the cones lie well inside the map: the gaussians sum to their amount
    // (a 3.2 sigma window of the wide tail keeps ~99.9%)
    let total = sum(&acc);
    let want: f64 = amt.iter().sum();
    assert!((total - want).abs() / want < 2e-3, "spray total {total}, wanted {want}");
    // no tail: tighter
    let mut acc2 = vec![0.0f64; h * w];
    {
        let mut a = PlaneMut::new(h, w, &mut acc2);
        ts::spray_accumulate(&px, &py, &amt, &sig, 0.0, 3.0, 0.0, 0.0, &mut a);
    }
    assert!((sum(&acc2) - want).abs() / want < 2e-3);
    // zero or negative amounts add nothing
    let mut acc3 = vec![0.0f64; h * w];
    {
        let mut a = PlaneMut::new(h, w, &mut acc3);
        ts::spray_accumulate(&px, &py, &[0.0, -1.0, 0.0], &sig, 0.2, 3.0, 0.0, 0.0, &mut a);
    }
    assert_eq!(sum(&acc3), 0.0);
    // peak sits at the centre of the first cone
    let peak = acc2[30 * w + 30];
    assert!(acc2.iter().all(|&v| v <= peak * 1.5));
}

#[test]
fn discs_conserve_volume() {
    let (h, w) = (60usize, 60usize);
    let tt = tooth(h, w);
    let tp = Plane::new(h, w, &tt);
    let mut acc = vec![0.0f64; h * w];
    let xs = [20.0, 31.3, 40.7, 10.25];
    let ys = [20.5, 30.0, 12.2, 40.0];
    let rs = [3.5, 0.3, 5.0, 2.0]; // one sub-pixel droplet
    let vs = [1.0, 0.25, 2.0, 0.75];
    {
        let mut a = PlaneMut::new(h, w, &mut acc);
        ts::splat_discs(&xs, &ys, &rs, &vs, 0, 0, &mut a, &tp, 0.5);
    }
    let want: f64 = vs.iter().sum();
    assert!((sum(&acc) - want).abs() < 1e-9, "discs: {} vs {}", sum(&acc), want);
    // a disc half off the map loses its outside share but never adds
    let mut acc2 = vec![0.0f64; h * w];
    {
        let mut a = PlaneMut::new(h, w, &mut acc2);
        ts::splat_discs(&[0.0], &[30.0], &[4.0], &[1.0], 0, 0, &mut a, &tp, 0.0);
    }
    assert!(sum(&acc2) < 1.0 && sum(&acc2) > 0.3);
}

#[test]
fn drops_conserve_volume() {
    let (h, w) = (80usize, 80usize);
    let tt = tooth(h, w);
    let tp = Plane::new(h, w, &tt);
    let mut g = Lcg(5);
    for &dome in &[1.0, 0.0] {
        let mut acc = vec![0.0f64; h * w];
        let n = 12;
        let mut xs = vec![];
        let mut ys = vec![];
        let mut rs = vec![];
        let mut es = vec![];
        let mut th = vec![];
        let mut vs = vec![];
        let mut ph = vec![];
        for i in 0..n {
            xs.push(20.0 + 40.0 * g.next());
            ys.push(20.0 + 40.0 * g.next());
            rs.push(if i == 3 { 0.4 } else { 1.0 + 4.0 * g.next() });
            es.push(if i % 3 == 0 { 1.0 } else { 1.0 + 2.0 * g.next() }); // round and streaked
            th.push(6.28 * g.next());
            vs.push(0.1 + g.next());
            ph.push(6.28 * g.next());
        }
        {
            let mut a = PlaneMut::new(h, w, &mut acc);
            ts::splat_drops(&xs, &ys, &rs, &es, &th, &vs, &ph, 0, 0, &mut a, &tp, 0.4, dome);
        }
        let want = sum(&vs);
        assert!((sum(&acc) - want).abs() < 1e-9, "drops (dome {dome}): {} vs {}", sum(&acc), want);
        assert!(acc.iter().all(|&v| v >= 0.0));
    }
}

#[test]
fn pen_raster_rules() {
    let (h, w) = (40usize, 60usize);
    let tt = vec![0.0f32; 400 * 400];
    let tp = Plane::new(400, 400, &tt);
    let px: Vec<f64> = (0..20).map(|i| 10.0 + i as f64 * 2.0).collect();
    let py = vec![20.0f64; 20];
    let rad = vec![2.0f64; 20];
    let ink = vec![0.8f64; 20];
    let skip = vec![-3.0f64; 20];
    let gap = vec![3.0f64; 20];
    let fill = vec![1.0f64; 20];
    for kind in 0..3i64 {
        let mut cov = vec![0.0f64; h * w];
        let mut amt = vec![0.0f64; h * w];
        {
            let mut c = PlaneMut::new(h, w, &mut cov);
            let mut a = PlaneMut::new(h, w, &mut amt);
            ts::pen_raster(
                &px, &py, &rad, &ink, &skip, &gap, &fill, kind, 0.8, 0.6, 3.0, 0.7, 0, 0, &mut c, &mut a, &tp,
            );
        }
        assert!(cov.iter().all(|&v| (0.0..=1.0).contains(&v)), "kind {kind}: coverage outside 0..1");
        // ink density is coverage * ink wherever one dab made it (here: at most ink)
        assert!(amt.iter().all(|&v| (0.0..=0.8 + 1e-12).contains(&v)));
        assert!(cov[20 * w + 30] > 0.99, "kind {kind}: centre of the line not covered");
        assert!((amt[20 * w + 30] - 0.8 * cov[20 * w + 30]).abs() < 1e-12);
        assert_eq!(cov[2 * w + 2], 0.0, "kind {kind}: far corner touched");
        assert_eq!(cov.iter().map(|&v| (v > 0.0) as i32).sum::<i32>() > 0, true);
    }
    // a skipping pen (tooth below the skip level) leaves paper bare where tooth < skip - 0.225
    let tt2 = vec![-1.0f32; 400 * 400];
    let tp2 = Plane::new(400, 400, &tt2);
    let mut cov = vec![0.0f64; h * w];
    let mut amt = vec![0.0f64; h * w];
    {
        let mut c = PlaneMut::new(h, w, &mut cov);
        let mut a = PlaneMut::new(h, w, &mut amt);
        let skip2 = vec![0.5f64; 20];
        ts::pen_raster(&px, &py, &rad, &ink, &skip2, &gap, &fill, 0, 1.0, 0.0, 0.0, 0.5, 0, 0, &mut c, &mut a, &tp2);
    }
    assert_eq!(sum(&cov), 0.0);
}

#[test]
fn feather_only_grows_and_respects_permeability() {
    let (h, w) = (30usize, 30usize);
    let mut cov = vec![0.0f64; h * w];
    let mut amt = vec![0.0f64; h * w];
    for x in 5..25 {
        cov[15 * w + x] = 1.0;
        amt[15 * w + x] = 0.6;
    }
    let before = cov.clone();
    // permeable everywhere: coverage creeps outward, never shrinks
    let perm = vec![0.9f64; h * w];
    {
        let mut c = PlaneMut::new(h, w, &mut cov);
        let mut a = PlaneMut::new(h, w, &mut amt);
        ts::feather(&mut c, &mut a, &Plane::new(h, w, &perm), 3, 0.82);
    }
    assert!(cov.iter().zip(&before).all(|(a, b)| a >= b));
    assert!(sum(&cov) > sum(&before));
    assert!(cov[14 * w + 15] > 0.0 && cov[13 * w + 15] > 0.0 && cov[12 * w + 15] > 0.0);
    assert_eq!(cov[10 * w + 15], 0.0); // 3 iterations reach 3 pixels, not 5
    assert!(cov.iter().all(|&v| v <= 1.0));
    // impermeable paper: nothing moves
    let mut cov2 = before.clone();
    let mut amt2 = vec![0.0f64; h * w];
    {
        let mut c = PlaneMut::new(h, w, &mut cov2);
        let mut a = PlaneMut::new(h, w, &mut amt2);
        ts::feather(&mut c, &mut a, &Plane::new(h, w, &vec![0.0f64; h * w]), 4, 0.82);
    }
    assert_eq!(cov2, before);
    // zero iterations: nothing moves
    let mut cov3 = before.clone();
    {
        let mut c = PlaneMut::new(h, w, &mut cov3);
        let mut a = PlaneMut::new(h, w, &mut amt2);
        ts::feather(&mut c, &mut a, &Plane::new(h, w, &perm), 0, 0.82);
    }
    assert_eq!(cov3, before);
}

fn wc_dummy<'a>(d3: &'a mut [Vec<f32>], d2: &'a mut [Vec<f32>]) -> WcLayer<'a> {
    let mut it3 = d3.iter_mut();
    let mut it2 = d2.iter_mut();
    WcLayer {
        k: SpecMut::new(1, 1, NB, it3.next().unwrap()),
        s: SpecMut::new(1, 1, NB, it3.next().unwrap()),
        water: PlaneMut::new(1, 1, it2.next().unwrap()),
        m: PlaneMut::new(1, 1, it2.next().unwrap()),
        g: PlaneMut::new(1, 1, it2.next().unwrap()),
        st: PlaneMut::new(1, 1, it2.next().unwrap()),
        dk: SpecMut::new(1, 1, NB, it3.next().unwrap()),
        ds: SpecMut::new(1, 1, NB, it3.next().unwrap()),
        dm: PlaneMut::new(1, 1, it2.next().unwrap()),
        dg: PlaneMut::new(1, 1, it2.next().unwrap()),
        dst: PlaneMut::new(1, 1, it2.next().unwrap()),
    }
}

// --------------------------------------------------------------------------
// sponge
// --------------------------------------------------------------------------

struct Film {
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
}

impl Film {
    fn new(h: usize, w: usize) -> Self {
        let n = h * w;
        Film {
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
        }
    }
    fn wet_patch(&mut self, y0: usize, y1: usize, x0: usize, x1: usize, g: &mut Lcg) {
        for y in y0..y1 {
            for x in x0..x1 {
                let i = y * self.w + x;
                for k in 0..NB {
                    self.film_k[i * NB + k] = (0.3 * (0.5 + g.next())) as f32;
                    self.film_s[i * NB + k] = (0.2 * (0.5 + g.next())) as f32;
                    self.base_k[i * NB + k] = (0.05 * g.next()) as f32;
                    self.base_s[i * NB + k] = (0.03 * g.next()) as f32;
                }
                self.film_vol[i] = (0.5 + 0.3 * g.next()) as f32;
                self.base_vol[i] = (0.1 * g.next()) as f32;
                self.wet[i] = 0.9;
                self.gloss[i] = 0.5;
                self.body[i] = 1.0;
                self.drate[i] = 0.001;
                self.top_id[i] = 3;
            }
        }
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
    /// (K, S per band, volume) of top + base.
    fn totals(&self) -> (Vec<f64>, Vec<f64>, f64) {
        let mut k = vec![0.0; NB];
        let mut s = vec![0.0; NB];
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

fn texture(t: usize, g: &mut Lcg) -> Vec<f64> {
    (0..t * t).map(|_| g.next()).collect()
}

fn params(fluid: bool, pickup: f64, water_lvl: f64) -> ts::SpongeParams {
    ts::SpongeParams {
        cx: 30.0,
        cy: 28.0,
        rot: 0.4,
        scale: 1.0,
        depth: 0.5,
        soft: 0.08,
        micro: 0.35,
        alpha: if fluid { 0.6 } else { 0.9 },
        pickup,
        cap: 1.0,
        p_gloss: 0.4,
        p_body: 0.7,
        p_drate: 0.002,
        fluid,
        water_lvl,
        wat_ratio: 0.6,
        gran: 0.5,
        stain: 0.3,
        sid: 7,
        x0: 5,
        y0: 5,
    }
}

#[test]
fn sponge_paste_conserves_paint() {
    let (h, w) = (60usize, 60usize);
    let mut g = Lcg(11);
    let tt = tooth(h, w);
    let tex = texture(40, &mut g);
    let mut film = Film::new(h, w);
    film.wet_patch(15, 40, 15, 40, &mut g); // older wet paint the sponge will pick up
    let mask = vec![0.0f32; 50 * 50]; // region (5, 5) .. (54, 54)
    let (before_k, before_s, before_v) = film.totals();
    let mut rk: Vec<f64> = (0..NB).map(|b| 0.5 + 0.1 * b as f64).collect();
    let mut rs: Vec<f64> = (0..NB).map(|b| 0.3 + 0.05 * b as f64).collect();
    let mut rv = vec![1.2f64];
    let (rk0, rs0, rv0) = (rk.clone(), rs.clone(), rv[0]);
    // dummy 1 x 1 water layer, as Python passes it for paste paints
    let mut d3 = vec![vec![0.0f32; NB]; 4];
    let mut d2 = vec![vec![0.0f32; 1]; 8];
    let mut wc = wc_dummy(&mut d3, &mut d2);
    let mut damp_v = vec![0.0f32; 1];
    let mut damp = PlaneMut::new(1, 1, &mut damp_v);
    let bbox = {
        let arrays = film.arrays();
        let mut cv = DenseWet::new(h, w, arrays, &tt, &tt);
        ts::sponge_dab(
            &params(false, 0.5, 0.0),
            &Plane::new(40, 40, &tex),
            &mut rk,
            &mut rs,
            &mut rv,
            &mut cv,
            &Plane::new(50, 50, &mask),
            &Plane::new(h, w, &tt),
            &mut wc,
            &mut damp,
        )
    };
    assert!(bbox.2 >= bbox.0 && bbox.3 >= bbox.1, "the dab touched nothing: {bbox:?}");
    assert!(bbox.0 >= 5 && bbox.2 <= 54, "bbox outside the region: {bbox:?}");
    let (after_k, after_s, after_v) = film.totals();
    // paint in the film + reservoir stays the same (to f32 rounding of the film)
    let tv = (after_v + rv[0] - before_v - rv0).abs() / (before_v + rv0);
    assert!(tv < 1e-5, "volume not conserved, rel {tv:e}");
    for b in 0..NB {
        let tk = (after_k[b] + rk[b] - before_k[b] - rk0[b]).abs() / (before_k[b] + rk0[b]);
        let ts_ = (after_s[b] + rs[b] - before_s[b] - rs0[b]).abs() / (before_s[b] + rs0[b]);
        assert!(tk < 1e-5 && ts_ < 1e-5, "band {b}: K rel {tk:e}, S rel {ts_:e}");
    }
    // the sponge both picked up (film lost) and deposited somewhere
    assert!(after_v != before_v);
    // a fully masked region does nothing
    let mask1 = vec![1.0f32; 50 * 50];
    let snapshot = film.totals();
    let mut rv2 = vec![1.2f64];
    let mut rk2 = rk0.clone();
    let mut rs2 = rs0.clone();
    let b2 = {
        let arrays = film.arrays();
        let mut cv = DenseWet::new(h, w, arrays, &tt, &tt);
        ts::sponge_dab(
            &params(false, 0.5, 0.0),
            &Plane::new(40, 40, &tex),
            &mut rk2,
            &mut rs2,
            &mut rv2,
            &mut cv,
            &Plane::new(50, 50, &mask1),
            &Plane::new(h, w, &tt),
            &mut wc,
            &mut damp,
        )
    };
    assert_eq!(b2, (w as i64, h as i64, -1, -1));
    assert_eq!(film.totals().2, snapshot.2);
    assert_eq!(rv2[0], 1.2);
}

#[test]
fn sponge_fluid_conserves_pigment_and_water() {
    let (h, w) = (60usize, 60usize);
    let mut g = Lcg(23);
    let tt = tooth(h, w);
    let tex = texture(40, &mut g);
    let mut film = Film::new(h, w);
    let n = h * w;
    let mut wk = vec![0.0f32; n * NB];
    let mut ws = vec![0.0f32; n * NB];
    let mut wwat = vec![0.0f32; n];
    let mut wm = vec![0.0f32; n];
    let mut wg = vec![0.0f32; n];
    let mut wst = vec![0.0f32; n];
    let mut dk = vec![0.0f32; n * NB];
    let mut ds = vec![0.0f32; n * NB];
    let mut dm = vec![0.0f32; n];
    let mut dg = vec![0.0f32; n];
    let mut dst = vec![0.0f32; n];
    let mut damp_v = vec![0.2f32; n];
    // settled pigment and a little suspended pigment under the sponge
    for i in 0..n {
        if (i / w) > 10 && (i / w) < 50 && (i % w) > 10 && (i % w) < 50 {
            for b in 0..NB {
                dk[i * NB + b] = (0.2 + 0.1 * g.next()) as f32;
                ds[i * NB + b] = (0.1 + 0.1 * g.next()) as f32;
                wk[i * NB + b] = (0.05 * g.next()) as f32;
                ws[i * NB + b] = (0.03 * g.next()) as f32;
            }
            dm[i] = (0.3 + 0.2 * g.next()) as f32;
            dg[i] = 0.1;
            dst[i] = (0.05 * g.next()) as f32;
            wm[i] = (0.1 * g.next()) as f32;
            wwat[i] = (0.1 * g.next()) as f32;
        }
    }
    let mask = vec![0.0f32; 50 * 50];
    let mut rk: Vec<f64> = (0..NB).map(|b| 0.5 + 0.1 * b as f64).collect();
    let mut rs: Vec<f64> = (0..NB).map(|b| 0.3 + 0.05 * b as f64).collect();
    let mut rv = vec![1.2f64];
    let tot = |dm: &[f32], wm: &[f32], dk: &[f32], wk: &[f32], rv: f64, rk: &[f64]| -> (f64, f64) {
        let m: f64 = dm.iter().chain(wm).map(|v| *v as f64).sum::<f64>() + rv;
        let k: f64 = dk.iter().chain(wk).map(|v| *v as f64).sum::<f64>() + rk.iter().sum::<f64>();
        (m, k)
    };
    for &(pickup, water_lvl) in &[(0.6, 0.1), (0.6, 0.9)] {
        // damp sponge (0.9) deposits water + pigment; a thirsty one (0.1) lifts pigment and water
        let (m0, k0) = tot(&dm, &wm, &dk, &wk, rv[0], &rk);
        let water0: f64 = wwat.iter().map(|v| *v as f64).sum();
        let bbox = {
            let arrays = film.arrays();
            let mut cv = DenseWet::new(h, w, arrays, &tt, &tt);
            let mut wc = WcLayer {
                k: SpecMut::new(h, w, NB, &mut wk),
                s: SpecMut::new(h, w, NB, &mut ws),
                water: PlaneMut::new(h, w, &mut wwat),
                m: PlaneMut::new(h, w, &mut wm),
                g: PlaneMut::new(h, w, &mut wg),
                st: PlaneMut::new(h, w, &mut wst),
                dk: SpecMut::new(h, w, NB, &mut dk),
                ds: SpecMut::new(h, w, NB, &mut ds),
                dm: PlaneMut::new(h, w, &mut dm),
                dg: PlaneMut::new(h, w, &mut dg),
                dst: PlaneMut::new(h, w, &mut dst),
            };
            let mut damp = PlaneMut::new(h, w, &mut damp_v);
            ts::sponge_dab(
                &params(true, pickup, water_lvl),
                &Plane::new(40, 40, &tex),
                &mut rk,
                &mut rs,
                &mut rv,
                &mut cv,
                &Plane::new(50, 50, &mask),
                &Plane::new(h, w, &tt),
                &mut wc,
                &mut damp,
            )
        };
        assert!(bbox.2 >= bbox.0, "the dab touched nothing");
        let (m1, k1) = tot(&dm, &wm, &dk, &wk, rv[0], &rk);
        assert!((m1 - m0).abs() / m0 < 1e-5, "pigment mass not conserved: {m0} -> {m1}");
        assert!((k1 - k0).abs() / k0 < 1e-5, "pigment K not conserved: {k0} -> {k1}");
        let water1: f64 = wwat.iter().map(|v| *v as f64).sum();
        if water_lvl > 0.5 {
            assert!(water1 > water0, "a damp sponge should add water");
        } else {
            assert!(water1 < water0, "a thirsty sponge should lift water");
        }
    }
}
