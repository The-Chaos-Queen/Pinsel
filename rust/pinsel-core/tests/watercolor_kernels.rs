//! Tests of the watercolor kernels on synthetic inputs (no Python):
//! known answers, conservation of water and pigment (in = out to float32
//! rounding, as in paste_backends.rs), strided windows, and serial versus
//! parallel runs giving bit-identical results.

use pinsel_core::watercolor as wc;
use wc::{Deposit, Pigment, Pl, PlM, Sp, SpM};

const NB: usize = 12;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn vec(&mut self, n: usize, lo: f64, hi: f64) -> Vec<f32> {
        (0..n).map(|_| (lo + (hi - lo) * self.next()) as f32).collect()
    }
    /// Like `vec`, but a fraction `zero` of the entries is exactly 0.
    fn vec0(&mut self, n: usize, lo: f64, hi: f64, zero: f64) -> Vec<f32> {
        (0..n).map(|_| if self.next() < zero { 0.0 } else { (lo + (hi - lo) * self.next()) as f32 }).collect()
    }
}

fn prm() -> Vec<f64> {
    // the PARAMS of pinsel/watercolor.py (absorbency 0.5)
    let mut a = vec![0.0f64; 32];
    a[0] = 0.20; // gravity
    a[1] = 0.14; // friction
    a[2] = 0.06; // h_visc
    a[3] = 0.004; // h_min
    a[4] = 0.012; // h_wet
    a[5] = 0.25; // wett_width
    a[6] = 0.95; // h_adv
    a[7] = 0.35; // adv_width
    a[8] = 0.35; // theta
    a[9] = 0.16 * 2.0 * 0.5; // absorb
    a[10] = 0.30; // capacity
    a[11] = 0.0055; // evap
    a[12] = 4.0; // edge_evap
    a[13] = 1.0 / 55.0; // evap_damp
    a[14] = 0.0030; // dep
    a[15] = 0.025; // dep_h0
    a[16] = 0.025; // dep_damp
    a[17] = 0.03; // des
    a[18] = 0.95; // bond
    a[19] = 6.0; // bond_time
    a[20] = 0.004; // soften
    a[21] = 0.45; // gran
    a[22] = 0.25; // salt_absorb
    a[23] = 4.0; // edge_dep
    a[24] = 0.6; // shear_lift
    a
}

fn sum32(v: &[f32]) -> f64 {
    v.iter().map(|&x| x as f64).sum()
}

fn close(a: f64, b: f64, tol: f64, what: &str) {
    assert!((a - b).abs() <= tol, "{what}: {a} vs {b} (tolerance {tol})");
}

/// A wet synthetic window with the arrays the flow kernels need.
struct Flow {
    n: usize,
    w: usize,
    h: Vec<f32>,
    s: Vec<f32>,
    hp: Vec<f32>,
    cxf: Vec<f32>,
    cyf: Vec<f32>,
    th: Vec<f32>,
    rho: Vec<f32>,
    mask: Vec<f32>,
    u: Vec<f32>,
    v: Vec<f32>,
    ur: Vec<f32>,
    vr: Vec<f32>,
    r: Vec<f32>,
    ax: Vec<f32>,
    ay: Vec<f32>,
    tiles: Vec<bool>,
    rowtile: Vec<f32>,
}

impl Flow {
    fn new(n: usize, w: usize, g: &mut Lcg) -> Self {
        let ntx = (w + wc::TS - 1) / wc::TS;
        let nty = (n + wc::TS - 1) / wc::TS;
        let mut mask = vec![0.0f32; n * w];
        for m in mask.iter_mut() {
            let q = g.next();
            *m = if q < 0.04 { 1.0 } else if q < 0.06 { 0.4 } else { 0.0 };
        }
        Flow {
            n,
            w,
            h: g.vec0(n * w, 0.0, 0.6, 0.4),
            s: g.vec(n * w, 0.0, 1.0),
            hp: g.vec(n * w, 0.0, 0.2),
            cxf: g.vec(n * w, 0.5, 1.5),
            cyf: g.vec(n * w, 0.5, 1.5),
            th: g.vec(n * w, 0.08, 0.97),
            rho: g.vec(n * w, 0.02, 1.0),
            mask,
            u: g.vec(n * (w + 1), -0.05, 0.05),
            v: g.vec((n + 1) * w, -0.05, 0.05),
            ur: vec![0.0; n * (w + 1)],
            vr: vec![0.0; (n + 1) * w],
            r: vec![0.0; n * w],
            ax: vec![0.0; n * (w + 1)],
            ay: vec![0.0; (n + 1) * w],
            tiles: vec![true; nty * ntx],
            rowtile: vec![0.0; n * ntx],
        }
    }

    /// One sub-step: flow1 then flow2, as `_flow` in Python.
    fn step(&mut self, tx: f64, ty: f64, use_tiles: bool) {
        let (n, w) = (self.n, self.w);
        let ntx = (w + wc::TS - 1) / wc::TS;
        let nty = (n + wc::TS - 1) / wc::TS;
        let p = prm();
        {
            let (h, s, hp, cxf, cyf, th, rho, mask) = (
                Pl::dense(n, w, &self.h),
                Pl::dense(n, w, &self.s),
                Pl::dense(n, w, &self.hp),
                Pl::dense(n, w, &self.cxf),
                Pl::dense(n, w, &self.cyf),
                Pl::dense(n, w, &self.th),
                Pl::dense(n, w, &self.rho),
                Pl::dense(n, w, &self.mask),
            );
            let tiles = Pl::dense(nty, ntx, &self.tiles);
            let u = PlM::dense(n, w + 1, &mut self.u);
            let v = PlM::dense(n + 1, w, &mut self.v);
            let ur = PlM::dense(n, w + 1, &mut self.ur);
            let vr = PlM::dense(n + 1, w, &mut self.vr);
            let r = PlM::dense(n, w, &mut self.r);
            let rowtile = PlM::dense(n, ntx, &mut self.rowtile);
            wc::flow1(&h, &s, &hp, &cxf, &cyf, &th, &rho, &mask, &u, &v, &ur, &vr, &r, &p, tx, ty, true, use_tiles, &tiles, &rowtile);
        }
        let h = PlM::dense(n, w, &mut self.h);
        let u = PlM::dense(n, w + 1, &mut self.u);
        let v = PlM::dense(n + 1, w, &mut self.v);
        let ur = Pl::dense(n, w + 1, &self.ur);
        let vr = Pl::dense(n + 1, w, &self.vr);
        let r = Pl::dense(n, w, &self.r);
        let ax = PlM::dense(n, w + 1, &mut self.ax);
        let ay = PlM::dense(n + 1, w, &mut self.ay);
        let tiles = Pl::dense(nty, ntx, &self.tiles);
        wc::flow2(&h, &u, &v, &ur, &vr, &r, &ax, &ay, use_tiles, &tiles);
    }
}

#[test]
fn flow_conserves_water_and_respects_walls() {
    let mut g = Lcg(11);
    for &(n, w) in &[(37usize, 53usize), (16, 16), (5, 40), (40, 5)] {
        let mut f = Flow::new(n, w, &mut g);
        let before = sum32(&f.h);
        for step in 0..6 {
            f.step(0.001, -0.002, step % 2 == 1);
        }
        // the only loss is the +-1e-8 dead band on tiny fluxes
        close(sum32(&f.h), before, 1e-5 * before.max(1.0), "total water");
        assert!(f.h.iter().all(|&x| x >= 0.0), "negative water");
        // walls: no flux through the outer faces
        for y in 0..n {
            assert_eq!(f.u[y * (w + 1)], 0.0);
            assert_eq!(f.u[y * (w + 1) + w], 0.0);
        }
        for x in 0..w {
            assert_eq!(f.v[x], 0.0);
            assert_eq!(f.v[n * w + x], 0.0);
        }
        // the accumulated volumes record exactly what moved: summed over all faces they are
        // antisymmetric bookkeeping, so each face accumulator stays finite
        assert!(f.ax.iter().chain(f.ay.iter()).all(|x| x.is_finite()));
    }
}

#[test]
fn still_water_stays_still_and_gradient_moves_it_downhill() {
    // flat water, no tilt, no relief: no flux at all
    let (n, w) = (12, 14);
    let mut g = Lcg(3);
    let mut f = Flow::new(n, w, &mut g);
    f.h = vec![0.3; n * w];
    f.hp = vec![0.0; n * w];
    f.mask = vec![0.0; n * w];
    f.u = vec![0.0; n * (w + 1)];
    f.v = vec![0.0; (n + 1) * w];
    f.step(0.0, 0.0, false);
    assert!(f.ur.iter().chain(f.vr.iter()).all(|&x| x == 0.0), "flat water must not flow");
    assert!(f.h.iter().all(|&x| x == 0.3));
    // a deep column in the middle spreads sideways (positive flux to the right of it, negative to the left)
    f.h[(n / 2) * w + w / 2] = 0.9;
    f.step(0.0, 0.0, false);
    let y = n / 2;
    assert!(f.ur[y * (w + 1) + w / 2 + 1] > 0.0, "flux away from the column (right)");
    assert!(f.ur[y * (w + 1) + w / 2] < 0.0, "flux away from the column (left)");
    assert!(f.h[y * w + w / 2] < 0.9, "the column loses water");
}

#[test]
fn outflow_is_water_plus_what_left() {
    let (n, w) = (6, 7);
    let h = vec![0.5f32; n * w];
    let mut ax = vec![0.0f32; n * (w + 1)];
    let mut ay = vec![0.0f32; (n + 1) * w];
    ax[2 * (w + 1) + 3] = 0.1; // face left of (2,2) is x=2 -> index 2; face right of (2,2) is index 3: flux +0.1 out of (2,2)
    ay[3 * w + 4] = -0.2; // face between (2,4) and (3,4), negative: pixel (3,4) loses 0.2
    let mut den = vec![0.0f32; n * w];
    wc::outflow(&Pl::dense(n, w, &h), &Pl::dense(n, w + 1, &ax), &Pl::dense(n + 1, w, &ay), &PlM::dense(n, w, &mut den));
    assert_eq!(den[2 * w + 2], 0.6);
    assert_eq!(den[3 * w + 4], 0.7);
    assert_eq!(den[w], 0.5);
}

struct Pig {
    n: usize,
    w: usize,
    h: Vec<f32>,
    den: Vec<f32>,
    k: Vec<f32>,
    s: Vec<f32>,
    m: Vec<f32>,
    g: Vec<f32>,
    st: Vec<f32>,
    salt: Vec<f32>,
    cf: Vec<f32>,
    acc: Vec<f32>,
}

impl Pig {
    fn new(n: usize, w: usize, vertical: bool, g: &mut Lcg) -> Self {
        let h = g.vec(n * w, 0.0, 0.5);
        let den = h.iter().map(|&x| x + 0.02).collect();
        Pig {
            n,
            w,
            h,
            den,
            k: g.vec0(n * w * NB, 0.0, 2.0, 0.3),
            s: g.vec0(n * w * NB, 0.0, 2.0, 0.3),
            m: g.vec0(n * w, 0.0, 1.0, 0.2),
            g: g.vec0(n * w, 0.0, 1.0, 0.2),
            st: g.vec0(n * w, 0.0, 1.0, 0.2),
            salt: g.vec0(n * w, 0.0, 1.0, 0.5),
            cf: g.vec(n * w, 0.1, 1.5),
            acc: if vertical { g.vec0((n + 1) * w, -0.08, 0.08, 0.3) } else { g.vec0(n * (w + 1), -0.08, 0.08, 0.3) },
        }
    }

    fn totals(&self) -> [f64; 5] {
        [sum32(&self.k), sum32(&self.s), sum32(&self.m), sum32(&self.g), sum32(&self.st)]
    }

    fn run(&mut self, vertical: bool, cmix: f64, salt_on: bool) {
        let (n, w) = (self.n, self.w);
        let h = Pl::dense(n, w, &self.h);
        let den = Pl::dense(n, w, &self.den);
        let salt = Pl::dense(n, w, &self.salt);
        let cf = Pl::dense(n, w, &self.cf);
        let k = SpM::new(n, w, NB, w * NB, &mut self.k);
        let s = SpM::new(n, w, NB, w * NB, &mut self.s);
        let m = PlM::dense(n, w, &mut self.m);
        let g = PlM::dense(n, w, &mut self.g);
        let st = PlM::dense(n, w, &mut self.st);
        let pg = Pigment { k: &k, s: &s, m: &m, g: &g, st: &st };
        if vertical {
            let acc = PlM::dense(n + 1, w, &mut self.acc);
            wc::ypass(&h, &den, &pg, &acc, &salt, &cf, cmix, 0.1, 0.012, salt_on);
        } else {
            let acc = PlM::dense(n, w + 1, &mut self.acc);
            wc::xpass(&h, &den, &pg, &acc, &salt, &cf, cmix, 0.1, 0.012, salt_on);
        }
    }
}

#[test]
fn pigment_passes_conserve_every_channel() {
    let mut g = Lcg(5);
    for &(n, w) in &[(30usize, 45usize), (9, 33), (33, 9), (4, 4)] {
        for vertical in [false, true] {
            for &(cmix, salt_on) in &[(0.0, false), (0.022, false), (0.022, true)] {
                let mut p = Pig::new(n, w, vertical, &mut g);
                let before = p.totals();
                let acc_before: f64 = p.acc.iter().map(|&x| (x as f64).abs()).sum();
                p.run(vertical, cmix, salt_on);
                let after = p.totals();
                for c in 0..5 {
                    close(after[c], before[c], 2e-5 * before[c].max(1.0), &format!("channel {c} vertical={vertical}"));
                }
                // consumed accumulators only shrink
                let acc_after: f64 = p.acc.iter().map(|&x| (x as f64).abs()).sum();
                assert!(acc_after <= acc_before);
            }
        }
    }
}

#[test]
fn pigment_follows_the_water_downstream() {
    // wet row, pigment only in the first pixel, water flowing to +x
    let (n, w) = (1, 6);
    let mut p = Pig::new(n, w, false, &mut Lcg(1));
    p.h = vec![0.4; w];
    p.den = vec![0.45; w];
    p.k = vec![0.0; w * NB];
    p.s = vec![0.0; w * NB];
    p.m = vec![0.0; w];
    p.g = vec![0.0; w];
    p.st = vec![0.0; w];
    p.salt = vec![0.0; w];
    p.cf = vec![1.0; w];
    p.k[0] = 1.0;
    p.m[0] = 1.0;
    p.acc = vec![0.0; w + 1];
    p.acc[1] = 0.05; // flux over the face between pixel 0 and 1
    p.run(false, 0.0, false);
    assert!(p.m[0] < 1.0 && p.m[1] > 0.0, "pigment moved downstream");
    assert!(p.k[NB] > 0.0, "spectral K moved with it");
    assert_eq!(p.acc[1], 0.0, "volume consumed");
    // pigment fraction = flux / water * thin-film factor
    let expect = (0.05f32 / 0.45f32) as f64 * (0.4 / (0.4 + 0.05));
    close(p.m[1] as f64, expect, 1e-6, "transferred mass");
}

#[test]
fn edge_matches_a_naive_box_blur() {
    let mut g = Lcg(7);
    for &(n, w, r) in &[(23usize, 31usize, 7i64), (9, 70, 2), (40, 40, 7), (1, 5, 1)] {
        let h = g.vec0(n * w, 0.0, 0.6, 0.5);
        let mask: Vec<f32> = (0..n * w).map(|_| if g.next() < 0.1 { 1.0 } else { 0.0 }).collect();
        let mut tmp = vec![0.0f32; n * w];
        let mut e = vec![0.0f32; n * w];
        wc::edge(&Pl::dense(n, w, &h), &Pl::dense(n, w, &mask), 0.012, r, &PlM::dense(n, w, &mut tmp), &PlM::dense(n, w, &mut e));
        let wet = |y: i64, x: i64| -> f64 {
            let (y, x) = (y.clamp(0, n as i64 - 1) as usize, x.clamp(0, w as i64 - 1) as usize);
            if h[y * w + x] as f64 > 0.012 && (mask[y * w + x] as f64) < 0.5 {
                1.0
            } else {
                0.0
            }
        };
        let inv = 1.0 / (2 * r + 1) as f64;
        for y in 0..n as i64 {
            for x in 0..w as i64 {
                let mut a = 0.0;
                for dy in -r..=r {
                    for dx in -r..=r {
                        a += wet(y + dy, x + dx);
                    }
                }
                close(e[y as usize * w + x as usize] as f64, 1.0 - a * inv * inv, 2e-5, "E");
            }
        }
    }
    // dry window: E = 1 everywhere
    let (n, w) = (10, 12);
    let (h, mask) = (vec![0.0f32; n * w], vec![0.0f32; n * w]);
    let (mut tmp, mut e) = (vec![0.0f32; n * w], vec![0.5f32; n * w]);
    wc::edge(&Pl::dense(n, w, &h), &Pl::dense(n, w, &mask), 0.012, 7, &PlM::dense(n, w, &mut tmp), &PlM::dense(n, w, &mut e));
    assert!(e.iter().all(|&x| x == 1.0));
}

/// Inputs of `_k_local` for an n x w window.
struct Local {
    n: usize,
    w: usize,
    h: Vec<f32>,
    s: Vec<f32>,
    s2: Vec<f32>,
    k: Vec<f32>,
    sg: Vec<f32>,
    m: Vec<f32>,
    g: Vec<f32>,
    st: Vec<f32>,
    dk: Vec<f32>,
    ds: Vec<f32>,
    dm: Vec<f32>,
    dg: Vec<f32>,
    dst: Vec<f32>,
    bond: Vec<f32>,
    salt: Vec<f32>,
    e: Vec<f32>,
    gr: Vec<f32>,
    mask: Vec<f32>,
    cxc: Vec<f32>,
    cyc: Vec<f32>,
    u: Vec<f32>,
    v: Vec<f32>,
    evm: Vec<f32>,
}

impl Local {
    fn new(n: usize, w: usize, g: &mut Lcg) -> Self {
        let c = n * w;
        let m = g.vec0(c, 0.0, 1.0, 0.3);
        let gg: Vec<f32> = m.iter().map(|&x| x * g.next() as f32).collect();
        let st: Vec<f32> = m.iter().map(|&x| x * g.next() as f32).collect();
        Local {
            n,
            w,
            h: g.vec0(c, 0.0, 0.6, 0.4),
            s: g.vec0(c, 0.0, 1.0, 0.2),
            s2: vec![0.0; c],
            k: g.vec0(c * NB, 0.0, 2.0, 0.3),
            sg: g.vec0(c * NB, 0.0, 2.0, 0.3),
            m,
            g: gg,
            st,
            dk: g.vec0(c * NB, 0.0, 2.0, 0.3),
            ds: g.vec0(c * NB, 0.0, 2.0, 0.3),
            dm: g.vec0(c, 0.0, 1.0, 0.3),
            dg: g.vec0(c, 0.0, 0.5, 0.3),
            dst: g.vec0(c, 0.0, 0.5, 0.3),
            bond: g.vec0(c, 0.0, 1.0, 0.3),
            salt: g.vec0(c, 0.0, 1.0, 0.8),
            e: g.vec(c, 0.0, 1.0),
            gr: g.vec(c, 0.0, 1.0),
            mask: (0..c).map(|_| if g.next() < 0.05 { 1.0 } else { 0.0 }).collect(),
            cxc: g.vec(c, 0.0, 3.0),
            cyc: g.vec(c, 0.0, 3.0),
            u: g.vec(n * (w + 1), -0.1, 0.1),
            v: g.vec((n + 1) * w, -0.1, 0.1),
            evm: g.vec(c, 0.5, 3.0),
        }
    }

    fn run(&mut self, dt: f64, pdt: f64) {
        let (n, w) = (self.n, self.w);
        let p = prm();
        let h = PlM::dense(n, w, &mut self.h);
        let s2 = PlM::dense(n, w, &mut self.s2);
        let s = Pl::dense(n, w, &self.s);
        let k = SpM::new(n, w, NB, w * NB, &mut self.k);
        let sg = SpM::new(n, w, NB, w * NB, &mut self.sg);
        let dk = SpM::new(n, w, NB, w * NB, &mut self.dk);
        let ds = SpM::new(n, w, NB, w * NB, &mut self.ds);
        let m = PlM::dense(n, w, &mut self.m);
        let g = PlM::dense(n, w, &mut self.g);
        let st = PlM::dense(n, w, &mut self.st);
        let dm = PlM::dense(n, w, &mut self.dm);
        let dg = PlM::dense(n, w, &mut self.dg);
        let dst = PlM::dense(n, w, &mut self.dst);
        let bond = PlM::dense(n, w, &mut self.bond);
        let pg = Pigment { k: &k, s: &sg, m: &m, g: &g, st: &st };
        let dp = Deposit { dk: &dk, ds: &ds, dm: &dm, dg: &dg, dst: &dst, bond: &bond };
        wc::local(
            &h,
            &s,
            &s2,
            &pg,
            &dp,
            &Pl::dense(n, w, &self.salt),
            &Pl::dense(n, w, &self.e),
            &Pl::dense(n, w, &self.gr),
            &Pl::dense(n, w, &self.mask),
            &Pl::dense(n, w, &self.cxc),
            &Pl::dense(n, w, &self.cyc),
            &Pl::dense(n, w + 1, &self.u),
            &Pl::dense(n + 1, w, &self.v),
            &Pl::dense(n, w, &self.evm),
            &p,
            dt,
            0.11 * dt / 0.5,
            pdt,
        );
    }
}

#[test]
fn local_conserves_pigment_and_only_removes_water() {
    let mut g = Lcg(21);
    for &(n, w) in &[(31usize, 29usize), (8, 8), (1, 13)] {
        for &pdt in &[0.0, 4.0] {
            let mut l = Local::new(n, w, &mut g);
            let pig = |l: &Local| -> [f64; 5] {
                [
                    sum32(&l.k) + sum32(&l.dk),
                    sum32(&l.sg) + sum32(&l.ds),
                    sum32(&l.m) + sum32(&l.dm),
                    sum32(&l.g) + sum32(&l.dg),
                    sum32(&l.st) + sum32(&l.dst),
                ]
            };
            let before = pig(&l);
            let water = sum32(&l.h);
            let h0 = l.h.clone();
            l.run(2.0, pdt);
            let after = pig(&l);
            for c in 0..5 {
                close(after[c], before[c], 3e-5 * before[c].max(1.0), &format!("suspended + deposited, channel {c}, pdt {pdt}"));
            }
            assert!(sum32(&l.h) <= water + 1e-6, "absorption and evaporation never add water");
            assert!(l.h.iter().zip(&h0).all(|(a, b)| a <= b), "per pixel too");
            assert!(l.s2.iter().all(|&x| (0.0..=1.0).contains(&x)), "saturation stays in 0..1");
        }
    }
}

#[test]
fn local_dry_paper_wicks_and_dries() {
    // no free water, saturation step: moisture spreads to the dry neighbour, total only falls
    let (n, w) = (5, 9);
    let mut g = Lcg(2);
    let mut l = Local::new(n, w, &mut g);
    l.h = vec![0.0; n * w];
    l.m = vec![0.0; n * w];
    l.mask = vec![0.0; n * w];
    l.salt = vec![0.0; n * w];
    l.s = vec![0.0; n * w];
    l.cxc = vec![1.0; n * w];
    l.cyc = vec![1.0; n * w];
    l.evm = vec![0.0; n * w]; // no evaporation: pure wicking
    for y in 0..n {
        for x in 0..4 {
            l.s[y * w + x] = 0.8;
        }
    }
    let before = sum32(&l.s);
    l.run(1.0, 0.0);
    close(sum32(&l.s2), before, 1e-4, "wicking conserves moisture");
    assert!(l.s2[2 * w + 4] > 0.0, "neighbour of the wet part gets wet");
    assert!(l.s2[2 * w + 3] < 0.8 && l.s2[2 * w] <= 0.8);
    assert_eq!(l.s2[2 * w + 8], 0.0);
}

#[test]
fn salt_diffusion_conserves_salt() {
    let mut g = Lcg(9);
    let (n, w) = (20, 26);
    let salt = g.vec0(n * w, 0.0, 1.0, 0.5);
    let h = g.vec0(n * w, 0.0, 0.5, 0.3);
    let s = g.vec(n * w, 0.0, 1.0);
    let cxc = g.vec(n * w, 0.0, 3.0);
    let cyc = g.vec(n * w, 0.0, 3.0);
    let mut s2 = vec![0.0f32; n * w];
    wc::salt(
        &Pl::dense(n, w, &salt),
        &PlM::dense(n, w, &mut s2),
        &Pl::dense(n, w, &h),
        &Pl::dense(n, w, &s),
        &Pl::dense(n, w, &cxc),
        &Pl::dense(n, w, &cyc),
        0.1,
        0.012,
    );
    // links are symmetric (both ends wet), so what leaves one pixel enters its neighbour
    close(sum32(&s2), sum32(&salt), 2e-5 * sum32(&salt), "salt");
    // dry pixels keep their salt exactly
    for i in 0..n * w {
        if !(h[i] as f64 > 0.012 || s[i] as f64 > 0.35) {
            assert_eq!(s2[i], salt[i]);
        }
    }
}

#[test]
fn active_finds_the_row_extent() {
    let (n, w) = (4, 10);
    let mut h = vec![0.0f32; n * w];
    let mut s = vec![0.0f32; n * w];
    let mut m = vec![0.0f32; n * w];
    h[w + 3] = 0.1;
    s[w + 7] = 0.5;
    m[3 * w + 9] = 1.0;
    let (mut r0, mut r1) = (vec![0i64; n], vec![0i64; n]);
    wc::active(&Pl::dense(n, w, &h), &Pl::dense(n, w, &s), &Pl::dense(n, w, &m), &PlM::new(n, 1, 1, &mut r0), &PlM::new(n, 1, 1, &mut r1));
    assert_eq!(r0, vec![w as i64, 3, w as i64, 9]);
    assert_eq!(r1, vec![-1, 7, -1, 9]);
}

#[test]
fn settle_all_deposits_everything() {
    let mut g = Lcg(4);
    let (n, w) = (10, 11);
    let mut l = Local::new(n, w, &mut g);
    let tot = |l: &Local| {
        [
            sum32(&l.k) + sum32(&l.dk),
            sum32(&l.sg) + sum32(&l.ds),
            sum32(&l.m) + sum32(&l.dm),
            sum32(&l.g) + sum32(&l.dg),
            sum32(&l.st) + sum32(&l.dst),
        ]
    };
    let before = tot(&l);
    {
        let k = SpM::new(n, w, NB, w * NB, &mut l.k);
        let sg = SpM::new(n, w, NB, w * NB, &mut l.sg);
        let dk = SpM::new(n, w, NB, w * NB, &mut l.dk);
        let ds = SpM::new(n, w, NB, w * NB, &mut l.ds);
        let (m, gg, st) = (PlM::dense(n, w, &mut l.m), PlM::dense(n, w, &mut l.g), PlM::dense(n, w, &mut l.st));
        let (dm, dg, dst) = (PlM::dense(n, w, &mut l.dm), PlM::dense(n, w, &mut l.dg), PlM::dense(n, w, &mut l.dst));
        let (h, s) = (PlM::dense(n, w, &mut l.h), PlM::dense(n, w, &mut l.s));
        let pg = Pigment { k: &k, s: &sg, m: &m, g: &gg, st: &st };
        wc::settle_all(&pg, &dk, &ds, &dm, &dg, &dst, &h, &s);
    }
    let after = tot(&l);
    for c in 0..5 {
        close(after[c], before[c], 1e-5 * before[c], "settled pigment");
    }
    for v in [&l.k, &l.sg, &l.m, &l.g, &l.st, &l.h, &l.s] {
        assert!(v.iter().all(|&x| x == 0.0), "nothing suspended, no water");
    }
}

#[test]
fn fibres_count_links_along_a_straight_fibre() {
    // one fibre along +x on row 3 starting at x = 5.5: 20 steps of 0.5, crossing x = 6 .. 15
    let (hh, ww) = (8i64, 30i64);
    let (mut cx, mut cy, mut cov) = (vec![0.0f64; 8 * 30], vec![0.0f64; 8 * 30], vec![0.0f64; 8 * 30]);
    wc::fibres(
        hh,
        ww,
        &[5.5],
        &[3.5],
        &[0.0],
        &[10.0],
        &[0.0],
        &[2.0],
        &PlM::dense(8, 30, &mut cx),
        &PlM::dense(8, 30, &mut cy),
        &PlM::dense(8, 30, &mut cov),
    );
    let row3: f64 = cx[3 * 30..4 * 30].iter().sum();
    close(row3, 2.0 * 10.0, 1e-12, "x links of the fibre (10 crossings of weight 2)");
    assert!(cy.iter().all(|&x| x == 0.0), "a horizontal fibre has no y links");
    close(cov.iter().sum::<f64>(), 2.0 * 20.0, 1e-12, "coverage (20 steps)");
    // fibres leaving the canvas add nothing outside it and do not panic
    let (mut cx, mut cy, mut cov) = (vec![0.0f64; 8 * 30], vec![0.0f64; 8 * 30], vec![0.0f64; 8 * 30]);
    wc::fibres(
        hh,
        ww,
        &[-30.0, 15.0],
        &[3.5, 12.0],
        &[0.0, 1.0],
        &[10.0, 20.0],
        &[0.0, 0.1],
        &[1.0, 1.0],
        &PlM::dense(8, 30, &mut cx),
        &PlM::dense(8, 30, &mut cy),
        &PlM::dense(8, 30, &mut cov),
    );
}

#[test]
fn strided_windows_give_the_same_answer_as_dense_copies() {
    // a window of a larger array (rows not contiguous) must behave like a compact copy
    let mut g = Lcg(31);
    let (n, w) = (17, 23);
    let (bn, bw) = (n + 5, w + 9);
    let mut l = Local::new(n, w, &mut g);
    let (y0, x0) = (2usize, 3usize);
    // embed h, s and the water-related inputs in larger arrays
    let embed = |src: &[f32], cols: usize, rows: usize, bcols: usize| -> Vec<f32> {
        let mut big = vec![-7.0f32; (rows + 5) * bcols];
        for y in 0..rows {
            for x in 0..cols {
                big[(y + y0) * bcols + x + x0] = src[y * cols + x];
            }
        }
        big
    };
    let hb = embed(&l.h, w, n, bw);
    let sb = embed(&l.s, w, n, bw);
    let off = y0 * bw + x0;
    let hv = Pl::new(n, w, bw, &hb[off..]);
    let sv = Pl::new(n, w, bw, &sb[off..]);
    for y in 0..n {
        for x in 0..w {
            assert_eq!(hv.at(y, x), l.h[y * w + x]);
            assert_eq!(sv.at(y, x), l.s[y * w + x]);
        }
    }
    // active() on strided windows equals active() on dense ones
    let m = vec![0.0f32; n * w];
    let mb = embed(&m, w, n, bw);
    let (mut a0, mut a1, mut b0, mut b1) = (vec![0i64; n], vec![0i64; n], vec![0i64; n], vec![0i64; n]);
    wc::active(&hv, &sv, &Pl::new(n, w, bw, &mb[off..]), &PlM::new(n, 1, 1, &mut a0), &PlM::new(n, 1, 1, &mut a1));
    wc::active(&Pl::dense(n, w, &l.h), &Pl::dense(n, w, &l.s), &Pl::dense(n, w, &m), &PlM::new(n, 1, 1, &mut b0), &PlM::new(n, 1, 1, &mut b1));
    assert_eq!((a0, a1), (b0, b1));
    // a strided 3-D view
    let kb: Vec<f32> = (0..bn * bw * NB).map(|i| i as f32).collect();
    let kv = Sp::new(n, w, NB, bw * NB, &kb[(y0 * bw + x0) * NB..]);
    assert_eq!(kv.at(3, 4, 5), ((3 + y0) * bw * NB + (4 + x0) * NB + 5) as f32);
    l.run(1.0, 0.0); // keep `l` used mutably
}

/// Run a closure with serial and with parallel loops and require identical bits.
fn same_serial_and_parallel<T: PartialEq + std::fmt::Debug>(what: &str, mut f: impl FnMut() -> T) {
    wc::set_force_serial(true);
    let a = f();
    wc::set_force_serial(false);
    let b = f();
    assert_eq!(a, b, "{what}: serial and parallel runs differ");
}

#[test]
fn parallel_and_serial_loops_agree_bit_for_bit() {
    // a window above SERIAL_BELOW, so the rayon path really runs
    let (n, w) = (310usize, 330usize);
    assert!(n * w >= wc::SERIAL_BELOW);
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    same_serial_and_parallel("flow", || {
        let mut f = Flow::new(n, w, &mut Lcg(77));
        for step in 0..3 {
            f.step(0.001, 0.0, step == 1);
        }
        (bits(&f.h), bits(&f.u), bits(&f.v), bits(&f.ax), bits(&f.ay), bits(&f.rowtile))
    });
    same_serial_and_parallel("xpass + ypass", || {
        let mut p = Pig::new(n, w, false, &mut Lcg(78));
        p.run(false, 0.022, true);
        let mut q = Pig::new(n, w, true, &mut Lcg(79));
        q.run(true, 0.022, true);
        (bits(&p.k), bits(&p.m), bits(&p.acc), bits(&q.k), bits(&q.m), bits(&q.acc))
    });
    same_serial_and_parallel("local", || {
        let mut l = Local::new(n, w, &mut Lcg(80));
        l.run(2.0, 3.0);
        (bits(&l.h), bits(&l.s2), bits(&l.k), bits(&l.dk), bits(&l.m), bits(&l.bond))
    });
    same_serial_and_parallel("edge", || {
        let mut g = Lcg(81);
        let h = g.vec0(n * w, 0.0, 0.5, 0.5);
        let mask = vec![0.0f32; n * w];
        let (mut tmp, mut e) = (vec![0.0f32; n * w], vec![0.0f32; n * w]);
        wc::edge(&Pl::dense(n, w, &h), &Pl::dense(n, w, &mask), 0.012, 7, &PlM::dense(n, w, &mut tmp), &PlM::dense(n, w, &mut e));
        bits(&e)
    });
    wc::set_force_serial(false);
}
