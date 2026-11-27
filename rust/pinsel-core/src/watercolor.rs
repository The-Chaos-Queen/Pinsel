//! Port of pinsel (Python 0.1) kernels: watercolor. See docs/RUST_PORTING_GUIDE.md.
//!
//! Every function mirrors one numba kernel of `pinsel/watercolor.py`, with the
//! same arguments in the same order, and is written to be bit-identical to
//! that kernel compiled without fastmath. Number types follow numba's typing:
//! the canvas arrays are float32; a float32 element mixed with a float64
//! scalar is computed in f64 and rounded on store; float32 op float32 stays
//! f32 (for example `F / da`, `mb / hb`, `tmp[a] - tmp[b]`, `abs(u) + abs(u)`).
//! Expressions keep the operation order of the Python source.
//!
//! Canvas access goes through the stride-aware views below (`Pl`, `PlM`,
//! `Sp`, `SpM`). The simulation kernels receive *windows* (`wc.water[y0:y1,
//! x0:x1]`, ...), whose rows are not contiguous, so `grid::Plane` (which
//! needs `len == h * w`) cannot hold them; these views add a row stride and
//! are otherwise used the same way (`at`, `set`, `px`). Kernels never index
//! raw slices with `y * w + x`.
//!
//! Parallelism: `prange` loops become `par_for` (rayon) over rows (or over
//! column blocks for the y-pass and the vertical blur). Every one of them
//! was checked for overlapping writes; see the notes at each kernel.

use rayon::prelude::*;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicI8, Ordering};

// --------------------------------------------------------------------------
// Stride-aware views
// --------------------------------------------------------------------------

#[inline]
fn span(h: usize, w: usize, c: usize, rs: usize) -> usize {
    if h == 0 || w == 0 || c == 0 {
        0
    } else {
        (h - 1) * rs + w * c
    }
}

/// Read-only H x W window with row stride `rs` (in elements).
#[derive(Clone, Copy)]
pub struct Pl<'a, T> {
    d: &'a [T],
    h: usize,
    w: usize,
    rs: usize,
}

impl<'a, T: Copy> Pl<'a, T> {
    pub fn new(h: usize, w: usize, rs: usize, d: &'a [T]) -> Self {
        assert!(rs >= w && d.len() >= span(h, w, 1, rs), "Pl: slice too short for the window");
        Pl { d, h, w, rs }
    }
    /// Contiguous H x W.
    pub fn dense(h: usize, w: usize, d: &'a [T]) -> Self {
        Self::new(h, w, w, d)
    }
    #[inline]
    pub fn height(&self) -> usize {
        self.h
    }
    #[inline]
    pub fn width(&self) -> usize {
        self.w
    }
    #[inline]
    pub fn at(&self, y: usize, x: usize) -> T {
        assert!(y < self.h && x < self.w);
        self.d[y * self.rs + x]
    }
}

/// Mutable H x W window. Methods take `&self` so that row-parallel kernels
/// can share it; callers must keep concurrent writes on disjoint cells (each
/// parallel kernel below documents why its rows are disjoint).
pub struct PlM<'a, T> {
    p: *mut T,
    h: usize,
    w: usize,
    rs: usize,
    _m: PhantomData<&'a mut [T]>,
}
unsafe impl<T: Send> Send for PlM<'_, T> {}
unsafe impl<T: Send> Sync for PlM<'_, T> {}

impl<'a, T: Copy> PlM<'a, T> {
    pub fn new(h: usize, w: usize, rs: usize, d: &'a mut [T]) -> Self {
        assert!(rs >= w && d.len() >= span(h, w, 1, rs), "PlM: slice too short for the window");
        PlM { p: d.as_mut_ptr(), h, w, rs, _m: PhantomData }
    }
    pub fn dense(h: usize, w: usize, d: &'a mut [T]) -> Self {
        Self::new(h, w, w, d)
    }
    #[inline]
    pub fn height(&self) -> usize {
        self.h
    }
    #[inline]
    pub fn width(&self) -> usize {
        self.w
    }
    #[inline]
    pub fn at(&self, y: usize, x: usize) -> T {
        assert!(y < self.h && x < self.w);
        // SAFETY: in bounds of the slice given to `new`.
        unsafe { *self.p.add(y * self.rs + x) }
    }
    #[inline]
    pub fn set(&self, y: usize, x: usize, v: T) {
        assert!(y < self.h && x < self.w);
        // SAFETY: in bounds; concurrent callers write disjoint cells.
        unsafe { *self.p.add(y * self.rs + x) = v }
    }
}

/// Read-only H x W x C window with row stride `rs` (in elements).
#[derive(Clone, Copy)]
pub struct Sp<'a, T> {
    d: &'a [T],
    h: usize,
    w: usize,
    c: usize,
    rs: usize,
}

impl<'a, T: Copy> Sp<'a, T> {
    pub fn new(h: usize, w: usize, c: usize, rs: usize, d: &'a [T]) -> Self {
        assert!(rs >= w * c && d.len() >= span(h, w, c, rs), "Sp: slice too short for the window");
        Sp { d, h, w, c, rs }
    }
    #[inline]
    pub fn height(&self) -> usize {
        self.h
    }
    #[inline]
    pub fn width(&self) -> usize {
        self.w
    }
    #[inline]
    pub fn channels(&self) -> usize {
        self.c
    }
    #[inline]
    pub fn at(&self, y: usize, x: usize, b: usize) -> T {
        assert!(y < self.h && x < self.w && b < self.c);
        self.d[y * self.rs + x * self.c + b]
    }
}

/// Mutable H x W x C window (same sharing contract as [`PlM`]).
pub struct SpM<'a, T> {
    p: *mut T,
    h: usize,
    w: usize,
    c: usize,
    rs: usize,
    _m: PhantomData<&'a mut [T]>,
}
unsafe impl<T: Send> Send for SpM<'_, T> {}
unsafe impl<T: Send> Sync for SpM<'_, T> {}

impl<'a, T: Copy> SpM<'a, T> {
    pub fn new(h: usize, w: usize, c: usize, rs: usize, d: &'a mut [T]) -> Self {
        assert!(rs >= w * c && d.len() >= span(h, w, c, rs), "SpM: slice too short for the window");
        SpM { p: d.as_mut_ptr(), h, w, c, rs, _m: PhantomData }
    }
    #[inline]
    pub fn height(&self) -> usize {
        self.h
    }
    #[inline]
    pub fn width(&self) -> usize {
        self.w
    }
    #[inline]
    pub fn channels(&self) -> usize {
        self.c
    }
    #[inline]
    pub fn at(&self, y: usize, x: usize, b: usize) -> T {
        assert!(y < self.h && x < self.w && b < self.c);
        // SAFETY: in bounds of the slice given to `new`.
        unsafe { *self.p.add(y * self.rs + x * self.c + b) }
    }
    #[inline]
    pub fn set(&self, y: usize, x: usize, b: usize, v: T) {
        assert!(y < self.h && x < self.w && b < self.c);
        // SAFETY: in bounds; concurrent callers write disjoint cells.
        unsafe { *self.p.add(y * self.rs + x * self.c + b) = v }
    }
}

// --------------------------------------------------------------------------
// Parallel loop helper
// --------------------------------------------------------------------------

/// Windows below this many cells run serially (as `_SERIAL_BELOW` in Python).
pub const SERIAL_BELOW: usize = 90_000;

static SERIAL: AtomicI8 = AtomicI8::new(-1); // -1: not decided yet

/// Force serial execution (tests compare serial and parallel runs; the
/// environment variable `PINSEL_RS_SERIAL=1` does the same for a whole run).
pub fn set_force_serial(on: bool) {
    SERIAL.store(on as i8, Ordering::Relaxed);
}

fn force_serial() -> bool {
    match SERIAL.load(Ordering::Relaxed) {
        -1 => {
            let on = std::env::var("PINSEL_RS_SERIAL").map(|v| !v.is_empty() && v != "0").unwrap_or(false);
            SERIAL.store(on as i8, Ordering::Relaxed);
            on
        }
        v => v == 1,
    }
}

/// `for i in prange(n)`: runs `f(i)` for every i, in parallel when the window
/// is big enough. The caller guarantees that different i write disjoint cells.
fn par_for<F: Fn(usize) + Sync + Send>(n: usize, cells: usize, f: F) {
    if cells >= SERIAL_BELOW && n > 1 && !force_serial() && rayon::current_num_threads() > 1 {
        (0..n).into_par_iter().for_each(f);
    } else {
        for i in 0..n {
            f(i);
        }
    }
}

// --------------------------------------------------------------------------
// Constants (indices into the packed float64 parameter vector)
// --------------------------------------------------------------------------

/// Activity tile size (px), `_TS`.
pub const TS: usize = 16;

const I_G: usize = 0;
const I_FR: usize = 1;
const I_HV: usize = 2;
const I_HMIN: usize = 3;
const I_HWET: usize = 4;
const I_WW: usize = 5;
const I_HADV: usize = 6;
const I_ADVW: usize = 7;
const I_THETA: usize = 8;
const I_ABS: usize = 9;
const I_CAP: usize = 10;
const I_EV: usize = 11;
const I_EDGE: usize = 12;
const I_EVD: usize = 13;
const I_DEP: usize = 14;
const I_H0: usize = 15;
const I_DEPD: usize = 16;
const I_DES: usize = 17;
const I_BOND: usize = 18;
const I_BT: usize = 19;
const I_SOFT: usize = 20;
const I_GRAN: usize = 21;
const I_SABS: usize = 22;
const I_EDEP: usize = 23;
const I_SHEAR: usize = 24;

// --------------------------------------------------------------------------
// Inline helpers
// --------------------------------------------------------------------------

#[inline]
fn smooth01(t: f64) -> f64 {
    if t <= 0.0 {
        0.0
    } else if t >= 1.0 {
        1.0
    } else {
        t * t * (3.0 - 2.0 * t)
    }
}

/// `_gate`: how readily free water may enter a pixel that has none.
#[inline]
fn gate(h_src: f32, s_dst: f32, th_dst: f32, rho_dst: f32, mask_dst: f32, prm: &[f64]) -> f64 {
    if mask_dst as f64 >= 0.5 {
        return 0.0;
    }
    let mut w = ((s_dst - th_dst) as f64) / prm[I_WW];
    if w <= 0.0 {
        w = 0.0;
    } else if w >= 1.0 {
        w = 1.0;
    } else {
        w = w * w;
    }
    w *= rho_dst as f64;
    let hadv = prm[I_HADV] * (1.0 + 1.6 * (th_dst as f64 - prm[I_THETA]));
    let a = smooth01((h_src as f64 - hadv) / prm[I_ADVW]);
    if a > w {
        w = a;
    }
    w * (1.0 - mask_dst as f64)
}

/// `_face`: new raw virtual-pipe flux a -> b over one face. `ha_pot` is
/// `h[a] + hp[a] + tilt` (f32 sum, then + tilt in f64), `hb_pot` is
/// `h[b] + hp[b]` (f32).
#[inline]
#[allow(clippy::too_many_arguments)]
fn face(
    f: f32,
    ha: f32,
    hb: f32,
    ha_pot: f64,
    hb_pot: f32,
    c: f32,
    sa: f32,
    sb: f32,
    tha: f32,
    thb: f32,
    ra: f32,
    rb: f32,
    ma: f32,
    mb: f32,
    prm: &[f64],
    g: f64,
    damp: f64,
) -> f64 {
    let hmin = prm[I_HMIN];
    if (ha as f64 <= hmin && hb as f64 <= hmin) || ma as f64 >= 0.5 || mb as f64 >= 0.5 {
        return 0.0;
    }
    let hav = 0.5 * ((ha + hb) as f64);
    let mob = hav / (hav + prm[I_HV]);
    let mut f = (f as f64 * damp + g * (c as f64) * (ha_pot - hb_pot as f64)) * mob;
    let hwet = prm[I_HWET];
    if f > 0.0 {
        if ha as f64 <= hmin {
            return 0.0;
        }
        if hb as f64 <= hwet {
            f *= gate(ha, sb, thb, rb, mb, prm);
        }
    } else if f < 0.0 {
        if hb as f64 <= hmin {
            return 0.0;
        }
        if ha as f64 <= hwet {
            f *= gate(hb, sa, tha, ra, ma, prm);
        }
    }
    f
}

#[inline]
fn tile_on(tiles: &Pl<bool>, y: usize, x: usize) -> bool {
    tiles.at(y / TS, x / TS)
}

// --------------------------------------------------------------------------
// Flow
// --------------------------------------------------------------------------

/// `_k_flow1`: raw pipe fluxes (ur, vr) and each pixel's outflow limiter r.
///
/// prange check: loop 1 row y writes `rowtile[y]`, `ur[y, :]`, `vr[y + 1, :]`
/// (its own bottom faces), plus `u[y, 0]`, `u[y, w]` and, only for y == 0 or
/// y == n-1, `v[0, :]` / `v[n, :]`; it reads `v[y + 1, :]` only for y < n-1,
/// which nobody writes. Loop 2 writes `r[y, :]` and reads the finished
/// ur/vr. Rows are disjoint; no reduction. Safe to parallelise by rows.
#[allow(clippy::too_many_arguments)]
pub fn flow1(
    h: &Pl<f32>,
    s: &Pl<f32>,
    hp: &Pl<f32>,
    cxf: &Pl<f32>,
    cyf: &Pl<f32>,
    th: &Pl<f32>,
    rho: &Pl<f32>,
    mask: &Pl<f32>,
    u: &PlM<f32>,
    v: &PlM<f32>,
    ur: &PlM<f32>,
    vr: &PlM<f32>,
    r: &PlM<f32>,
    prm: &[f64],
    tx: f64,
    ty: f64,
    wall: bool,
    use_tiles: bool,
    tiles: &Pl<bool>,
    rowtile: &PlM<f32>,
) {
    let (n, w) = (h.height(), h.width());
    let g = prm[I_G];
    let damp = 1.0 - prm[I_FR];
    par_for(n, n * w, |y| {
        for t in 0..rowtile.width() {
            rowtile.set(y, t, 0.0);
        }
        if wall {
            u.set(y, 0, 0.0);
            u.set(y, w, 0.0);
            if y == 0 {
                for x in 0..w {
                    v.set(0, x, 0.0);
                }
            }
            if y == n - 1 {
                for x in 0..w {
                    v.set(n, x, 0.0);
                }
            }
        }
        ur.set(y, 0, u.at(y, 0));
        ur.set(y, w, u.at(y, w));
        if y == 0 {
            for x in 0..w {
                vr.set(0, x, v.at(0, x));
            }
        }
        // x faces of this row
        for x in 0..w.saturating_sub(1) {
            if use_tiles && !(tile_on(tiles, y, x) || tile_on(tiles, y, x + 1)) {
                ur.set(y, x + 1, u.at(y, x + 1));
                continue;
            }
            let f = face(
                u.at(y, x + 1),
                h.at(y, x),
                h.at(y, x + 1),
                (h.at(y, x) + hp.at(y, x)) as f64 + tx,
                h.at(y, x + 1) + hp.at(y, x + 1),
                cxf.at(y, x),
                s.at(y, x),
                s.at(y, x + 1),
                th.at(y, x),
                th.at(y, x + 1),
                rho.at(y, x),
                rho.at(y, x + 1),
                mask.at(y, x),
                mask.at(y, x + 1),
                prm,
                g,
                damp,
            );
            ur.set(y, x + 1, f as f32);
            let a = f.abs();
            let t = x / TS;
            if a > rowtile.at(y, t) as f64 {
                rowtile.set(y, t, a as f32);
            }
        }
        // y faces: bottom face owned by this row
        if y + 1 < n {
            for x in 0..w {
                if use_tiles && !(tile_on(tiles, y, x) || tile_on(tiles, y + 1, x)) {
                    vr.set(y + 1, x, v.at(y + 1, x));
                    continue;
                }
                let fb = face(
                    v.at(y + 1, x),
                    h.at(y, x),
                    h.at(y + 1, x),
                    (h.at(y, x) + hp.at(y, x)) as f64 + ty,
                    h.at(y + 1, x) + hp.at(y + 1, x),
                    cyf.at(y, x),
                    s.at(y, x),
                    s.at(y + 1, x),
                    th.at(y, x),
                    th.at(y + 1, x),
                    rho.at(y, x),
                    rho.at(y + 1, x),
                    mask.at(y, x),
                    mask.at(y + 1, x),
                    prm,
                    g,
                    damp,
                );
                vr.set(y + 1, x, fb as f32);
                let a = fb.abs();
                let t = x / TS;
                if a > rowtile.at(y, t) as f64 {
                    rowtile.set(y, t, a as f32);
                }
            }
        } else {
            for x in 0..w {
                vr.set(n, x, v.at(n, x));
            }
        }
    });
    // outflow limiter: no pixel gives away more than 60 % of its water
    par_for(n, n * w, |y| {
        for x in 0..w {
            let mut out = 0.0f64;
            let mut f = ur.at(y, x + 1);
            if f > 0.0 {
                out += f as f64;
            }
            f = ur.at(y, x);
            if f < 0.0 {
                out -= f as f64;
            }
            f = vr.at(y + 1, x);
            if f > 0.0 {
                out += f as f64;
            }
            f = vr.at(y, x);
            if f < 0.0 {
                out -= f as f64;
            }
            let avail = 0.6 * h.at(y, x) as f64;
            if out > avail && out > 0.0 {
                r.set(y, x, (avail / out) as f32);
            } else {
                r.set(y, x, 1.0);
            }
        }
    });
}

/// `_k_flow2`: limit the raw fluxes, store them, accumulate the moved
/// volumes and move the water.
///
/// prange check: row y writes `u[y, x+1]`, `ax[y, x+1]`, `v[y+1, x]`,
/// `ay[y+1, x]` and `h[y, x]`; it reads ur/vr/r (not written here) and `h`
/// only at its own cell. Rows are disjoint; no reduction.
#[allow(clippy::too_many_arguments)]
pub fn flow2(
    h: &PlM<f32>,
    u: &PlM<f32>,
    v: &PlM<f32>,
    ur: &Pl<f32>,
    vr: &Pl<f32>,
    r: &Pl<f32>,
    ax: &PlM<f32>,
    ay: &PlM<f32>,
    use_tiles: bool,
    tiles: &Pl<bool>,
) {
    let (n, w) = (h.height(), h.width());
    par_for(n, n * w, |y| {
        for x in 0..w {
            let mut d = 0.0f64;
            let on = !use_tiles || tile_on(tiles, y, x);
            if x > 0 && (on || tile_on(tiles, y, x - 1)) {
                let mut f = ur.at(y, x);
                f *= if f > 0.0 { r.at(y, x - 1) } else { r.at(y, x) };
                d += f as f64;
            }
            if x + 1 < w && (on || tile_on(tiles, y, x + 1)) {
                let mut f = ur.at(y, x + 1);
                f *= if f > 0.0 { r.at(y, x) } else { r.at(y, x + 1) };
                let mut fd = f as f64;
                if -1e-8 < fd && fd < 1e-8 {
                    fd = 0.0;
                }
                u.set(y, x + 1, fd as f32);
                ax.set(y, x + 1, (ax.at(y, x + 1) as f64 + fd) as f32);
                d -= fd;
            }
            if y > 0 && (on || tile_on(tiles, y - 1, x)) {
                let mut f = vr.at(y, x);
                f *= if f > 0.0 { r.at(y - 1, x) } else { r.at(y, x) };
                d += f as f64;
            }
            if y + 1 < n && (on || tile_on(tiles, y + 1, x)) {
                let mut f = vr.at(y + 1, x);
                f *= if f > 0.0 { r.at(y, x) } else { r.at(y + 1, x) };
                let mut fd = f as f64;
                if -1e-8 < fd && fd < 1e-8 {
                    fd = 0.0;
                }
                v.set(y + 1, x, fd as f32);
                ay.set(y + 1, x, (ay.at(y + 1, x) as f64 + fd) as f32);
                d -= fd;
            }
            if d != 0.0 {
                let hv = h.at(y, x) as f64 + d;
                h.set(y, x, if hv > 0.0 { hv as f32 } else { 0.0 });
            }
        }
    });
}

/// `_k_outflow`: den = water before the accumulated outflows left.
/// Row y writes only `den[y, :]`.
pub fn outflow(h: &Pl<f32>, ax: &Pl<f32>, ay: &Pl<f32>, den: &PlM<f32>) {
    let (n, w) = (h.height(), h.width());
    par_for(n, n * w, |y| {
        for x in 0..w {
            let mut o = 0.0f64;
            if x + 1 < w {
                let f = ax.at(y, x + 1);
                if f > 0.0 {
                    o += f as f64;
                }
            }
            if x > 0 {
                let f = ax.at(y, x);
                if f < 0.0 {
                    o -= f as f64;
                }
            }
            if y + 1 < n {
                let f = ay.at(y + 1, x);
                if f > 0.0 {
                    o += f as f64;
                }
            }
            if y > 0 {
                let f = ay.at(y, x);
                if f < 0.0 {
                    o -= f as f64;
                }
            }
            den.set(y, x, (h.at(y, x) as f64 + o) as f32);
        }
    });
}

// --------------------------------------------------------------------------
// Pigment transport
// --------------------------------------------------------------------------

/// The pigment channels moved with the water: K, S (nb each), m, g, st.
pub struct Pigment<'a, 'b> {
    pub k: &'b SpM<'a, f32>,
    pub s: &'b SpM<'a, f32>,
    pub m: &'b PlM<'a, f32>,
    pub g: &'b PlM<'a, f32>,
    pub st: &'b PlM<'a, f32>,
}

impl Pigment<'_, '_> {
    /// `_load`: pixel (y, x) as 2 * nb + 3 float64 channels.
    #[inline]
    fn load(&self, buf: &mut [f64], y: usize, x: usize, nb: usize) {
        for b in 0..nb {
            buf[b] = self.k.at(y, x, b) as f64;
            buf[nb + b] = self.s.at(y, x, b) as f64;
        }
        buf[2 * nb] = self.m.at(y, x) as f64;
        buf[2 * nb + 1] = self.g.at(y, x) as f64;
        buf[2 * nb + 2] = self.st.at(y, x) as f64;
    }

    /// Transfer between pixel a = (ya, xa) and b = (yb, xb): `a_old`, `b_old`
    /// hold the pre-update values of the two pixels.
    #[inline]
    fn transfer(&self, fa: f64, fb: f64, a_old: &[f64], b_old: &[f64], a: (usize, usize), b: (usize, usize), nb: usize) {
        for i in 0..nb {
            let t = fa * a_old[i] - fb * b_old[i];
            self.k.set(a.0, a.1, i, (self.k.at(a.0, a.1, i) as f64 - t) as f32);
            self.k.set(b.0, b.1, i, (self.k.at(b.0, b.1, i) as f64 + t) as f32);
            let t = fa * a_old[nb + i] - fb * b_old[nb + i];
            self.s.set(a.0, a.1, i, (self.s.at(a.0, a.1, i) as f64 - t) as f32);
            self.s.set(b.0, b.1, i, (self.s.at(b.0, b.1, i) as f64 + t) as f32);
        }
        let t = fa * a_old[2 * nb] - fb * b_old[2 * nb];
        self.m.set(a.0, a.1, (self.m.at(a.0, a.1) as f64 - t) as f32);
        self.m.set(b.0, b.1, (self.m.at(b.0, b.1) as f64 + t) as f32);
        let t = fa * a_old[2 * nb + 1] - fb * b_old[2 * nb + 1];
        self.g.set(a.0, a.1, (self.g.at(a.0, a.1) as f64 - t) as f32);
        self.g.set(b.0, b.1, (self.g.at(b.0, b.1) as f64 + t) as f32);
        let t = fa * a_old[2 * nb + 2] - fb * b_old[2 * nb + 2];
        self.st.set(a.0, a.1, (self.st.at(a.0, a.1) as f64 - t) as f32);
        self.st.set(b.0, b.1, (self.st.at(b.0, b.1) as f64 + t) as f32);
    }
}

/// `_link_terms`: pigment-carrying fractions over one link (fa, fb).
#[inline]
#[allow(clippy::too_many_arguments)]
fn link_terms(
    f: f32,
    ha: f32,
    hb: f32,
    da: f32,
    db: f32,
    ma: f64,
    mb: f32,
    sa: f32,
    sb: f32,
    c: f32,
    cmix: f64,
    csalt: f64,
    hwet: f64,
    salt_on: bool,
) -> (f64, f64) {
    if c <= 0.0 {
        return (0.0, 0.0);
    }
    let mut fa = 0.0f64;
    let mut fb = 0.0f64;
    if f > 0.0 {
        if da as f64 > 1e-9 {
            fa = ((f / da) as f64) * (ha as f64 / (ha as f64 + 0.05));
        }
    } else if f < 0.0 && db as f64 > 1e-9 {
        fb = (((-f) / db) as f64) * (hb as f64 / (hb as f64 + 0.05));
    }
    if ha as f64 > hwet && hb as f64 > hwet {
        if cmix > 0.0 {
            let ca = ma / ha as f64;
            let cb = mb / hb; // f32 / f32
            let dc = ca - cb as f64;
            let tol = 0.015 * (ca + cb as f64) + 1e-5;
            if dc > tol || dc < -tol {
                let mut q = cmix * c as f64;
                if q > 0.06 {
                    q = 0.06;
                }
                let hm = if ha < hb { ha } else { hb };
                let qq = q * hm as f64;
                fa += qq / da as f64;
                fb += qq / db as f64;
            }
        }
        if salt_on {
            let mut d = csalt * ((sa - sb) as f64);
            if d > 0.035 {
                d = 0.035;
            }
            if d < -0.035 {
                d = -0.035;
            }
            if d > 0.0 {
                fa += d * ha as f64 / da as f64;
            } else {
                fb -= d * hb as f64 / db as f64;
            }
        }
    }
    if fa < 2.5e-3 && fb < 2.5e-3 {
        return (0.0, 0.0);
    }
    (fa, fb)
}

/// `_k_xpass`: move pigment over x faces; accumulated volumes `ax` are
/// consumed. Along a row the pre-update pigment of the current pixel is kept,
/// so every transfer uses old values (exact upwinding).
///
/// prange check: row y reads and writes only row y of K, S, m, g, st and
/// `ax[y, :]`; h, den, salt, cxf are read only. Rows are disjoint.
#[allow(clippy::too_many_arguments)]
pub fn xpass(
    h: &Pl<f32>,
    den: &Pl<f32>,
    pg: &Pigment,
    ax: &PlM<f32>,
    salt: &Pl<f32>,
    cxf: &Pl<f32>,
    cmix: f64,
    csalt: f64,
    hwet: f64,
    salt_on: bool,
) {
    let (n, w) = (h.height(), h.width());
    let nb = pg.k.channels();
    let nc = 2 * nb + 3;
    par_for(n, n * w, |y| {
        let mut a = vec![0.0f64; nc];
        let mut b = vec![0.0f64; nc];
        let mut have = false;
        for x in 0..w.saturating_sub(1) {
            let f = ax.at(y, x + 1);
            let ha = h.at(y, x);
            let hb = h.at(y, x + 1);
            if f == 0.0 && (ha as f64 <= hwet || hb as f64 <= hwet) {
                have = false;
                continue;
            }
            let ma = if have { a[2 * nb] } else { pg.m.at(y, x) as f64 };
            let (fa, fb) = link_terms(
                f,
                ha,
                hb,
                den.at(y, x),
                den.at(y, x + 1),
                ma,
                pg.m.at(y, x + 1),
                salt.at(y, x),
                salt.at(y, x + 1),
                cxf.at(y, x),
                cmix,
                csalt,
                hwet,
                salt_on,
            );
            if fa == 0.0 && fb == 0.0 {
                have = false;
                continue;
            }
            ax.set(y, x + 1, 0.0);
            if !have {
                pg.load(&mut a, y, x, nb);
            }
            pg.load(&mut b, y, x + 1, nb);
            pg.transfer(fa, fb, &a, &b, (y, x), (y, x + 1), nb);
            a.copy_from_slice(&b);
            have = true;
        }
    });
}

/// `_k_ypass`: move pigment over y faces; blocks of 16 columns are
/// independent (a column's state only depends on that column).
///
/// prange check: block bi reads and writes only columns xa0..xb0 of K, S, m,
/// g, st, and `ay[:, xa0..xb0]`; the other arguments are read only. Column
/// blocks are disjoint. (Rows cannot be split here: a column is a sequential
/// chain down y.)
#[allow(clippy::too_many_arguments)]
pub fn ypass(
    h: &Pl<f32>,
    den: &Pl<f32>,
    pg: &Pigment,
    ay: &PlM<f32>,
    salt: &Pl<f32>,
    cyf: &Pl<f32>,
    cmix: f64,
    csalt: f64,
    hwet: f64,
    salt_on: bool,
) {
    const BW: usize = 16;
    let (n, w) = (h.height(), h.width());
    let nb = pg.k.channels();
    let nc = 2 * nb + 3;
    let nblk = (w + BW - 1) / BW;
    par_for(nblk, n * w, |bi| {
        let xa0 = bi * BW;
        let xb0 = w.min(xa0 + BW);
        let mut a = vec![0.0f64; BW * nc];
        let mut b = vec![0.0f64; nc];
        let mut have = [false; BW];
        for y in 0..n.saturating_sub(1) {
            let y1 = y + 1;
            for x in xa0..xb0 {
                let k = x - xa0;
                let f = ay.at(y1, x);
                let ha = h.at(y, x);
                let hb = h.at(y1, x);
                if f == 0.0 && (ha as f64 <= hwet || hb as f64 <= hwet) {
                    have[k] = false;
                    continue;
                }
                let ma = if have[k] { a[k * nc + 2 * nb] } else { pg.m.at(y, x) as f64 };
                let (fa, fb) = link_terms(
                    f,
                    ha,
                    hb,
                    den.at(y, x),
                    den.at(y1, x),
                    ma,
                    pg.m.at(y1, x),
                    salt.at(y, x),
                    salt.at(y1, x),
                    cyf.at(y, x),
                    cmix,
                    csalt,
                    hwet,
                    salt_on,
                );
                if fa == 0.0 && fb == 0.0 {
                    have[k] = false;
                    continue;
                }
                ay.set(y1, x, 0.0);
                if !have[k] {
                    pg.load(&mut a[k * nc..(k + 1) * nc], y, x, nb);
                }
                pg.load(&mut b, y1, x, nb);
                pg.transfer(fa, fb, &a[k * nc..(k + 1) * nc], &b, (y, x), (y1, x), nb);
                a[k * nc..(k + 1) * nc].copy_from_slice(&b);
                have[k] = true;
            }
        }
    });
}

// --------------------------------------------------------------------------
// Edge zone, local processes, salt, activity, settling
// --------------------------------------------------------------------------

/// `_k_edge`: E = 1 - box_blur(wet mask, R).
///
/// prange check: pass 1 row y writes `tmp[y, :]`; pass 2 (after the barrier)
/// column block bi writes `E[:, xa0..xb0]` and only reads tmp. Disjoint.
pub fn edge(h: &Pl<f32>, mask: &Pl<f32>, hwet: f64, r: i64, tmp: &PlM<f32>, e: &PlM<f32>) {
    let (n, w) = (h.height(), h.width());
    if n == 0 || w == 0 {
        return;
    }
    let inv = 1.0 / (2 * r + 1) as f64;
    let wi = w as i64;
    let ni = n as i64;
    let clampx = move |v: i64| v.max(0).min(wi - 1) as usize;
    let clampy = move |v: i64| v.max(0).min(ni - 1) as usize;
    par_for(n, n * w, |y| {
        let wet = |xx: usize| h.at(y, xx) as f64 > hwet && (mask.at(y, xx) as f64) < 0.5;
        let mut acc = 0.0f64;
        for x in -r..=r {
            if wet(clampx(x)) {
                acc += 1.0;
            }
        }
        for x in 0..w {
            tmp.set(y, x, (acc * inv) as f32);
            let xo = clampx(x as i64 - r);
            let xi = clampx(x as i64 + r + 1);
            if wet(xo) {
                acc -= 1.0;
            }
            if wet(xi) {
                acc += 1.0;
            }
        }
    });
    const BW: usize = 32;
    let nblk = (w + BW - 1) / BW;
    par_for(nblk, n * w, |bi| {
        let xa0 = bi * BW;
        let xb0 = w.min(xa0 + BW);
        let mut acc = [0.0f64; BW];
        for x in xa0..xb0 {
            let mut a = 0.0f64;
            for yy in -r..=r {
                a += tmp.at(clampy(yy), x) as f64;
            }
            acc[x - xa0] = a;
        }
        for y in 0..n {
            for x in xa0..xb0 {
                let k = x - xa0;
                e.set(y, x, (1.0 - acc[k] * inv) as f32);
                let yo = clampy(y as i64 - r);
                let yi = clampy(y as i64 + r + 1);
                acc[k] += (tmp.at(yi, x) - tmp.at(yo, x)) as f64;
            }
        }
    });
}

/// `_capw`: capillary exchange over one link.
#[inline]
fn capw(d: f64, c: f32, si: f32, sj: f32) -> f64 {
    let sa = 0.5 * ((si + sj) as f64);
    let mut wgt = d * c as f64 * sa * sa * 2.0;
    if wgt > 0.12 {
        wgt = 0.12;
    }
    wgt * ((sj - si) as f64)
}

/// The deposited-pigment and bond arrays `_k_local` writes.
pub struct Deposit<'a, 'b> {
    pub dk: &'b SpM<'a, f32>,
    pub ds: &'b SpM<'a, f32>,
    pub dm: &'b PlM<'a, f32>,
    pub dg: &'b PlM<'a, f32>,
    pub dst: &'b PlM<'a, f32>,
    pub bond: &'b PlM<'a, f32>,
}

/// `_k_local`: per-pixel processes (absorption, evaporation, capillary
/// wicking, salt; adsorption / desorption and bonding when `pdt > 0`).
///
/// prange check: row y writes only cells (y, x) of h, s2, K, S, m, g, st, dK,
/// dS, dm, dg, dst, bond. Neighbour reads are of `s`, mask, cxc, cyc, u, v,
/// which are not written (the new saturation goes to s2). Rows are disjoint.
#[allow(clippy::too_many_arguments)]
pub fn local(
    h: &PlM<f32>,
    s: &Pl<f32>,
    s2: &PlM<f32>,
    pg: &Pigment,
    dp: &Deposit,
    salt: &Pl<f32>,
    e: &Pl<f32>,
    gr: &Pl<f32>,
    mask: &Pl<f32>,
    cxc: &Pl<f32>,
    cyc: &Pl<f32>,
    u: &Pl<f32>,
    v: &Pl<f32>,
    evm: &Pl<f32>,
    prm: &[f64],
    dt: f64,
    dcap: f64,
    pdt: f64,
) {
    let (n, w) = (h.height(), h.width());
    let nb = pg.k.channels();
    let absr = prm[I_ABS];
    let cap = prm[I_CAP];
    let ev = prm[I_EV];
    let ke = prm[I_EDGE];
    let evd = prm[I_EVD];
    let kdep = prm[I_DEP];
    let h0 = prm[I_H0];
    let kdd = prm[I_DEPD];
    let kdes = prm[I_DES];
    let bstr = prm[I_BOND];
    let tb = prm[I_BT];
    let soft = prm[I_SOFT];
    let ag = prm[I_GRAN];
    let hwet = prm[I_HWET];
    let sab = prm[I_SABS];
    let kedge_dep = prm[I_EDEP];
    let kshear = prm[I_SHEAR];
    let (m, g, st) = (pg.m, pg.g, pg.st);
    par_for(n, n * w, |y| {
        for x in 0..w {
            let sv0 = s.at(y, x);
            let mut sv = sv0 as f64;
            // -- capillary wicking through the fibre network
            if (mask.at(y, x) as f64) < 0.5 {
                let mut acc = 0.0f64;
                if x > 0 && (mask.at(y, x - 1) as f64) < 0.5 {
                    acc += capw(dcap, cxc.at(y, x - 1), sv0, s.at(y, x - 1));
                }
                if x + 1 < w && (mask.at(y, x + 1) as f64) < 0.5 {
                    acc += capw(dcap, cxc.at(y, x), sv0, s.at(y, x + 1));
                }
                if y > 0 && (mask.at(y - 1, x) as f64) < 0.5 {
                    acc += capw(dcap, cyc.at(y - 1, x), sv0, s.at(y - 1, x));
                }
                if y + 1 < n && (mask.at(y + 1, x) as f64) < 0.5 {
                    acc += capw(dcap, cyc.at(y, x), sv0, s.at(y + 1, x));
                }
                sv += acc;
            }
            let mut hv = h.at(y, x) as f64;
            let mv = m.at(y, x);
            if hv <= 0.0 && sv <= 0.0 && mv as f64 <= 0.0 {
                s2.set(y, x, 0.0);
                continue;
            }
            let em = evm.at(y, x) as f64;
            let ed = e.at(y, x) as f64;
            // -- absorption into the paper
            if hv > 0.0 && sv < 1.0 {
                let mut a = dt * absr * (1.0 - sv) * cap;
                if a > hv {
                    a = hv;
                }
                hv -= a;
                sv += a / cap;
                if sv > 1.0 {
                    sv = 1.0;
                }
            }
            // -- evaporation (faster near the boundary of the wet area)
            if hv > 0.0 {
                hv -= dt * ev * em * (1.0 + ke * ed);
                if hv < 2e-4 {
                    if hv > 0.0 {
                        sv += hv / cap;
                        if sv > 1.0 {
                            sv = 1.0;
                        }
                    }
                    hv = 0.0;
                }
            } else {
                sv -= dt * evd * em * (1.0 + 0.4 * ke * ed);
                if sv < 0.004 {
                    sv = 0.0;
                }
            }
            // -- salt soaks up free water
            let sl = salt.at(y, x) as f64;
            if sl > 0.0 && hv > 0.0 {
                let mut take = dt * sab * sl;
                if take > hv {
                    take = hv;
                }
                hv -= take;
            }
            if pdt <= 0.0 {
                h.set(y, x, hv as f32);
                s2.set(y, x, sv as f32);
                continue;
            }
            // -- pigment adsorption / desorption
            let dmv = dp.dm.at(y, x);
            let mut gam = 0.0f64;
            if mv as f64 > 1e-9 {
                gam = (g.at(y, x) / mv) as f64; // f32 / f32
                if gam > 1.0 {
                    gam = 1.0;
                }
            }
            let mut gamd = 0.0f64;
            let mut stnd = 0.0f64;
            if dmv as f64 > 1e-9 {
                gamd = (dp.dg.at(y, x) / dmv) as f64;
                stnd = (dp.dst.at(y, x) / dmv) as f64;
                if gamd > 1.0 {
                    gamd = 1.0;
                }
                if stnd > 1.0 {
                    stnd = 1.0;
                }
            }
            let val = 2.0 * gr.at(y, x) as f64 - 1.0; // +1 valley, -1 peak
            let mut fdes = 0.0f64;
            let mut fdep;
            if hv > hwet {
                fdep = pdt * kdep * (1.0 + ag * gam * val) * (1.0 + kedge_dep * ed) / (hv + h0);
                if dmv as f64 > 1e-9 {
                    let mut wf = hv / 0.08;
                    if wf > 1.0 {
                        wf = 1.0;
                    }
                    // flowing water lifts loosely settled pigment (shear)
                    let sp = u.at(y, x).abs() + u.at(y, x + 1).abs() + v.at(y, x).abs() + v.at(y + 1, x).abs();
                    let mut spd = (sp as f64) / (hv + 0.05);
                    if spd > 2.0 {
                        spd = 2.0;
                    }
                    fdes = pdt
                        * (kdes + kshear * spd)
                        * (1.0 - 0.92 * stnd)
                        * (1.0 - bstr * dp.bond.at(y, x) as f64)
                        * (1.0 - ag * gamd * val)
                        * wf;
                }
                if sl > 0.0 {
                    fdep /= 1.0 + 8.0 * sl;
                }
            } else if sv > 0.03 {
                fdep = pdt * kdd * (1.0 + ag * gam * val) * (1.25 - sv);
            } else {
                fdep = 1.0;
            }
            if fdep > 1.0 {
                fdep = 1.0;
            }
            if fdep < 0.0 {
                fdep = 0.0;
            }
            if fdes > 0.5 {
                fdes = 0.5;
            }
            if fdes < 0.0 {
                fdes = 0.0;
            }
            if (fdep > 0.0 && mv as f64 > 0.0) || fdes > 0.0 {
                for b in 0..nb {
                    let kv = pg.k.at(y, x, b) as f64;
                    let dkv = dp.dk.at(y, x, b) as f64;
                    let t = kv * fdep - dkv * fdes;
                    pg.k.set(y, x, b, (kv - t) as f32);
                    dp.dk.set(y, x, b, (dkv + t) as f32);
                    let sv_ = pg.s.at(y, x, b) as f64;
                    let dsv = dp.ds.at(y, x, b) as f64;
                    let t = sv_ * fdep - dsv * fdes;
                    pg.s.set(y, x, b, (sv_ - t) as f32);
                    dp.ds.set(y, x, b, (dsv + t) as f32);
                }
                let md = mv as f64 * fdep;
                let mu = dmv as f64 * fdes;
                let gv = g.at(y, x) as f64;
                let dgv = dp.dg.at(y, x) as f64;
                let t = gv * fdep - dgv * fdes;
                g.set(y, x, (gv - t) as f32);
                dp.dg.set(y, x, (dgv + t) as f32);
                let stv = st.at(y, x) as f64;
                let dstv = dp.dst.at(y, x) as f64;
                let t = stv * fdep - dstv * fdes;
                st.set(y, x, (stv - t) as f32);
                dp.dst.set(y, x, (dstv + t) as f32);
                m.set(y, x, (mv as f64 - md + mu) as f32);
                let ndm = dmv as f64 + md - mu;
                dp.dm.set(y, x, ndm as f32);
                if ndm > 1e-9 {
                    // fresh deposit is unbonded: mass-weighted bond
                    let bd = dp.bond.at(y, x) as f64;
                    dp.bond.set(y, x, (bd * (dmv as f64 - mu) / ndm) as f32);
                }
            }
            // -- bonding of dried deposits / slow loosening under water
            if hv <= hwet && sv < 0.06 {
                let q = pdt / tb;
                let q = if 1.0 < q { 1.0 } else { q };
                let bd = dp.bond.at(y, x) as f64;
                dp.bond.set(y, x, (bd + q * (1.0 - bd)) as f32);
            } else if hv > hwet {
                let bd = dp.bond.at(y, x) as f64;
                dp.bond.set(y, x, (bd - pdt * soft * bd) as f32);
            }
            h.set(y, x, hv as f32);
            s2.set(y, x, sv as f32);
        }
    });
}

/// `_k_salt`: brine spreads along fibres through wet paper (reads `salt`,
/// writes `s2`; row y writes only `s2[y, :]`).
#[allow(clippy::too_many_arguments)]
pub fn salt(
    salt: &Pl<f32>,
    s2: &PlM<f32>,
    h: &Pl<f32>,
    s: &Pl<f32>,
    cxc: &Pl<f32>,
    cyc: &Pl<f32>,
    d: f64,
    hwet: f64,
) {
    let (n, w) = (salt.height(), salt.width());
    let wet = |y: usize, x: usize| h.at(y, x) as f64 > hwet || s.at(y, x) as f64 > 0.35;
    // D * c ** 2, with c ** 2 evaluated in float32 (numba: float32 ** 2)
    let link = |c: f32| -> f64 {
        let q = d * ((c * c) as f64);
        if 0.12 < q {
            0.12
        } else {
            q
        }
    };
    par_for(n, n * w, |y| {
        for x in 0..w {
            let si = salt.at(y, x);
            if !wet(y, x) {
                s2.set(y, x, si);
                continue;
            }
            let mut acc = 0.0f64;
            if x > 0 && wet(y, x - 1) {
                let c = link(cxc.at(y, x - 1));
                acc += c * ((salt.at(y, x - 1) - si) as f64);
            }
            if x + 1 < w && wet(y, x + 1) {
                let c = link(cxc.at(y, x));
                acc += c * ((salt.at(y, x + 1) - si) as f64);
            }
            if y > 0 && wet(y - 1, x) {
                let c = link(cyc.at(y - 1, x));
                acc += c * ((salt.at(y - 1, x) - si) as f64);
            }
            if y + 1 < n && wet(y + 1, x) {
                let c = link(cyc.at(y, x));
                acc += c * ((salt.at(y + 1, x) - si) as f64);
            }
            s2.set(y, x, (si as f64 + acc) as f32);
        }
    });
}

/// `_k_active`: per-row extent of pixels that still hold water or suspended
/// pigment. Row y writes `rowx0[y]`, `rowx1[y]`.
pub fn active(h: &Pl<f32>, s: &Pl<f32>, m: &Pl<f32>, rowx0: &PlM<i64>, rowx1: &PlM<i64>) {
    let (n, w) = (h.height(), h.width());
    par_for(n, n * w, |y| {
        let mut a = w as i64;
        let mut b = -1i64;
        for x in 0..w {
            if h.at(y, x) as f64 > 1e-6 || s.at(y, x) as f64 > 2e-3 || m.at(y, x) as f64 > 1e-9 {
                if (x as i64) < a {
                    a = x as i64;
                }
                b = x as i64;
            }
        }
        rowx0.set(y, 0, a);
        rowx1.set(y, 0, b);
    });
}

/// `_k_settle_all`: deposit every bit of suspended pigment and remove all
/// water. Row y writes only row y of every array.
#[allow(clippy::too_many_arguments)]
pub fn settle_all(
    pg: &Pigment,
    dk: &SpM<f32>,
    ds: &SpM<f32>,
    dm: &PlM<f32>,
    dg: &PlM<f32>,
    dst: &PlM<f32>,
    h: &PlM<f32>,
    s: &PlM<f32>,
) {
    let (n, w) = (h.height(), h.width());
    let nb = pg.k.channels();
    par_for(n, n * w, |y| {
        for x in 0..w {
            if pg.m.at(y, x) != 0.0 {
                dm.set(y, x, dm.at(y, x) + pg.m.at(y, x));
                dg.set(y, x, dg.at(y, x) + pg.g.at(y, x));
                dst.set(y, x, dst.at(y, x) + pg.st.at(y, x));
                pg.m.set(y, x, 0.0);
                pg.g.set(y, x, 0.0);
                pg.st.set(y, x, 0.0);
            }
            for b in 0..nb {
                if pg.k.at(y, x, b) != 0.0 || pg.s.at(y, x, b) != 0.0 {
                    dk.set(y, x, b, dk.at(y, x, b) + pg.k.at(y, x, b));
                    ds.set(y, x, b, ds.at(y, x, b) + pg.s.at(y, x, b));
                    pg.k.set(y, x, b, 0.0);
                    pg.s.set(y, x, b, 0.0);
                }
            }
            h.set(y, x, 0.0);
            s.set(y, x, 0.0);
        }
    });
}

// --------------------------------------------------------------------------
// Paper structure
// --------------------------------------------------------------------------

/// `_k_fibres`: rasterise curved fibres 4-connectedly; count fibre links per
/// pipe. Sequential (fibres overlap each other's cells).
#[allow(clippy::too_many_arguments)]
pub fn fibres(
    hh: i64,
    ww: i64,
    xs: &[f64],
    ys: &[f64],
    ang: &[f64],
    lens: &[f64],
    curv: &[f64],
    wts: &[f64],
    cx: &PlM<f64>,
    cy: &PlM<f64>,
    cov: &PlM<f64>,
) {
    for f in 0..xs.len() {
        let mut x = xs[f];
        let mut y = ys[f];
        let mut a = ang[f];
        let l = lens[f];
        let c = curv[f];
        let wt = wts[f];
        let mut ix = x.floor() as i64;
        let mut iy = y.floor() as i64;
        let mut t = 0.0f64;
        while t < l {
            x += a.cos() * 0.5;
            y += a.sin() * 0.5;
            a += c * 0.5;
            t += 0.5;
            let jx = x.floor() as i64;
            let jy = y.floor() as i64;
            if jx != ix {
                let lx = ix.min(jx);
                if 0 <= iy && iy < hh && 0 <= lx && lx < ww - 1 {
                    let (yy, xx) = (iy as usize, lx as usize);
                    cx.set(yy, xx, cx.at(yy, xx) + wt);
                }
                ix = jx;
            }
            if jy != iy {
                let ly = iy.min(jy);
                if 0 <= ix && ix < ww && 0 <= ly && ly < hh - 1 {
                    let (yy, xx) = (ly as usize, ix as usize);
                    cy.set(yy, xx, cy.at(yy, xx) + wt);
                }
                iy = jy;
            }
            if 0 <= ix && ix < ww && 0 <= iy && iy < hh {
                let (yy, xx) = (iy as usize, ix as usize);
                cov.set(yy, xx, cov.at(yy, xx) + wt);
            }
        }
    }
}
