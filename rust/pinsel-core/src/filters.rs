//! Port of the scipy.ndimage / scipy.spatial / numpy.fft calls pinsel makes.
//! See docs/RUST_PORTING_GUIDE.md and `pinsel/_filters.py`.
//!
//! Everything here follows the C code of scipy.ndimage (1.18) operation by
//! operation, so results are bit-identical (not just close):
//!
//! * a 1-D pass reads a line (or, for non-last axes, a row of the other axes
//!   at once), extends it by the boundary mode, accumulates in **f64** and
//!   stores in the array dtype (f32 or f64). Several axes are separate
//!   passes, so an f32 array is rounded to f32 between passes, as in scipy;
//! * correlate1d detects symmetric / antisymmetric weights and then pairs
//!   the taps (`iline[-j] + iline[j]`, farthest pair first), as scipy does;
//! * gaussian kernels use numpy's pairwise summation for the normalisation.
//!
//! The arrays are C-contiguous slices with an explicit `shape`; the
//! bindings make them contiguous.

pub use rustfft::num_complex::Complex64;
use rustfft::FftPlanner;

// ---------------------------------------------------------------------------
// Element types and boundary modes
// ---------------------------------------------------------------------------

/// f32 or f64 array element. All arithmetic is done in f64.
pub trait Elem: Copy + Send + Sync + Default {
    fn to_f64(self) -> f64;
    fn from_f64(v: f64) -> Self;
}
impl Elem for f32 {
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v as f32
    }
}
impl Elem for f64 {
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v
    }
}

/// Boundary modes (scipy names). `Reflect` is scipy's default for most
/// filters (d c b a | a b c d | d c b a), `Nearest` repeats the edge,
/// `Wrap` and `GridWrap` are periodic (identical for filters), `Constant`
/// pads with `cval`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Reflect,
    Nearest,
    Wrap,
    GridWrap,
    Constant,
}

impl Mode {
    pub fn parse(s: &str) -> Option<Mode> {
        Some(match s {
            "reflect" | "grid-mirror" => Mode::Reflect,
            "nearest" => Mode::Nearest,
            "wrap" => Mode::Wrap,
            "grid-wrap" => Mode::GridWrap,
            "constant" => Mode::Constant,
            _ => return None,
        })
    }
}

/// Map an out-of-range index of a line of length `n` into `0..n`
/// (None = constant padding).
#[inline]
fn map_index(i: i64, n: usize, mode: Mode) -> Option<usize> {
    let n = n as i64;
    if (0..n).contains(&i) {
        return Some(i as usize);
    }
    match mode {
        Mode::Reflect => {
            let m = i.rem_euclid(2 * n);
            Some(if m >= n { 2 * n - 1 - m } else { m } as usize)
        }
        Mode::Nearest => Some(i.clamp(0, n - 1) as usize),
        Mode::Wrap | Mode::GridWrap => Some(i.rem_euclid(n) as usize),
        Mode::Constant => None,
    }
}

fn split_axis(shape: &[usize], axis: usize) -> (usize, usize, usize) {
    let outer: usize = shape[..axis].iter().product();
    let n = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    (outer, n, inner)
}

// ---------------------------------------------------------------------------
// numpy pairwise summation (np.sum of a contiguous float64 vector)
// ---------------------------------------------------------------------------

fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for &v in a {
            res += v;
        }
        res
    } else if n <= 128 {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

/// `np.sum(a)` for a contiguous float64 vector (pairwise over all of it;
/// checked against numpy 2.5 for n = 1..400).
pub fn np_sum(a: &[f64]) -> f64 {
    pairwise_sum(a)
}

// ---------------------------------------------------------------------------
// Gaussian kernel (scipy.ndimage._filters._gaussian_kernel1d, order 0)
// ---------------------------------------------------------------------------

/// Radius used by `gaussian_filter1d`: `int(truncate * sd + 0.5)`.
pub fn gaussian_radius(sigma: f64, truncate: f64) -> usize {
    (truncate * sigma + 0.5) as usize
}

/// `_gaussian_kernel1d(sigma, 0, radius)`, in x = -radius..=radius order.
/// (The Python reverses it before correlate1d; it is symmetric.)
pub fn gaussian_kernel1d(sigma: f64, radius: usize) -> Vec<f64> {
    let sigma2 = sigma * sigma;
    let r = radius as i64;
    let mut phi: Vec<f64> = (-r..=r)
        .map(|x| {
            let x2 = (x * x) as f64;
            (-0.5 / sigma2 * x2).exp()
        })
        .collect();
    let s = np_sum(&phi);
    for v in phi.iter_mut() {
        *v /= s;
    }
    phi
}

// ---------------------------------------------------------------------------
// correlate1d
// ---------------------------------------------------------------------------

/// scipy's NI_Correlate1D symmetry test (tolerance DBL_EPSILON): returns
/// 1 for symmetric, -1 for antisymmetric, 0 otherwise (odd sizes only).
fn symmetry(w: &[f64]) -> i32 {
    let fs = w.len();
    if fs % 2 == 0 {
        return 0;
    }
    let c = fs / 2;
    let mut sym = 1;
    for ii in 1..=c {
        if (w[c + ii] - w[c - ii]).abs() > f64::EPSILON {
            sym = 0;
            break;
        }
    }
    if sym == 0 {
        sym = -1;
        for ii in 1..=c {
            if (w[c + ii] + w[c - ii]).abs() > f64::EPSILON {
                sym = 0;
                break;
            }
        }
    }
    sym
}

/// One correlate1d pass along `axis`: `dst = correlate(src, w)`.
pub fn correlate1d_pass<T: Elem>(
    src: &[T],
    dst: &mut [T],
    shape: &[usize],
    axis: usize,
    w: &[f64],
    mode: Mode,
    cval: f64,
) {
    let (outer, n, inner) = split_axis(shape, axis);
    if src.is_empty() {
        return;
    }
    let fs = w.len();
    let size1 = (fs / 2) as i64;
    let size2 = (fs as i64) - size1 - 1;
    let sym = symmetry(w);
    let c = fs / 2;
    let mut acc = vec![0.0f64; inner];
    // tap rows for one output row: index of the source row or None (cval)
    let row_of = |o: usize, k: i64| -> Option<usize> {
        map_index(k, n, mode).map(|kk| (o * n + kk) * inner)
    };
    let get = |r: Option<usize>, e: usize| -> f64 {
        match r {
            Some(base) => src[base + e].to_f64(),
            None => cval,
        }
    };
    for o in 0..outer {
        for k in 0..n {
            let ki = k as i64;
            let centre = (o * n + k) * inner;
            if sym > 0 {
                for e in 0..inner {
                    acc[e] = src[centre + e].to_f64() * w[c];
                }
                for jj in -size1..0 {
                    let wj = w[(c as i64 + jj) as usize];
                    let ra = row_of(o, ki + jj);
                    let rb = row_of(o, ki - jj);
                    for e in 0..inner {
                        acc[e] += (get(ra, e) + get(rb, e)) * wj;
                    }
                }
            } else if sym < 0 {
                for a in acc.iter_mut() {
                    *a = 0.0;
                }
                for jj in -size1..0 {
                    let wj = w[(c as i64 + jj) as usize];
                    let ra = row_of(o, ki + jj);
                    let rb = row_of(o, ki - jj);
                    for e in 0..inner {
                        acc[e] += (get(ra, e) - get(rb, e)) * wj;
                    }
                }
            } else {
                for a in acc.iter_mut() {
                    *a = 0.0;
                }
                for jj in -size1..=size2 {
                    let wj = w[(size1 + jj) as usize];
                    let r = row_of(o, ki + jj);
                    for e in 0..inner {
                        acc[e] += get(r, e) * wj;
                    }
                }
            }
            for e in 0..inner {
                dst[centre + e] = T::from_f64(acc[e]);
            }
        }
    }
}

/// `scipy.ndimage.correlate1d(a, w, axis, mode=mode, cval=cval)`.
pub fn correlate1d<T: Elem>(a: &[T], shape: &[usize], axis: usize, w: &[f64], mode: Mode, cval: f64) -> Vec<T> {
    let mut out = vec![T::default(); a.len()];
    correlate1d_pass(a, &mut out, shape, axis, w, mode, cval);
    out
}

/// `scipy.ndimage.gaussian_filter(a, sigma, mode=mode, truncate=truncate)`
/// with one sigma per axis; axes with sigma <= 1e-15 are skipped.
pub fn gaussian_filter<T: Elem>(a: &[T], shape: &[usize], sigmas: &[f64], mode: Mode, truncate: f64) -> Vec<T> {
    let kernels: Vec<Option<Vec<f64>>> = sigmas
        .iter()
        .map(|&sd| if sd > 1e-15 { Some(gaussian_kernel1d(sd, gaussian_radius(sd, truncate))) } else { None })
        .collect();
    gaussian_filter_with(a, shape, &kernels, mode)
}

/// Gaussian filter with the per-axis kernels given (`None` = skip the
/// axis). The kernels are in `x = -r..=r` order, as `_gaussian_kernel1d`
/// returns them. The Python shim passes kernels computed with numpy so
/// that they match scipy bit for bit even where numpy's SIMD `exp` differs
/// from the system libm.
pub fn gaussian_filter_with<T: Elem>(a: &[T], shape: &[usize], kernels: &[Option<Vec<f64>>], mode: Mode) -> Vec<T> {
    let mut cur: Vec<T> = a.to_vec();
    let mut tmp: Vec<T> = vec![T::default(); a.len()];
    for (axis, k) in kernels.iter().enumerate() {
        if let Some(k) = k {
            let mut w = k.clone();
            w.reverse();
            correlate1d_pass(&cur, &mut tmp, shape, axis, &w, mode, 0.0);
            std::mem::swap(&mut cur, &mut tmp);
        }
    }
    cur
}

/// `scipy.ndimage.sobel(a, axis=axis, mode=mode)`: derivative [-1, 0, 1]
/// along `axis`, then smoothing [1, 2, 1] along every other axis in order.
pub fn sobel<T: Elem>(a: &[T], shape: &[usize], axis: usize, mode: Mode) -> Vec<T> {
    let mut cur = correlate1d(a, shape, axis, &[-1.0, 0.0, 1.0], mode, 0.0);
    let mut tmp = vec![T::default(); a.len()];
    for ii in 0..shape.len() {
        if ii != axis {
            correlate1d_pass(&cur, &mut tmp, shape, ii, &[1.0, 2.0, 1.0], mode, 0.0);
            std::mem::swap(&mut cur, &mut tmp);
        }
    }
    cur
}

// ---------------------------------------------------------------------------
// uniform_filter / maximum_filter
// ---------------------------------------------------------------------------

/// `scipy.ndimage.uniform_filter(a, size, mode=mode)` (same size on every
/// axis with size > 1). scipy keeps a running mean along each line.
pub fn uniform_filter<T: Elem>(a: &[T], shape: &[usize], size: usize, mode: Mode, cval: f64) -> Vec<T> {
    let mut cur: Vec<T> = a.to_vec();
    if size <= 1 || a.is_empty() {
        return cur;
    }
    let mut tmp = vec![T::default(); a.len()];
    let fsz = size as f64;
    let size1 = (size / 2) as i64;
    for axis in 0..shape.len() {
        let (outer, n, inner) = split_axis(shape, axis);
        let mut run = vec![0.0f64; inner];
        for o in 0..outer {
            let tap = |k: i64| -> Option<usize> { map_index(k, n, mode).map(|kk| (o * n + kk) * inner) };
            for r in run.iter_mut() {
                *r = 0.0;
            }
            // running (un-normalised) window sum, divided on output
            for ll in 0..size as i64 {
                let t = tap(ll - size1);
                for e in 0..inner {
                    run[e] += match t {
                        Some(b) => cur[b + e].to_f64(),
                        None => cval,
                    };
                }
            }
            for e in 0..inner {
                tmp[o * n * inner + e] = T::from_f64(run[e] / fsz);
            }
            for ll in 1..n as i64 {
                let add = tap(ll + size as i64 - 1 - size1);
                let sub = tap(ll - 1 - size1);
                for e in 0..inner {
                    let x = match add {
                        Some(b) => cur[b + e].to_f64(),
                        None => cval,
                    };
                    let y = match sub {
                        Some(b) => cur[b + e].to_f64(),
                        None => cval,
                    };
                    run[e] += x - y;
                    tmp[(o * n + ll as usize) * inner + e] = T::from_f64(run[e] / fsz);
                }
            }
        }
        std::mem::swap(&mut cur, &mut tmp);
    }
    cur
}

/// `scipy.ndimage.maximum_filter(a, size=size, mode=mode)` (square
/// footprint, separable; max is exact so the order does not matter).
pub fn maximum_filter<T: Elem>(a: &[T], shape: &[usize], size: usize, mode: Mode, cval: f64) -> Vec<T> {
    let mut cur: Vec<T> = a.to_vec();
    if size <= 1 || a.is_empty() {
        return cur;
    }
    let mut tmp = vec![T::default(); a.len()];
    let size1 = (size / 2) as i64;
    for axis in 0..shape.len() {
        let (outer, n, inner) = split_axis(shape, axis);
        for o in 0..outer {
            for k in 0..n as i64 {
                let dst = (o * n + k as usize) * inner;
                for e in 0..inner {
                    let mut best = f64::NEG_INFINITY;
                    let mut first = true;
                    for j in 0..size as i64 {
                        let v = match map_index(k + j - size1, n, mode) {
                            Some(kk) => cur[(o * n + kk) * inner + e].to_f64(),
                            None => cval,
                        };
                        if first || v > best || v.is_nan() {
                            best = v;
                            first = false;
                        }
                    }
                    tmp[dst + e] = T::from_f64(best);
                }
            }
        }
        std::mem::swap(&mut cur, &mut tmp);
    }
    cur
}

// ---------------------------------------------------------------------------
// binary morphology (cross structure, border_value = 0)
// ---------------------------------------------------------------------------

/// One binary erosion (`dilate = false`) or dilation with the 4-connected
/// cross (`generate_binary_structure(2, 1)`), `border_value = 0`, on an
/// H x W boolean image stored as bytes (0/1).
fn binary_step(a: &[u8], h: usize, w: usize, dilate: bool) -> Vec<u8> {
    let mut out = vec![0u8; a.len()];
    for y in 0..h {
        for x in 0..w {
            let at = |yy: i64, xx: i64| -> u8 {
                if yy < 0 || xx < 0 || yy >= h as i64 || xx >= w as i64 {
                    0
                } else {
                    a[yy as usize * w + xx as usize]
                }
            };
            let (yi, xi) = (y as i64, x as i64);
            let c = [at(yi, xi), at(yi - 1, xi), at(yi + 1, xi), at(yi, xi - 1), at(yi, xi + 1)];
            out[y * w + x] = if dilate {
                c.iter().any(|&v| v != 0) as u8
            } else {
                c.iter().all(|&v| v != 0) as u8
            };
        }
    }
    out
}

/// `binary_erosion(a, iterations=n)` / `binary_dilation` with the default
/// cross structure and `border_value=0` (2-D).
pub fn binary_morph(a: &[u8], h: usize, w: usize, dilate: bool, iterations: usize) -> Vec<u8> {
    let mut cur = a.to_vec();
    for _ in 0..iterations {
        cur = binary_step(&cur, h, w, dilate);
    }
    cur
}

pub fn binary_opening(a: &[u8], h: usize, w: usize, iterations: usize) -> Vec<u8> {
    binary_morph(&binary_morph(a, h, w, false, iterations), h, w, true, iterations)
}

pub fn binary_closing(a: &[u8], h: usize, w: usize, iterations: usize) -> Vec<u8> {
    binary_morph(&binary_morph(a, h, w, true, iterations), h, w, false, iterations)
}

// ---------------------------------------------------------------------------
// spline prefilter and zoom
// ---------------------------------------------------------------------------

fn init_causal_wrap(c: &mut [f64], z: f64) {
    let n = c.len();
    let mut z_i = z;
    for i in 1..n {
        c[0] += z_i * c[n - i];
        z_i *= z;
    }
    c[0] /= 1.0 - z_i;
}

fn init_anticausal_wrap(c: &mut [f64], z: f64) {
    let n = c.len();
    let mut z_i = z;
    for i in 0..n - 1 {
        c[n - 1] += z_i * c[i];
        z_i *= z;
    }
    c[n - 1] *= z / (z_i - 1.0);
}

/// Cubic spline pole. scipy has it as a literal; `sqrt(3) - 2` is 1 ulp off.
fn pole3() -> f64 {
    -0.26794919243112270647
}

/// Cubic B-spline prefilter of one line, periodic ("grid-wrap") boundary.
fn spline_filter_line3_wrap(c: &mut [f64]) {
    let n = c.len();
    if n <= 1 {
        return;
    }
    let z = pole3();
    let weight = (1.0 - z) * (1.0 - 1.0 / z);
    for v in c.iter_mut() {
        *v *= weight;
    }
    init_causal_wrap(c, z);
    for i in 1..n {
        c[i] += z * c[i - 1];
    }
    init_anticausal_wrap(c, z);
    for i in (0..n - 1).rev() {
        c[i] = z * (c[i + 1] - c[i]);
    }
}

/// `scipy.ndimage.spline_filter(a, order=3, output=float64, mode='grid-wrap')`.
pub fn spline_filter3_wrap<T: Elem>(a: &[T], shape: &[usize]) -> Vec<f64> {
    let mut cur: Vec<f64> = a.iter().map(|v| v.to_f64()).collect();
    let mut line: Vec<f64> = Vec::new();
    for axis in 0..shape.len() {
        let (outer, n, inner) = split_axis(shape, axis);
        line.resize(n, 0.0);
        for o in 0..outer {
            for e in 0..inner {
                for k in 0..n {
                    line[k] = cur[(o * n + k) * inner + e];
                }
                spline_filter_line3_wrap(&mut line);
                for k in 0..n {
                    cur[(o * n + k) * inner + e] = line[k];
                }
            }
        }
    }
    cur
}

/// Spline weights of scipy's `spline_coefficients` for order 1 and 3.
#[inline]
fn spline_weights(x: f64, order: usize) -> ([f64; 4], i64) {
    let start = if order & 1 == 1 {
        x.floor() as i64 - (order as i64) / 2
    } else {
        (x + 0.5).floor() as i64 - (order as i64) / 2
    };
    let mut r = [0.0f64; 4];
    for hh in 0..=order {
        let y = (start as f64 - x + hh as f64).abs();
        r[hh] = match order {
            1 => {
                if y > 1.0 {
                    0.0
                } else {
                    1.0 - y
                }
            }
            _ => {
                if y < 1.0 {
                    (y * y * (y - 2.0) * 3.0 + 4.0) / 6.0
                } else if y < 2.0 {
                    let y = 2.0 - y;
                    (y * y * y) / 6.0
                } else {
                    0.0
                }
            }
        };
    }
    (r, start)
}

/// Output shape of `scipy.ndimage.zoom`: `int(round(n * z))` per axis.
pub fn zoom_shape(shape: &[usize], zoom: &[f64]) -> Vec<usize> {
    shape.iter().zip(zoom).map(|(&n, &z)| (n as f64 * z).round_ties_even() as usize).collect()
}

/// `scipy.ndimage.zoom(a, zoom, order=order, mode=mode, cval=cval)` for
/// order 1 (any mode: pinsel uses `constant`) and order 3 with
/// `grid-wrap`, `grid_mode=False`, `prefilter=True`. Returns (data, shape).
pub fn zoom<T: Elem>(
    a: &[T],
    shape: &[usize],
    zoom: &[f64],
    order: usize,
    mode: Mode,
    cval: f64,
) -> (Vec<T>, Vec<usize>) {
    assert!(order == 1 || (order == 3 && mode == Mode::GridWrap), "zoom: unsupported order/mode");
    let rank = shape.len();
    let oshape = zoom_shape(shape, zoom);
    let total: usize = oshape.iter().product();
    if total == 0 || a.is_empty() {
        return (vec![T::default(); total], oshape);
    }
    let filtered: Option<Vec<f64>> = if order == 3 { Some(spline_filter3_wrap(a, shape)) } else { None };
    // per-axis zoom factor: (in - 1) / (out - 1), or 1 when out == 1
    let zf: Vec<f64> = (0..rank)
        .map(|j| {
            let div = oshape[j] as f64 - 1.0;
            if div != 0.0 {
                (shape[j] as f64 - 1.0) / div
            } else {
                1.0
            }
        })
        .collect();
    // per axis, per output coordinate: None (constant region) or the
    // spline weights and the source index of each tap (None = padding)
    struct Ax {
        w: [f64; 4],
        idx: [Option<usize>; 4],
    }
    let mut axes: Vec<Vec<Option<Ax>>> = Vec::with_capacity(rank);
    for j in 0..rank {
        let len = shape[j];
        let mut v = Vec::with_capacity(oshape[j]);
        for kk in 0..oshape[j] {
            let mut cc = kk as f64 * zf[j];
            // map_coordinate
            match mode {
                Mode::Constant => {
                    if cc < 0.0 || cc > (len as f64) - 1.0 {
                        cc = -1.0;
                    }
                }
                Mode::GridWrap => {
                    let sz = len as f64;
                    if cc < 0.0 {
                        if len <= 1 {
                            cc = 0.0;
                        } else {
                            cc += sz * ((-cc / sz) as i64) as f64;
                            if cc < 0.0 {
                                cc += sz;
                            }
                        }
                    } else if cc > (len as f64) - 1.0 {
                        if len <= 1 {
                            cc = 0.0;
                        } else {
                            cc -= sz * ((cc / sz) as i64) as f64;
                        }
                    }
                }
                Mode::Nearest => {
                    if cc <= 0.0 {
                        cc = 0.0;
                    } else if cc >= (len as f64) - 1.0 {
                        cc = (len as f64) - 1.0;
                    }
                }
                _ => panic!("zoom: unsupported mode"),
            }
            if cc > -1.0 {
                let (w, start) = spline_weights(cc, order);
                let mut idx = [None; 4];
                for hh in 0..=order {
                    idx[hh] = map_index(start + hh as i64, len, mode);
                }
                v.push(Some(Ax { w, idx }));
            } else {
                v.push(None);
            }
        }
        axes.push(v);
    }
    let istr: Vec<usize> = (0..rank).map(|j| shape[j + 1..].iter().product()).collect();
    let ntap = order + 1;
    let mut out = vec![T::default(); total];
    let mut pos = vec![0usize; rank];
    let mut tap = vec![0usize; rank];
    for o in out.iter_mut() {
        let cells: Vec<&Option<Ax>> = (0..rank).map(|j| &axes[j][pos[j]]).collect();
        if cells.iter().any(|c| c.is_none()) {
            *o = T::from_f64(cval);
        } else {
            let cells: Vec<&Ax> = cells.into_iter().map(|c| c.as_ref().unwrap()).collect();
            let mut t = 0.0f64;
            let ncombo = ntap.pow(rank as u32);
            for combo in 0..ncombo {
                // last axis varies fastest (scipy's odometer)
                let mut c = combo;
                for j in (0..rank).rev() {
                    tap[j] = c % ntap;
                    c /= ntap;
                }
                let mut off = 0usize;
                let mut pad = false;
                for j in 0..rank {
                    match cells[j].idx[tap[j]] {
                        Some(i) => off += i * istr[j],
                        None => pad = true,
                    }
                }
                let mut coeff = if pad {
                    cval
                } else {
                    match &filtered {
                        Some(f) => f[off],
                        None => a[off].to_f64(),
                    }
                };
                for j in 0..rank {
                    coeff *= cells[j].w[tap[j]];
                }
                t += coeff;
            }
            *o = T::from_f64(t);
        }
        // advance output position
        let mut l = rank;
        while l > 0 {
            l -= 1;
            pos[l] += 1;
            if pos[l] < oshape[l] {
                break;
            }
            pos[l] = 0;
        }
    }
    (out, oshape)
}

// ---------------------------------------------------------------------------
// FFT (noise synthesis): numpy.fft.rfft2 / irfft2 with rustfft
// ---------------------------------------------------------------------------

/// `np.fft.fftfreq(n)`: integer bins times `1.0 / n`.
pub fn fftfreq(n: usize) -> Vec<f64> {
    let val = 1.0 / (n as f64);
    let nn = (n - 1) / 2 + 1;
    let mut r: Vec<f64> = (0..nn as i64).map(|i| i as f64 * val).collect();
    r.extend((-((n / 2) as i64)..0).map(|i| i as f64 * val));
    r
}

/// `np.fft.rfftfreq(n)`.
pub fn rfftfreq(n: usize) -> Vec<f64> {
    let val = 1.0 / (n as f64);
    (0..(n / 2 + 1) as i64).map(|i| i as f64 * val).collect()
}

/// `np.fft.rfft2(a)` for an H x W real array: H x (W/2+1) complex,
/// row-major (re, im) pairs.
pub fn rfft2(a: &[f64], h: usize, w: usize) -> Vec<Complex64> {
    let wc = w / 2 + 1;
    let mut planner = FftPlanner::<f64>::new();
    let fw = planner.plan_fft_forward(w);
    let fh = planner.plan_fft_forward(h);
    // rows: real -> complex, keep the first wc bins
    let mut half = vec![Complex64::new(0.0, 0.0); h * wc];
    let mut buf = vec![Complex64::new(0.0, 0.0); w];
    for y in 0..h {
        for x in 0..w {
            buf[x] = Complex64::new(a[y * w + x], 0.0);
        }
        fw.process(&mut buf);
        half[y * wc..(y + 1) * wc].copy_from_slice(&buf[..wc]);
    }
    // columns: complex FFT along axis 0
    let mut col = vec![Complex64::new(0.0, 0.0); h];
    for x in 0..wc {
        for y in 0..h {
            col[y] = half[y * wc + x];
        }
        fh.process(&mut col);
        for y in 0..h {
            half[y * wc + x] = col[y];
        }
    }
    half
}

/// `np.fft.irfft2(spec, s=(h, w))` for an H x (W/2+1) complex array
/// (row-major): H x W real, scaled by 1 / (H * W).
pub fn irfft2(spec: &[Complex64], h: usize, w: usize) -> Vec<f64> {
    let wc = w / 2 + 1;
    let mut planner = FftPlanner::<f64>::new();
    let fw = planner.plan_fft_inverse(w);
    let fh = planner.plan_fft_inverse(h);
    let mut half = spec.to_vec();
    // inverse complex FFT along axis 0
    let mut col = vec![Complex64::new(0.0, 0.0); h];
    for x in 0..wc {
        for y in 0..h {
            col[y] = half[y * wc + x];
        }
        fh.process(&mut col);
        for y in 0..h {
            half[y * wc + x] = col[y];
        }
    }
    // c2r along axis 1: hermitian-extend each row, inverse FFT, real part
    let norm = 1.0 / ((h * w) as f64);
    let mut out = vec![0.0f64; h * w];
    let mut buf = vec![Complex64::new(0.0, 0.0); w];
    for y in 0..h {
        for x in 0..wc {
            buf[x] = half[y * wc + x];
        }
        // pocketfft's c2r ignores the imaginary part of the DC (and, for
        // even w, the Nyquist) bin
        buf[0].im = 0.0;
        if w % 2 == 0 {
            buf[w / 2].im = 0.0;
        }
        for x in wc..w {
            buf[x] = buf[w - x].conj();
        }
        fw.process(&mut buf);
        for x in 0..w {
            out[y * w + x] = buf[x].re * norm;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// k-d tree (scipy.spatial.cKDTree queries)
// ---------------------------------------------------------------------------

/// Exact nearest-neighbour search over `n` points of dimension `d`.
/// Neighbours are ordered by (distance, index). Distances are computed as
/// `sqrt(sum (a_i - b_i)^2)` accumulated in dimension order, like cKDTree
/// with p = 2.
pub struct KdTree {
    d: usize,
    pts: Vec<f64>,
    n: usize,
    // implicit balanced tree over `order`
    order: Vec<u32>,
    nodes: Vec<Node>,
}

struct Node {
    lo: u32,
    hi: u32,
    split_dim: u32,
    split: f64,
    left: u32,
    right: u32,
    leaf: bool,
}

const LEAF: usize = 16;

impl KdTree {
    pub fn new(pts: &[f64], d: usize) -> KdTree {
        let n = pts.len() / d;
        let mut t = KdTree { d, pts: pts.to_vec(), n, order: (0..n as u32).collect(), nodes: Vec::new() };
        if n > 0 {
            t.build(0, n);
        }
        t
    }

    fn build(&mut self, lo: usize, hi: usize) -> u32 {
        let id = self.nodes.len() as u32;
        self.nodes.push(Node { lo: lo as u32, hi: hi as u32, split_dim: 0, split: 0.0, left: 0, right: 0, leaf: true });
        if hi - lo <= LEAF {
            return id;
        }
        let d = self.d;
        // widest dimension
        let mut best = 0;
        let mut bw = -1.0;
        for k in 0..d {
            let (mut mn, mut mx) = (f64::INFINITY, f64::NEG_INFINITY);
            for &i in &self.order[lo..hi] {
                let v = self.pts[i as usize * d + k];
                mn = mn.min(v);
                mx = mx.max(v);
            }
            if mx - mn > bw {
                bw = mx - mn;
                best = k;
            }
        }
        if bw <= 0.0 {
            return id;
        }
        let pts = &self.pts;
        self.order[lo..hi].sort_unstable_by(|&a, &b| {
            pts[a as usize * d + best].partial_cmp(&pts[b as usize * d + best]).unwrap().then(a.cmp(&b))
        });
        let mid = (lo + hi) / 2;
        let split = self.pts[self.order[mid] as usize * d + best];
        let l = self.build(lo, mid);
        let r = self.build(mid, hi);
        let nd = &mut self.nodes[id as usize];
        nd.leaf = false;
        nd.split_dim = best as u32;
        nd.split = split;
        nd.left = l;
        nd.right = r;
        id
    }

    #[inline]
    fn dist2(&self, i: usize, q: &[f64]) -> f64 {
        let mut s = 0.0;
        for k in 0..self.d {
            let diff = self.pts[i * self.d + k] - q[k];
            s += diff * diff;
        }
        s
    }

    /// k nearest neighbours of `q`: (distance, index), sorted by
    /// (distance, index); missing neighbours (k > n) are (inf, n).
    pub fn query(&self, q: &[f64], k: usize) -> Vec<(f64, usize)> {
        let mut best: Vec<(f64, usize)> = Vec::with_capacity(k + 1);
        if self.n > 0 {
            self.search(0, q, k, &mut best);
        }
        let mut out: Vec<(f64, usize)> = best.into_iter().map(|(d2, i)| (d2.sqrt(), i)).collect();
        while out.len() < k {
            out.push((f64::INFINITY, self.n));
        }
        out
    }

    fn search(&self, id: u32, q: &[f64], k: usize, best: &mut Vec<(f64, usize)>) {
        let nd = &self.nodes[id as usize];
        if nd.leaf {
            for &i in &self.order[nd.lo as usize..nd.hi as usize] {
                let d2 = self.dist2(i as usize, q);
                self.offer(best, k, d2, i as usize);
            }
            return;
        }
        let diff = q[nd.split_dim as usize] - nd.split;
        let (near, far) = if diff < 0.0 { (nd.left, nd.right) } else { (nd.right, nd.left) };
        self.search(near, q, k, best);
        // the far side can hold a point at distance diff^2 or more; use
        // <= so exact ties are never pruned
        if best.len() < k || diff * diff <= best[best.len() - 1].0 {
            self.search(far, q, k, best);
        }
    }

    fn offer(&self, best: &mut Vec<(f64, usize)>, k: usize, d2: f64, i: usize) {
        if best.len() == k {
            let w = best[k - 1];
            if !(d2 < w.0 || (d2 == w.0 && i < w.1)) {
                return;
            }
        }
        let pos = best.partition_point(|&(bd, bi)| bd < d2 || (bd == d2 && bi < i));
        best.insert(pos, (d2, i));
        if best.len() > k {
            best.pop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairwise_matches_naive_for_short() {
        let a = [1.0, 2.0, 3.0];
        assert_eq!(np_sum(&a), 6.0);
    }

    #[test]
    fn gaussian_kernel_sums_to_one() {
        for &s in &[0.35, 1.0, 2.5, 7.3] {
            let k = gaussian_kernel1d(s, gaussian_radius(s, 4.0));
            assert!((k.iter().sum::<f64>() - 1.0).abs() < 1e-14);
            assert_eq!(k.len() % 2, 1);
        }
    }

    #[test]
    fn constant_image_unchanged_by_gaussian() {
        let a = vec![3.5f64; 7 * 5];
        for mode in [Mode::Reflect, Mode::Nearest, Mode::Wrap] {
            let o = gaussian_filter(&a, &[7, 5], &[2.0, 9.0], mode, 4.0);
            for v in o {
                assert!((v - 3.5).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn sobel_of_ramp() {
        // ramp along axis 1: derivative * smoothing = 8 inside
        let (h, w) = (5usize, 6usize);
        let a: Vec<f64> = (0..h * w).map(|i| (i % w) as f64).collect();
        let o = sobel(&a, &[h, w], 1, Mode::Nearest);
        assert_eq!(o[2 * w + 2], 8.0);
        let o0 = sobel(&a, &[h, w], 0, Mode::Nearest);
        assert_eq!(o0[2 * w + 2], 0.0);
    }

    #[test]
    fn uniform_and_maximum() {
        let a = vec![1.0f64, 2.0, 6.0, 2.0, 1.0];
        let u = uniform_filter(&a, &[5], 3, Mode::Reflect, 0.0);
        assert!((u[2] - 10.0 / 3.0).abs() < 1e-12);
        let m = maximum_filter(&a, &[5], 3, Mode::Reflect, 0.0);
        assert_eq!(m, vec![2.0, 6.0, 6.0, 6.0, 2.0]);
    }

    #[test]
    fn zoom_linear_ramp_is_linear() {
        let a: Vec<f64> = (0..4).map(|i| i as f64).collect();
        let (o, s) = zoom(&a, &[4], &[2.0], 1, Mode::Constant, 0.0);
        assert_eq!(s, vec![8]);
        for (i, v) in o.iter().enumerate().take(7) {
            assert!((v - i as f64 * 3.0 / 7.0).abs() < 1e-12);
        }
    }

    #[test]
    fn cubic_zoom_keeps_constants() {
        let a = vec![2.0f64; 6 * 5];
        let (o, s) = zoom(&a, &[6, 5], &[3.0, 3.0], 3, Mode::GridWrap, 0.0);
        assert_eq!(s, vec![18, 15]);
        for v in o {
            assert!((v - 2.0).abs() < 1e-12);
        }
    }

    #[test]
    fn fft_roundtrip() {
        let (h, w) = (6usize, 9usize);
        let a: Vec<f64> = (0..h * w).map(|i| ((i * 37 % 11) as f64) - 5.0).collect();
        let s = rfft2(&a, h, w);
        let b = irfft2(&s, h, w);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-12);
        }
    }

    #[test]
    fn kdtree_matches_brute_force() {
        let mut x = 12345u64;
        let mut r = || {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((x >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let pts: Vec<f64> = (0..300 * 3).map(|_| (r() * 8.0).floor()).collect(); // many ties
        let t = KdTree::new(&pts, 3);
        for _ in 0..50 {
            let q = [r() * 8.0, r() * 8.0, r() * 8.0];
            let got = t.query(&q, 5);
            let mut all: Vec<(f64, usize)> =
                (0..300).map(|i| (t.dist2(i, &q), i)).collect();
            all.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for (g, e) in got.iter().zip(all.iter().take(5)) {
                assert_eq!(g.1, e.1);
            }
        }
    }
}
