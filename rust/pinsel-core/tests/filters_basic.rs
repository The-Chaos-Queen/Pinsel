//! Checks with a right answer for the filters port (no Python needed).
//! Bit-identity against scipy is tested by rust/harness/test_filters_parity.py.

use pinsel_core::filters::*;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

fn field(h: usize, w: usize, seed: u64) -> Vec<f64> {
    let mut r = Lcg(seed);
    (0..h * w).map(|_| r.next() - 0.5).collect()
}

#[test]
fn gaussian_conserves_total_mass_with_reflect_and_wrap() {
    // reflect and wrap are mass conserving for a symmetric kernel; the sum
    // of the filtered image equals the sum of the input to rounding
    let (h, w) = (37, 29);
    let a = field(h, w, 1);
    let total: f64 = a.iter().sum();
    for mode in [Mode::Reflect, Mode::Wrap] {
        for sg in [0.5, 2.0, 11.0] {
            let o = gaussian_filter(&a, &[h, w], &[sg, sg], mode, 4.0);
            let t: f64 = o.iter().sum();
            assert!((t - total).abs() < 1e-10, "{mode:?} sigma {sg}: {t} vs {total}");
        }
    }
}

#[test]
fn gaussian_impulse_response_is_the_kernel() {
    let n = 41;
    let mut a = vec![0.0f64; n];
    a[20] = 1.0;
    let sg = 2.0;
    let o = gaussian_filter(&a, &[n], &[sg], Mode::Reflect, 4.0);
    let k = gaussian_kernel1d(sg, gaussian_radius(sg, 4.0));
    let r = gaussian_radius(sg, 4.0);
    for (i, &kv) in k.iter().enumerate() {
        assert_eq!(o[20 - r + i], kv);
    }
}

#[test]
fn f32_is_rounded_between_axis_passes() {
    let (h, w) = (9, 11);
    let a64 = field(h, w, 2);
    let a32: Vec<f32> = a64.iter().map(|&v| v as f32).collect();
    let got = gaussian_filter(&a32, &[h, w], &[1.5, 1.5], Mode::Nearest, 4.0);
    // same thing by hand: f64 pass along axis 0, round to f32, f64 pass along axis 1
    let a_as64: Vec<f64> = a32.iter().map(|&v| v as f64).collect();
    let p0 = gaussian_filter(&a_as64, &[h, w], &[1.5, 0.0], Mode::Nearest, 4.0);
    let p0r: Vec<f64> = p0.iter().map(|&v| (v as f32) as f64).collect();
    let p1 = gaussian_filter(&p0r, &[h, w], &[0.0, 1.5], Mode::Nearest, 4.0);
    for (g, e) in got.iter().zip(&p1) {
        assert_eq!(*g, (*e as f32));
    }
}

#[test]
fn constants_survive_every_filter() {
    let a = vec![2.5f64; 12 * 10];
    let s = [12usize, 10];
    for v in sobel(&a, &s, 0, Mode::Nearest) {
        assert_eq!(v, 0.0);
    }
    for v in uniform_filter(&a, &s, 3, Mode::Reflect, 0.0) {
        assert!((v - 2.5).abs() < 1e-14);
    }
    for v in maximum_filter(&a, &s, 5, Mode::Reflect, 0.0) {
        assert_eq!(v, 2.5);
    }
    let (z, zs) = zoom(&a, &s, &[2.3, 1.7], 1, Mode::Constant, 0.0);
    assert_eq!(zs, vec![28, 17]);
    // constant mode: an output pixel whose source coordinate rounds just
    // past the last row becomes 0 (scipy does the same); everything else is 2.5
    assert!(z.iter().all(|&v| (v - 2.5).abs() < 1e-12 || v == 0.0));
    assert!(z.iter().filter(|&&v| v != 0.0).count() > z.len() * 9 / 10);
}

#[test]
fn zoom_cubic_gridwrap_reproduces_the_samples_at_integer_scale_points() {
    // zoom 1 + k/(n-1)... at output points that fall on input samples the
    // spline interpolates the input (prefilter + synthesis are inverse)
    let n = 9;
    let a = field(1, n, 3);
    let (z, _) = zoom(&a, &[n], &[3.0], 3, Mode::GridWrap, 0.0);
    // output length 27, factor 8/26: sample j sits at output 26*j/8 (not an integer
    // except j = 0 and 4): check those
    assert!((z[0] - a[0]).abs() < 1e-13);
    assert!((z[13] - a[4]).abs() < 1e-13);
    assert!((z[26] - a[8]).abs() < 1e-13);
}

#[test]
fn kdtree_ties_go_to_the_lower_index_and_match_brute_force() {
    let mut r = Lcg(7);
    let pts: Vec<f64> = (0..400 * 3).map(|_| (r.next() * 6.0).floor()).collect();
    let t = KdTree::new(&pts, 3);
    for _ in 0..100 {
        let q = [(r.next() * 6.0).floor(), (r.next() * 6.0).floor(), (r.next() * 6.0).floor()];
        let got = t.query(&q, 6);
        let mut all: Vec<(f64, usize)> = (0..400)
            .map(|i| {
                let mut s = 0.0;
                for k in 0..3 {
                    let d = pts[i * 3 + k] - q[k];
                    s += d * d;
                }
                (s.sqrt(), i)
            })
            .collect();
        all.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(got, all[..6].to_vec());
    }
}

#[test]
fn fft_roundtrip_and_dc() {
    let (h, w) = (10usize, 13usize);
    let a = field(h, w, 4);
    let spec = rfft2(&a, h, w);
    let back = irfft2(&spec, h, w);
    for (x, y) in a.iter().zip(&back) {
        assert!((x - y).abs() < 1e-13);
    }
    let sum: f64 = a.iter().sum();
    assert!((spec[0].re - sum).abs() < 1e-12 && spec[0].im.abs() < 1e-12);
}

#[test]
fn binary_morphology_laws() {
    let (h, w) = (20, 25);
    let mut r = Lcg(9);
    let a: Vec<u8> = (0..h * w).map(|_| (r.next() < 0.6) as u8).collect();
    let er = binary_morph(&a, h, w, false, 1);
    let di = binary_morph(&a, h, w, true, 1);
    for i in 0..a.len() {
        assert!(er[i] <= a[i] && a[i] <= di[i]);
    }
    let op = binary_opening(&a, h, w, 1);
    assert_eq!(binary_opening(&op, h, w, 1), op);
    let cl = binary_closing(&a, h, w, 1);
    assert_eq!(binary_closing(&cl, h, w, 1), cl);
}
