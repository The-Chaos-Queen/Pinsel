//! Synthetic-input checks for the painterly stroke-growing kernels (no Python).

use pinsel_core::grid::{Plane, Spec, SpecMut};
use pinsel_core::painterly::{bil, densify, field_dir, grow_strokes, GrowParams};

const H: usize = 40;
const W: usize = 64;

fn gp(max_steps: i64, back_steps: i64) -> GrowParams {
    GrowParams {
        step: 2.0,
        min_steps: 4,
        max_steps,
        back_steps,
        fc: 1.0,
        edge_stop: 50.0,
        turn_cos: -1.0,
        dab_len: 6.0,
        margin: 0.0,
        color_mix: 0.5,
    }
}

/// Reference with value `c` in every pixel, a field pointing along +x.
fn flat() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let rf: Vec<f32> = (0..H * W).flat_map(|_| [50.0f32, 10.0, -20.0]).collect();
    let cvl: Vec<f32> = (0..H * W).flat_map(|_| [90.0f32, 40.0, 30.0]).collect();
    let v: Vec<f32> = (0..H * W).flat_map(|_| [1.0f32, 0.0]).collect();
    (rf, cvl, v, vec![1.0f32; H * W])
}

#[test]
fn bilinear_samples_clamp_and_interpolate() {
    let data: Vec<f32> = (0..H * W).map(|i| (i % W) as f32 + 100.0 * (i / W) as f32).collect();
    let img = Spec::new(H, W, 1, &data);
    assert_eq!(bil(&img, 3.0, 2.0, 0), 203.0);
    assert!((bil(&img, 3.5, 2.0, 0) - 203.5).abs() < 1e-12);
    assert!((bil(&img, 3.0, 2.25, 0) - 228.0).abs() < 1e-9);
    // clamped at all four borders
    assert_eq!(bil(&img, -5.0, -5.0, 0), 0.0);
    assert_eq!(bil(&img, 1e6, 1e6, 0), data[H * W - 1] as f64);
    assert_eq!(bil(&img, 1e6, -3.0, 0), (W - 1) as f64);
}

#[test]
fn field_dir_follows_the_doubled_angle_and_weak_field_is_zero() {
    let v: Vec<f32> = (0..H * W).flat_map(|_| [0.0f32, 1.0]).collect(); // doubled angle 90 deg
    let f = Spec::new(H, W, 2, &v);
    let (dx, dy, m) = field_dir(&f, 10.0, 10.0, 0.0);
    assert!((m - 1.0).abs() < 1e-12);
    assert!((dx - (std::f64::consts::FRAC_PI_4).cos()).abs() < 1e-12);
    assert!((dy - (std::f64::consts::FRAC_PI_4).sin()).abs() < 1e-12);
    let weak: Vec<f32> = vec![0.01f32; H * W * 2];
    let f = Spec::new(H, W, 2, &weak);
    let (dx, dy, m) = field_dir(&f, 5.0, 5.0, 0.3);
    assert_eq!((dx, dy), (0.0, 0.0));
    assert!(m < 0.04);
}

#[test]
fn stroke_grows_both_ways_along_a_straight_field() {
    let (rf, cvl, v, region) = flat();
    let (sx, sy) = (vec![32.0f64], vec![20.0f64]);
    let max_steps = 12;
    let mut ctrl = vec![0.0f64; (max_steps as usize + 2) * 2];
    let mut nctrl = vec![0i64; 1];
    let mut cols = vec![0.0f64; 3];
    grow_strokes(
        &sx,
        &sy,
        &[0.0],
        &Spec::new(H, W, 3, &rf),
        &Spec::new(H, W, 3, &cvl),
        &Spec::new(H, W, 2, &v),
        &Plane::new(H, W, &region),
        &gp(max_steps, 4),
        &mut SpecMut::new(1, max_steps as usize + 2, 2, &mut ctrl),
        &mut nctrl,
        &mut SpecMut::new(1, 1, 3, &mut cols),
    );
    // 4 back points, the seed, 8 forward points (max_steps - nb)
    assert_eq!(nctrl[0], 4 + 1 + 8);
    let pt = |k: usize| (ctrl[2 * k], ctrl[2 * k + 1]);
    for k in 0..nctrl[0] as usize {
        let (x, y) = pt(k);
        assert_eq!(y, 20.0);
        assert!((x - (24.0 + 2.0 * k as f64)).abs() < 1e-12, "k={k} x={x}");
    }
    // flat reference: the stroke colour is the reference colour
    assert!((cols[0] - 50.0).abs() < 1e-9 && (cols[1] - 10.0).abs() < 1e-9 && (cols[2] + 20.0).abs() < 1e-9);
}

#[test]
fn stroke_stops_at_the_border_and_at_the_region_edge() {
    let (rf, cvl, v, mut region) = flat();
    for y in 0..H {
        for x in 40..W {
            region[y * W + x] = 0.0;
        }
    }
    let max_steps = 100;
    let mut ctrl = vec![0.0f64; (max_steps as usize + 2) * 2];
    let mut nctrl = vec![0i64; 1];
    let mut cols = vec![0.0f64; 3];
    grow_strokes(
        &[32.0],
        &[20.0],
        &[0.0],
        &Spec::new(H, W, 3, &rf),
        &Spec::new(H, W, 3, &cvl),
        &Spec::new(H, W, 2, &v),
        &Plane::new(H, W, &region),
        &gp(max_steps, 0),
        &mut SpecMut::new(1, max_steps as usize + 2, 2, &mut ctrl),
        &mut nctrl,
        &mut SpecMut::new(1, 1, 3, &mut cols),
    );
    // forward points x = 34, 36, 38 (x = 40 rounds into the empty region)
    assert_eq!(nctrl[0], 1 + 3);
    assert_eq!((ctrl[0], ctrl[2], ctrl[4], ctrl[6]), (32.0, 34.0, 36.0, 38.0));
}

#[test]
fn dead_end_makes_a_dab_along_the_local_direction() {
    let (rf, cvl, v, _) = flat();
    let region = vec![0.0f32; H * W]; // nothing may be painted: every walk stops at once
    let max_steps = 10;
    let mut ctrl = vec![0.0f64; (max_steps as usize + 2) * 2];
    let mut nctrl = vec![0i64; 1];
    let mut cols = vec![0.0f64; 3];
    grow_strokes(
        &[32.0],
        &[20.0],
        &[0.0],
        &Spec::new(H, W, 3, &rf),
        &Spec::new(H, W, 3, &cvl),
        &Spec::new(H, W, 2, &v),
        &Plane::new(H, W, &region),
        &gp(max_steps, 3),
        &mut SpecMut::new(1, max_steps as usize + 2, 2, &mut ctrl),
        &mut nctrl,
        &mut SpecMut::new(1, 1, 3, &mut cols),
    );
    assert_eq!(nctrl[0], 2);
    assert_eq!((ctrl[0], ctrl[1], ctrl[2], ctrl[3]), (29.0, 20.0, 35.0, 20.0));
}

#[test]
fn edge_stop_ends_a_stroke_on_a_colour_edge() {
    let (mut rf, cvl, v, region) = flat();
    for y in 0..H {
        for x in 41..W {
            rf[(y * W + x) * 3] = 95.0; // a bright step in L at x > 40
        }
    }
    let mut p = gp(60, 0);
    p.edge_stop = 10.0;
    let mut ctrl = vec![0.0f64; 62 * 2];
    let mut nctrl = vec![0i64; 1];
    let mut cols = vec![0.0f64; 3];
    grow_strokes(
        &[32.0],
        &[20.0],
        &[0.0],
        &Spec::new(H, W, 3, &rf),
        &Spec::new(H, W, 3, &cvl),
        &Spec::new(H, W, 2, &v),
        &Plane::new(H, W, &region),
        &p,
        &mut SpecMut::new(1, 62, 2, &mut ctrl),
        &mut nctrl,
        &mut SpecMut::new(1, 1, 3, &mut cols),
    );
    let last = ctrl[2 * (nctrl[0] as usize - 1)];
    assert!(last < 41.0, "stroke crossed the edge: last x = {last}");
    assert!(nctrl[0] >= 4);
}

#[test]
fn densify_reproduces_a_straight_polyline_and_endpoints() {
    let k = 6usize;
    let sub = 4i64;
    let mut ctrl = vec![0.0f64; 2 * (k + 2) * 2];
    for s in 0..2 {
        for i in 0..k {
            let base = (s * (k + 2) + i) * 2;
            ctrl[base] = 3.0 + 2.5 * i as f64;
            ctrl[base + 1] = 7.0 + s as f64 * 10.0 - 1.5 * i as f64;
        }
    }
    let nctrl = vec![k as i64, k as i64];
    let mo = (k + 1) * sub as usize + 2;
    let mut out = vec![-1.0f64; 2 * mo * 2];
    let mut nout = vec![0i64; 2];
    densify(
        &Spec::new(2, k + 2, 2, &ctrl),
        &nctrl,
        sub,
        &mut SpecMut::new(2, mo, 2, &mut out),
        &mut nout,
    );
    assert_eq!(nout, vec![(k as i64 - 1) * sub + 1; 2]);
    for s in 0..2usize {
        for j in 0..nout[s] as usize {
            let t = j as f64 / sub as f64; // in control-point units
            let (x, y) = (out[(s * mo + j) * 2], out[(s * mo + j) * 2 + 1]);
            assert!((x - (3.0 + 2.5 * t)).abs() < 1e-9, "s={s} j={j}");
            assert!((y - (7.0 + s as f64 * 10.0 - 1.5 * t)).abs() < 1e-9);
        }
    }
}

#[test]
fn densify_short_strokes_are_straight_segments() {
    let ctrl = vec![1.0, 2.0, 5.0, 10.0, 0.0, 0.0];
    let nctrl = vec![2i64];
    let sub = 4i64;
    let mut out = vec![0.0f64; 8 * 2];
    let mut nout = vec![0i64; 1];
    densify(&Spec::new(1, 3, 2, &ctrl), &nctrl, sub, &mut SpecMut::new(1, 8, 2, &mut out), &mut nout);
    assert_eq!(nout[0], 5);
    assert_eq!((out[0], out[1]), (1.0, 2.0));
    assert_eq!((out[8], out[9]), (5.0, 10.0));
    assert!((out[4] - 3.0).abs() < 1e-12 && (out[5] - 6.0).abs() < 1e-12);
}

#[test]
fn densify_curve_passes_through_control_points() {
    // k = 3 exercises both end-point reflections at once
    let ctrl = vec![0.0, 0.0, 4.0, 3.0, 9.0, 1.0, 0.0, 0.0];
    let nctrl = vec![3i64];
    let sub = 5i64;
    let mut out = vec![0.0f64; 14 * 2];
    let mut nout = vec![0i64; 1];
    densify(&Spec::new(1, 4, 2, &ctrl), &nctrl, sub, &mut SpecMut::new(1, 14, 2, &mut out), &mut nout);
    assert_eq!(nout[0], 11);
    for (i, want) in [(0usize, (0.0, 0.0)), (5, (4.0, 3.0)), (10, (9.0, 1.0))] {
        assert!((out[i * 2] - want.0).abs() < 1e-12 && (out[i * 2 + 1] - want.1).abs() < 1e-12);
    }
}
