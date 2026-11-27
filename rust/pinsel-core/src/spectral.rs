//! Kubelka-Munk layer equations: `pinsel.spectral.km_layer` / `km_over`.
//!
//! numba unifies the arguments to float64 (they are reassigned float64
//! literals in the clamps), so callers pass f32 canvas values cast to f64.

/// Reflectance R and transmittance T of a layer with optical depths K*x, S*x.
#[inline(always)]
pub fn km_layer(k: f64, s: f64) -> (f64, f64) {
    let k = if k < 1e-5 { 1e-5 } else { k };
    let s = if s < 1e-5 { 1e-5 } else { s };
    let a = 1.0 + k / s;
    let b = (a * a - 1.0).sqrt();
    let bs = b * s;
    if bs > 25.0 {
        return (a - b, 0.0);
    }
    let sh = bs.sinh();
    let ch = bs.cosh();
    let d = a * sh + b * ch;
    (sh / d, b / d)
}

/// Reflectance of a layer (K*x, S*x) over a substrate of reflectance `rg`.
#[inline(always)]
pub fn km_over(k: f64, s: f64, rg: f64) -> f64 {
    let (r, t) = km_layer(k, s);
    r + t * t * rg / (1.0 - r * rg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opaque_thick_layer_hides_the_substrate() {
        let a = km_over(5.0, 50.0, 0.0);
        let b = km_over(5.0, 50.0, 1.0);
        assert!((a - b).abs() < 1e-9, "thick layer should hide what is below");
    }

    #[test]
    fn vanishing_layer_shows_the_substrate() {
        let r = km_over(0.0, 0.0, 0.6);
        assert!((r - 0.6).abs() < 1e-3, "an empty layer should leave Rg, got {r}");
    }
}
