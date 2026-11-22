"""
Spectral colour science for pinsel.

Paint colour is not RGB.  Every paint in pinsel is described by its
Kubelka-Munk absorption (K) and scattering (S) coefficients sampled in
``NB`` wavelength bands between 380 and 740 nm.  Mixing paints means adding
their K and S (weighted by volume), which reproduces subtractive pigment
behaviour: blue + yellow gives green, a touch of phthalo dominates a mix, a
transparent glaze tints whatever lies underneath.

This module provides
  * the band layout (``NB``, ``BAND_CENTERS``),
  * a reflectance -> linear sRGB matrix built from analytic CIE 1931
    colour matching functions under D65,
  * sRGB <-> linear conversion helpers,
  * an RGB -> smooth reflectance spectrum fit (sigmoid-quadratic, after
    Jakob & Hanika 2019),
  * Kubelka-Munk helpers in numpy and numba.
"""
from __future__ import annotations

import math
import numpy as np
from numba import njit

# --------------------------------------------------------------------------
# Band layout
# --------------------------------------------------------------------------
NB = 12
BAND_EDGES = np.linspace(380.0, 740.0, NB + 1)
BAND_CENTERS = 0.5 * (BAND_EDGES[:-1] + BAND_EDGES[1:])

# --------------------------------------------------------------------------
# CIE 1931 2deg colour matching functions (Wyman, Sloan & Shirley 2013 fit)
# --------------------------------------------------------------------------

def _g(lam, mu, s1, s2):
    s = np.where(lam < mu, s1, s2)
    return np.exp(-0.5 * ((lam - mu) / s) ** 2)


def cie_xyz_bar(lam):
    lam = np.asarray(lam, dtype=np.float64)
    x = (1.056 * _g(lam, 599.8, 37.9, 31.0) + 0.362 * _g(lam, 442.0, 16.0, 26.7)
         - 0.065 * _g(lam, 501.1, 20.4, 26.2))
    y = 0.821 * _g(lam, 568.8, 46.9, 40.5) + 0.286 * _g(lam, 530.9, 16.3, 31.1)
    z = 1.217 * _g(lam, 437.0, 11.8, 36.0) + 0.681 * _g(lam, 459.0, 26.0, 13.8)
    return np.stack([x, y, z], axis=-1)


# CIE D65 relative spectral power, 380..740 nm in 10 nm steps
_D65 = np.array([
    50.0, 54.6, 82.8, 91.5, 93.4, 86.7, 104.9, 117.0, 117.8, 114.9, 115.9,
    108.8, 109.4, 107.8, 104.8, 107.7, 104.4, 104.0, 100.0, 96.3, 95.8,
    88.7, 90.0, 89.6, 87.7, 83.3, 83.7, 80.0, 80.2, 82.3, 78.3, 69.7,
    71.6, 74.3, 61.6, 69.9, 75.1])
_D65_LAM = np.arange(380.0, 741.0, 10.0)

XYZ_TO_LINEAR_SRGB = np.array([
    [3.2406, -1.5372, -0.4986],
    [-0.9689, 1.8758, 0.0415],
    [0.0557, -0.2040, 1.0570]])
LINEAR_SRGB_TO_XYZ = np.linalg.inv(XYZ_TO_LINEAR_SRGB)


def _build_band_matrix():
    """3 x NB matrix mapping band reflectance -> linear sRGB (white -> 1,1,1)."""
    lam = np.arange(380.0, 740.0, 1.0) + 0.5
    d65 = np.interp(lam, _D65_LAM, _D65)
    cmf = cie_xyz_bar(lam) * d65[:, None]
    band_xyz = np.zeros((NB, 3))
    for b in range(NB):
        sel = (lam >= BAND_EDGES[b]) & (lam < BAND_EDGES[b + 1])
        band_xyz[b] = cmf[sel].sum(axis=0)
    band_xyz /= band_xyz[:, 1].sum()
    M = XYZ_TO_LINEAR_SRGB @ band_xyz.T          # 3 x NB
    white = M.sum(axis=1)
    M = M / white[:, None]                       # perfect reflector -> (1,1,1)
    return M.astype(np.float64), band_xyz


SPEC_TO_RGB, BAND_XYZ = _build_band_matrix()
SPEC_TO_RGB32 = SPEC_TO_RGB.astype(np.float32)


# --------------------------------------------------------------------------
# sRGB transfer functions
# --------------------------------------------------------------------------

def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, None)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def parse_color(color):
    """Accept '#rrggbb', 'rrggbb', (r,g,b) 0..255 ints or 0..1 floats -> sRGB floats."""
    if isinstance(color, str):
        s = color.lstrip('#')
        if len(s) == 3:
            s = ''.join(ch * 2 for ch in s)
        return np.array([int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])
    c = np.asarray(color, dtype=np.float64)
    if c.max() > 1.0:
        c = c / 255.0
    return c[:3]


def spectrum_to_linear_rgb(R):
    R = np.asarray(R, dtype=np.float64)
    return R @ SPEC_TO_RGB.T


def spectrum_to_srgb(R):
    return np.clip(linear_to_srgb(spectrum_to_linear_rgb(R)), 0, 1)


def spectrum_to_hex(R):
    c = np.round(spectrum_to_srgb(R) * 255).astype(int)
    return '#%02x%02x%02x' % tuple(c)


# CIELAB (D65) for perceptual comparisons
_XN = LINEAR_SRGB_TO_XYZ @ np.ones(3)


def linear_rgb_to_lab(rgb):
    rgb = np.asarray(rgb, dtype=np.float64)
    xyz = rgb @ LINEAR_SRGB_TO_XYZ.T / _XN
    e = 216 / 24389
    k = 24389 / 27
    f = np.where(xyz > e, np.cbrt(np.maximum(xyz, 0)), (k * xyz + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


# --------------------------------------------------------------------------
# RGB -> smooth reflectance spectrum
# --------------------------------------------------------------------------
_T = (BAND_CENTERS - 560.0) / 180.0      # normalised wavelength in [-1, 1]


def _sig_spec(c):
    z = c[0] + c[1] * _T + c[2] * _T * _T
    return 0.5 + 0.5 * z / np.sqrt(1.0 + z * z)


def rgb_to_spectrum(color, lo=0.002, hi=0.985):
    """Fit a smooth, bounded reflectance spectrum whose sRGB matches ``color``.

    Uses the sigmoid-of-quadratic model from Jakob & Hanika (2019), fitted by
    damped Gauss-Newton in CIELAB, falling back gracefully for colours outside
    the gamut of physically plausible reflectances.
    """
    target = srgb_to_linear(parse_color(color))
    target = np.clip(target, 1e-4, 1.0)
    lab_t = linear_rgb_to_lab(target)
    best, best_err = None, 1e18
    lum = float(np.clip(target @ np.array([0.2126, 0.7152, 0.0722]), 0.01, 0.99))
    starts = [np.array([math.copysign(1, lum - .5) * min(5, abs(2 * lum - 1) / max(1e-3, math.sqrt(1 - (2 * lum - 1) ** 2))), 0.0, 0.0])]
    for c1 in (-6, 6):
        for c2 in (-8, 8):
            starts.append(np.array([0.0, c1, c2]))

    def resid(c):
        R = lo + (hi - lo) * _sig_spec(c)
        return linear_rgb_to_lab(spectrum_to_linear_rgb(R)) - lab_t

    for c in starts:
        c = c.astype(np.float64)
        lam_d = 1e-3
        r = resid(c)
        err = float(r @ r)
        for _ in range(60):
            J = np.zeros((3, 3))
            for j in range(3):
                dc = np.zeros(3); dc[j] = 1e-4
                J[:, j] = (resid(c + dc) - r) / 1e-4
            A = J.T @ J + lam_d * np.eye(3)
            step = np.linalg.solve(A, -J.T @ r)
            cn = c + step
            rn = resid(cn)
            en = float(rn @ rn)
            if en < err:
                c, r, err = cn, rn, en
                lam_d = max(lam_d * 0.3, 1e-7)
                if err < 1e-6:
                    break
            else:
                lam_d *= 10
                if lam_d > 1e6:
                    break
        if err < best_err:
            best, best_err = c, err
    return lo + (hi - lo) * _sig_spec(best)


# --------------------------------------------------------------------------
# Kubelka-Munk
# --------------------------------------------------------------------------

def ks_from_reflectance(R):
    """K/S ratio of an opaque layer with infinite-thickness reflectance R."""
    R = np.clip(np.asarray(R, dtype=np.float64), 1e-4, 0.9999)
    return (1 - R) ** 2 / (2 * R)


def reflectance_from_ks(ks):
    ks = np.maximum(np.asarray(ks, dtype=np.float64), 0)
    return 1 + ks - np.sqrt(ks * ks + 2 * ks)


def km_layer_np(K, S):
    """Reflectance and transmittance of a layer with optical depths K, S."""
    K = np.maximum(np.asarray(K, dtype=np.float64), 1e-5)
    S = np.maximum(np.asarray(S, dtype=np.float64), 1e-5)
    a = 1 + K / S
    b = np.sqrt(a * a - 1)
    bs = np.minimum(b * S, 30.0)
    sh, ch = np.sinh(bs), np.cosh(bs)
    d = a * sh + b * ch
    return sh / d, b / d


def km_composite_np(K, S, Rg):
    R, T = km_layer_np(K, S)
    return R + T * T * Rg / (1 - R * Rg)


@njit(cache=True, fastmath=True, inline='always')
def km_layer(K, S):
    if K < 1e-5:
        K = 1e-5
    if S < 1e-5:
        S = 1e-5
    a = 1.0 + K / S
    b = math.sqrt(a * a - 1.0)
    bs = b * S
    if bs > 25.0:
        return a - b, 0.0
    sh = math.sinh(bs)
    ch = math.cosh(bs)
    d = a * sh + b * ch
    return sh / d, b / d


@njit(cache=True, fastmath=True, inline='always')
def km_over(K, S, Rg):
    R, T = km_layer(K, S)
    return R + T * T * Rg / (1.0 - R * Rg)
