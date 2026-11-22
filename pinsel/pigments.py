"""
Pigments, paints and paint mixing.

A :class:`Paint` stores Kubelka-Munk absorption ``K`` and scattering ``S``
per unit *volume* of paint (12 spectral bands), together with the properties
of its binder / medium (oil, acrylic, gouache, watercolour, ink).

Paints mix by volume:  K_mix = sum(v_i K_i) / sum(v_i)  (same for S), which is
exactly what happens on the canvas, too.

    >>> from pinsel import pigments as pg
    >>> green = pg.mix(pg.CADMIUM_YELLOW, 3, pg.PHTHALO_BLUE, 1)
    >>> tint  = pg.mix(pg.TITANIUM_WHITE, 4, pg.ULTRAMARINE, 1)
    >>> glaze = pg.ALIZARIN_CRIMSON.thinned(0.7)
    >>> wc    = pg.ULTRAMARINE.as_medium('watercolor')
"""
from __future__ import annotations

import copy
import numpy as np

from .spectral import (NB, BAND_CENTERS, rgb_to_spectrum, ks_from_reflectance,
                       km_composite_np, spectrum_to_srgb, spectrum_to_hex,
                       parse_color)

# --------------------------------------------------------------------------
# Media (binders)
# --------------------------------------------------------------------------
MEDIA = {
    #             paste/fluid  wet gloss  dry gloss  body   dry time(s)  S boost  viscosity
    'oil':        dict(fluid=False, gloss=0.85, gloss_dry=0.55, body=0.5, dry_time=3.0e5, s_boost=1.0, viscosity=0.75),
    'acrylic':    dict(fluid=False, gloss=0.55, gloss_dry=0.35, body=0.4, dry_time=900.0, s_boost=1.0, viscosity=0.6),
    'gouache':    dict(fluid=False, gloss=0.08, gloss_dry=0.02, body=0.15, dry_time=400.0, s_boost=1.7, viscosity=0.35),
    'watercolor': dict(fluid=True, gloss=0.25, gloss_dry=0.0, body=0.0, dry_time=240.0, s_boost=1.0, viscosity=0.0),
    'ink':        dict(fluid=True, gloss=0.3, gloss_dry=0.05, body=0.0, dry_time=120.0, s_boost=1.0, viscosity=0.0),
}


class Paint:
    """A paint: spectral K and S per unit volume plus medium properties."""

    def __init__(self, K, S, name='paint', medium='oil', granulation=0.0,
                 staining=0.3, **overrides):
        self.K = np.asarray(K, dtype=np.float64).reshape(NB).copy()
        self.S = np.asarray(S, dtype=np.float64).reshape(NB).copy()
        self.name = name
        self.granulation = float(granulation)
        self.staining = float(staining)
        self._set_medium(medium)
        for k, v in overrides.items():
            setattr(self, k, v)

    # -- medium handling ----------------------------------------------------
    def _set_medium(self, medium):
        if medium not in MEDIA:
            raise ValueError(f'unknown medium {medium!r}; choose from {list(MEDIA)}')
        self.medium = medium
        for k, v in MEDIA[medium].items():
            setattr(self, k, v)

    def as_medium(self, medium):
        """Same pigment(s), different binder: 'oil', 'acrylic', 'gouache',
        'watercolor' or 'ink'."""
        p = copy.deepcopy(self)
        old = MEDIA[self.medium]['s_boost']
        p._set_medium(medium)
        p.S = self.S / old * p.s_boost
        if medium == 'ink':           # inks are dyes / very fine dispersions
            p.granulation = self.granulation * 0.15
        return p

    def copy(self, **changes):
        p = copy.deepcopy(self)
        for k, v in changes.items():
            setattr(p, k, v)
        return p

    # -- modifications --------------------------------------------------------
    def thinned(self, fraction=0.5):
        """Dilute with clear medium (glazing medium / water): pigment per volume
        drops, so thin layers become transparent."""
        f = float(np.clip(1.0 - fraction, 0.0, 1.0))
        # lean, thinned paint also builds much less relief
        return self.copy(K=self.K * f, S=self.S * f, body=self.body * max(0.25, f),
                         name=f'{self.name} thinned')

    def stronger(self, factor=1.5):
        return self.copy(K=self.K * factor, S=self.S * factor)

    def with_body(self, body):
        """Change how much relief (impasto height) a unit of paint builds."""
        return self.copy(body=float(body))

    def impasto(self, factor=2.0):
        """Mix with impasto gel: more body, slightly more transparent."""
        return self.copy(body=self.body * factor * 1.5, K=self.K / factor ** 0.3,
                         S=self.S / factor ** 0.3, name=f'{self.name} impasto')

    # -- appearance -------------------------------------------------------------
    def reflectance(self, thickness=4.0, ground=None):
        """Band reflectance of a layer of ``thickness`` volume units over a
        ground (white by default)."""
        g = np.full(NB, 0.9) if ground is None else np.asarray(ground)
        return km_composite_np(self.K * thickness, self.S * thickness, g)

    def rgb(self, thickness=4.0, ground=None):
        return spectrum_to_srgb(self.reflectance(thickness, ground))

    def hex(self, thickness=4.0, ground=None):
        return spectrum_to_hex(self.reflectance(thickness, ground))

    def over(self, substrate, thickness=0.4):
        """Preview this paint layered over ``substrate`` as sRGB floats (3,).

        ``substrate`` may be another :class:`Paint` (uses its masstone
        reflectance), a hex/RGB colour, or a 12-band reflectance spectrum.
        """
        if isinstance(substrate, Paint):
            bg = substrate.reflectance(4.0)
        elif isinstance(substrate, np.ndarray) and substrate.shape == (NB,):
            bg = substrate
        else:
            bg = rgb_to_spectrum(substrate)
        return spectrum_to_srgb(self.reflectance(thickness, ground=bg))

    def over_hex(self, substrate, thickness=0.4):
        """Preview this paint layered over ``substrate`` as a ``'#rrggbb'`` string."""
        c = np.round(self.over(substrate, thickness=thickness) * 255).astype(int)
        return '#%02x%02x%02x' % tuple(c)

    def __repr__(self):
        return f'<Paint {self.name!r} {self.medium} {self.hex()}>'

    # -- construction -------------------------------------------------------------
    @classmethod
    def from_color(cls, color, opacity=0.7, name=None, medium='oil', **kw):
        """Create a paint whose masstone matches an sRGB colour.

        ``opacity`` (0..1) sets the scattering power: 1 = covers in one thin
        coat like titanium white, 0 = a transparent dye.
        """
        R = rgb_to_spectrum(color)
        S = 0.05 + 6.0 * float(opacity) ** 1.5
        Sv = np.full(NB, S) * (560.0 / BAND_CENTERS) ** 0.3
        K = Sv * ks_from_reflectance(R)
        p = cls(K, Sv, name=name or str(color), medium='oil', **kw)
        return p.as_medium(medium) if medium != 'oil' else p

    @classmethod
    def from_glaze(cls, color, scatter=0.08, strength=1.0, name=None, medium='oil',
                   red_tail=True, **kw):
        """Create a transparent paint whose thin glaze over white looks like
        ``color``.  ``red_tail=False`` suppresses deep-red reflectance (for
        phthalo / prussian type pigments that absorb red completely)."""
        Rw = 0.9
        Rg = np.minimum(rgb_to_spectrum(color), Rw * 0.995)
        if not red_tail:
            Rg = _clip_red_tail(Rg)
        K = -np.log(Rg / Rw) / (2 * 0.2) * strength
        S = np.full(NB, scatter * strength) * (560.0 / BAND_CENTERS) ** 0.5
        p = cls(K, S, name=name or str(color), medium='oil', **kw)
        return p.as_medium(medium) if medium != 'oil' else p


def _clip_red_tail(R):
    """Make reflectance non-increasing beyond its minimum in the red."""
    R = R.copy()
    idx = np.where(BAND_CENTERS > 560)[0]
    m = idx[np.argmin(R[idx])]
    R[m:] = np.minimum(R[m:], R[m])
    return R


def mix(*args):
    """Mix paints by volume.

    ``mix(a, 2, b, 1)``  or  ``mix([(a, 2), (b, 1)])``  or  ``mix(a, b)``.
    The medium of the first paint wins; other properties are volume weighted.
    """
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        pairs = [(p, w) if isinstance(p, Paint) else p for p, w in args[0]] \
            if args[0] and isinstance(args[0][0], (list, tuple)) else [(p, 1.0) for p in args[0]]
    else:
        pairs, i = [], 0
        while i < len(args):
            p = args[i]
            w = 1.0
            if i + 1 < len(args) and not isinstance(args[i + 1], Paint):
                w = float(args[i + 1]); i += 1
            pairs.append((p, w)); i += 1
    total = sum(w for _, w in pairs)
    base = pairs[0][0]
    K = sum(p.K * w for p, w in pairs) / total
    S = sum(p.S * w for p, w in pairs) / total
    out = base.copy(K=K, S=S, name=' + '.join(p.name for p, _ in pairs))
    for attr in ('granulation', 'staining', 'body', 'gloss', 'gloss_dry', 'viscosity'):
        setattr(out, attr, sum(getattr(p, attr) * w for p, w in pairs) / total)
    return out


# --------------------------------------------------------------------------
# Pigment library
# --------------------------------------------------------------------------

def _keep_red(R):
    """Warm pigments reflect everything beyond their absorption edge: make the
    reflectance non-decreasing past its long-wavelength maximum."""
    R = R.copy()
    idx = np.where(BAND_CENTERS > 500)[0]
    m = idx[np.argmax(R[idx])]
    R[m:] = np.maximum(R[m:], R[m])
    return R


def _absorber(kvals, name, scatter=0.08, strength=1.0, **kw):
    """Transparent pigment from a hand-shaped absorption spectrum (12 bands)."""
    K = np.asarray(kvals, dtype=np.float64) * strength
    S = np.full(NB, scatter) * (560.0 / BAND_CENTERS) ** 0.5
    return Paint(K, S, name=name, **kw)


def _opaque(hexcol, S, name, slope=0.3, warm=False, **kw):
    R = rgb_to_spectrum(hexcol)
    if warm:
        R = _keep_red(R)
    Sv = np.full(NB, S) * (560.0 / BAND_CENTERS) ** slope
    return Paint(Sv * ks_from_reflectance(R), Sv, name=name, **kw)


def _glaze(hexcol, name, scatter=0.08, strength=1.0, warm=False, **kw):
    p = Paint.from_glaze(hexcol, scatter=scatter, strength=strength, name=name, **kw)
    if warm:
        Rw = 0.9
        Rg = _keep_red(Rw * np.exp(-2 * 0.2 * p.K / strength))
        p.K = -np.log(Rg / Rw) / (2 * 0.2) * strength
    return p


def _white():
    # warm-ish titanium white: slight absorption in the violet
    R = np.array([0.80, 0.90, 0.935, 0.945, 0.95, 0.952, 0.953, 0.953, 0.952, 0.95, 0.948, 0.945])
    S = np.full(NB, 7.0) * (560.0 / BAND_CENTERS) ** 0.4
    return Paint(S * ks_from_reflectance(R), S, name='titanium white', staining=0.0)


TITANIUM_WHITE = _white()
ZINC_WHITE = TITANIUM_WHITE.copy(S=TITANIUM_WHITE.S * 0.4, K=TITANIUM_WHITE.K * 0.3, name='zinc white')
UNBLEACHED_TITANIUM = _opaque('#e6dcc6', 5.0, 'unbleached titanium')
IVORY_BLACK = Paint(np.full(NB, 9.0) * np.linspace(1.08, 0.94, NB), np.full(NB, 0.35), name='ivory black')
MARS_BLACK = Paint(np.full(NB, 10.0), np.full(NB, 0.8), name='mars black', granulation=0.3)
LAMP_BLACK = Paint(np.full(NB, 11.0) * np.linspace(0.92, 1.06, NB), np.full(NB, 0.25), name='lamp black', granulation=0.2)

CADMIUM_YELLOW_LIGHT = _opaque('#fbe600', 2.6, 'cadmium yellow light', warm=True)
CADMIUM_YELLOW = _opaque('#ffc300', 2.6, 'cadmium yellow', warm=True)
CADMIUM_ORANGE = _opaque('#ff7300', 2.6, 'cadmium orange', warm=True)
CADMIUM_RED_LIGHT = _opaque('#e8361e', 2.4, 'cadmium red light', warm=True)
CADMIUM_RED = _opaque('#c4162a', 2.4, 'cadmium red', warm=True)
NAPLES_YELLOW = _opaque('#f2d48a', 3.0, 'naples yellow', warm=True)
HANSA_YELLOW = _glaze('#f6e21c', 'hansa yellow', scatter=0.35, strength=1.1, staining=0.6, warm=True)
QUINACRIDONE_GOLD = _glaze('#d6901c', 'quinacridone gold', staining=0.7, warm=True)

YELLOW_OCHRE = _opaque('#c8962e', 1.6, 'yellow ochre', granulation=0.45, staining=0.2, warm=True)
RAW_SIENNA = _glaze('#c77a2c', 'raw sienna', scatter=0.5, granulation=0.5, staining=0.3, warm=True)
BURNT_SIENNA = _glaze('#b5532d', 'burnt sienna', scatter=0.45, granulation=0.4, staining=0.35, warm=True)
VENETIAN_RED = _opaque('#a2412a', 2.0, 'venetian red', granulation=0.3, warm=True)
RAW_UMBER = _opaque('#4f4232', 1.2, 'raw umber', granulation=0.6)
BURNT_UMBER = _opaque('#4a2c1f', 1.1, 'burnt umber', granulation=0.5)

ALIZARIN_CRIMSON = _absorber([5.5, 5.5, 6.5, 9.5, 13.0, 14.0, 9.5, 2.8, 0.6, 0.3, 0.25, 0.25], 'alizarin crimson', scatter=0.06, staining=0.8)
QUINACRIDONE_MAGENTA = _absorber([2.0, 2.2, 3.0, 6.0, 10.0, 13.0, 11.0, 4.0, 0.8, 0.3, 0.25, 0.25], 'quinacridone magenta', scatter=0.06, staining=0.8)
PERMANENT_ROSE = _absorber([3.6, 3.8, 4.5, 6.5, 9.0, 10.0, 6.0, 1.6, 0.4, 0.2, 0.2, 0.2], 'permanent rose', scatter=0.08, staining=0.7)

ULTRAMARINE = _absorber([0.45, 0.25, 0.4, 1.7, 4.2, 6.6, 8.2, 8.2, 6.8, 4.5, 2.8, 2.0], 'ultramarine blue', scatter=0.35, granulation=0.9, staining=0.15)
COBALT_BLUE = _opaque('#1f4fae', 1.0, 'cobalt blue', slope=0.5, granulation=0.7, staining=0.1)
CERULEAN_BLUE = _opaque('#2f86bf', 2.2, 'cerulean blue', granulation=0.75, staining=0.1)
PHTHALO_BLUE = _absorber([3.0, 1.2, 0.45, 0.55, 1.8, 6.0, 13.0, 19.0, 23.0, 25.0, 23.0, 19.0], 'phthalo blue', scatter=0.06, strength=1.4, staining=1.0)
PRUSSIAN_BLUE = _absorber([4.0, 3.0, 2.2, 2.0, 2.7, 5.0, 9.0, 13.0, 16.0, 18.0, 18.0, 17.0], 'prussian blue', scatter=0.05, strength=1.3, staining=0.9)
INDANTHRONE_BLUE = _absorber([2.2, 1.6, 1.4, 2.2, 4.0, 6.5, 8.5, 9.0, 8.0, 6.5, 5.0, 4.0], 'indanthrone blue', scatter=0.08, strength=1.3, staining=0.6)

PHTHALO_GREEN = _absorber([12.0, 9.0, 4.0, 1.4, 0.55, 1.2, 5.0, 12.0, 20.0, 24.0, 24.0, 22.0], 'phthalo green', scatter=0.06, strength=1.3, staining=1.0)
VIRIDIAN = _absorber([5.0, 4.0, 2.2, 1.0, 0.6, 1.0, 2.6, 5.0, 7.0, 8.0, 8.0, 7.5], 'viridian', scatter=0.3, granulation=0.5, staining=0.3)
CHROMIUM_OXIDE_GREEN = _opaque('#5e7a3c', 2.4, 'chromium oxide green', granulation=0.2)
SAP_GREEN = _absorber([9.0, 8.0, 6.5, 4.0, 1.6, 0.8, 1.0, 2.6, 4.5, 4.0, 2.5, 1.5], 'sap green', scatter=0.12, staining=0.6)
TERRE_VERTE = _opaque('#7d8a68', 0.6, 'terre verte', granulation=0.6)

DIOXAZINE_PURPLE = _absorber([1.2, 0.8, 1.6, 4.0, 10.0, 16.0, 18.0, 15.0, 8.0, 3.0, 1.2, 0.8], 'dioxazine purple', scatter=0.05, strength=1.4, staining=0.9)
COBALT_VIOLET = _opaque('#9a5fb5', 1.0, 'cobalt violet', granulation=0.8)

PIGMENTS = {
    'titanium_white': TITANIUM_WHITE, 'zinc_white': ZINC_WHITE,
    'unbleached_titanium': UNBLEACHED_TITANIUM,
    'ivory_black': IVORY_BLACK, 'mars_black': MARS_BLACK, 'lamp_black': LAMP_BLACK,
    'cadmium_yellow_light': CADMIUM_YELLOW_LIGHT, 'cadmium_yellow': CADMIUM_YELLOW,
    'cadmium_orange': CADMIUM_ORANGE, 'cadmium_red_light': CADMIUM_RED_LIGHT,
    'cadmium_red': CADMIUM_RED, 'naples_yellow': NAPLES_YELLOW,
    'hansa_yellow': HANSA_YELLOW, 'quinacridone_gold': QUINACRIDONE_GOLD,
    'yellow_ochre': YELLOW_OCHRE, 'raw_sienna': RAW_SIENNA, 'burnt_sienna': BURNT_SIENNA,
    'venetian_red': VENETIAN_RED, 'raw_umber': RAW_UMBER, 'burnt_umber': BURNT_UMBER,
    'alizarin_crimson': ALIZARIN_CRIMSON, 'quinacridone_magenta': QUINACRIDONE_MAGENTA,
    'permanent_rose': PERMANENT_ROSE,
    'ultramarine': ULTRAMARINE, 'cobalt_blue': COBALT_BLUE, 'cerulean_blue': CERULEAN_BLUE,
    'phthalo_blue': PHTHALO_BLUE, 'prussian_blue': PRUSSIAN_BLUE,
    'indanthrone_blue': INDANTHRONE_BLUE,
    'phthalo_green': PHTHALO_GREEN, 'viridian': VIRIDIAN,
    'chromium_oxide_green': CHROMIUM_OXIDE_GREEN, 'sap_green': SAP_GREEN,
    'terre_verte': TERRE_VERTE,
    'dioxazine_purple': DIOXAZINE_PURPLE, 'cobalt_violet': COBALT_VIOLET,
}

# Convenience mixtures painters buy in tubes
PAYNES_GREY = mix(ULTRAMARINE, 2, IVORY_BLACK, 1).copy(name="payne's grey")
INDIGO = mix(PRUSSIAN_BLUE, 2, IVORY_BLACK, 1, ALIZARIN_CRIMSON, 0.3).copy(name='indigo')
SEPIA = mix(BURNT_UMBER, 3, IVORY_BLACK, 1).copy(name='sepia')
HOOKERS_GREEN = mix(PHTHALO_GREEN, 1, HANSA_YELLOW, 1.2, BURNT_SIENNA, 0.4).copy(name="hooker's green")
PIGMENTS.update({'paynes_grey': PAYNES_GREY, 'indigo': INDIGO, 'sepia': SEPIA,
                 'hookers_green': HOOKERS_GREEN})


def get(name, medium='oil'):
    """Look up a library pigment by name, e.g. ``get('ultramarine', 'watercolor')``."""
    key = name.lower().replace(' ', '_').replace("'", '')
    p = PIGMENTS[key]
    return p.as_medium(medium) if medium != p.medium else p.copy()


def paint(color_or_name, medium='oil', opacity=0.7):
    """Get a library pigment by name or make a paint from a colour."""
    if isinstance(color_or_name, Paint):
        return color_or_name.as_medium(medium)
    if isinstance(color_or_name, str) and not color_or_name.startswith('#'):
        try:
            return get(color_or_name, medium)
        except KeyError:
            pass
    return Paint.from_color(color_or_name, opacity=opacity, medium=medium)
