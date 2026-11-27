"""
pinsel -- a physically inspired painting engine.

    import pinsel as ps
    from pinsel import pigments as pg, path as P

    cv = ps.Canvas(800, 600, ground='canvas')
    brush = ps.Brush.round(12).load(pg.ULTRAMARINE)
    cv.stroke(brush, P.spline([(100, 300), (400, 200), (700, 350)]))
    cv.save('out.png')

Modules: canvas, brush, pigments, path, spectral, grounds, noise,
tools (knife, airbrush, splatter, sponge, pens, stencils),
watercolor (fluid simulation), painterly (auto-painter, palette solver).
"""
from .canvas import Canvas
from .brush import Brush, SHAPES
from .pigments import Paint, mix, paint, get, PIGMENTS
from . import pigments, path, spectral, noise, grounds
from . import tools, watercolor, painterly, pastel, monotype
from .tools import OilCrayon
from .pastel import SoftPastel, Charcoal, KneadedEraser
from .monotype import MonotypePlate

__version__ = '0.2.0'
