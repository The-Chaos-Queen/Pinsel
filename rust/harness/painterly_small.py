"""Reduced painterly run for PINSEL_AB=1 (every call of every ported kernel
compared on identical inputs): 240 x 180 canvas, two styles, so the A/B copies
of the canvas stay cheap. Uses the helpers of examples/painterly_demo.py.

    PINSEL_AB=1 venv/bin/python rust/harness/painterly_small.py 2>&1 | grep "A/B"
    bash rust/harness/verify.sh $PWD/rust/harness/painterly_small.py
"""
import importlib.util
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np
from PIL import Image

import pinsel as ps
from pinsel.painterly import PaletteSolver, DEFAULT_PALETTE

spec = importlib.util.spec_from_file_location('painterly_demo', os.path.join(os.getcwd(), 'examples', 'painterly_demo.py'))
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)

OUT = os.path.join(os.getcwd(), 'output')
os.makedirs(OUT, exist_ok=True)
W, H = 240, 180
ref, masks = demo.landscape(W, H, seed=3, return_masks=True)
ground = ps.Canvas(8, 8, ground='canvas', seed=7).ground.reflectance.astype(np.float64)
solver = PaletteSolver(DEFAULT_PALETTE, max_pigments=3, thickness=3.0, ground=ground, cache_step=1.0)
for style in ('impressionist', 'expressionist'):
    cv, res = demo.paint_style(ref, masks, style, W, H, solver=solver)
    print(style, res)
    cv.save(os.path.join(OUT, f'painterly_small_{style}.png'), **demo.RENDER)
