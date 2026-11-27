"""Seed spread: one mark per seed (0..19) for three random-looking tools.

  PINSEL_BACKEND=numba|rust venv/bin/python rust/harness/seed_spread.py

The canvas keeps its fixed seed (scenes.CANVAS_SEED), so only the tool's seed
changes. Per tool: covered pixels and mean linear RGB for each seed, plus mean
and sample standard deviation (ddof=1) over the 20 seeds. Writes
rust/harness/results/seed_spread_<backend>.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import scenes as S  # noqa: E402  (also puts the repo root on sys.path)
from pinsel import _backend, path as P, pigments as pg, tools as T  # noqa: E402

SEEDS = list(range(20))
LINE = P.line((60, 128), (196, 128))


def splatter_flick(cv, s):
    sp = T.Splatter(seed=s).load(pg.CADMIUM_RED.thinned(0.5))
    sp.flick(cv, (40, 128), direction=0.0, spread=25.0, distance=(10, 200), seed=s)


def airbrush_spray(cv, s):
    ab = T.Airbrush(seed=s).load(pg.CERULEAN_BLUE)
    ab.spray(cv, LINE, seed=s)


def sponge_dab(cv, s):
    sp = T.Sponge(size=30, seed=s).load(pg.YELLOW_OCHRE)
    sp.dab(cv, 128, 128, pressure=0.6, seed=s)


TOOLS = {'splatter_flick': splatter_flick, 'airbrush_spray': airbrush_spray, 'sponge_dab': sponge_dab}


def main():
    out = {'backend': _backend.NAME, 'canvas_seed': S.CANVAS_SEED, 'seeds': SEEDS, 'tools': {}}
    for name, fn in TOOLS.items():
        rows = []
        for s in SEEDS:
            cv = S.new_canvas('canvas')
            fn(cv, s)
            S.finish(cv)
            m = S.measure(cv.render_linear(), S.blank_render('canvas'), 'y')
            rows.append({'seed': s, 'covered_px': m['covered_px'], 'mean_linear_rgb': m['mean_linear_rgb']})
        px = np.array([r['covered_px'] for r in rows], dtype=np.float64)
        rgb = np.array([r['mean_linear_rgb'] for r in rows], dtype=np.float64)
        out['tools'][name] = {
            'per_seed': rows,
            'covered_px_mean': float(px.mean()), 'covered_px_std': float(px.std(ddof=1)),
            'mean_rgb_mean': rgb.mean(axis=0).tolist(), 'mean_rgb_std': rgb.std(axis=0, ddof=1).tolist(),
        }
        t = out['tools'][name]
        print(f"{name}: covered px {t['covered_px_mean']:.1f} +- {t['covered_px_std']:.1f} "
              f"(min {int(px.min())}, max {int(px.max())})")
        print(f"  mean RGB {[round(v, 5) for v in t['mean_rgb_mean']]} "
              f"+- {[round(v, 6) for v in t['mean_rgb_std']]}")
    S.RES.mkdir(parents=True, exist_ok=True)
    path = S.RES / f'seed_spread_{_backend.NAME}.json'
    path.write_text(json.dumps(out, indent=1))
    print(f"-> {path}")


if __name__ == '__main__':
    main()
