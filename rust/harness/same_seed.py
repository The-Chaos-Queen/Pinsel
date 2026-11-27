"""Whole-image same-seed test for stage A (Derek's review, point 2).

While the Python glue and its RNG are unchanged, an example must paint the
same picture with the numba and the Rust kernels. This runs one example
under each backend, collects the PNGs it writes, and compares them:
fraction of identical pixels, fraction with CIEDE2000 > 1, p99 and max dE.

  python rust/harness/same_seed.py examples/quickstart.py [more examples ...]

Run from the repository root, inside an environment with pinsel (editable),
numba and pinsel_core installed.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output'
RES = ROOT / 'rust' / 'harness' / 'results'


def srgb_to_lab(img):
    c = np.asarray(img, dtype=np.float64)[..., :3] / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    e = (6 / 29) ** 3
    f = np.where(xyz > e, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def run(example, backend):
    OUT.mkdir(exist_ok=True)
    before = {p: p.stat().st_mtime_ns for p in OUT.glob('*.png')}
    env = dict(os.environ, PINSEL_BACKEND=backend)
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, str(ROOT / example)], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-4000:], sep='\n')
        raise SystemExit(f'{example} failed under {backend}')
    new = [p for p in OUT.glob('*.png') if before.get(p) != p.stat().st_mtime_ns]
    dst = RES / Path(example).stem / backend
    dst.mkdir(parents=True, exist_ok=True)
    for p in new:
        shutil.copy2(p, dst / p.name)
    return dt, sorted(p.name for p in new)


def compare(a_path, b_path):
    from pinsel.painterly import delta_e2000
    a = np.asarray(Image.open(a_path).convert('RGB'))
    b = np.asarray(Image.open(b_path).convert('RGB'))
    if a.shape != b.shape:
        return {'error': f'shape {a.shape} vs {b.shape}'}
    same = np.all(a == b, axis=-1)
    de = delta_e2000(srgb_to_lab(a), srgb_to_lab(b))
    return {'pixels': int(same.size), 'identical': float(same.mean()),
            'frac_dE_gt_1': float((de > 1.0).mean()), 'p99_dE': float(np.percentile(de, 99)),
            'max_dE': float(de.max())}


def main(examples):
    report = {}
    for ex in examples:
        t_nb, files = run(ex, 'numba')
        t_rs, files_rs = run(ex, 'rust')
        name = Path(ex).stem
        rows = {f: compare(RES / name / 'numba' / f, RES / name / 'rust' / f) for f in files if f in files_rs}
        report[name] = {'seconds_numba': round(t_nb, 2), 'seconds_rust': round(t_rs, 2), 'images': rows}
        print(f'== {name}: numba {t_nb:.1f} s, rust {t_rs:.1f} s')
        for f, r in rows.items():
            if 'error' in r:
                print(f'   {f}: {r["error"]}')
            else:
                print(f'   {f}: identical {r["identical"]:.4%}, dE>1 {r["frac_dE_gt_1"]:.4%}, '
                      f'p99 dE {r["p99_dE"]:.3f}, max dE {r["max_dE"]:.2f}')
    RES.mkdir(parents=True, exist_ok=True)
    (RES / 'same_seed.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:] or ['examples/quickstart.py'])
