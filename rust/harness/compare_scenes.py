"""Compare two scenes_<backend>.json files from scenes.py.

  python rust/harness/compare_scenes.py results/scenes_numba.json results/scenes_rust.json

Per scene: covered-area ratio (B / A), max |difference| of the mean linear RGB,
and max |difference| of the luminance profile. A scene is identical when all
three are exactly equal. The last line counts identical and differing scenes.
"""
from __future__ import annotations

import json
import sys


def load(path):
    with open(path) as f:
        return json.load(f)


def area_ratio(a, b):
    if a == 0:
        return 1.0 if b == 0 else float('inf')
    return b / a


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    da, db = load(sys.argv[1]), load(sys.argv[2])
    print(f"A = {da['backend']} ({sys.argv[1]})")
    print(f"B = {db['backend']} ({sys.argv[2]})")
    sa, sb = da['scenes'], db['scenes']
    missing = sorted(set(sa) ^ set(sb))
    if missing:
        print(f"only in one file (not compared): {', '.join(missing)}")
    header = f"{'scene':18s} {'area A':>7s} {'area B':>7s} {'area B/A':>9s} {'max dRGB':>10s} {'max dProf':>10s}  status"
    print(header)
    identical = differ = 0
    for name in sorted(set(sa) & set(sb)):
        a, b = sa[name], sb[name]
        ratio = area_ratio(a['covered_px'], b['covered_px'])
        d_rgb = max(abs(x - y) for x, y in zip(a['mean_linear_rgb'], b['mean_linear_rgb']))
        d_prof = max(abs(x - y) for x, y in zip(a['profile_luma'], b['profile_luma']))
        same = a['covered_px'] == b['covered_px'] and d_rgb == 0 and d_prof == 0
        identical += same
        differ += not same
        print(f"{name:18s} {a['covered_px']:7d} {b['covered_px']:7d} {ratio:9.4f} "
              f"{d_rgb:10.3g} {d_prof:10.3g}  {'identical' if same else 'DIFFERS'}")
    print(f"identical: {identical}, differ: {differ}")


if __name__ == '__main__':
    main()
