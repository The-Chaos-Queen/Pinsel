"""First stroke at which two runs' canvases differ (Derek, after stage 0).

A chaotic engine stays bit-identical until the first threshold flips; a
kernel bug differs from the first stroke, or from the first use of a tool.

  python rust/harness/first_divergence.py REF_TRACE OTHER_TRACE [...]

Traces come from PINSEL_TRACE=<prefix> (one line per call:
`index region_hash full_hash_or_-`).
"""
import sys


def load(p):
    rows = []
    with open(p, encoding='utf-8') as f:
        for line in f:
            i, reg, full = line.split()
            rows.append((int(i), reg, full))
    return rows


def first_divergence(a, b):
    n = min(len(a), len(b))
    first_reg = next((a[k][0] for k in range(n) if a[k][1] != b[k][1]), None)
    first_full = next((a[k][0] for k in range(n) if a[k][2] != b[k][2]), None)
    differing = sum(a[k][1] != b[k][1] for k in range(n))
    return {'calls': (len(a), len(b)), 'first_region_diff': first_reg,
            'first_full_canvas_diff_checkpoint': first_full, 'calls_with_region_diff': differing}


if __name__ == '__main__':
    ref = load(sys.argv[1])
    for other in sys.argv[2:]:
        r = first_divergence(ref, load(other))
        print(f'{other}: first differing stroke {r["first_region_diff"]}, '
              f'first differing full-canvas checkpoint {r["first_full_canvas_diff_checkpoint"]}, '
              f'strokes with any difference {r["calls_with_region_diff"]} of {r["calls"][0]}')
