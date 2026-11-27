"""
Numba kernels: bristle/paint exchange, Kubelka-Munk rendering, drying.

Conventions
-----------
* Pixel (y, x) has its centre at integer coordinates (x, y).
* ``film_K``/``film_S`` (H, W, NB) hold the *optical depth* of the wet paint
  film (K*thickness, S*thickness), i.e. extensive quantities that simply add
  when paints mix.  ``film_vol`` (H, W) is the physical paint volume.
* Per-pixel film properties ``wet``, ``gloss``, ``body``, ``drate`` are
  volume-weighted averages of what was deposited.
"""
from __future__ import annotations

import math
import numpy as np
from ._jit import njit, prange

from .spectral import km_over, NB


# --------------------------------------------------------------------------
# Small helpers usable from other kernels (tools, watercolour)
# --------------------------------------------------------------------------
@njit(cache=True, fastmath=True, inline='always')
def deposit_px(film_K, film_S, film_vol, wet, gloss, body, drate,
               y, x, dv, Kpv, Spv, p_gloss, p_body, p_drate):
    """Add ``dv`` volume of paint with per-volume K/S (``Kpv``, ``Spv``)."""
    if dv <= 0.0:
        return
    fv = film_vol[y, x]
    nv = fv + dv
    for b in range(film_K.shape[2]):
        film_K[y, x, b] += Kpv[b] * dv
        film_S[y, x, b] += Spv[b] * dv
    wet[y, x] = (wet[y, x] * fv + dv) / nv
    gloss[y, x] = (gloss[y, x] * fv + p_gloss * dv) / nv
    body[y, x] = (body[y, x] * fv + p_body * dv) / nv
    drate[y, x] = (drate[y, x] * fv + p_drate * dv) / nv
    film_vol[y, x] = nv


@njit(cache=True, fastmath=True, inline='always')
def move_film(film_K, film_S, film_vol, y0, x0, y1, x1, frac):
    """Move fraction ``frac`` of the wet film from pixel (y0,x0) to (y1,x1)."""
    if frac <= 0.0:
        return
    for b in range(film_K.shape[2]):
        dk = film_K[y0, x0, b] * frac
        ds = film_S[y0, x0, b] * frac
        film_K[y0, x0, b] -= dk
        film_S[y0, x0, b] -= ds
        film_K[y1, x1, b] += dk
        film_S[y1, x1, b] += ds
    dv = film_vol[y0, x0] * frac
    film_vol[y0, x0] -= dv
    film_vol[y1, x1] += dv


@njit(cache=True, fastmath=True, inline='always')
def _clamp01(v):
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


# --------------------------------------------------------------------------
# Paste media (oil, acrylic, gouache): bristle brush stroke
# --------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def paste_stroke(px, py, pp, pa, ds, speed,
                 boff, bthr, bfollow, brad, bsplay, bdep, tips,
                 res_K, res_S, res_vol, cap, tip_K, tip_S, tip_vol, alpha_tip,
                 size, sx0, sx1, sy0, sy1, sgamma, splay,
                 alpha, pickup, plough, contact_soft,
                 p_gloss, p_body, p_drate,
                 film_K, film_S, film_vol, wet, gloss, body, drate,
                 tooth, mask, tooth_fill,
                 base_K, base_S, base_vol, top_id, sid):
    H, W = film_vol.shape
    nb = film_K.shape[2]
    N = boff.shape[0]
    M = px.shape[0]
    bx0, by0, bx1, by1 = W, H, -1, -1
    tK = np.empty(nb)
    tS = np.empty(nb)
    spd_dep = 1.0 / math.sqrt(max(speed, 0.05))
    spd_depth = 1.0 / max(speed, 0.05) ** 0.3
    for j in range(M):
        cx = px[j]
        cy = py[j]
        p = pp[j]
        a = pa[j]
        ca = math.cos(a)
        sa = math.sin(a)
        # direction of motion
        j0 = j - 1 if j > 0 else 0
        j1 = j + 1 if j < M - 1 else M - 1
        mx = px[j1] - px[j0]
        my = py[j1] - py[j0]
        ml = math.sqrt(mx * mx + my * my)
        if ml > 1e-9:
            mx /= ml
            my /= ml
        else:
            mx, my = 1.0, 0.0
        nx = -my
        ny = mx
        pg = p ** sgamma if p > 0 else 0.0
        sx = size * (sx0 + (sx1 - sx0) * pg)
        sy = size * (sy0 + (sy1 - sy0) * pg)
        # footprint width across the motion direction (for pickup normalisation)
        across = 2.0 * math.sqrt((sx * (ca * nx + sa * ny)) ** 2 + (sy * (-sa * nx + ca * ny)) ** 2) + 1.0
        pick_base = 2.0 * pickup * across / N * ds
        dep_base = alpha * ds * spd_dep
        dep_tip = alpha_tip * ds
        mig = 0.25 * alpha_tip * ds
        for i in range(N):
            # loosely held (picked-up) paint slowly works into the belly
            if tip_vol[i] > 1e-9:
                for b in range(nb):
                    mk = tip_K[i, b] * mig
                    ms = tip_S[i, b] * mig
                    tip_K[i, b] -= mk
                    tip_S[i, b] -= ms
                    res_K[i, b] += mk
                    res_S[i, b] += ms
                mv = tip_vol[i] * mig
                tip_vol[i] -= mv
                res_vol[i] += mv
            spl = 1.0 + splay * bsplay[i] * p * p
            lx = boff[i, 0] * sx * spl
            ly = boff[i, 1] * sy * spl
            gx = cx + lx * ca - ly * sa
            gy = cy + lx * sa + ly * ca
            if j == 0 or tips[i, 0] != tips[i, 0]:
                tips[i, 0] = gx
                tips[i, 1] = gy
            else:
                f = bfollow[i]
                tips[i, 0] += (gx - tips[i, 0]) * f
                tips[i, 1] += (gy - tips[i, 1]) * f
            thr = bthr[i]
            if p <= thr:
                continue
            cp = _clamp01((p - thr) / (1.0 - thr + 1e-6))
            tx = tips[i, 0]
            ty = tips[i, 1]
            r = brad[i] * (0.55 + 0.45 * cp)
            r2 = r * r
            x0 = int(math.floor(tx - r))
            x1 = int(math.ceil(tx + r))
            y0 = int(math.floor(ty - r))
            y1 = int(math.ceil(ty + r))
            if x1 < 0 or y1 < 0 or x0 >= W or y0 >= H:
                continue
            if x0 < 0:
                x0 = 0
            if y0 < 0:
                y0 = 0
            if x1 > W - 1:
                x1 = W - 1
            if y1 > H - 1:
                y1 = H - 1
            wsum = 0.0
            for Y in range(y0, y1 + 1):
                dy = Y - ty
                for X in range(x0, x1 + 1):
                    dx = X - tx
                    d2 = dx * dx + dy * dy
                    if d2 < r2:
                        u = 1.0 - d2 / r2
                        wsum += u * u
            if wsum <= 1e-9:
                continue
            inv = 1.0 / wsum
            v = res_vol[i]
            load = _clamp01(v / (cap[i] + 1e-12))
            depth = 2.6 * (0.42 * cp + 0.58 * math.sqrt(load)) * spd_depth
            level = 1.0 - depth
            press = 0.3 + 0.7 * cp
            if x0 < bx0:
                bx0 = x0
            if y0 < by0:
                by0 = y0
            if x1 > bx1:
                bx1 = x1
            if y1 > by1:
                by1 = y1
            for Y in range(y0, y1 + 1):
                dy = Y - ty
                for X in range(x0, x1 + 1):
                    dx = X - tx
                    d2 = dx * dx + dy * dy
                    if d2 >= r2:
                        continue
                    u = 1.0 - d2 / r2
                    w = u * u * inv
                    m = 1.0 - mask[Y, X]
                    if m <= 0.0:
                        continue
                    # a new stroke lies on top of older wet paint: push the
                    # previous stroke's paint down into the base layer
                    if top_id[Y, X] != sid:
                        if film_vol[Y, X] > 0.0 or film_K[Y, X, 0] > 0.0:
                            for b in range(nb):
                                base_K[Y, X, b] += film_K[Y, X, b]
                                base_S[Y, X, b] += film_S[Y, X, b]
                                film_K[Y, X, b] = 0.0
                                film_S[Y, X, b] = 0.0
                            base_vol[Y, X] += film_vol[Y, X]
                            film_vol[Y, X] = 0.0
                        top_id[Y, X] = sid
                    fvt = film_vol[Y, X]
                    fvb = base_vol[Y, X]
                    fv = fvt + fvb
                    t = tooth[Y, X] * math.exp(-fv * tooth_fill)
                    contact = _clamp01((t - level) / contact_soft + 0.5)
                    wc = w * contact * m * press
                    if wc <= 0.0:
                        continue
                    tv = tip_vol[i]
                    dvb = v * dep_base * wc * bdep[i]
                    if dvb > 0.5 * v:
                        dvb = 0.5 * v
                    dvt = tv * dep_tip * wc
                    if dvt > 0.5 * tv:
                        dvt = 0.5 * tv
                    dv = dvb + dvt
                    pf = 0.0
                    if fv > 1e-7 and pickup > 0.0:
                        room = 1.0 - (v + tv) / (1.6 * cap[i] + 1e-12)
                        if room > 0.0:
                            pf = pick_base * wc * wet[Y, X] * room
                            if pf > 0.6:
                                pf = 0.6
                    pfb = pf * math.exp(-fvt * 2.5)
                    pv = fvt * pf + fvb * pfb
                    if dv <= 0.0 and pv <= 0.0:
                        continue
                    fdep = dvb / v if v > 1e-12 else 0.0
                    ftip = dvt / tv if tv > 1e-12 else 0.0
                    for b in range(nb):
                        dk = res_K[i, b] * fdep
                        dsg = res_S[i, b] * fdep
                        tk = tip_K[i, b] * ftip
                        tsg = tip_S[i, b] * ftip
                        qk = film_K[Y, X, b] * pf
                        qs = film_S[Y, X, b] * pf
                        bk = base_K[Y, X, b] * pfb
                        bs = base_S[Y, X, b] * pfb
                        film_K[Y, X, b] += dk + tk - qk
                        film_S[Y, X, b] += dsg + tsg - qs
                        base_K[Y, X, b] -= bk
                        base_S[Y, X, b] -= bs
                        res_K[i, b] -= dk
                        res_S[i, b] -= dsg
                        tip_K[i, b] += qk + bk - tk
                        tip_S[i, b] += qs + bs - tsg
                    tip_vol[i] = tv + pv - dvt
                    base_vol[Y, X] = fvb * (1.0 - pfb)
                    keep = fv - pv
                    nv = keep + dv
                    if dv > 0.0 and nv > 1e-12:
                        wet[Y, X] = (wet[Y, X] * keep + dv) / nv
                        gloss[Y, X] = (gloss[Y, X] * keep + p_gloss * dv) / nv
                        body[Y, X] = (body[Y, X] * keep + p_body * dv) / nv
                        drate[Y, X] = (drate[Y, X] * keep + p_drate * dv) / nv
                    film_vol[Y, X] = fvt * (1.0 - pf) + dv
                    v = v - dvb
                    res_vol[i] = v
                    # ploughing: stiff bristles push wet paint aside -> ridges
                    if plough > 0.0 and nv > 0.05:
                        lat = dx * nx + dy * ny
                        side = 1.0 if lat >= 0 else -1.0
                        X2 = int(round(tx + nx * side * (r + 0.9)))
                        Y2 = int(round(ty + ny * side * (r + 0.9)))
                        if 0 <= X2 < W and 0 <= Y2 < H and (X2 != X or Y2 != Y) and mask[Y2, X2] < 0.999:
                            fr = plough * contact * u * 0.25 * wet[Y, X] * (1.0 - mask[Y2, X2])
                            if fr > 0.5:
                                fr = 0.5
                            move_film(film_K, film_S, film_vol, Y, X, Y2, X2, fr)
                            if wet[Y2, X2] < wet[Y, X]:
                                wet[Y2, X2] = wet[Y, X]
                                gloss[Y2, X2] = gloss[Y, X]
                                body[Y2, X2] = body[Y, X]
                                drate[Y2, X2] = drate[Y, X]
    return bx0, by0, bx1, by1


# --------------------------------------------------------------------------
# Fluid media (watercolour, ink): bristle brush exchanging water + pigment
# --------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def fluid_stroke(px, py, pp, pa, ds, speed,
                 boff, bthr, bfollow, brad, bsplay, bdep, tips,
                 res_K, res_S, res_vol, res_water, wcap,
                 size, sx0, sx1, sy0, sy1, sgamma, splay,
                 alpha, lift, contact_soft, gran, stain,
                 wc_water, wc_K, wc_S, wc_m, wc_g, wc_st,
                 tooth, mask, water_ref):
    H, W = wc_water.shape
    nb = wc_K.shape[2]
    N = boff.shape[0]
    M = px.shape[0]
    bx0, by0, bx1, by1 = W, H, -1, -1
    spd_dep = 1.0 / math.sqrt(max(speed, 0.05))
    spd_depth = 1.0 / max(speed, 0.05) ** 0.3
    for j in range(M):
        cx = px[j]
        cy = py[j]
        p = pp[j]
        a = pa[j]
        ca = math.cos(a)
        sa = math.sin(a)
        j0 = j - 1 if j > 0 else 0
        j1 = j + 1 if j < M - 1 else M - 1
        mx = px[j1] - px[j0]
        my = py[j1] - py[j0]
        ml = math.sqrt(mx * mx + my * my)
        if ml > 1e-9:
            mx /= ml
            my /= ml
        else:
            mx, my = 1.0, 0.0
        nx = -my
        ny = mx
        pg = p ** sgamma if p > 0 else 0.0
        sx = size * (sx0 + (sx1 - sx0) * pg)
        sy = size * (sy0 + (sy1 - sy0) * pg)
        across = 2.0 * math.sqrt((sx * (ca * nx + sa * ny)) ** 2 + (sy * (-sa * nx + ca * ny)) ** 2) + 1.0
        lift_base = lift * across / N * ds
        dep_base = alpha * ds * spd_dep
        for i in range(N):
            spl = 1.0 + splay * bsplay[i] * p * p
            lx = boff[i, 0] * sx * spl
            ly = boff[i, 1] * sy * spl
            gx = cx + lx * ca - ly * sa
            gy = cy + lx * sa + ly * ca
            if j == 0 or tips[i, 0] != tips[i, 0]:
                tips[i, 0] = gx
                tips[i, 1] = gy
            else:
                f = bfollow[i]
                tips[i, 0] += (gx - tips[i, 0]) * f
                tips[i, 1] += (gy - tips[i, 1]) * f
            thr = bthr[i]
            if p <= thr:
                continue
            cp = _clamp01((p - thr) / (1.0 - thr + 1e-6))
            tx = tips[i, 0]
            ty = tips[i, 1]
            r = brad[i] * (0.55 + 0.45 * cp)
            r2 = r * r
            x0 = int(math.floor(tx - r))
            x1 = int(math.ceil(tx + r))
            y0 = int(math.floor(ty - r))
            y1 = int(math.ceil(ty + r))
            if x1 < 0 or y1 < 0 or x0 >= W or y0 >= H:
                continue
            x0 = max(x0, 0)
            y0 = max(y0, 0)
            x1 = min(x1, W - 1)
            y1 = min(y1, H - 1)
            wsum = 0.0
            for Y in range(y0, y1 + 1):
                dy = Y - ty
                for X in range(x0, x1 + 1):
                    dx = X - tx
                    d2 = dx * dx + dy * dy
                    if d2 < r2:
                        u = 1.0 - d2 / r2
                        wsum += u * u
            if wsum <= 1e-9:
                continue
            inv = 1.0 / wsum
            if x0 < bx0:
                bx0 = x0
            if y0 < by0:
                by0 = y0
            if x1 > bx1:
                bx1 = x1
            if y1 > by1:
                by1 = y1
            wv = res_water[i]
            sat_b = wv / (wcap[i] + 1e-12)
            depth = 2.6 * (0.35 * cp + 0.65 * math.sqrt(_clamp01(sat_b))) * spd_depth
            level = 1.0 - depth
            press = 0.35 + 0.65 * cp
            for Y in range(y0, y1 + 1):
                dy = Y - ty
                for X in range(x0, x1 + 1):
                    dx = X - tx
                    d2 = dx * dx + dy * dy
                    if d2 >= r2:
                        continue
                    u = 1.0 - d2 / r2
                    w = u * u * inv
                    m = 1.0 - mask[Y, X]
                    if m <= 0.0:
                        continue
                    cw = wc_water[Y, X]
                    t = tooth[Y, X] * math.exp(-cw * 2.0)
                    contact = _clamp01((t - level) / contact_soft + 0.5)
                    wc = w * contact * m * press
                    if wc <= 0.0:
                        continue
                    sat_c = cw / water_ref
                    diff = sat_b - 0.8 * sat_c
                    if diff > 0.0 and wv > 1e-9:
                        # brush -> paper: water carries pigment along
                        f = dep_base * wc * (0.4 + diff) * bdep[i]
                        if f > 0.5:
                            f = 0.5
                        dw = wv * f
                        dm = res_vol[i] * f
                        for b in range(nb):
                            dk = res_K[i, b] * f
                            dsg = res_S[i, b] * f
                            wc_K[Y, X, b] += dk
                            wc_S[Y, X, b] += dsg
                            res_K[i, b] -= dk
                            res_S[i, b] -= dsg
                        wc_water[Y, X] = cw + dw
                        wc_m[Y, X] += dm
                        wc_g[Y, X] += dm * gran
                        wc_st[Y, X] += dm * stain
                        res_vol[i] -= dm
                        wv -= dw
                        res_water[i] = wv
                    elif diff < 0.0 and lift > 0.0:
                        # thirsty brush lifts water and suspended pigment
                        room = 1.0 - wv / (1.3 * wcap[i] + 1e-12)
                        if room > 0.0:
                            f = lift_base * wc * (-diff) * room
                            if f > 0.5:
                                f = 0.5
                            dw = cw * f
                            mm = wc_m[Y, X]
                            for b in range(nb):
                                dk = wc_K[Y, X, b] * f
                                dsg = wc_S[Y, X, b] * f
                                wc_K[Y, X, b] -= dk
                                wc_S[Y, X, b] -= dsg
                                res_K[i, b] += dk
                                res_S[i, b] += dsg
                            wc_water[Y, X] = cw - dw
                            wc_m[Y, X] = mm * (1 - f)
                            wc_g[Y, X] *= (1 - f)
                            wc_st[Y, X] *= (1 - f)
                            res_vol[i] += mm * f
                            wv += dw
                            res_water[i] = wv
                    sat_b = wv / (wcap[i] + 1e-12)
    return bx0, by0, bx1, by1


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
@njit(parallel=True, cache=True, fastmath=True)
def render_reflectance(below_R, base_K, base_S, film_K, film_S, has_wc, wc_K, wc_S, wc_dK, wc_dS,
                       wc_water, wc_damp, M, out):
    """Composite all layers with Kubelka-Munk and project to linear sRGB."""
    H, W, nb = below_R.shape
    for y in prange(H):
        for x in range(W):
            r = 0.0
            g = 0.0
            bl = 0.0
            wetd = 0.0
            if has_wc:
                wetd = wc_water[y, x] + 0.5 * wc_damp[y, x]
                if wetd > 1.0:
                    wetd = 1.0
            for k in range(nb):
                R = below_R[y, x, k]
                if has_wc:
                    dk = wc_dK[y, x, k]
                    dsg = wc_dS[y, x, k]
                    if dk + dsg > 1e-6:
                        R = km_over(dk, dsg, R)
                bk = base_K[y, x, k]
                bsg = base_S[y, x, k]
                if bk + bsg > 1e-6:
                    R = km_over(bk, bsg, R)
                K = film_K[y, x, k]
                S = film_S[y, x, k]
                if has_wc:
                    K += wc_K[y, x, k]
                    S += wc_S[y, x, k]
                if K + S > 1e-6:
                    R = km_over(K, S, R)
                if wetd > 0.0:
                    R = R ** (1.0 + 0.45 * wetd) * (1.0 - 0.05 * wetd)
                r += M[0, k] * R
                g += M[1, k] * R
                bl += M[2, k] * R
            out[y, x, 0] = r
            out[y, x, 1] = g
            out[y, x, 2] = bl


@njit(parallel=True, cache=True, fastmath=True)
def bake_film(sel, below_R, film_K, film_S, film_vol, wet, gloss, body, drate,
              h_dry, gloss_dry, gloss_dry_factor, base_K, base_S, base_vol):
    """Dry the wet film where ``sel`` is True: fold it into below_R/h_dry."""
    H, W, nb = below_R.shape
    for y in prange(H):
        for x in range(W):
            if not sel[y, x]:
                continue
            btot = 0.0
            for k in range(nb):
                btot += base_K[y, x, k] + base_S[y, x, k]
            if btot > 1e-7:
                for k in range(nb):
                    below_R[y, x, k] = km_over(base_K[y, x, k], base_S[y, x, k], below_R[y, x, k])
            tot = 0.0
            for k in range(nb):
                tot += film_K[y, x, k] + film_S[y, x, k]
            fv = film_vol[y, x] + base_vol[y, x]
            if tot > 1e-7:
                for k in range(nb):
                    below_R[y, x, k] = km_over(film_K[y, x, k], film_S[y, x, k], below_R[y, x, k])
            if fv > 1e-7:
                hd = fv * body[y, x]
                old = h_dry[y, x]
                h_dry[y, x] = old + hd
                # dried gloss: thick fresh layers dominate
                wgt = fv / (fv + 0.3)
                gloss_dry[y, x] = gloss_dry[y, x] * (1 - wgt) + gloss[y, x] * gloss_dry_factor * wgt
            for k in range(nb):
                film_K[y, x, k] = 0.0
                film_S[y, x, k] = 0.0
                base_K[y, x, k] = 0.0
                base_S[y, x, k] = 0.0
            film_vol[y, x] = 0.0
            base_vol[y, x] = 0.0
            wet[y, x] = 0.0
            gloss[y, x] = 0.0
            body[y, x] = 0.0
            drate[y, x] = 0.0


@njit(parallel=True, cache=True, fastmath=True)
def bake_deposit(sel, below_R, dK, dS):
    """Fold deposited watercolour pigment into below_R where ``sel``."""
    H, W, nb = below_R.shape
    for y in prange(H):
        for x in range(W):
            if not sel[y, x]:
                continue
            for k in range(nb):
                if dK[y, x, k] + dS[y, x, k] > 1e-7:
                    below_R[y, x, k] = km_over(dK[y, x, k], dS[y, x, k], below_R[y, x, k])
                dK[y, x, k] = 0.0
                dS[y, x, k] = 0.0
