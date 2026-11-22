"""Emberwing. A mixed-media moth study, drawn in three passes on one sheet."""
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pinsel as ps
from pinsel import path as P, watercolor as wc


@lru_cache(None)
def color(hexcode, medium='oil'):
    return ps.Paint.from_color(hexcode, opacity=.9, medium=medium)


def paint():
    cv = ps.Canvas(740, 486, ground='cold_press', ground_color='#efdfbd',
                   texture_scale=.85, seed=32)
    rng = np.random.default_rng(721)
    y, x = np.mgrid[:cv.H, :cv.W]
    angle = np.radians(-10)
    rotation = np.array([[np.cos(angle), np.sin(angle)],
                         [-np.sin(angle), np.cos(angle)]])

    def place(points, side=1):
        pts = np.asarray(points, float).copy()
        pts[:, 0] *= side
        if side == -1:
            pts[:, 1] += .00018 * pts[:, 0] ** 2
        return pts @ rotation + [369, 243]

    def curve(points, side=1, closed=False):
        return P.spline(place(points, side), closed=closed)

    def rub(path, tint, size, pressure, side=.16):
        stick = ps.OilCrayon(size, side=side).load(color(tint))
        cv.stroke(stick, path, pressure=pressure)

    def wash(area, tint, strength, water=.46):
        wc.wash(cv, area, color(tint, 'watercolor'), concentration=strength,
                water=water, rough=.25, variation=.22, seed=int(rng.integers(99999)))

    fore = [(6,-32),(45,-59),(117,-89),(199,-112),(253,-117),
            (265,-104),(252,-77),(222,-50),(177,-23),(114,10),
            (63,27),(23,28),(10,13)]
    hind = [(12,-13),(108,-29),(167,-12),(185,19),(181,58),
            (158,91),(127,108),(94,119),(63,111),(36,86),(21,55)]
    body = [(0,-49),(9,-47),(12,-39),(18,-32),(18,-14),(12,5),
            (11,35),(8,63),(2,81),(-4,75),(-9,54),(-12,16),
            (-13,2),(-18,-15),(-16,-33),(-10,-40),(-8,-47)]
    fore_masks = {s: wc.polygon(curve(fore,s,True))(cv) for s in (-1,1)}
    fore_area = np.maximum(*fore_masks.values())
    hind_area = np.maximum(*[wc.polygon(curve(hind,s,True))(cv) for s in (-1,1)])
    body_area = wc.polygon(curve(body,closed=True))(cv)
    silhouette = np.maximum.reduce([fore_area,hind_area,body_area])

    # Lay the pale, branching ribbons down first. The stencil keeps the
    # crayon on the wing while pressure controls what the wash can enter.
    bands = [
        ([(23,-52),(35,-30),(39,-17),(54,-7),(63,11),(69,30)], 7),
        ([(65,-73),(76,-54),(94,-40),(91,-23),(102,-4),(118,12)], 14),
        ([(94,-40),(106,-52),(119,-54),(127,-66),(124,-94)], 10),
        ([(90,-22),(73,-18),(60,-7),(46,0),(30,-1)], 7),
        ([(144,-102),(141,-78),(147,-63),(165,-52),(169,-40),(158,-12)], 13),
        ([(148,-63),(162,-70),(179,-68),(196,-80),(202,-96),(198,-118)], 9),
        ([(196,-80),(206,-65),(221,-59),(232,-42)], 11),
        ([(219,-116),(224,-103),(244,-95),(260,-95)], 9),
        ([(106,-52),(100,-62),(94,-68)], 3.8),
    ]
    for s in (-1,1):
        cv.set_mask(1-fore_masks[s])
        for points, size in bands:
            pts = np.asarray(points,float) + rng.normal(0,1.8,(len(points),2))
            path = P.wobble(curve(pts,s), amount=1.6, scale=9, seed=rng)
            rub(path, '#f7e5b8', size, P.swell(.65,.99,.48,.44))
        for points in [[(26,-49),(107,-84),(183,-106),(231,-114)],
                       [(252,-77),(222,-52),(172,-23),(114,7)]]:
            rub(curve(points,s), '#ead6aa', 3.4, P.swell(.3,.78))
    cv.clear_mask()
    cv.dry()
    steps = [cv.to_image(relief=.5)]

    # A dilute grey-green tide around the specimen, pooled most deeply below it.
    tide = np.exp(-((x-389)/284)**2-((y-267)/134)**2) * (1-silhouette)
    wash(tide, '#929a83', .68, .30)
    wash(hind_area*(1-fore_area)*(1-body_area), '#e27448', 4.0)
    wash(hind_area*(1-fore_area)*(1-body_area)*np.clip((y-211)/175,0,1),
         '#b74935', 2.7)
    wash(fore_area*(1-body_area), '#59392f', 4.2)
    shade = .5+.5*np.sin(x/47 + .7*np.cos(y/24))
    wash(fore_area*shade*(1-body_area), '#362f36', 2.1)
    wash(body_area, '#533c39', 4.6)
    spots = np.zeros(cv.shape,np.float32)
    for s in (-1,1):
        for cx,cy,rx,ry,tilt in [(51,52,6,9,-24),(75,72,11,15,-27),
                                 (106,94,9,6,15),(127,61,13,17,23),
                                 (158,39,8,12,38),(94,35,5,6,0)]:
            cx,cy = place([(cx+rng.uniform(-2,2),cy)],s)[0]
            spot = wc.ellipse(cx,cy,rx,ry,angle=s*tilt-10)(cv)
            np.maximum(spots,spot,out=spots)
    wash(spots*hind_area*(1-fore_area), '#30343c', 5.2)
    cv.wait(1.1)
    cv.dry()
    steps.append(cv.to_image(relief=.5))

    # Dip-pen marks are laid after the sheet is dry. Leave gaps in the
    # contours, so the eye can still follow the softer watercolor edges.
    @lru_cache(None)
    def nib(tint,width):
        return ps.tools.Pen(width=width,feather=.08,pooling=.15,seed=26).load(color(tint,'ink'))

    def line(points,tint,width,side=1):
        nib(tint,width).line(cv,curve(points,side))

    for s in (-1,1):
        line(fore[1:6], '#5b4a3f',1.5,s)
        line(fore[6:11], '#694d3a',1.1,s)
        # Short ranks of copper scales sit inside the dark pools. Vary their
        # density from patch to patch and leave the cream ribbons clear.
        for cx,cy in [(47,-33),(77,-39),(112,-38),(137,-66),(168,-46),
                      (184,-88),(218,-77),(231,-105)]:
            for _ in range(26):
                u,v = rng.normal(cx,7),rng.normal(cy,4)
                start = place([(u,v)],s)[0]
                ix,iy = np.rint(start).astype(int)
                if fore_masks[s][iy,ix] < .98 or cv.wax[iy,ix] > .12:
                    continue
                nib(rng.choice(['#896449','#79513d','#966e51']),.95).line(
                    cv,place([(u,v),(u+rng.uniform(1.5,4.2),v-1)],s))
        for end in [(177,45),(161,75),(130,101),(94,115),(64,107)]:
            ex,ey = end
            line([(.37*ex,.37*ey+8),(.69*ex,.69*ey),(ex,ey)], '#ab6949',1.1,s)
        # Thin fringe, drawn as individual hairs with uneven spacing.
        edge = curve(fore[5:12],s)
        for idx in np.linspace(1,len(edge)-2,72,dtype=int):
            p = edge[idx]
            q = p + [s*rng.uniform(.7,3),rng.uniform(1.2,4)]
            nib('#8c7154',.9).line(cv,[p,q])
        # Feathered antennae: paired barbs shorten toward the tips.
        stem = curve([(5,-46),(17,-67),(48,-89),(60,-114)],s)
        nib('#463d35',1.9).line(cv,stem)
        for f in np.linspace(.11,.92,20):
            i = min(len(stem)-2,int(f*(len(stem)-1)))
            t = stem[i+1]-stem[i]
            t /= np.linalg.norm(t)
            normal = np.array([-t[1],t[0]])
            for direction in (-1,1):
                tip = stem[i] + normal*direction*(1+6*np.sin(np.pi*f)) - t*5
                nib('#67533f',1.0).line(cv,[tip,stem[i]])
        for row in range(6):
            yy = 5 + 10*row
            rub(curve([(1,yy),(6,yy-1),(10-row*.6,yy-4)],s),
                '#c77b4a',2.6,P.swell(.22,.85))
        for _ in range(110):
            ux,uy = rng.normal(7,4),rng.uniform(-37,3)
            path = curve([(ux,uy),(ux+2,uy+2),(ux+rng.uniform(3,7),uy+rng.uniform(4,10))],s)
            rub(path,rng.choice(['#b77d52','#d6a873','#776252']),1.15,rng.uniform(.45,.78))
        rub(curve([(6,-42),(8,-43),(9,-44)],s),'#cd9b70',2.2,.79)
    final = cv.to_image(relief=.5)
    steps.append(final)
    strip = Image.new('RGB',(3*370+12,243),'#f1e7d2')
    for i,step in enumerate(steps):
        strip.paste(step.resize((370,243),Image.Resampling.LANCZOS),(i*376,0))
    return final,strip


if __name__ == '__main__':
    output = ROOT/'output'
    output.mkdir(exist_ok=True)
    final,steps = paint()
    final.save(output/'tiger_moth.png')
    steps.save(output/'tiger_moth_process.png')
