"""It Was There a Second Ago. Two small additions to the same marbled sheet."""
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from untitled_marbling import ROOT, make_sheet
import pinsel as ps
from pinsel import path as P, watercolor as wc


LIGHT = dict(relief=.48, specular=.02)


def draw_readings(out):
    cv = make_sheet()
    base = cv.to_image(**LIGHT)
    with Image.open(out/'untitled_marbling.png') as original:
        np.testing.assert_array_equal(np.asarray(base), np.asarray(original))
    master = cv.get_state()

    # One dark eye. A pinprick of the original uninked paper shows through it.
    y,x = np.mgrid[:cv.H,:cv.W]
    eye = wc.ellipse(99,192,5.8,4.85,angle=-20)(cv)
    eye *= 1-wc.ellipse(97.4,190.5,1.3,1.1)(cv)
    ps.tools.glaze(cv,ps.Paint.from_color('#182029'),thickness=1.05,region=eye,
                   feather=.20,variation=0)
    tadpole = cv.to_image(**LIGHT)
    tadpole.save(out/'marbling_tadpole.png')

    cv.set_state(master)
    np.testing.assert_array_equal(np.asarray(cv.to_image(**LIGHT)),np.asarray(base))
    # Catch only the exposed paper. The old indigo lines supply the petal folds.
    paper = np.clip((np.asarray(base,dtype=float)[:,:,0]-55)/155,0,1)

    def veil(region, tint, amount, feather=0):
        ps.tools.glaze(cv,ps.Paint.from_glaze(tint),thickness=amount,
                       region=region,feather=feather,variation=0)

    def petal(points):
        return wc.polygon(P.spline(points,closed=True))(cv)

    def stroke(points,tint,size,peak):
        stick=ps.OilCrayon(size).load(ps.Paint.from_color(tint))
        cv.stroke(stick,P.spline(points),pressure=P.taper(.17,.36,peak,start=.18))

    def soft_band(cx,cy,rx,ry):
        return np.exp(-((x-cx)/rx)**2-((y-cy)/ry)**2)

    petals = [
        [(312,231),(346,215),(359,174),(384,152),(428,145),(466,151),
         (431,135),(382,126),(340,139),(322,168),(311,202),(284,219)],
        [(330,248),(372,234),(385,188),(408,165),(448,166),(484,182),
         (503,205),(501,180),(471,158),(421,148),(381,158),(364,180),(355,227)],
        [(337,249),(392,264),(450,253),(466,225),(453,203),(428,187),
         (449,186),(479,197),(494,223),(478,254),(448,272),(396,279)],
        [(322,249),(381,281),(436,294),(486,289),(525,276),(541,288),
         (526,317),(485,325),(438,315),(384,296)],
        [(274,237),(321,258),(356,287),(406,312),(427,321),(414,332),
         (386,338),(353,329),(323,304),(301,272)],
    ]
    washed=np.zeros(cv.shape,float)
    for pts in petals:
        washed=np.maximum(washed,ndi.gaussian_filter(petal(pts),3))
    veil(washed*paper,'#efafbd',.24)
    veil(washed*paper*soft_band(358,247,104,57),'#d98091',.22)

    # The marbling already drew the long style and its rounded end.
    style=petal([(90,181),(103,177),(118,181),(131,192),(156,199),
                 (206,207),(259,219),(316,235),(352,245),(318,242),
                 (259,228),(210,217),(156,208),(127,207),(112,218),(94,219),(83,207)])
    veil(style*paper,'#eca756',.34,feather=.7)
    veil(style*paper*soft_band(106,199,24,17),'#d3883f',.18)
    stroke([(116,204),(158,202),(194,208),(232,216),(277,226),(322,239)],
           '#cc964e',1.5,.72)
    # Three folds catch the warm reflected colour, then disappear into the paper.
    for points in [[(348,233),(358,218),(365,189),(374,171),(392,159)],
                   [(357,251),(389,255),(413,254),(429,250)],
                   [(359,279),(386,291),(415,300),(450,308),(474,308)]]:
        stroke(points,'#c88796',1.25,.62)
    rng=np.random.default_rng(34)
    for cx,cy,dx,dy in [(363,182,3,9),(408,267,15,2.4),(474,312,16,4),(378,318,11,3)]:
        for _ in range(15):
            sx,sy=rng.normal(cx,dx),rng.normal(cy,dy)
            ix,iy=int(sx),int(sy)
            if paper[iy,ix]*washed[iy,ix]>.3:
                stroke([(sx,sy),(sx+1.3,sy+.5),(sx+rng.uniform(2,4),sy+1)],
                       '#b8757f',rng.uniform(1,2.6),rng.uniform(.55,.88))
    lily=cv.to_image(**LIGHT)
    lily.save(out/'marbling_lily.png')

    # Keep the panels at identical scale. The narrow gutter belongs to the mount.
    w,h=base.size
    pair=Image.new('RGB',(w*2+16,h),'#e7dfcf')
    pair.paste(lily,(0,0))
    pair.paste(tadpole,(w+16,0))
    pair.save(out/'it_was_there_a_second_ago.png')


if __name__=='__main__':
    draw_readings(ROOT/'output')
