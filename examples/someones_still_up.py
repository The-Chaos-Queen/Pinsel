"""Someone's Still Up. Two impressions from one inked and wiped plate."""
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pinsel import path as P
from pinsel.monotype import MonotypePlate


class Studio:
    def __init__(self, update=lambda image, caption: None):
        self.w, self.h = 680, 482
        self.rng = np.random.default_rng(308)
        self.y, self.x = np.mgrid[:self.h,:self.w]
        self.plate = MonotypePlate(self.w,self.h,seed=308)
        self.update = update
        grain = ndi.gaussian_filter(self.rng.standard_normal((self.h,self.w)),1)
        self.edge = ((self.x>19+2*grain)&(self.x<660+2*grain)&
                     (self.y>18+2*grain)&(self.y<464+2*grain)).astype('f')
        self.window=self.poly([(224,78),(552,74),(551,349),(225,356)])

    def poly(self, pts):
        img=Image.new('L',(self.w*2,self.h*2))
        ImageDraw.Draw(img).polygon([tuple(v) for v in (2*np.asarray(pts,float))],fill=255)
        return np.asarray(img.resize((self.w,self.h),Image.Resampling.LANCZOS),dtype='f')/255

    def stroke(self, pts, width):
        img=Image.new('L',(self.w*2,self.h*2))
        ImageDraw.Draw(img).line([tuple(v) for v in 2*P.spline(pts)],fill=255,
                                 width=max(1,round(width*2)),joint='curve')
        return np.asarray(img.resize((self.w,self.h),Image.Resampling.LANCZOS),dtype='f')/255

    def ink(self, mask, color, load=.8):
        # Compose in the reader's coordinates; work backwards on the actual plate.
        self.plate.ink(np.ascontiguousarray((mask*self.edge)[:,::-1]),color,load=load)

    def wipe(self, mask, strength):
        self.plate.wipe(np.ascontiguousarray((mask*self.edge)[:,::-1]),strength=strength)

    def line(self, pts, width, color, load=.8):
        self.ink(self.stroke(pts,width),color,load)

    def show(self, caption):
        self.update(self.plate.proof(),caption)


def roll(s):
    s.show('Glass plate  ·  Before inking')
    # A brayer skips at the edges. Leave the window opening clear on the glass.
    night=1-s.window
    for i in range(5):
        left=16+129*i
        stripe=np.clip((s.x-left+6)/10,0,1)*np.clip((left+142-s.x)/11,0,1)
        load=.75+.12*np.sin(s.y/82+i*.6)+.09*np.sin(s.x/7+i)
        s.ink(stripe*load*night,['#25383e','#314348','#304349','#354248','#343949'][i],1)
        s.show(f'Rolling the night ink  ·  Pass {i+1} of 5')
    # Thin courses of weatherboard keep the house readable in darkness.
    for i in range(9):
        yy=49+i*44
        trace=s.stroke([(12,yy+4),(210,yy),(430,yy-3),(670,yy-8)],2)
        s.ink(trace*night,'#687171',.28)
        if i%2==0:
            s.wipe(s.stroke([(32,yy+7),(112,yy+6),(182,yy+4)],1.3)*night,.32)
    s.show('The wall  ·  Scratches through the roller ink')


def light(s):
    s.ink(s.window,'#e7b86e',.82)
    s.show('The lit room  ·  A warm rectangle left in the dark')
    lower=np.clip((s.y-99)/210,0,1)
    s.ink(s.window*lower,'#c77c4d',.53)
    for cx,cy,rx,ry in [(280,176,56,63),(344,197,70,65)]:
        glow=np.exp(-((s.x-cx)/rx)**2-((s.y-cy)/ry)**2)
        s.wipe(glow*s.window,.44)
    s.show('A rag lifts a pool of lamplight')
    s.ink(s.poly([(225,82),(258,81),(252,330),(226,350)])*s.window,'#b37f5a',.56)
    s.ink(s.poly([(223,285),(549,275),(550,322),(224,333)]),'#ae674b',.54)
    s.line([(227,288),(313,286),(438,281),(550,277)],2,'#f0bd79',.72)
    # A shade listing a few degrees, with its dark underside and a lit seam.
    s.line([(313,77),(314,112),(312,134)],2.6,'#4a4841',.92)
    s.ink(s.poly([(284,130),(326,124),(355,165),(268,177)]),'#c77449',1.1)
    s.ink(s.poly([(284,130),(301,128),(291,172),(268,177)]),'#ab5d48',.7)
    s.ink(s.poly([(268,176),(354,165),(349,171),(278,182)]),'#ffcf82',.98)
    s.line([(284,131),(303,128),(326,125)],2,'#fbd69b',.9)
    s.line([(301,130),(298,150),(293,170)],1.5,'#f1c78c',.78)
    s.show('A crooked lampshade  ·  Vermilion and butter yellow')


def leaf(s,base,tip,breadth):
    a,b=np.array(base,float),np.array(tip,float)
    d=b-a
    n=np.array([-d[1],d[0]])/max(1,np.linalg.norm(d))
    one=P.bezier(a,a+.20*d+n*breadth,a+.66*d+n*breadth*.88,b)
    two=P.bezier(b,b-.23*d-n*breadth*.48,a+.34*d-n*breadth*1.05,a)
    return s.poly(np.vstack([one,two]))


def kitchen(s):
    # A sleeve already through the doorway. The fading forearm gets very little ink.
    sleeve=s.poly([(552,153),(533,159),(522,176),(510,200),(518,214),
                   (537,209),(551,197)])
    s.ink(sleeve,'#6d8280',.5)
    s.ink(s.poly([(510,200),(518,196),(535,204),(537,212),(518,220)]),'#c3bba0',.52)
    s.ink(s.poly(P.spline([(519,216),(528,215),(532,220),(528,229),(515,239),
                           (506,244),(498,243),(491,237),(474,232),(467,229),
                           (472,225),(490,227),(503,232)],closed=True)),'#b87358',.29)
    s.wipe(s.stroke([(468,229),(481,231),(492,234)],4),.76)
    s.show('At the edge of the room  ·  An elbow slips out of view')
    # A folded tea towel lies behind the terracotta pot.
    towel=s.poly([(267,307),(334,303),(354,312),(354,329),(283,336),(273,321)])
    s.ink(towel,'#c5c3a5',.88)
    for j in (0,3,11):
        s.line([(282+j,307),(292+j,317),(296+j,333)],1.7,'#718684',.65)
    s.line([(282,336),(306,331),(354,329)],2.3,'#6c7973',.54)
    # Stems spring from a common point, then fork. Each leaf has two unequal sides.
    root=(433,295)
    branches=[
        ([(433,296),(429,267),(424,243),(418,221),(415,192)],
         [((428,259),(405,246),8),((426,251),(442,231),7),((422,237),(398,226),9),
          ((419,222),(435,207),7),((418,215),(403,198),8),((415,200),(419,182),5)]),
        ([(434,295),(443,271),(451,254),(454,233)],
         [((441,272),(463,259),8),((446,263),(430,249),6),((451,253),(471,239),7),
          ((454,242),(446,228),5),((454,240),(461,224),5)]),
        ([(432,295),(420,280),(410,269),(400,253)],
         [((421,283),(402,285),7),((415,276),(422,263),5),((409,267),(389,260),6),
          ((402,257),(392,246),5)])]
    for i,(branch,leaves) in enumerate(branches):
        s.line(branch,2.5,'#484e42',1.1)
        for j,(a,b,width) in enumerate(leaves):
            s.ink(leaf(s,a,b,width),['#495846','#52654e','#60765a'][(i+j)%3],.98)
            if j%2==0:
                middle=tuple(.4*np.array(a)+.6*np.array(b))
                s.line([a,middle],1.1,'#a1a474',.62)
        s.show(f'Basil on the sill  ·  Stem {i+1} of 3')
    s.ink(s.poly([(411,289),(453,287),(447,323),(425,325)]),'#a65f43',1.24)
    s.ink(s.poly([(408,287),(455,285),(455,293),(409,296)]),'#d58a55',1.05)
    s.line([(422,299),(425,309),(427,317)],2.6,'#e3a469',.78)
    s.line([(444,297),(442,307),(440,322)],3.7,'#914e40',.71)
    s.show('Pot and tea towel  ·  A fingernail catches the rim')
    # Worn joinery and two reflected slashes on the pane. Rag cuts display glass.
    s.ink(s.poly([(218,71),(556,67),(553,77),(222,84)]),'#293c41',1.22)
    s.ink(s.poly([(214,75),(227,81),(229,350),(216,358)]),'#5d6762',1.03)
    s.ink(s.poly([(544,77),(556,70),(556,348),(543,349)]),'#273c43',1.18)
    s.ink(s.poly([(218,335),(554,325),(564,339),(216,351)]),'#805b4d',1.04)
    s.ink(s.poly([(207,351),(562,338),(578,345),(575,352),(205,368)]),'#66716b',1.04)
    s.line([(214,352),(323,348),(441,344),(562,340)],2.8,'#d2ad87',.93)
    s.ink(s.poly([(370,79),(376,78),(377,333),(370,334)]),'#5a544a',.98)
    s.line([(377,83),(377,193),(378,330)],1.4,'#eac18a',.88)
    for path in [[(240,116),(257,104),(270,95)],[(242,127),(251,120),(256,116)],
                 [(481,113),(490,106),(510,94)]]:
        s.wipe(s.stroke(path,2)*s.window,.43)
    s.show('The pane  ·  A sash, a sill and three wiped reflections')


def garden(s):
    # Curved wiping marks carry the lamplight into the grain of the wall.
    grain=s.rng.random((s.h,s.w))
    cloud=ndi.gaussian_filter(s.rng.normal(size=(s.h,s.w)),6)
    cloud=cloud/max(.001,cloud.std())
    haze=np.exp(-((s.x-368)/195)**2-((s.y-235)/144)**2)*(1-s.window)
    s.wipe(haze*np.clip(cloud+.25,0,1)*(grain>.83),.41)
    # Rag ridges around the light are much less regular than brush lines.
    for k in range(22):
        a=s.rng.uniform(235,351)
        b=s.rng.uniform(112,257)
        ridge=s.stroke([(a,b),(a+4,b-2),(a+s.rng.uniform(9,23),b-4)],s.rng.uniform(.8,2))
        s.wipe(ridge*s.window*(grain>.42),s.rng.uniform(.18,.40))
    s.show('Wiping the plate  ·  The lamplight breaks into little rag marks')
    # A back bank of stems is visible through the darker, nearer foliage.
    for i in range(23):
        bx=s.rng.uniform(0,680)
        if 235<bx<500:
            top=s.rng.uniform(368,430)
        else:
            top=s.rng.uniform(228,366)
        lean=s.rng.uniform(-49,55)
        stem=[(bx,478),(bx+lean*.24,top+77),(bx+lean*.64,top+31),(bx+lean,top)]
        s.line(stem,s.rng.uniform(1,2.1),'#70796b',.33)
    s.show('Outside the window  ·  The farthest stems')
    # Three arching fronds. Broken gold lines pick out the side facing the house.
    fronds=[([(55,475),(76,397),(95,337),(127,276),(154,231)],1),
            ([(140,476),(152,429),(179,379),(218,337),(249,310)],.78),
            ([(603,474),(600,424),(586,370),(568,319),(549,289)],.83)]
    for j,(points,scale) in enumerate(fronds):
        path=P.spline(points)
        s.line(points,3*scale,'#243a38',1.15)
        for t in np.linspace(.13,.95,15):
            k=int(t*(len(path)-2))
            p=path[k]
            tangent=path[k+1]-p
            tangent/=max(.01,np.linalg.norm(tangent))
            normal=np.array([-tangent[1],tangent[0]])
            for side in [-1,1]:
                reach=scale*(8+21*np.sin(np.pi*t))*s.rng.uniform(.78,1.12)
                tip=p+normal*reach*side+tangent*reach*.80
                s.ink(leaf(s,p,tip,scale*(2.2+4*np.sin(np.pi*t))),
                      '#344d44' if side>0 else '#283e39',1.04)
                if k%3 and side==(1 if j!=2 else -1):
                    s.line([p+.2*(tip-p),p+.7*(tip-p)],1.0,'#8c936f',.36)
        s.show(f'Garden silhouettes  ·  Frond {j+1} of 3')
    # Seed heads above the fronds: curved spokes, with individual flecks of ink at their tips.
    for i,(bx,by,tx,ty,size) in enumerate([(48,474,79,200,23),(73,479,122,235,15),
                                             (648,480,629,195,21),(635,475,595,245,14)]):
        s.line([(bx,by),(bx+(tx-bx)*.16,by-84),(tx-4,ty+69),(tx,ty+13)],2.2,'#242f34',1.18)
        for ang in np.linspace(-2.83,-.27,12):
            ray=size*s.rng.uniform(.64,1.15)
            tip=np.array([tx+np.cos(ang)*ray,ty+np.sin(ang)*ray*.6])
            s.line([(tx,ty+14),(.55*tx+.45*tip[0],ty+4),tip],1.1,'#555c4d',.89)
            for _ in range(3):
                xx,yy=tip+s.rng.normal(0,1.8,2)
                s.line([(xx,yy),(xx+1.5,yy-.5)],1.6,'#b6a37d',.62)
    s.show('Last marks  ·  Seed heads in the light from the house')


def mat_pair(first,ghost):
    # Keep the two pulls equally sized, in the same light, on the same support.
    mat=Image.new('RGB',(first.width*2+22,first.height+16),'#e0d6c2')
    mat.paste(first,(6,8))
    mat.paste(ghost,(first.width+16,8))
    return mat


def print_pair(s):
    output=ROOT/'output'
    output.mkdir(exist_ok=True)
    s.show('Ready to print  ·  Looking down at the inked glass')
    first=s.plate.pull()
    first.save(output/'someones_still_up_first.png')
    s.update(first,'First pull  ·  The printed window faces the other way')
    s.show('The bare press  ·  Only the ink left after the first pull')
    ghost=s.plate.pull()
    ghost.save(output/'someones_still_up_ghost.png')
    s.update(ghost,'Ghost pull  ·  A fresh sheet on the same glass')
    pair=mat_pair(first,ghost)
    pair.save(output/'someones_still_up.png')
    s.update(pair,'Drying rack  ·  First pull on the left, ghost on the right')


def save_replay(frames,dest):
    from PIL import ImageFont
    try:
        font=ImageFont.truetype('segoeui.ttf',15)
    except OSError:
        font=ImageFont.load_default()
    cards=[]
    durations=[]
    for frame,caption in frames:
        card=Image.new('RGB',(680,516),'#292b2a')
        small=frame.copy()
        small.thumbnail((680,482),Image.Resampling.LANCZOS)
        card.paste(small,((680-small.width)//2,(482-small.height)//2))
        ImageDraw.Draw(card).text((12,490),caption,font=font,fill='#f6ebd1')
        cards.append(card.quantize(colors=192,method=Image.Quantize.MEDIANCUT))
        durations.append(2200 if 'pull' in caption.lower() else 800)
    durations[-1]=4200
    cards[0].save(dest,save_all=True,append_images=cards[1:],duration=durations,loop=0,
                  disposal=2,optimize=False)


if __name__=='__main__':
    frames=[]
    studio=Studio(lambda im,caption: frames.append((im,caption)))
    for step in (roll,light,kitchen,garden,print_pair):
        step(studio)
    save_replay(frames,ROOT/'output'/'someones_still_up_replay.gif')
