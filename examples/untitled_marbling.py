"""One untitled indigo marbling sheet. Two readings can start from the same lift."""
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pinsel as ps
from pinsel import noise, path as P, pigments as pg


def resample(field, x, y):
    return ndi.map_coordinates(field, (y,x), order=1, mode='nearest', prefilter=False)


class Bath:
    def __init__(self, width, height, seed):
        self.rng = np.random.default_rng(seed)
        self.y,self.x = np.indices((height,width), dtype=np.float32)
        self.ink = np.zeros((height,width),np.float32)
        self.eddy = noise.fbm(self.ink.shape,scale=132,octaves=4,seed=self.rng)

    def touch(self, cx, cy, radius, density, phase):
        # A new disk pushes the old surface radially outward. In a round bath
        # the inverse radius is sqrt(r²-R²); here a gently varying R describes
        # how unevenly the drop spreads in different directions.
        x=(self.x-cx)*.94
        y=(self.y-cy)/.94
        theta=np.arctan2(y,x)
        r=np.hypot(x,y)
        R=radius*(1+.064*np.sin(3*theta+phase)+.032*np.sin(5*theta-phase)
                  +.021*np.cos(8*theta+.7*phase))
        scale=np.sqrt(np.maximum(0,r*r-R*R))/np.maximum(r,.01)
        previous=resample(self.ink,cx+(self.x-cx)*scale,cy+(self.y-cy)*scale)
        cover=np.clip(R-r+.5,0,1)
        inside=np.maximum(0,1-(r/np.maximum(R,1))**2)
        pigment=density*(.83+.13*self.eddy+.10*np.sin(theta*3+phase)*inside)
        pigment+=density*.24*np.exp(-((r-R+1.4)/2.4)**2)
        self.ink=np.clip(previous*(1-cover)+pigment*cover,0,1).astype('f')

    def draw_bristle(self, points):
        # Backtrace the surface once along one continuous gesture. The local
        # return flow curls the flanks of a band as the tip drags its middle.
        route=P.resample(P.spline(points),5.0)
        x,y=self.x.copy(),self.y.copy()
        for i in range(len(route)-1,0,-1):
            cx,cy=(route[i]+route[i-1])*.5
            vx,vy=(route[i]-route[i-1])*1.36
            width=27+13*np.sin(np.pi*i/len(route))
            dx=(x-cx)/width
            dy=(y-cy)/width
            envelope=np.exp(-.5*(dx*dx+dy*dy))
            stream=vx*dy-vy*dx
            x-=envelope*(vx-dy*stream)
            y-=envelope*(vy+dx*stream)
        self.ink=resample(self.ink,x,y)

    def view(self):
        # A quick view of the water surface. The lifted sheet is rendered
        # separately through Pinsel's spectral pigments on actual paper relief.
        tint=1-np.exp(-2*self.ink)
        water=np.array([227,227,214])
        blue=np.array([43,61,92])
        return Image.fromarray(np.uint8(water+(blue-water)*tint[...,None]))

    def lift(self):
        height,width=self.ink.shape
        cv=ps.Canvas(width,height,ground='cold_press',ground_color='#f6efdc',
                     texture_scale=.82,seed=142)
        # Contact mirrors the floating pattern on the sheet. Exposed paper
        # tooth and a fine fringe at the margin interrupt the transfer.
        mass=self.ink[:,::-1].copy()
        texture=noise.spectral_noise(mass.shape,beta=.9,seed=71)
        mass*=np.clip(.89+.105*texture+.055*cv.tooth,.4,1.18)
        border=np.minimum.reduce([self.x,width-1-self.x,self.y,height-1-self.y])
        mass*=np.clip(border-15+.85*texture,0,1)
        indigo=pg.mix(pg.INDIGO,1,pg.ULTRAMARINE,1.7).as_medium('watercolor')
        ps.tools.glaze(cv,indigo,thickness=.88,region=np.clip(mass,0,1),
                       feather=0,variation=0,wet=False)
        return cv


def make_sheet(capture=lambda image,label:None):
    bath=Bath(836,570,324)
    capture(bath.view(),'Water')
    for i,radius in enumerate([106,112,97,88,100,79,75,67,61,54,46,33]):
        t=i/11
        cx=340+119*t+34*np.sin(7*t+.3)
        cy=289-30*t+36*np.sin(5*t+.1)
        phase=.8*i+.4*np.sin(i)
        bath.touch(cx,cy,radius,[.62,.39,.79,.44,.65,.48][i%6],phase)
        capture(bath.view(),f'Indigo  ·  {i+1} / 12')
        bath.touch(cx+5.7*np.cos(i*.8),cy+6.3*np.sin(i*.85),radius*(1.19+.16*np.sin(i)),
                   0,phase+.17)
        capture(bath.view(),f'Clear  ·  {i+1} / 12')
    bath.draw_bristle([(47,441),(160,393),(295,383),(403,341),(490,259),
                       (488,188),(422,163),(357,191),(337,252),(382,284),
                       (472,275),(581,228),(733,199)])
    capture(bath.view(),'One bristle')
    canvas=bath.lift()
    capture(canvas.to_image(relief=.48,specular=.02),'Paper lift')
    return canvas


def save_session(output):
    output.mkdir(exist_ok=True)
    frames=[]
    try:
        font=ImageFont.truetype('segoeui.ttf',16)
    except OSError:
        font=ImageFont.load_default()
    def record(image,label):
        card=Image.new('RGB',(626,460),'#272d30')
        card.paste(image.resize((626,427),Image.Resampling.LANCZOS))
        ImageDraw.Draw(card).text((12,433),label,font=font,fill='#f5ecdb')
        frames.append(card.quantize(colors=200))
    sheet=make_sheet(record)
    sheet.save(output/'untitled_marbling.png',relief=.48,specular=.02)
    frames[0].save(output/'untitled_marbling_replay.gif',save_all=True,
                   append_images=frames[1:],duration=[600]*25+[1500,3600],loop=0,
                   disposal=2,optimize=False)


if __name__=='__main__':
    save_session(ROOT/'output')
