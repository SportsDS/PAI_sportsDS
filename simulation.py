"""Color Courier: a deterministic 2-D differential-drive robot simulation.

Perception: classify parcel color from a simulated RGB camera image.
Planning: A* grid search around polygonal obstacles.
Control: waypoint-following differential-drive kinematics.
Rendering: Pillow frames, encoded to MP4 with ffmpeg.
"""
from __future__ import annotations
import argparse, heapq, math, os, subprocess, tempfile
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 960, 540
WORLD_W, WORLD_H = 12.0, 6.75
SCALE = min((W-96) / WORLD_W, (H-54) / WORLD_H)
COLORS = {"red": (230, 83, 76), "blue": (67, 139, 222), "yellow": (245, 190, 56)}
BINS = {"red": (10.35, 1.0), "blue": (10.35, 3.35), "yellow": (10.35, 5.7)}
START = (1.0, 3.4)
PARCELS_INIT = [(2.1, 1.15, "blue"), (2.0, 5.55, "yellow"), (3.3, 3.4, "red")]
OBSTACLES = [(4.2, 0.0, 5.1, 2.0), (4.2, 2.65, 5.1, 4.05), (4.2, 4.7, 5.1, 6.75),
             (6.5, 1.55, 7.35, 2.45), (6.5, 4.35, 7.35, 5.2)]
CELL = 0.15


def world_to_px(x, y): return int(75+x*67), int(435-y*44)
def hsv_like_classify(rgb):
    """Nearest class prototype from RGB sensor pixels; returns label and confidence."""
    v = np.asarray(rgb, dtype=float)
    distances = {k: float(np.linalg.norm(v-np.asarray(c))) for k,c in COLORS.items()}
    ordered = sorted(distances, key=distances.get)
    confidence = 1.0 - distances[ordered[0]] / (distances[ordered[1]] + 1e-9)
    return ordered[0], max(0.0, confidence)


def camera_sample(parcel_color):
    """Simulated camera RGB observation with fixed, documented sensor noise."""
    rng = np.random.default_rng(sum(ord(c) for c in parcel_color) + 19)
    measured = np.clip(np.array(COLORS[parcel_color]) + rng.normal(0, 7, 3), 0, 255)
    return measured.astype(int)


def free(x, y):
    if not (0.35 <= x <= 11.55 and 0.35 <= y <= 6.4): return False
    return not any(x1-0.25 <= x <= x2+0.25 and y1-0.25 <= y <= y2+0.25 for x1,y1,x2,y2 in OBSTACLES)


def astar(start, goal):
    """8-connected A* on occupancy grid; returns world-coordinate waypoints."""
    def cell(p): return (round(p[0]/CELL), round(p[1]/CELL))
    def xy(c): return (c[0]*CELL, c[1]*CELL)
    s,g=cell(start),cell(goal); q=[(0.0,0.0,s)]; prev={}; cost={s:0.0}; seen=set()
    dirs=[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    while q:
        _,gc,u=heapq.heappop(q)
        if u in seen: continue
        if u==g: break
        seen.add(u)
        for dx,dy in dirs:
            v=(u[0]+dx,u[1]+dy); vx,vy=xy(v)
            if not free(vx,vy): continue
            step=CELL*(math.sqrt(2) if dx and dy else 1)
            ng=gc+step
            if ng < cost.get(v,float('inf')):
                cost[v]=ng; prev[v]=u
                h=math.hypot((g[0]-v[0])*CELL,(g[1]-v[1])*CELL)
                heapq.heappush(q,(ng+h,ng,v))
    if g not in cost: raise RuntimeError(f"No path {start} -> {goal}")
    nodes=[g]
    while nodes[-1]!=s: nodes.append(prev[nodes[-1]])
    nodes.reverse()
    pts=[xy(c) for c in nodes]
    # prune collinear grid steps
    pruned=[pts[0]]
    for i in range(1,len(pts)-1):
        a,b,c=pts[i-1:i+2]
        if abs((b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0]))>1e-6: pruned.append(b)
    pruned.append(pts[-1]); pruned[0]=start; pruned[-1]=goal
    return pruned

@dataclass
class Robot:
    x: float=START[0]; y: float=START[1]; theta: float=0.0; carrying: str|None=None

class Simulation:
    def __init__(self):
        self.robot=Robot(); self.parcels=[list(p) for p in PARCELS_INIT]
        self.target_index=0; self.path=[]; self.waypoint=0; self.state="SCAN"; self.time=0.0
        self.total= len(self.parcels); self.delivered=0; self.delivered_indices=set(); self.log=[]; self.last_class=None
        self.target=None; self.load_next()
    def load_next(self):
        if self.target_index>=len(self.parcels): self.state="DONE"; return
        p=self.parcels[self.target_index]
        measured=camera_sample(p[2]); label,conf=hsv_like_classify(measured)
        self.last_class=(label,conf,measured.tolist()); self.target=(float(p[0]),float(p[1]),p[2])
        self.path=astar((self.robot.x,self.robot.y),(self.target[0],self.target[1])); self.waypoint=1
        self.state="NAVIGATE_TO_PARCEL"; self.log.append(f"camera_rgb={measured.tolist()} -> {label} ({conf:.2f})")
    def dt_step(self,dt):
        self.time+=dt
        if self.state in ("DONE","WAIT"): return
        r=self.robot
        if self.state.startswith("NAVIGATE"):
            goal=self.path[self.waypoint]
            dx,dy=goal[0]-r.x,goal[1]-r.y; dist=math.hypot(dx,dy)
            if dist < 0.08:
                self.waypoint+=1
                if self.waypoint>=len(self.path):
                    if self.state=="NAVIGATE_TO_PARCEL":
                        self.state="PICK_UP"; self._event_timer=0.0
                    else:
                        self.state="DROP"; self._event_timer=0.0
                return
            desired=math.atan2(dy,dx); err=(desired-r.theta+math.pi)%(2*math.pi)-math.pi
            omega=float(np.clip(3.2*err,-2.4,2.4)); v=min(0.72,1.6*dist)*max(0.15,math.cos(err))
            r.theta += omega*dt; r.x += v*math.cos(r.theta)*dt; r.y += v*math.sin(r.theta)*dt
        elif self.state=="PICK_UP":
            self._event_timer+=dt
            if self._event_timer>=0.85:
                r.carrying=self.last_class[0]; self.state="NAVIGATE_TO_BIN"
                bx,by=BINS[r.carrying]; self.path=astar((r.x,r.y),(bx-0.55,by)); self.waypoint=1
        elif self.state=="DROP":
            self._event_timer+=dt
            if self._event_timer>=0.85:
                r.carrying=None; self.delivered_indices.add(self.target_index); self.delivered+=1; self.target_index+=1; self.load_next()

    def render(self,frame_i,fps):
        im=Image.new('RGB',(W,H),(247,249,252)); d=ImageDraw.Draw(im)
        try: font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18); small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
        except: font=small=ImageFont.load_default()
        # field
        d.rounded_rectangle((27,27,W-27,H-27),radius=18,fill=(252,253,255),outline=(210,219,230),width=2)
        for x1,y1,x2,y2 in OBSTACLES:
            p1=world_to_px(x1,y2); p2=world_to_px(x2,y1)
            d.rounded_rectangle((p1[0],p1[1],p2[0],p2[1]),radius=10,fill=(87,101,120))
            d.text((p1[0]+8,p1[1]+8),'OBSTACLE',font=small,fill=(235,240,245))
        # bins
        for color,(x,y) in BINS.items():
            px,py=world_to_px(x,y); col=COLORS[color]
            d.rounded_rectangle((px-40,py-32,px+40,py+32),radius=9,fill=(255,255,255),outline=col,width=4)
            d.text((px-23,py-9),color.upper(),font=small,fill=col)
        # path to target / bin
        if self.path:
            points=[world_to_px(*p) for p in self.path[self.waypoint-1:]] if self.waypoint>0 else []
            if len(points)>1: d.line(points,fill=(62,159,130),width=4)
        # parcels with target focus halo
        for i,(x,y,c) in enumerate(self.parcels):
            if i in self.delivered_indices: continue
            if self.target_index==i and self.state not in ('DONE',):
                px,py=world_to_px(x,y); d.ellipse((px-22,py-22,px+22,py+22),outline=(35,167,128),width=3)
            if self.target_index==i and self.state in ('PICK_UP','NAVIGATE_TO_BIN','DROP'): continue
            px,py=world_to_px(x,y); d.rounded_rectangle((px-13,py-13,px+13,py+13),radius=5,fill=COLORS[c],outline=(255,255,255),width=2)
        # robot body/wheels
        px,py=world_to_px(self.robot.x,self.robot.y); ang=self.robot.theta
        d.ellipse((px-20,py-20,px+20,py+20),fill=(35,51,74),outline=(20,32,48),width=2)
        for side in (-1,1):
            wx=px+int(-math.sin(ang)*side*15); wy=py+int(math.cos(ang)*side*15)
            d.rounded_rectangle((wx-9,wy-4,wx+9,wy+4),radius=3,fill=(18,24,32))
        d.line((px,py,px+int(25*math.cos(ang)),py-int(25*math.sin(ang))),fill=(245,248,250),width=4)
        if self.robot.carrying:
            d.rectangle((px-8,py-29,px+8,py-13),fill=COLORS[self.robot.carrying],outline='white',width=2)
        # sidebar header and status
        d.rounded_rectangle((50,43,655,103),radius=12,fill=(255,255,255),outline=(220,226,234),width=1)
        d.text((68,52),'COLOR COURIER',font=font,fill=(31,47,67)); d.text((68,79),'AI perception  •  A* planning  •  differential-drive control',font=small,fill=(83,99,117))
        labels={'SCAN':'RGB VISION: CLASSIFY PARCEL','NAVIGATE_TO_PARCEL':'A* ROUTE: APPROACH PARCEL','PICK_UP':'PICKUP: GRIPPER ACTION','NAVIGATE_TO_BIN':'A* ROUTE: DELIVER TO MATCHING BIN','DROP':'RELEASE: SORT PARCEL','DONE':'TASK COMPLETE'}
        d.rounded_rectangle((675,43,910,103),radius=12,fill=(32,48,69))
        d.text((692,55),labels.get(self.state,self.state),font=small,fill=(255,255,255))
        d.text((692,79),f'Delivered {self.delivered}/{self.total}',font=small,fill=(186,226,211))
        # lower banner
        d.rounded_rectangle((50,H-82,910,H-46),radius=8,fill=(255,255,255),outline=(224,229,235),width=1)
        explanation={'NAVIGATE_TO_PARCEL':'RGB sensor observation is matched to the nearest color prototype.', 'PICK_UP':'The robot closes its simulated gripper around the parcel.', 'NAVIGATE_TO_BIN':'A* replans a collision-free route to the color-matched destination.', 'DROP':'Parcel is released into its matching bin.', 'DONE':'All parcels have been sorted successfully.'}.get(self.state,'The robot selects the next parcel and estimates its color.')
        d.text((67,H-72),explanation,font=small,fill=(53,67,83))
        # simulated camera swatch
        if self.last_class:
            label,conf,rgb=self.last_class
            d.text((675,123),f'RGB {rgb} → {label} ({conf:.2f})',font=small,fill=(46,67,87))
        d.text((52,116),f't = {self.time:05.1f} s',font=small,fill=(96,107,121))
        return im

def render_video(out,seconds=75,fps=12):
    sim=Simulation(); steps=int(seconds*fps); dt=1.0/fps
    out=Path(out); out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        for i in range(steps):
            # fixed intro/outro holds for legibility
            if i >= int(70*fps) and sim.state=='DONE': pass
            else: sim.dt_step(dt)
            frame=sim.render(i,fps)
            frame.save(Path(td)/f'{i:06d}.png')
        subprocess.run(['ffmpeg','-y','-loglevel','error','-framerate',str(fps),'-i',f'{td}/%06d.png','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],check=True)
    return sim

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--video',default='deliverables/color_courier_demo.mp4'); ap.add_argument('--seconds',type=int,default=75); ap.add_argument('--fps',type=int,default=12); args=ap.parse_args()
    sim=render_video(args.video,args.seconds,args.fps)
    print(f'video={args.video} state={sim.state} delivered={sim.delivered}/{sim.total} elapsed_sim_s={sim.time:.1f}')
    for item in sim.log: print(item)
if __name__=='__main__': main()
