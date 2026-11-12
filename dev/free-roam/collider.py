#!/usr/bin/env python3
"""Occupancy from opaque splat centres in the exported frame. No capture-path carve.
The viewer removes disconnected components of <=8 cells before collision starts.
"""
import base64,json,re,struct,sys
from pathlib import Path
import numpy as np
ply=Path(sys.argv[1]);scene=json.loads(Path(sys.argv[2]).read_text());package=Path(sys.argv[3]);fields=[]
with ply.open('rb') as f:
    while True:
        line=f.readline().decode('ascii').strip()
        if line.startswith('format ') and line!='format binary_little_endian 1.0':raise ValueError('collision input must be binary little-endian PLY')
        if line.startswith('element vertex '):count=int(line.split()[-1])
        if line.startswith('property '):
            _,t,name=line.split();fields.append((name,{'float':'<f4','double':'<f8','uchar':'u1','int':'<i4'}[t]))
        if line=='end_header':break
    offset=f.tell()
a=np.memmap(ply,dtype=np.dtype(fields),mode='r',offset=offset,shape=(count,))
m=np.array(scene.get('transform_from_source',np.eye(4).flatten().tolist())).reshape(4,4).T
mpu=scene.get('meters_per_unit',1); edge=.05/mpu
# Reject transparent haze and enormous splats; a small opaque surface still collides.
mask=np.isfinite(a['opacity']) & (a['opacity']>1.38629436)
points=np.column_stack([a[k][mask] for k in ('x','y','z')]);points=points@m[:3,:3].T+m[:3,3]
scales=np.column_stack([a[k][mask] for k in ('scale_0','scale_1','scale_2')]);size=np.exp(scales.max(axis=1))*np.linalg.norm(m[:3,0])*mpu
points=points[np.isfinite(points).all(axis=1)&(size<.5)]
if len(points)<8:raise ValueError('no opaque finite surface points')
lo,hi=np.quantile(points,[.001,.999],axis=0);lo-=edge*2;hi+=edge*2
# Limit storage, not movement. Far outliers cannot allocate unbounded grids.
while np.prod(np.ceil((hi-lo)/edge))>40_000_000:edge*=1.25
n=np.ceil((hi-lo)/edge).astype(int);idx=np.floor((points-lo)/edge).astype(int);idx=idx[((idx>=0)&(idx<n)).all(axis=1)]
grid=np.zeros(tuple(n[::-1]),dtype=np.uint8);grid[idx[:,2],idx[:,1],idx[:,0]]=1
# Close a one-voxel sampling gap only when opposite neighbours both support it.
for axis in range(3):
    left=[slice(None)]*3;right=left.copy();mid=left.copy();left[axis]=slice(None,-2);right[axis]=slice(2,None);mid[axis]=slice(1,-1)
    grid[tuple(mid)] |= grid[tuple(left)] & grid[tuple(right)]
out=bytearray(b'VXO1'+bytes(4));chunks=0
for z in range(0,int(n[2]),32):
 for y in range(0,int(n[1]),32):
  for x in range(0,int(n[0]),32):
   part=grid[z:z+32,y:y+32,x:x+32]
   if not part.any():continue
   block=np.zeros((32,32,32),dtype=np.uint8);block[:part.shape[0],:part.shape[1],:part.shape[2]]=part
   out+=struct.pack('<iii',x//32,y//32,z//32)+np.packbits(block.ravel(),bitorder='little').tobytes();chunks+=1
struct.pack_into('<I',out,4,chunks)
(package/'data/collide-00.js').write_text('window.__SPLAT_DATA=window.__SPLAT_DATA||{};window.__SPLAT_DATA["collide-00"]="'+base64.b64encode(out).decode()+'";\n')
p=package/'index.html';text=p.read_text();match=re.search(r'^const CFG = (.*);$',text,re.M);cfg=json.loads(match[1]);cfg['collide']={'key':'collide','parts':1,'bytes':len(out),'origin':lo.tolist(),'edge':edge,'dims':n.tolist(),'chunk':32,'cells':int(grid.sum()),'radiusM':.2,'stopM':.2,'source':'opaque model centres (alpha > 0.8, sigma < 0.5 m); no path carve'};cfg['up']=scene.get('up',[0,1,0]);cfg['floorHeight']=scene.get('floor',{}).get('y');cfg['floorExtent']=scene.get('floor',{}).get('extent')
build_path=package/'build.json'
if build_path.exists():
 build=json.loads(build_path.read_text());build['collide']=cfg['collide'];build_path.write_text(json.dumps(build,indent=2)+'\n')
# Inline-script safe: no literal '<' (a title's </script> or <!-- cannot end or alter the script element).
p.write_text(text[:match.start(1)]+json.dumps(cfg,separators=(',',':')).replace('<','\\u003c')+text[match.end(1):])
print(json.dumps(cfg['collide']))
