#!/usr/bin/env python3
"""Bare PLY fallback: preserve source coordinates, assume metres, read its up comment.
Pass --data to package.sh when capture scale/ground truth cameras are available.
"""
import json,re,sys
from pathlib import Path
import numpy as np

def vertical_axis(comment):
    """The up vector of a 'Vertical axis:' comment, as splat-train-serde's importer reads it:
    x -> +X, y -> -Y, z -> -Z (its axis signs), else three numbers; None for anything else."""
    s=comment.lower().split('vertical axis:',1)[1].strip()
    named={'x':[1.,0.,0.],'y':[0.,-1.,0.],'z':[0.,0.,-1.]}
    if s in named: return named[s]
    parts=[]
    for word in filter(None,re.split(r'[\s,\[\]]+',s)):
        try: parts.append(float(word))
        except ValueError: pass
    return parts if len(parts)==3 else None

def horizontal(u):
    """Forward on the floor: the reference axis least aligned with up (-Z first), minus its up part (viewer levelBasis)."""
    ref=np.array([0.,0.,-1.])
    for r in ([1.,0.,0.],[0.,1.,0.]):
        if abs(u@r)<abs(u@ref)-1e-9: ref=np.array(r)
    f=ref-u*(u@ref); return f/np.linalg.norm(f)

ply=Path(sys.argv[1]); dest=Path(sys.argv[2]); names=[];types=[];up=[0.,1.,0.]
with ply.open('rb') as f:
    if f.readline()!=b'ply\n': raise ValueError('not a PLY')
    while True:
        line=f.readline().decode('ascii').strip()
        if line.startswith('format ') and line!='format binary_little_endian 1.0': raise ValueError('bare fallback requires binary little-endian PLY; use --data otherwise')
        if line.lower().startswith('comment vertical axis:'):
            axis=vertical_axis(line[len('comment '):])
            if axis is not None: up=axis  # the last valid comment wins, as in the importer
        if line.startswith('element vertex '): count=int(line.split()[-1])
        if line.startswith('property '):
            _,typ,name=line.split(); names.append(name);types.append({'float':'<f4','double':'<f8','uchar':'u1','int':'<i4'}[typ])
        if line=='end_header': break
    offset=f.tell()
a=np.memmap(ply,dtype=np.dtype(list(zip(names,types))),mode='r',offset=offset,shape=(count,))
sample=a[::max(1,count//100000)]
p=np.column_stack([sample[k] for k in ('x','y','z')]);finite=np.isfinite(p).all(axis=1)
if 'opacity' in names: finite &= sample['opacity']>0
p=p[finite]
if len(p)<10:raise ValueError('too few finite opaque points')
u=np.array(up,dtype=float);norm=np.linalg.norm(u)
if not np.isfinite(u).all() or not np.isfinite(norm) or norm<1e-9: raise ValueError(f'vertical axis must be a finite nonzero vector, got {up}')
u/=norm
lo,hi=np.quantile(p,[.01,.99],axis=0);center=np.median(p,axis=0);floor=float(np.quantile(p@u,.03));eye=center+u*(floor+1.5-center@u)
f=horizontal(u)
cam={'position':eye.tolist(),'look_at':(eye+f).tolist(),'vfov_deg':65}
s={'transform_from_source':np.eye(4).flatten().tolist(),'schema':1,'splat':'scene.spz','meters_per_unit':1,'up':u.tolist(),'radius':float(np.linalg.norm(hi-lo)/2),'focus':center.tolist(),'start_camera':cam,'capture_cameras':[], 'bounds':{'min':lo.tolist(),'max':hi.tolist()},'floor':{'detected':False,'y':floor},'source':{'ply':ply.name},'render_mode':'mip' if b'SplatRenderMode: mip' in ply.open('rb').read(offset) else 'standard','scale_estimate':{'note':'bare PLY: metres assumed; provide --data for solved orientation and scale'}}
dest.write_text(json.dumps(s,indent=2)+'\n')
