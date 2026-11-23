"""NAO R2022b leg geometry; coupled 11-DoF numerical double-foot IK.

Torso frame x forward, y left, z up. Sole reference is FSR box bottom center;
orientation is foot-frame orientation (not the FSR's rotated X/Y axes).
"""
import numpy as np


def rotation(axis, angle):
    axis=np.asarray(axis,dtype=float);axis=axis/np.linalg.norm(axis)
    x,y,z=axis;k=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    return np.eye(3)+np.sin(angle)*k+(1-np.cos(angle))*(k@k)


def forward(q):
    poses=[]
    for side in range(2):
        sign=1 if side==0 else -1
        p=np.array([0.,sign*.05,-.085]);r=np.eye(3)
        axes=((0,1,-sign),(1,0,0),(0,1,0),(0,1,0),(0,1,0),(1,0,0))
        offsets=((0,0,0),(0,0,0),(0,0,0),(0,0,-.1),(0,0,-.1029),(0,0,0))
        for j,(axis,offset) in enumerate(zip(axes,offsets)):
            p=p+r@offset;r=r@rotation(axis,q[6*side+j])
        p=p+r@np.array([.02,sign*.007,-.046])
        t=np.eye(4);t[:3,:3]=r;t[:3,3]=p;poses.append(t)
    return np.asarray(poses)


def expand(x):
    return np.r_[x[:6],x[0],x[6:]]


def errors(q, targets):
    actual=forward(q);out=[]
    for a,b in zip(actual,targets):
        d=b[:3,:3]@a[:3,:3].T
        # sin(angle)*axis, sufficient for locally continuous small steps.
        w=.5*np.array([d[2,1]-d[1,2],d[0,2]-d[2,0],d[1,0]-d[0,1]])
        out.extend(b[:3,3]-a[:3,3]);out.extend(.15*w)
    return np.asarray(out)


def solve(targets, previous, lower, upper, max_step=.12):
    indices=[0,1,2,3,4,5,7,8,9,10,11]
    lo=np.asarray(lower)[indices].copy();hi=np.asarray(upper)[indices].copy()
    lo[0]=max(lower[0],lower[6]);hi[0]=min(upper[0],upper[6])
    initial=np.asarray(previous)[indices].copy()
    lo=np.maximum(lo,initial-max_step);hi=np.minimum(hi,initial+max_step)
    x=np.clip(initial,lo,hi)
    for _ in range(40):
        e=errors(expand(x),targets)
        if np.max(np.abs(e))<1e-6:break
        eps=1e-5
        jac=np.column_stack([(errors(expand(x+np.eye(11)[j]*eps),targets)-e)/eps for j in range(11)])
        dx=np.linalg.solve(jac.T@jac+1e-5*np.eye(11),-jac.T@e)
        found=False
        for scale in (1.,.5,.25,.1):
            candidate=np.clip(x+scale*dx,lo,hi)
            if np.linalg.norm(errors(expand(candidate),targets))<np.linalg.norm(e):
                x=candidate;found=True;break
        if not found:break
    q=expand(x);actual=forward(q)
    pos=float(np.max(np.linalg.norm(actual[:,:3,3]-targets[:,:3,3],axis=1)))
    angles=[np.arccos(np.clip((np.trace(b[:3,:3]@a[:3,:3].T)-1)/2,-1,1)) for a,b in zip(actual,targets)]
    return q,dict(position_error_m=pos,orientation_error_rad=float(max(angles)))
