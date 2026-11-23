"""Once-per-control-step stage transitions and baselineV2/projection reward."""
import math
from collections import deque
import numpy as np
from .stationary_reward import stationary_terms,stand_terms

def wrap(a):return math.atan2(math.sin(a),math.cos(a))

class Stages:
    def __init__(self,cfg,error):
        self.cfg=cfg;self.count=0
        if cfg['turn_stand_walk'].get('stand_yaw_stability')!='signed_mean_one_gait_period':
            raise ValueError('Expected cycle-mean Stand yaw stability contract')
        self.yaw_dt=cfg['webots']['control_time_step_ms']/1000.
        self.yaw_period=1./cfg['environment']['gait_frequency_hz']
        self.yaw_samples=math.ceil(self.yaw_period/self.yaw_dt)
        self.yaw_window=deque(maxlen=self.yaw_samples)
        self.mean_wz=None
        self.stage='stand' if abs(math.degrees(error))<=cfg['turn_stand_walk']['angle_tolerance_deg'] else 'turn'

    def command(self,error,speed):
        if self.stage=='walk':return np.array([speed,0.,0.],np.float32)
        if self.stage=='stand':return np.zeros(3,np.float32)
        return np.array([0.,0.,math.copysign(self.cfg['environment']['heading_max_rate'],error)],np.float32)

    def advance(self,error,speed,wz,ready,fell):
        if self.stage=='walk':return
        p=self.cfg['stationary_reward']
        if self.cfg.get('_expert_collection_stability'):
            p=dict(p)
            p.update(self.cfg.get('expert_collection_stability', {}))
        thresholds=self.cfg['turn_stand_walk']
        enter=thresholds['angle_tolerance_deg']
        hold=thresholds.get('stand_angle_tolerance_deg',enter)
        if not 0 < enter <= hold:raise ValueError('Expected 0 < Turn entry tolerance <= Stand hold tolerance')
        tolerance=hold if self.stage=='stand' else enter
        aligned=abs(error)<=math.radians(tolerance)
        if not aligned or fell:
            self.stage='turn';self.count=0;self.yaw_window.clear();self.mean_wz=None;return
        if self.stage=='turn':
            self.stage='stand';self.count=0;self.yaw_window.clear();self.mean_wz=None;return
        # Only Stand-command transitions enter the causal rolling window.
        self.yaw_window.append(float(wz))
        if len(self.yaw_window)<self.yaw_samples:
            self.count=0;self.mean_wz=None;return
        # Partial oldest sample makes the duration exactly one gait period.
        oldest_dt=self.yaw_period-(self.yaw_samples-1)*self.yaw_dt
        self.mean_wz=(sum(self.yaw_window)*self.yaw_dt
                      -self.yaw_window[0]*(self.yaw_dt-oldest_dt))/self.yaw_period
        # Translation is constrained by Reward, not by the Stand -> Walk gate.
        # Keep speed/ready arguments for existing callers and recorded data.
        stable=abs(self.mean_wz)<=p['yaw_tolerance_rad_s']
        self.count=self.count+1 if stable else 0
        if self.count>=self.cfg['turn_stand_walk']['stand_steps']:self.stage='walk'

def reward_inputs(old_position,position,target,old_heading,heading,wz,command_wz,speed,ready,dt):
    delta=np.asarray(position)[:2]-np.asarray(old_position)[:2]
    # Body forward at the start of this transition, not the global target ray.
    projection=float(delta@np.array([math.cos(old_heading),math.sin(old_heading)]))
    return [wrap(target-old_heading),wrap(target-heading),float(wz-command_wz),float(speed),float(ready),projection,float(dt)]

def motion_reward(mode,values,cfg,fell=False):
    old,new,yaw,speed,ready,projection,dt=values
    p=cfg['reward']
    if mode not in (0,1,2) or dt<=0 or speed<0 or not np.isfinite(values).all():
        raise ValueError('Invalid bounded Reward input')
    bound=p['task_reward_bound']
    terms=dict(turn_hold=0.,turn_heading_progress=0.,turn_heading_error=0.,
               turn_linear_velocity=0.,turn_yaw_rate_error=0.,walk_forward_progress=0.)
    if mode==2:
        error=float(np.clip((1.-projection/dt/p['walk_speed_reference'])/2.,0.,1.))
        value=bound*(1.-2.*error)
        terms['walk_forward_progress']=value
    else:
        stage='turn' if mode==0 else 'stand'
        linear=speed/p[stage+'_translation_error_scale']
        angular=abs(yaw)/p[stage+'_yaw_error_scale']
        total=linear+angular
        factor=1. if p.get('turn_stand_unclipped',False) else (min(total,1.)/total if total>0 else 0.)
        terms['turn_hold']=bound
        terms['turn_linear_velocity']=-2.*bound*linear*factor
        terms['turn_yaw_rate_error']=-2.*bound*angular*factor
        value=sum(terms.values())
    terms.update(turn_total=value if mode!=2 else 0.,walk_total=value if mode==2 else 0.,
        turn_weighted=value if mode!=2 else 0.,walk_weighted=value if mode==2 else 0.,
        stand_total=value if mode==1 else 0.)
    return value,terms

def revalue(inputs,modes,cfg):
    if np.asarray(inputs).shape!=(len(modes),7) or not np.isfinite(inputs).all():
        raise ValueError('Expected finite V25 reward inputs [N,7]')
    return np.asarray([motion_reward(int(m),v,cfg)[0] for m,v in zip(modes,inputs)])


def step_reward_inputs(arrays):
    """Reconstruct instantaneous speed from immutable per-transition raw deltas."""
    inputs=np.asarray(arrays['reward_inputs'],dtype=float).copy()
    delta=np.asarray(arrays['cycle_actual'],dtype=float)
    dt=np.asarray(arrays['cycle_dt'],dtype=float).reshape(-1)
    if inputs.shape!=(len(dt),7) or delta.shape!=(len(dt),3):
        raise ValueError('Raw per-step displacement/dt required; cannot reuse window speed')
    if not np.isfinite(delta).all() or not np.isfinite(dt).all() or np.any(dt<=0):
        raise ValueError('Invalid raw per-step displacement/dt')
    if not np.allclose(inputs[:,6],dt):
        raise ValueError('Raw control time mismatch')
    inputs[:,3]=np.linalg.norm(delta[:,:2],axis=1)/dt
    inputs[:,4]=1.
    walk=np.asarray(arrays['modes'])==2
    if np.any(walk):
        command=np.asarray(arrays['observations'])[walk,5:7]
        target=np.asarray(arrays['cycle_target'],dtype=float)[walk,:2]
        length=np.linalg.norm(target,axis=1)
        # Retained Walk demonstrations command positive body X and zero body Y.
        # Their world command displacement therefore records body-forward axis.
        if np.any(command[:,0]<=0) or not np.allclose(command[:,1],0.) or np.any(length<=0):
            raise ValueError('Cannot reconstruct body-forward axis from raw Walk command')
        inputs[walk,5]=np.sum(delta[walk,:2]*target/length[:,None],axis=1)
    return inputs
