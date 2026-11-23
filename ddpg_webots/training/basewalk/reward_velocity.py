"""Task-boundary EMA and instantaneous Reward velocity, with one causal clock."""
import numpy as np
from .core import OBS_DIM,vector


class RewardVelocity:
    def __init__(self,source,alpha=.2):
        if source not in ('command_ema','instantaneous') or not 0 < alpha <= 1:
            raise ValueError('Invalid Reward velocity source or alpha')
        self.source,self.alpha=source,float(alpha)
        self.reset()

    def reset(self):
        self.key=None;self.value=None;self.count=0

    def advance(self,velocity,command,mode):
        v=vector(velocity,3,'Reward velocity sample')
        c=vector(command,3,'Reward source command')
        if int(mode) not in (0,1,2):raise ValueError('Invalid source task')
        key=(int(mode),tuple(c.tolist()))
        if self.source=='instantaneous' or self.value is None or key!=self.key:
            self.value=v.copy();self.count=1
        else:
            self.value+=self.alpha*(v-self.value);self.count+=1
        self.key=key
        return self.value.copy()


def sequence_velocities(arrays,cfg):
    obs=np.asarray(arrays.get('observations',arrays.get('obs')))
    nxt=np.asarray(arrays.get('next_observations',arrays.get('next_obs')))
    if obs.ndim!=2 or obs.shape[1:]!=(OBS_DIM,) or nxt.shape!=obs.shape or not len(obs):
        raise ValueError('Reward velocity requires aligned State50 arrays')
    if not np.isfinite(obs).all() or not np.isfinite(nxt).all():raise ValueError('Nonfinite State')
    source=cfg['reward']['biped_tracking']['velocity_source']
    after=np.c_[nxt[:,47:49],nxt[:,46]/cfg['environment']['yaw_rate_scale']].astype(np.float32)
    if source=='instantaneous':return after
    terminal=np.asarray(arrays.get('terminals',arrays.get('terminal',np.zeros(len(obs))))).reshape(-1)
    modes=np.asarray(arrays['modes']).reshape(-1)
    if terminal.shape!=(len(obs),) or modes.shape!=(len(obs),):raise ValueError('Invalid transition labels')
    f=RewardVelocity(source,cfg['environment']['velocity_filter']);result=np.empty_like(after)
    for i in range(len(obs)):
        reset=i==0 or bool(terminal[i-1]) or not np.allclose(obs[i],nxt[i-1],rtol=0,atol=1e-6)
        if reset:
            if (not np.allclose(obs[i,34:46],0,atol=1e-6) or abs(float(obs[i,9]))>1e-6
                    or abs(float(obs[i,8])-1)>1e-6):
                raise ValueError('Cannot reconstruct EMA from a partial/discontinuous trajectory')
            f.reset()
        result[i]=f.advance(after[i],obs[i,5:8],modes[i])
    return result
