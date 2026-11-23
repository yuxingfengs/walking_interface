"""Walk-only torso XY commands on a persistent control-step clock."""
import numpy as np


class WalkCommands:
    def __init__(self,cfg,start_stage='walk',*,state=None,**unused):
        spec=cfg['training']['walk_commands']
        self.hold=spec['hold_steps']
        xy=np.asarray(spec['commands_m_s'],dtype=np.float32)
        limits=np.asarray(cfg['environment']['command_ranges'][:2])
        if (type(self.hold) is not int or self.hold<1 or xy.ndim!=2 or xy.shape[1]!=2 or
                not len(xy) or not np.isfinite(xy).all() or
                np.any(np.linalg.norm(xy,axis=1)==0) or np.any(xy<limits[:,0]) or np.any(xy>limits[:,1])):
            raise ValueError('Invalid Walk commands or hold duration')
        self.commands=np.column_stack((xy,np.zeros(len(xy),dtype=np.float32)))
        self.step=0 if state is None else state['step']
        if type(self.step) is not int or self.step<0:raise ValueError('Invalid Walk clock')
        self.stage='walk';self.count=0;self.count_fall_as_attempt=True

    @property
    def limit(self):return self.hold

    @property
    def cycles(self):return self.step%self.hold

    def state(self):return dict(stage='walk',step=self.step)

    def command(self,error,speed):return self.commands[self.step//self.hold%len(self.commands)].copy()

    def advance_phase(self,before,after,*,fell=False):self.step+=1
