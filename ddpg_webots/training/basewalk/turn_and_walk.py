"""Three attempt-counted command stages: bounded Turn, forward Walk, lateral Walk."""
import copy
import numpy as np
from .gradual_turn import GradualTurnStages


class TurnAndWalkStages:
    def __init__(self, cfg, start_stage='turn', *, cycle_limit=None, state=None,
                 count_fall_as_attempt=False, turn_only=False, fixed_turn_sign=None):
        spec = cfg['training']['turn_and_walk']
        if not spec['enabled'] or turn_only or not count_fall_as_attempt or fixed_turn_sign is not None:
            raise ValueError('Turn and Walk requires mixed stages and counted falls')
        turn_cfg = copy.deepcopy(cfg)
        turn_cfg['training']['gradual_turn'] = dict(enabled=True, intervals_per_half_range=5, endpoint_attempts=20)
        turn = GradualTurnStages(turn_cfg, turn_only=True, count_fall_as_attempt=True)
        # Enter from zero, keep the original bidirectional loop, then exit at zero.
        rates = [.022,.044,.066,.088] + turn.rates + [.066,.044,.022,0.]
        self.commands = [np.array([0.,0.,rate],np.float32) for rate in rates]
        self.holds = [1]*4 + list(turn.holds) + [1]*4
        self.labels = ['turn']*len(self.holds)
        hold = spec['walk_attempts_per_level']
        if type(hold) is not int or hold < 1:
            raise ValueError('Positive integer Walk hold required')
        for axis,key in [(0,'vx_levels_m_s'),(1,'vy_levels_m_s')]:
            levels = spec[key]
            diagonal = axis == 1 and 'lateral_vx_m_s' in spec
            if (len(levels)<3 or levels[0]!=0 or levels[-1]!=0 or
                    not np.isfinite(levels).all() or (not diagonal and min(levels)<0)):
                raise ValueError('Walk levels must be finite and return to zero; legacy levels must be nonnegative')
            bounds=cfg['environment']['command_ranges'][axis]
            if min(levels)<bounds[0] or max(levels)>bounds[1]:
                raise ValueError('Walk command outside environment range')
            for value in levels:
                command=np.zeros(3,np.float32);command[axis]=value
                if diagonal:
                    forward = spec['lateral_vx_m_s']
                    forward_bounds = cfg['environment']['command_ranges'][0]
                    if not np.isfinite(forward) or not max(0., forward_bounds[0]) < forward <= forward_bounds[1]:
                        raise ValueError('Lateral-stage forward speed must be positive and within command range')
                    command[0] = forward
                self.commands.append(command);self.holds.append(hold);self.labels.append('walk')
        self.count_fall_as_attempt=True
        self.count=0
        self.index=0
        self.cycles=0
        if state is not None:
            index,cycles=state.get('turn_walk_index'),state.get('cycles')
            if (type(index) is not int or not 0<=index<len(self.commands) or
                    type(cycles) is not int or not 0<=cycles<self.holds[index] or
                    state.get('stage')!=self.labels[index]):
                raise ValueError('Invalid Turn and Walk continuation state')
            self.index,self.cycles=index,cycles

    @property
    def stage(self):return self.labels[self.index]

    @property
    def limit(self):return self.holds[self.index]

    def command(self,error,speed):return self.commands[self.index].copy()

    def state(self):
        return dict(stage=self.stage,cycles=self.cycles,turn_walk_index=self.index,
                    turn_sign=-1. if self.commands[self.index][2]<0 else 1.)

    def advance_phase(self,before,after,*,fell=False):
        if not fell and after>=before:return
        self.cycles+=1
        if self.cycles>=self.limit:
            self.cycles=0
            self.index=(self.index+1)%len(self.commands)
