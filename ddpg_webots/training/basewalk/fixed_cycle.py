"""Fixed Turn/Walk schedule with optional counting of interrupted attempts."""
import numpy as np
class FixedCycleStages:
    def __init__(self,cfg,start_stage="turn", *, cycle_limit=None, state=None,
                 count_fall_as_attempt=False, turn_only=False, fixed_turn_sign=None):
        if fixed_turn_sign is not None and fixed_turn_sign not in (-1., 1.):
            raise ValueError("Fixed Turn direction requires a sign of ±1")
        self.fixed_turn_sign = fixed_turn_sign
        self.cfg=cfg
        self.turn_rate = cfg['environment']['heading_max_rate']
        if (turn_only or cycle_limit is not None) and cfg.get('training', {}).get('online_turn_rate_rad_s') is not None:
            self.turn_rate = cfg['training']['online_turn_rate_rad_s']
            low, high = cfg['environment']['command_ranges'][2]
            if not np.isfinite(self.turn_rate) or not 0 < self.turn_rate <= min(-low, high):
                raise ValueError('Online Turn rate must be positive and within the command range')
        self.limit = int(cfg['fixed_cycle_commands']['cycles_per_command']) if cycle_limit is None else cycle_limit
        if type(self.limit) is not int:
            raise ValueError('Fixed-cycle limit must be an integer')
        if self.limit<=0:raise ValueError('Positive cycle count required')
        if type(count_fall_as_attempt) is not bool:
            raise ValueError('count_fall_as_attempt must be boolean')
        self.count_fall_as_attempt = count_fall_as_attempt
        self.order=tuple(cfg['fixed_cycle_commands']['order'])
        if self.order!=('turn','walk'):raise ValueError('Expected Turn-Walk')
        if turn_only:
            self.order = ('turn',)
        if start_stage not in self.order:raise ValueError('Invalid fixed-cycle initial stage')
        self.index=self.order.index(start_stage);self.stage=start_stage;self.cycles=0;self.count=0;self.turn_sign=1. if fixed_turn_sign is None else fixed_turn_sign
        if state is not None:
            stage, cycles, turn_sign = state['stage'], state['cycles'], state['turn_sign']
            if stage not in self.order or type(cycles) is not int or not 0 <= cycles < self.limit or turn_sign not in (-1., 1.):
                raise ValueError('Invalid fixed-cycle continuation state')
            if fixed_turn_sign is not None and turn_sign != fixed_turn_sign:
                raise ValueError("Continuation Turn direction differs from configured direction")
            self.index = self.order.index(stage)
            self.stage, self.cycles, self.turn_sign = stage, cycles, turn_sign

    def state(self):
        return dict(stage=self.stage, cycles=self.cycles, turn_sign=self.turn_sign)
    def command(self,error,speed):
        if self.stage=='walk':return np.array([self.cfg['reward']['walk_speed_reference'],0,0],np.float32)
        if self.stage=='turn':return np.array([0,0,self.turn_sign*self.turn_rate],np.float32)
        return np.zeros(3,np.float32)
    def advance_phase(self,before,after, *, fell=False):
        if fell:
            if not self.count_fall_as_attempt:
                return
        elif after >= before:
            return
        self.cycles+=1
        if self.cycles<self.limit:return
        self.cycles=0
        if self.stage=='turn' and self.fixed_turn_sign is None:self.turn_sign*=-1
        self.index=(self.index+1)%len(self.order)
        self.stage=self.order[self.index]


class TurnFallWalkReset:
    """Online-only reset admission; state survives world reloads."""
    def __init__(self, threshold=0, state=None):
        if type(threshold) is not int or threshold < 0:
            raise ValueError('Turn fall reset threshold must be a nonnegative integer')
        self.threshold = threshold
        self.falls = int((state or {}).get('falls', 0))
        self.pending_walk = bool((state or {}).get('pending_walk', False))

    def observe(self, source_stage, fell):
        # A real Walk transition breaks the streak, including a Walk fall.
        if source_stage == 'walk':
            self.falls = 0
        elif self.threshold and source_stage == 'turn' and fell:
            self.falls += 1
            if self.falls >= self.threshold:
                self.pending_walk = True

    def next_start(self):
        if self.pending_walk:
            self.pending_walk = False
            self.falls = 0
            return 'walk'
        return 'turn'

    def state(self):
        return dict(falls=self.falls, pending_walk=self.pending_walk)
