"""Fixed Turn/Walk schedule: a completed cycle or fall counts once."""
from pathlib import Path
import runpy
import math
import numpy as np


def online_stages(cfg, state=None):
    if 'source_hashes' in cfg:
        recipe = Path(cfg['reward_module_path']).with_name('training.py')
    else:
        folder = Path(cfg['project_layout']['experiment_folder'])
        recipe = (folder/'training/legacy_fixed.py' if folder == Path(cfg['project_layout']['project_root'])
                  else folder/'training.py')
    schedule = runpy.run_path(str(recipe))['command_schedule']
    return schedule(cfg, state)


def walk_heading(index):
    # Bit reversal covers opposite directions first, then bisects all remaining gaps.
    fraction, weight = 0., .5
    while index:
        index, bit = divmod(index, 2)
        fraction += bit * weight
        weight *= .5
    return 2 * math.pi * fraction


class FixedCycleStages:
    def __init__(self,cfg,start_stage="turn", *, cycle_limit=None, state=None,
                 turn_only=False, fixed_turn_sign=None):
        if fixed_turn_sign is not None and fixed_turn_sign not in (-1., 1.):
            raise ValueError("Fixed Turn direction requires a sign of ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â±1")
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
        self.order=tuple(cfg['fixed_cycle_commands']['order'])
        if self.order!=('turn','walk'):raise ValueError('Expected Turn-Walk')
        if turn_only:
            self.order = ('turn',)
        if start_stage not in self.order:raise ValueError('Invalid fixed-cycle initial stage')
        self.index=self.order.index(start_stage);self.stage=start_stage;self.cycles=0;self.count=0;self.turn_sign=1. if fixed_turn_sign is None else fixed_turn_sign
        self.omni = cfg['fixed_cycle_commands'].get('walk_headings') == 'dyadic_circle'
        self.walk_index = 0
        if self.omni:
            speed = cfg['reward']['walk_speed_reference']
            if not 0 < speed <= min(min(-lo, hi) for lo, hi in cfg['environment']['command_ranges'][:2]):
                raise ValueError('Omnidirectional walk speed exceeds the planar command range')
        if state is not None:
            if self.omni:
                self.walk_index = state['walk_index']
                if type(self.walk_index) is not int or self.walk_index < 0:
                    raise ValueError('Invalid Walk heading index')
            stage, cycles, turn_sign = state['stage'], state['cycles'], state['turn_sign']
            if stage not in self.order or type(cycles) is not int or not 0 <= cycles < self.limit or turn_sign not in (-1., 1.):
                raise ValueError('Invalid fixed-cycle continuation state')
            if fixed_turn_sign is not None and turn_sign != fixed_turn_sign:
                raise ValueError("Continuation Turn direction differs from configured direction")
            self.index = self.order.index(stage)
            self.stage, self.cycles, self.turn_sign = stage, cycles, turn_sign

    def state(self):
        state = dict(stage=self.stage, cycles=self.cycles, turn_sign=self.turn_sign)
        if self.omni:
            state['walk_index'] = self.walk_index
        return state
    def command(self,error,speed):
        if self.stage == 'walk':
            heading = walk_heading(self.walk_index) if self.omni else 0.
            speed = self.cfg['reward']['walk_speed_reference']
            return np.array([speed*math.cos(heading), speed*math.sin(heading), 0], np.float32)
        if self.stage=='turn':return np.array([0,0,self.turn_sign*self.turn_rate],np.float32)
        return np.zeros(3,np.float32)
    def advance_phase(self,before,after, *, fell=False):
        if not fell and after >= before:
            return
        self.cycles+=1
        if self.cycles<self.limit:return
        self.cycles=0
        if self.omni and self.stage == 'walk':
            self.walk_index += 1
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
