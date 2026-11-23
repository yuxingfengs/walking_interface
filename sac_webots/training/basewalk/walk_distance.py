"""Walk-only horizontal path length, stopping at a gait-cycle boundary."""
import numpy as np

class WalkDistance:
    def __init__(self,target):
        if target<=0:raise ValueError('Walk distance must be positive')
        self.target=float(target)
        self.reset()
    def reset(self):
        self.distance=0.
    def advance(self,stage,before,after,phase_before,phase_after):
        if stage!='walk':return False
        self.distance+=float(np.linalg.norm(np.asarray(after)[:2]-np.asarray(before)[:2]))
        return self.distance>=self.target and phase_after<phase_before
