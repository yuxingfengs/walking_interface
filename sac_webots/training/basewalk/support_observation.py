"""Causal normal-load difference. No history feature or hidden zero fallback."""
import numpy as np


class SupportObservation:
    def __init__(self, config, settled_samples):
        self.cfg=config
        samples=np.asarray(settled_samples[-16:],dtype=float)
        if samples.shape!=(16,2,3) or not np.isfinite(samples).all():
            raise ValueError('Support State requires 16 settled LFsr/RFsr samples')
        median=np.median(samples[:,:,2],axis=0)
        if np.any(abs(median)<1e-6):raise ValueError('Cannot calibrate support State with unloaded foot')
        self.sign=np.sign(median)
        self.loads=np.maximum(0.,samples[-1,:,2]*self.sign)
        if not 0<config['filter_alpha']<=1 or config['epsilon']<=0:
            raise ValueError('Invalid support filter/epsilon')

    def advance(self, forces):
        forces=np.asarray(forces,dtype=float)
        if forces.shape!=(2,3) or not np.isfinite(forces).all():raise ValueError('Invalid LFsr/RFsr feedback')
        self.loads+=self.cfg['filter_alpha']*(np.maximum(0.,forces[:,2]*self.sign)-self.loads)

    @property
    def value(self):
        return float((self.loads[0]-self.loads[1])/(self.loads.sum()+self.cfg['epsilon']))
