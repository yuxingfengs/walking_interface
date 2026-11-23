"""Experimental aggregate-foot-force feedback, physical teacher only.

No center-of-pressure estimate: Webots supplies one 3D force per foot.
The oscillator gate and gains are engineering hypotheses, not B-Human code.
"""
import numpy as np


class ContactExpert:
    def __init__(self, config, stationary_samples):
        self.cfg = config
        samples = np.asarray(stationary_samples[-16:], dtype=float)
        if samples.shape != (16, 2, 3) or not np.isfinite(samples).all():
            raise ValueError('Need 16 finite stationary force samples for calibration')
        med = np.median(samples[:, :, 2], axis=0)
        if np.any(np.abs(med) < config['contact_force_n']):
            raise ValueError('Both feet must be loaded during force calibration')
        self.sign = np.sign(med)
        self.loads = np.maximum(0., samples[-1, :, 2] * self.sign)
        self.support = int(np.argmax(self.loads))
        self.gyro = np.zeros(2)
        self.pending = False
        self.wait_s = 0.
        self.confirmations = 0
        self.switches = 0
        self.timeouts = 0
        self.calibration = dict(raw_median_z=med.tolist(),normal_sign=self.sign.tolist(),
            initial_support=('left','right')[self.support],method='settled double-support local Z',
            limitations='Aggregate normal force only; no CoP or slip measurement')

    def advance(self, phase, dt, frequency, forces):
        forces = np.asarray(forces, dtype=float)
        if forces.shape != (2,3) or not np.isfinite(forces).all():
            raise ValueError('Invalid foot force feedback')
        alpha = self.cfg['filter_alpha']
        self.loads = (1-alpha)*self.loads + alpha*np.maximum(0., forces[:,2]*self.sign)
        step = dt*frequency
        boundary = .5 if phase < .5 else 1.
        if phase+step < boundary and not self.pending:
            return (phase+step)%1.
        # At a half-cycle boundary wait briefly for the opposite foot to load.
        self.pending = True
        other = 1-self.support
        total = float(self.loads.sum())
        ready = (self.loads[other] >= self.cfg['contact_force_n'] and
                 self.loads[other]/max(total,1e-9) >= self.cfg['load_fraction'])
        self.confirmations = self.confirmations+1 if ready else 0
        self.wait_s += dt
        confirmed = self.confirmations >= self.cfg['confirmation_steps']
        timed_out = self.wait_s >= self.cfg['max_switch_delay_s']
        if confirmed or timed_out:
            if confirmed:
                self.support = other
                self.switches += 1
            else:
                self.support = int(np.argmax(self.loads))
                self.timeouts += 1
            self.pending = False
            self.wait_s = 0.
            self.confirmations = 0
            return boundary%1.
        return boundary-1e-7

    def stabilize(self, targets, gyro):
        gyro = np.asarray(gyro,dtype=float)
        if not np.isfinite(gyro).all():
            raise ValueError('Invalid gyro feedback')
        a = self.cfg['filter_alpha']
        self.gyro = (1-a)*self.gyro+a*gyro
        result = np.array(targets,copy=True)
        if self.loads.sum() < self.cfg['contact_force_n']:
            return result
        weights = self.loads/self.loads.sum()
        dp = np.clip(self.cfg['pitch_gyro_gain']*self.gyro[1],
                     -self.cfg['max_pitch_correction'],self.cfg['max_pitch_correction'])
        dr = np.clip(self.cfg['roll_gyro_gain']*self.gyro[0],
                     -self.cfg['max_roll_correction'],self.cfg['max_roll_correction'])
        for foot, weight in enumerate(weights):
            result[foot*6+4] += weight*dp
            result[foot*6+5] += weight*dr
        return result

    def diagnostics(self):
        return dict(load_left_n=float(self.loads[0]),load_right_n=float(self.loads[1]),
                    support=('left','right')[self.support],waiting=self.pending,
                    confirmed_switches=self.switches,switch_timeouts=self.timeouts)
