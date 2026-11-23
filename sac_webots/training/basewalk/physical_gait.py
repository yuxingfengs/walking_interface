"""Independent contact-feedback foot-pose teacher; no global pose input.

Conservative engineering controller inspired by traditional NAO walking, not a
port of B-Human. All displacements below are in a locally reanchored torso frame.
"""
import numpy as np
from .core import DEFAULT_POSE
from .contact_expert import ContactExpert
from .footstep_ik import forward, rotation, solve


def smooth(u):
    u=float(np.clip(u,0.,1.))
    return u*u*u*(10.+u*(-15.+6.*u))


class PhysicalGait:
    def __init__(self, cfg, force_cfg, samples, joints, lower, upper, action_scale):
        self.cfg=cfg
        calibration=ContactExpert(force_cfg,samples)
        self.sign=calibration.sign
        self.calibration=calibration.calibration
        self.loads=calibration.loads.copy()
        self.q=np.asarray(joints,dtype=float).copy()
        self.q[[0,6]]=np.mean(self.q[[0,6]])
        self.lower=np.maximum(lower,DEFAULT_POSE-action_scale)
        self.upper=np.minimum(upper,DEFAULT_POSE+action_scale)
        self.support=int(np.argmax(self.loads))
        self.steps=0;self.phase=0.;self.t=0.;self.confirm=0
        self.command_filtered=np.zeros(3);self.command_active=np.zeros(3)
        self.balance=np.zeros(2);self.residual={};self.state='transfer'
        self.events=[];self.lost=0;self.obstruction=0;self.touched=False
        self._reanchor(self.q)

    def _reanchor(self, measured):
        self.feet=forward(measured)
        self.body=np.zeros(3);self.body_yaw=0.
        self.center=self.feet[:,:3,3].mean(axis=0)
        self.half_width=abs(self.feet[0,1,3]-self.feet[1,1,3])/2
        if not .025<=self.half_width<=.09:
            raise RuntimeError('Invalid measured stance width')
        self.shift=np.zeros(3)
        self.shift[1]=self.cfg['initial_shift_fraction']*self.feet[self.support,1,3]
        self.t=0.;self.confirm=0;self.state='transfer';self.touched=False

    def _start_swing(self):
        self.state='swing';self.t=0.;self.confirm=0
        self.command_active=self.command_filtered.copy()
        other=1-self.support
        self.swing_from=self.feet[other].copy()
        duration=self.cfg['shift_seconds']+self.cfg['swing_seconds']
        yaw=float(np.clip(self.command_active[2]*duration,-self.cfg['max_step_yaw_rad'],self.cfg['max_step_yaw_rad']))
        # Each alternating foot lands ahead of its old location by two step
        # advances. Rotate both position and sole orientation, not a hip angle.
        advance=np.clip(self.command_active[:2]*duration,
                        [-self.cfg['max_step_forward_m'],-self.cfg['max_step_lateral_m']],
                        [self.cfg['max_step_forward_m'],self.cfg['max_step_lateral_m']])
        center=self.center.copy();center[:2]+=2*advance
        offset=np.array([0.,self.half_width*(1 if other==0 else -1),0.])
        self.swing_to=self.swing_from.copy()
        self.swing_to[:3,3]=center+rotation((0,0,1),yaw)@offset
        self.swing_to[:3,:3]=rotation((0,0,1),yaw)
        self.swing_yaw_from=float(np.arctan2(self.swing_from[1,0],self.swing_from[0,0]))
        self.yaw_to=yaw/2
        self.foot_yaw_to=yaw
        self.body_from=self.body.copy()
        self.body_to=self.body.copy();self.body_to[0]+=advance[0]
        self.events.append('swing_started')

    def update(self, command, joints, forces, roll_pitch, gyro, dt):
        """Exactly once BEFORE issuing each actuator target."""
        command=np.asarray(command,dtype=float);joints=np.asarray(joints,dtype=float)
        forces=np.asarray(forces,dtype=float);rp=np.asarray(roll_pitch,dtype=float)
        gyro=np.asarray(gyro,dtype=float)
        if command.shape!=(3,) or joints.shape!=(12,) or forces.shape!=(2,3) or rp.shape!=(2,) or gyro.shape!=(2,):
            raise ValueError('Invalid controller feedback shape')
        if not np.isfinite(np.r_[command,joints,forces.ravel(),rp,gyro,dt]).all() or dt<=0:
            raise ValueError('Invalid controller feedback value')
        if np.max(abs(rp))>self.cfg['max_tilt_rad']:
            raise RuntimeError('Tilt limit exceeded')
        a=self.cfg['force_filter_alpha']
        self.loads=(1-a)*self.loads+a*np.maximum(0.,forces[:,2]*self.sign)
        total=float(self.loads.sum())
        if total<self.cfg['contact_force_n']:
            raise RuntimeError('Both feet lost contact')
        rate=np.asarray(self.cfg['command_acceleration'])*dt
        requested=np.clip(command,-np.asarray(self.cfg['command_limits']),self.cfg['command_limits'])
        self.command_filtered+=np.clip(requested-self.command_filtered,-rate,rate)
        correction=np.clip(-self.cfg['tilt_gain']*rp-self.cfg['gyro_gain_s']*gyro,
                           -self.cfg['max_balance_angle_rad'],self.cfg['max_balance_angle_rad'])
        self.balance+=self.cfg['balance_filter_alpha']*(correction-self.balance)
        self.t+=dt;other=1-self.support
        if self.state=='transfer':
            ratio=float(self.loads[self.support]/total)
            if self.t>=self.cfg['shift_seconds']:
                delta=self.cfg['load_shift_gain_m_s']*(self.cfg['support_fraction']-ratio)*dt
                self.shift[1]+=np.sign(self.feet[self.support,1,3])*max(delta,0.)
                self.shift[1]=float(np.clip(self.shift[1],-self.cfg['maximum_shift_m'],self.cfg['maximum_shift_m']))
            self.body=self.shift*smooth(self.t/self.cfg['shift_seconds'])
            ready=ratio>=self.cfg['support_fraction'] and self.loads[self.support]>=self.cfg['contact_force_n']
            self.confirm=self.confirm+1 if ready else 0
            if self.t>=self.cfg['shift_seconds'] and self.confirm>=self.cfg['confirmation_steps']:
                self._start_swing()
            elif self.t>self.cfg['shift_seconds']+self.cfg['transfer_timeout_s']:
                raise RuntimeError('Transfer timeout: measured support never unloaded opposite foot')
        elif self.state=='swing':
            self.lost=self.lost+1 if self.loads[self.support]<self.cfg['contact_force_n'] else 0
            if self.lost>=2:raise RuntimeError('Support foot lost during swing')
            u=min(self.t/self.cfg['swing_seconds'],1.)
            # Unexpected contact during ascent/mid-swing is a controlled rejection.
            self.obstruction=self.obstruction+1 if .25<u<.65 and self.loads[other]/total>.3 else 0
            if self.obstruction>=3:raise RuntimeError('Swing obstruction: foot did not clear floor')
            b=smooth(u)
            self.feet[other,:3,3]=(1-b)*self.swing_from[:3,3]+b*self.swing_to[:3,3]
            self.feet[other,2,3]+=self.cfg['lift_m']*64*u**3*(1-u)**3
            yaw=(1-b)*self.swing_yaw_from+b*self.foot_yaw_to
            self.feet[other,:3,:3]=rotation((0,0,1),yaw)
            self.body=(1-b)*self.body_from+b*self.body_to;self.body_yaw=b*self.yaw_to
            contact=self.loads[other]>=self.cfg['contact_force_n']
            self.confirm=self.confirm+1 if u>=.8 and contact else 0
            if self.confirm>=self.cfg['confirmation_steps'] or u>=1.:
                self.state='landing';self.t=0.;self.confirm=0
                self.landing_z=float(self.feet[other,2,3]);self.events.append('landing_started')
        elif self.state=='landing':
            contact=self.loads[other]>=self.cfg['contact_force_n']
            self.confirm=self.confirm+1 if contact else 0
            if not contact:
                self.feet[other,2,3]=self.landing_z-min(self.cfg['maximum_landing_search_m'],self.cfg['landing_search_m_s']*self.t)
            if self.confirm>=self.cfg['confirmation_steps']:
                self.support=other;self.steps+=1;self.events.append('touchdown_confirmed')
                # Reconstruct achieved geometry from encoders at the boundary;
                # planned contact positions cannot drift indefinitely.
                self._reanchor(joints)
            elif self.t>self.cfg['landing_timeout_s']:
                raise RuntimeError('Touchdown timeout')
        self.phase=((self.steps%2)+({'transfer':.2,'swing':.2+.7*min(self.t/self.cfg['swing_seconds'],1.),'landing':.95}[self.state]))/2
        r=rotation((0,0,1),self.body_yaw)@rotation((1,0,0),self.balance[0])@rotation((0,1,0),self.balance[1])
        targets=self.feet.copy()
        for f in targets:
            f[:3,3]=r.T@(f[:3,3]-self.body)
            f[:3,:3]=r.T@f[:3,:3]
        q,self.residual=solve(targets,self.q,self.lower,self.upper,self.cfg['max_joint_step_rad'])
        if self.residual['position_error_m']>self.cfg['ik_position_tolerance_m'] or self.residual['orientation_error_rad']>self.cfg['ik_orientation_tolerance_rad']:
            raise RuntimeError('Unreachable foot pose: '+str(self.residual))
        self.q=q
        return q.copy()

    def diagnostics(self):
        return dict(state=self.state,support=self.support,steps=self.steps,phase=self.phase,
            loads=self.loads.tolist(),filtered_command=self.command_filtered.tolist(),
            active_command=self.command_active.tolist(),body=self.body.tolist(),balance=self.balance.tolist(),
            feet=self.feet.tolist(),**self.residual)
