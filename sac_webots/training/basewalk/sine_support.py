"""Original joint-space sine gait with bounded support-aware turning modulation.

No IK, load-transfer gate, phase freeze or world-pose input. Pressure only
modulates the turning component and distributes bounded gyro corrections.
"""
import math
import numpy as np
from .sine_expert import three_stage_targets
from .contact_expert import ContactExpert
from .core import DEFAULT_POSE,joint_targets


class SineSupport:
    def __init__(self,cfg,force_cfg,samples,joints,lower,upper,action_scale):
        self.cfg=cfg;self.scale=action_scale;self.lower=lower;self.upper=upper
        calibration=ContactExpert(force_cfg,samples)
        self.calibration=calibration.calibration;self.sign=calibration.sign
        self.loads=calibration.loads.copy();self.support=int(np.argmax(self.loads))
        self.q=np.asarray(joints,dtype=float).copy();self.q[[0,6]]=self.q[[0,6]].mean()
        self.phase=0.;self.elapsed=0.;self.steps=0;self.state='step';self.events=[]
        self.gyro=np.zeros(2);self.balance=np.zeros(2);self.modulation=1.
        self.weights=np.array([0.,1.,0.]);self.target_weights=self.weights.copy()
        self.blend_from=self.weights.copy();self.blend_elapsed=0.
        self.confirm=0;self.airborne=0.;self.slew_limited=False;self.action_clipped=False

    def update(self,command,joints,forces,roll_pitch,gyro,dt):
        command=np.asarray(command,dtype=float);forces=np.asarray(forces,dtype=float)
        gyro=np.asarray(gyro,dtype=float);tilt=np.asarray(roll_pitch,dtype=float)
        if command.shape!=(3,) or forces.shape!=(2,3) or gyro.shape!=(2,) or tilt.shape!=(2,):
            raise ValueError('Invalid sine feedback shape')
        if not np.isfinite(np.r_[command,forces.ravel(),gyro,tilt,dt]).all() or dt<=0:
            raise ValueError('Invalid sine feedback value')
        if command[0]<0 or abs(command[1])>1e-8:
            raise ValueError('V27 validates forward/turn/step only')
        if max(abs(tilt))>self.cfg['max_tilt_rad']:raise RuntimeError('Tilt limit exceeded')
        alpha=self.cfg['load_filter_alpha']
        self.loads+=(alpha*(np.maximum(0.,forces[:,2]*self.sign)-self.loads))
        total=float(self.loads.sum());self.airborne=self.airborne+dt if total<self.cfg['contact_force_n'] else 0.
        if self.airborne>=self.cfg['airborne_timeout_s']:raise RuntimeError('Both feet lost contact')
        fractions=self.loads/max(total,1e-9)
        other=1-self.support
        ready=fractions[other]>=self.cfg['support_fraction'] and self.loads[other]>=self.cfg['contact_force_n']
        self.confirm=self.confirm+1 if ready else 0
        if self.confirm>=self.cfg['support_confirmation_steps']:
            self.support=other;self.steps+=1;self.confirm=0;self.events.append('support_switch')
        state='walk' if command[0]>1e-8 else ('turn' if abs(command[2])>1e-8 else 'step')
        requested=np.eye(3)[{'turn':0,'step':1,'walk':2}[state]]
        if not np.array_equal(requested,self.target_weights):
            self.blend_from=self.weights.copy();self.target_weights=requested;self.blend_elapsed=0.
        self.blend_elapsed+=dt
        u=min(self.blend_elapsed/self.cfg['command_blend_seconds'],1.);u=u*u*(3-2*u)
        self.weights=(1-u)*self.blend_from+u*self.target_weights;self.state=state
        self.phase=(self.phase+dt*self.cfg['frequency_hz'])%1.;self.elapsed+=dt
        # The existing sine raises the left knee for sin(phase)>0; right foot
        # is then the expected support. Reduce yaw/differential amplitude when
        # measured loading contradicts that expectation. Never reverse its sign.
        expected=math.sin(2*math.pi*self.phase)
        agreement=np.clip((fractions[1]-fractions[0])*expected,-1.,1.)
        desired=1.-self.cfg['support_modulation_strength']*max(0.,-agreement)
        self.modulation+=(1-math.exp(-dt/self.cfg['support_modulation_tau_s']))*(desired-self.modulation)
        wz=float(np.clip(command[2],-self.cfg['max_yaw_command'],self.cfg['max_yaw_command']))
        # Preserve the old turn command while smoothly leaving Turn for Step.
        if abs(wz)>1e-8:self.last_turn_wz=wz
        turn_wz=getattr(self,'last_turn_wz',0.)*self.modulation
        vx=min(command[0]/self.cfg['walk_speed_reference'],1.)
        if command[0]>1e-8:self.last_walk_scale=vx
        params=[self.cfg['turn_differential'],self.cfg['hip_yaw_amplitude'],
                self.cfg['walk_forward']*getattr(self,'last_walk_scale',1.),self.cfg['hold_knee_lift']]
        modes=('turn','hold','walk')
        targets=[three_stage_targets(self.phase,self.elapsed,params,stage,
            [0,0,turn_wz] if stage=='turn' else [command[0],0,wz*self.modulation] if stage=='walk' else [0,0,0],
            self.cfg['yaw_rate_reference']) for stage in modes]
        target=sum(w*q for w,q in zip(self.weights,targets))
        self.gyro+=self.cfg['gyro_filter_alpha']*(gyro-self.gyro)
        self.balance=np.clip(self.gyro*np.array([self.cfg['roll_gyro_gain'],self.cfg['pitch_gyro_gain']]),
                             -self.cfg['max_balance_rad'],self.cfg['max_balance_rad'])
        for side in range(2):
            target[6*side+4]+=fractions[side]*self.balance[1]
            target[6*side+5]+=fractions[side]*self.balance[0]
        bounded,_=joint_targets((target-DEFAULT_POSE)/self.scale,self.scale,self.lower,self.upper)
        self.action_clipped=not np.allclose(bounded,target,atol=1e-6)
        self.slew_limited=bool(max(abs(bounded-self.q))>self.cfg['max_joint_step_rad'])
        self.q+=np.clip(bounded-self.q,-self.cfg['max_joint_step_rad'],self.cfg['max_joint_step_rad'])
        return self.q.copy()

    def diagnostics(self):
        return dict(state=self.state,support=self.support,steps=self.steps,phase=self.phase,
            loads=self.loads.tolist(),mode_weights=self.weights.tolist(),turn_modulation=self.modulation,
            balance=self.balance.tolist(),action_clipped=self.action_clipped,slew_limited=self.slew_limited)
