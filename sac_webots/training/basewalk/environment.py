"""FiLM experiment: IMU/encoders plus explicitly privileged Supervisor yaw."""
import math
import numpy as np
from .training_design import TrackingAverage, turn_weight_bucket, turn_stand_bucket
from .heading_control import (heading_command, turn_then_forward_command,
                              turn_then_forward_phase, blended_turn_walk_command,
                              heading_blend_weights)
from .heading_estimator import HeadingEstimator
from .network_inputs import SHARED_VELOCITY_EMA, SHARED_GYRO_EMA, SHARED_YAW_ONLY, uses_body_gyro_observation
from .stationary_reward import NetSpeedWindow
from .report_diagnostics import TruthTracker, JointTruthTracker, capture_joint_truth, diagnostic_row
from .core import (JOINTS, DEFAULT_POSE, MINIMUM, MAXIMUM, Commands,
                   joint_targets, projected_gravity, observation, reward,
                   progress_reward, staged_navigation_reward,
                   blended_navigation_reward)
from .webots_devices import WebotsDevices


class WebotsBaseWalk(WebotsDevices):
    def __init__(self, supervisor, config):
        self.robot, self.config = supervisor, config
        self.continuous_turn = 'yaw_ramp' in config.get('training', {})
        self.actor_body_gyro = uses_body_gyro_observation(config)
        self.cfg = dict(config['environment'])
        if not config.get('support_observation') or not config.get('contact_expert',{}).get('enabled'):
            raise ValueError('V28 requires support Observation and LFsr/RFsr enabled')
        if not config.get('physical_gait',{}).get('imu_full_axes'):
            raise ValueError('V28 requires full-axis IMU')
        self.ms = config['webots']['control_time_step_ms']
        self.dt = self.ms/1000.
        self.reward_tracking = TrackingAverage(config.get('velocity_tracking', {}).get('window_samples', 1))
        from .reward_history import CommandVelocityWindow
        self.reward_velocity_window = CommandVelocityWindow(config.get('velocity_tracking', {}).get('window_samples', 1))
        if config['reward'].get('biped_tracking',{}).get('revision') == 'velocity_widths_variants_v1':
            from .reward_velocity import RewardVelocity
            self.reward_velocity_filter = RewardVelocity(config['reward']['biped_tracking']['velocity_source'],self.cfg['velocity_filter'], continuous_turn=self.continuous_turn)
        yaw_tau = config['reward'].get('biped_tracking', {}).get('turn_yaw_lowpass_tau_s')
        self.turn_heading_window = None
        heading_seconds = config['reward'].get('biped_tracking', {}).get('turn_heading_window_seconds')
        if heading_seconds not in (None, 0):
            from .turn_heading_window import TurnHeadingWindow
            self.turn_heading_window = TurnHeadingWindow(heading_seconds)
        self.turn_yaw_filter = None
        if yaw_tau is not None:
            from .reward_velocity import RewardVelocity
            self.turn_yaw_filter = RewardVelocity('command_ema', -math.expm1(-self.dt / yaw_tau), continuous_turn=self.continuous_turn)
        self._initialize_devices(supervisor, config)
        self.has_reset = False
        self.last_action = np.zeros(12, np.float32)
        self.phase = 0.
        self.elapsed = 0.
        self.previous_pose = None
        self.heading_estimator = HeadingEstimator()
        self.yaw = 0.
        self.truth_yaw = 0.
        self.turn_origin = None
        self.walking_phase = False
        self.motion_stage = 'turn'
        self.motion_stage_started_at = 0.
        self.turn_weight, self.walk_weight = 1., 0.

    def _command(self):
        if hasattr(self, 'expert_fixed_command'):
            result=np.asarray(self.expert_fixed_command,dtype=np.float32).copy()
            self.walk_weight=float(np.linalg.norm(result[:2]) > 1e-8);self.turn_weight=1.-self.walk_weight
            self.walking_phase=bool(self.walk_weight)
            if self.walk_weight:self.turn_origin=None
            elif self.turn_origin is None:self.turn_origin=self.position[:2].copy()
            return result
        if self.config.get('turn_stand_walk'):
            result=self.stages.command(math.atan2(math.sin(self.heading_reference-self.yaw),math.cos(self.heading_reference-self.yaw)),float(self.commands.current[0]))
            self.walking_phase=self.stages.stage=='walk'
            self.walk_weight=float(self.walking_phase);self.turn_weight=1.-self.walk_weight
            if self.walking_phase:self.turn_origin=None
            elif self.turn_origin is None:self.turn_origin=self.position[:2].copy()
            return result
        raise ValueError('V25 requires turn_stand_walk configuration')

    def _motion_mode(self, command):
        if np.linalg.norm(command[:2]) > 1e-8:
            return 2
        return 0 if self.continuous_turn or abs(command[2]) > 1e-8 else 1

    def _observation(self):
        heading_error = math.atan2(math.sin(self.heading_reference-self.yaw),
                                   math.cos(self.heading_reference-self.yaw))
        command = self._command()
        state = observation(self.gravity, self.gyro_value, command, self.phase,
                            self.q, self.qvel, self.last_action, self.cfg,
                            yaw_rate=float(self.velocity[2]), body_velocity=self.velocity[:2], heading_error=heading_error,
                            support_load_difference=self.support_feedback.value)
        if self.actor_body_gyro:
            gyro_z = self.actor_gyro_z * self.cfg['yaw_rate_scale']
            state = np.append(state, np.float32(gyro_z))
        if self.config.get('actor_observation') in (SHARED_VELOCITY_EMA, SHARED_GYRO_EMA, SHARED_YAW_ONLY):
            mode = self._motion_mode(command)
            key = self.reward_velocity_filter.command_key(command, mode)
            valid = self.reward_velocity_filter.key == key
            planar = (self.reward_velocity_filter.value[:2]
                      if valid and self.config.get("actor_observation") != SHARED_YAW_ONLY else (0., 0.))
            yaw_filter = self.turn_yaw_filter
            yaw_ema = yaw_filter.value[2] if yaw_filter.key == key else 0.
            state = np.append(state, np.asarray(
                [yaw_ema * self.cfg['yaw_rate_scale'], valid, *planar], dtype=np.float32))
        return state

    def reset(self, command=None, randomize=False):
        if self.turn_heading_window is not None:
            self.turn_heading_window.reset()
        if self.cfg.get('reset_mode', 'node') == 'world_reload':
            if self.has_reset:
                raise RuntimeError('world_reload requires a fresh controller before reset')
        else:
            self.node.loadState('basewalk_initial')
        self.has_reset = True
        if hasattr(self,'support_feedback'):del self.support_feedback
        self.foot_history=[]
        self.node.resetPhysics()
        for name in ('LShoulderPitch', 'RShoulderPitch'):
            self.robot.getDevice(name).setPosition(1.1)
        for motor, angle in zip(self.motors, DEFAULT_POSE):
            motor.setPosition(float(angle))
        for _ in range(self.cfg['settle_steps']):
            self._tick()
        from .support_observation import SupportObservation
        self.support_feedback=SupportObservation(self.config['support_observation'],self.foot_history)
        self.previous_pose = None
        self.heading_estimator.reset(0., self.cfg.get('yaw_rate_bias', 0.))
        self.diagnostic_truth = TruthTracker()
        self.joint_truth_tracker = JointTruthTracker()
        self._feedback()
        if self.cfg.get('yaw_source') == 'integrated_yaw_rate':
            self.heading_estimator.reset(0., self.cfg.get('yaw_rate_bias', 0.))
            self.yaw = 0.
        if self.height < self.cfg['fall_height'] or -self.gravity[2] < math.cos(self.cfg['fall_tilt_rad']):
            raise RuntimeError(f"Default reset posture is not upright: height={self.height:.4f}, gravity={self.gravity.tolist()}, joint_angles={self.q.tolist() if hasattr(self, 'q') else self._positions().tolist()}")
        self.q, self.qvel = self._positions(), np.zeros(12, np.float32)
        self.last_action = np.zeros(12, np.float32)
        self.filtered_velocity = self.velocity.copy()
        self.reward_tracking.reset()
        self.reward_velocity_window.reset()
        if hasattr(self,'reward_velocity_filter'):self.reward_velocity_filter.reset()
        if self.turn_yaw_filter is not None:
            self.turn_yaw_filter.reset()
        from .stationary_reward import PerStepPlanarSpeed
        if self.config['stationary_reward'].get('speed_measure') != 'per_step_planar_displacement':
            raise ValueError('V30 requires per-step planar displacement speed')
        # Keep the adapter name for archive consumers; there is no rolling window.
        self.net_speed_window = PerStepPlanarSpeed(self.dt, self.position)
        # Stage stability is independent of the instantaneous Reward speed.
        self.stand_speed_window = NetSpeedWindow(
            self.config['turn_stand_walk']['stand_speed_window_samples'], self.dt)
        self.phase = float(self.cfg.get('initial_gait_phase', 0.)) % 1.
        self.elapsed = 0.
        self.turn_origin = None
        self.walking_phase = False
        self.motion_stage = 'turn'
        self.motion_stage_started_at = 0.
        self.turn_weight, self.walk_weight = 1., 0.
        self.commands.reset(command)
        self.commands.advance(self.dt, randomize)
        self.start_position = self.position.copy()
        target_deg = getattr(self, 'direction_target_deg', None)
        self.heading_reference = (self.yaw + math.radians(target_deg)
                                  if self.cfg.get('navigation_mode') in ('turn_then_forward', 'turn_hold_forward', 'blended_turn_walk', 'turn_only') and target_deg is not None
                                  else self.yaw)
        from .turn_stand_walk import Stages, wrap
        if self.config.get('fixed_cycle_commands',{}).get('enabled'):
            from .fixed_cycle import FixedCycleStages
            persistent = getattr(self, 'persistent_fixed_cycle_stages', None)
            self.stages = (persistent if persistent is not None else
                           FixedCycleStages(self.config, getattr(self, 'fixed_cycle_start_stage', 'turn')))
        else:
            self.stages=Stages(self.config,wrap(self.heading_reference-self.yaw))
        return self._observation()

    def step(self, candidate, randomize=False):
        landing_gate = landing_error = None
        if self.config.get('landing_alignment'):
            import torch
            from .landing_projection import project_landing_actions
            with torch.no_grad():
                projected, gate, error = project_landing_actions(
                    torch.as_tensor(self._observation()), torch.as_tensor(candidate),
                    self.config, diagnostics=True)
            landing_gate = gate.tolist()
            landing_error = torch.rad2deg(error).tolist()
            candidate = projected.numpy()
        qtarget, executed = joint_targets(candidate, self.cfg['action_scale'], self.lower, self.upper)
        for motor, angle in zip(self.motors, qtarget):
            motor.setPosition(float(angle))
        phase_before=float(self.phase)
        old_q = self.q.copy()
        old_position = self.position.copy()
        old_heading = self.yaw
        previous_truth_yaw = self.truth_yaw
        previous_body_gyro_z = getattr(self, 'actor_gyro_z', None)
        previous_gravity = self.gravity  # _feedback replaces this sample after the action.
        active_command = self._command()  # Exactly the command present in the source Observation.
        source_mode = self._motion_mode(active_command)
        source_stage = ('turn', 'stand', 'walk')[source_mode]
        source_turn_weight = float(getattr(self, 'turn_weight', 1.))
        source_walk_weight = float(getattr(self, 'walk_weight', 0.))
        reverse_foot_yaw = self.config['reward'].get('biped_tracking', {}).get('turn_reverse_foot_yaw_weight')
        if source_mode == 0 and reverse_foot_yaw:
            from .stance_foot_yaw import relative_headings, reverse_cost
            if not hasattr(self, 'truth_foot_nodes'):
                self.truth_foot_nodes = [self.node.getFromProtoDef(name) for name in ('LFsr', 'RFsr')]
            before_contact = np.array([bool(node.getContactPoints(False)) for node in self.truth_foot_nodes])
            before_heading = relative_headings(previous_gravity, old_q)
            before_world_yaw = self.truth_yaw
        self._tick()
        self.elapsed += self.dt
        self.q = self._positions()
        self.qvel = (self.q-old_q)/self.dt
        self._feedback()
        if self.config.get('stationary_reward'):
            self.net_speed_window.advance(self.position)
            self.stand_speed_window.advance(self.position)
        alpha = self.cfg['velocity_filter']
        self.filtered_velocity += alpha*(self.velocity-self.filtered_velocity)
        turn_heading_error = None
        if self.turn_heading_window is not None:
            yaw_change = self.truth_yaw - previous_truth_yaw
            delta = math.atan2(math.sin(yaw_change), math.cos(yaw_change))
            turn_heading_error = self.turn_heading_window.advance(
                float(active_command[2]) if source_mode == 0 else 0.0, delta, self.dt)
        foot_yaw_cost = None
        if source_mode == 0 and reverse_foot_yaw:
            after_contact = np.array([bool(node.getContactPoints(False)) for node in self.truth_foot_nodes])
            foot_yaw_cost = reverse_cost(before_heading, relative_headings(self.gravity, self.q),
                                         self.truth_yaw-before_world_yaw, before_contact, after_contact,
                                         self.dt, active_command[2])
        fell = self.height < self.cfg['fall_height'] or -self.gravity[2] < math.cos(self.cfg['fall_tilt_rad'])
        reward_command=active_command.copy();reward_velocity=self.filtered_velocity.copy()
        blended=True;stand=source_stage=='stand';walking=source_stage=='walk'
        from .turn_stand_walk import reward_inputs, motion_reward
        saved_reward_inputs=reward_inputs(old_position,self.position,self.heading_reference,old_heading,self.yaw,
            self.velocity[2],active_command[2],self.net_speed_window.speed,self.net_speed_window.ready,self.dt)
        if self.config['reward'].get('biped_tracking'):
            from .reward_dispatch import (
                motion_reward as biped_motion_reward, uses_body_gyro_reward, uses_heading_yaw_tracking,
            )
            tracking_velocity = (self.filtered_velocity if self.config['reward']['biped_tracking'].get('velocity_source') == 'filtered' else self.velocity)
            if self.config['reward']['biped_tracking'].get('velocity_source') == 'causal_mean':
                tracking_velocity = self.reward_velocity_window.advance(self.velocity, active_command)
            if hasattr(self,'reward_velocity_filter'):
                tracking_velocity = self.reward_velocity_filter.advance(
                    np.array([*self.velocity[:2], self.actor_gyro_z], dtype=np.float32)
                    if self.config['reward']['biped_tracking'].get('yaw_ema_use_body_gyro') else self.velocity,
                    active_command, source_mode)
            yaw_sample = self.velocity
            if self.config['reward']['biped_tracking'].get('yaw_ema_use_body_gyro'):
                yaw_sample = np.array([*self.velocity[:2], self.actor_gyro_z], dtype=np.float32)
            if self.turn_yaw_filter is not None:
                if source_mode == 0:
                    tracking_velocity[2] = self.turn_yaw_filter.advance(
                        yaw_sample, active_command, source_mode)[2]
                else:
                    self.turn_yaw_filter.reset()
            if source_mode == 0 and self.config['reward']['biped_tracking'].get('turn_yaw_use_raw_gyro'):
                tracking_velocity = tracking_velocity.copy()
                tracking_velocity[2] = self.actor_gyro_z
            value,terms=biped_motion_reward(active_command,tracking_velocity,self.gyro_value,
                self.gravity,source_mode,self.config,actual_yaw_rate=float(self.velocity[2]),
                previous_gravity=previous_gravity,body_gyro_z=getattr(self,'actor_gyro_z',None),
                action=executed.astype(np.float32),previous_action=self.last_action,joint_position=self.q,
                reverse_foot_yaw_cost=foot_yaw_cost,turn_heading_error=turn_heading_error,
                previous_body_gyro_z=previous_body_gyro_z,reward_dt=self.dt,
                gait_phase=phase_before)
            reward_velocity=tracking_velocity.copy()
            if source_mode == 0:
                if 'world_gyro_z' in terms:
                    reward_velocity[2] = terms['world_gyro_z']
                elif uses_heading_yaw_tracking(self.config) and self.turn_yaw_filter is None:
                    reward_velocity[2] = self.velocity[2]
                elif uses_body_gyro_reward(self.config):
                    reward_velocity[2] = self.actor_gyro_z
        else:
            value,terms=motion_reward(source_mode,saved_reward_inputs,self.config,fell)
        terms['fall']=-self.config['reward']['fall_penalty'] if fell else 0.
        value+=terms['fall']
        terms.update(turn_weight=source_turn_weight,walk_weight=source_walk_weight)
        if not hasattr(self,'expert_fixed_command') and not self.config.get('fixed_cycle_commands',{}).get('enabled'):
            self.stages.advance(saved_reward_inputs[1],self.stand_speed_window.speed,self.velocity[2],self.stand_speed_window.ready,fell)
        self.last_action = executed.astype(np.float32)
        if (self.cfg.get('phase_runs_at_zero_command', False) or
                np.linalg.norm(active_command) > 1e-5):
            if hasattr(self, 'contact_controller'):
                self.phase = self.contact_controller.advance(self.phase, self.dt,
                    self.cfg['gait_frequency_hz'], self.foot_forces)
            else:
                self.phase = (self.phase+self.dt*self.cfg['gait_frequency_hz']) % 1.
        if self.config.get('fixed_cycle_commands',{}).get('enabled') and not hasattr(self,'expert_fixed_command'):
            persistent = getattr(self, 'persistent_fixed_cycle_stages', None)
            self.stages.advance_phase(phase_before, float(self.phase), fell=fell and persistent is not None)
        # Score the transition against its source command before resampling.
        self.commands.advance(self.dt, randomize)
        truncated = not self.cfg.get('disable_time_limit', False) and self.elapsed + 1e-8 >= self.cfg['episode_seconds']
        stage=source_stage;replay_mode=source_mode;replay_bucket=source_stage
        info = dict(support_load_difference=self.support_feedback.value,foot_normal_loads=self.support_feedback.loads.tolist(),reward_inputs=saved_reward_inputs,hold_steps=self.stages.count,next_stage=self.stages.stage,cycle_actual=[float(self.position[0]-old_position[0]),float(self.position[1]-old_position[1]),math.atan2(math.sin(self.yaw-old_heading),math.cos(self.yaw-old_heading))],
                    cycle_target=[self.dt*(math.cos(old_heading)*active_command[0]-math.sin(old_heading)*active_command[1]),self.dt*(math.sin(old_heading)*active_command[0]+math.cos(old_heading)*active_command[1]),self.dt*float(active_command[2])],
                    cycle_dt=self.dt, fell=bool(fell), elapsed=self.elapsed, position=self.position.tolist(), yaw=self.yaw,
                    heading_error=math.atan2(math.sin(self.heading_reference-self.yaw), math.cos(self.heading_reference-self.yaw)),
                    truth_yaw=getattr(self, 'truth_yaw', self.yaw),
                    actor_yaw_rate=float(self.velocity[2]),
                    motion_phase=stage,
                    turn_weight=source_turn_weight, walk_weight=source_walk_weight,
                    replay_mode=replay_mode, replay_bucket=replay_bucket,
                    velocity=self.filtered_velocity.tolist(), command=active_command.tolist(),
                    target_command=[*self.commands.target[:2].tolist(), float(active_command[2])], executed_action=self.last_action.copy(),
                    reward_terms=terms, reward_velocity=reward_velocity.tolist(), reward_command=reward_command.tolist(),
                    actual_velocity=self.velocity.tolist())
        if turn_heading_error is not None:
            info['turn_heading_error'] = turn_heading_error
        if landing_gate is not None:
            info['landing_gate'] = landing_gate
            info['landing_target_yaw_error_deg'] = landing_error
        info['report_diagnostics'] = diagnostic_row(self)
        if self.config['reward'].get('biped_tracking', {}).get('velocity_source') == 'causal_mean':
            info['reward_history_count'] = self.reward_velocity_window.count
        if hasattr(self,'reward_velocity_filter'):
            info['reward_history_count'] = self.reward_velocity_filter.count
        return self._observation(), value, bool(fell), truncated, info
