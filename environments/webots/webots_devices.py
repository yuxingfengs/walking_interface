"""NAO motor and sensor setup, simulation ticks and Supervisor feedback."""
import math
import numpy as np
from .core import Commands, JOINTS, MAXIMUM, MINIMUM, projected_gravity
from .report_diagnostics import TruthTracker, capture_joint_truth


class WebotsDevices:
    def _initialize_devices(self, supervisor, config):
        self.node = supervisor.getFromDef('NAO')
        self.commands = Commands(self.cfg, np.random.default_rng(config['seed']))
        self.motors = [supervisor.getDevice(j) for j in JOINTS]
        self.sensors = [supervisor.getDevice(j+'S') for j in JOINTS]
        self.lower = np.maximum(MINIMUM, [m.getMinPosition() for m in self.motors]) + self.cfg['joint_margin']
        self.upper = np.minimum(MAXIMUM, [m.getMaxPosition() for m in self.motors]) - self.cfg['joint_margin']
        if config.get('landing_alignment'):
            from .landing_projection import joint_bounds
            lower, upper = joint_bounds(config)
            if not (np.allclose(self.lower, lower, rtol=0, atol=1e-6)
                    and np.allclose(self.upper, upper, rtol=0, atol=1e-6)):
                raise ValueError('Webots joint limits differ from the landing Action projection')
        self.foot_sensors = []
        self.foot_history = []
        if config.get('contact_expert', {}).get('enabled'):
            self.foot_sensors = [supervisor.getDevice(n) for n in ('LFsr','RFsr')]
            for sensor in self.foot_sensors:
                sensor.enable(self.ms)
        self.imu, self.gyro = supervisor.getDevice('inertial unit'), supervisor.getDevice('gyro')
        if config.get('physical_gait', {}).get('imu_full_axes'):
            imu_node=self.node.getFromProtoDef('INERTIAL_UNIT')
            if imu_node is None:raise RuntimeError('Cannot configure NAO IMU axes')
            for name in ('xAxis','yAxis','zAxis'):
                field=imu_node.getField(name)
                if field is None:raise RuntimeError('Missing IMU field '+name)
                if not field.getSFBool():raise RuntimeError('IMU axis configuration failed: '+name)
        # Use only the built-in X/Y gyro. Z may be disabled or NaN.
        for sensor in [*self.sensors, self.imu, self.gyro]:
            sensor.enable(self.ms)
        for motor in self.motors:
            motor.setVelocity(motor.getMaxVelocity()*self.cfg['motor_velocity_scale'])
        for name in ('LShoulderPitch', 'RShoulderPitch'):
            supervisor.getDevice(name).setPosition(1.1)
        # Save all robot joint transforms before stepping, not just translation.
        self._tick()  # Commit sensor-channel configuration before saving the state.
        if self.cfg.get('reset_mode', 'node') == 'node':
            self.node.saveState('basewalk_initial')

    def _tick(self):
        if self.robot.step(self.ms) == -1:
            raise RuntimeError('Webots simulation ended before completion')
        if self.foot_sensors:
            from .foot_feedback import read_foot_forces
            self.foot_forces = read_foot_forces(self.foot_sensors)
            if not np.isfinite(self.foot_forces).all():
                raise RuntimeError('Nonfinite LFsr/RFsr force')
            self.foot_history.append(self.foot_forces.copy())
            self.foot_history = self.foot_history[-32:]
            if hasattr(self,'support_feedback'):
                self.support_feedback.advance(self.foot_forces)

    def _positions(self):
        return np.array([s.getValue() for s in self.sensors], np.float32)

    def _feedback(self):
        self.position = np.asarray(self.node.getPosition(), dtype=float)
        rotation = np.asarray(self.node.getOrientation()).reshape(3, 3)
        truth_yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        now = self.robot.getTime()
        # Webots getVelocity() returned NaN in the restored-body validation run.
        # Use one explicit causal estimator throughout, never substitute silently.
        self.velocity = np.zeros(3, np.float32)
        if self.previous_pose is not None:
            old_position, old_truth_yaw, old_time = self.previous_pose
            dt = now-old_time
            if dt <= 0:
                raise RuntimeError('Non-increasing simulation time')
            local_v = rotation.T @ ((self.position-old_position)/dt)
            delta_yaw = math.atan2(math.sin(truth_yaw-old_truth_yaw), math.cos(truth_yaw-old_truth_yaw))
            self.velocity[:] = local_v[0], local_v[1], delta_yaw/dt
            if self.cfg.get('yaw_source') == 'integrated_yaw_rate':
                self.yaw, self.velocity[2] = self.heading_estimator.update(float(self.velocity[2]), dt)
            else:
                self.yaw = truth_yaw
        elif self.cfg.get('yaw_source') != 'integrated_yaw_rate':
            self.yaw = truth_yaw
        self.truth_yaw = truth_yaw
        self.previous_pose = (self.position.copy(), truth_yaw, now)
        if not hasattr(self, 'diagnostic_truth'):
            self.diagnostic_truth = TruthTracker()
        self.truth_diagnostics = self.diagnostic_truth.update(self.position, rotation, now)
        capture_joint_truth(self, now)
        self.gravity = projected_gravity(self.imu.getRollPitchYaw()[:2])
        gyro = self.gyro.getValues()
        self.gyro_value = np.asarray(gyro[:2], np.float32)
        if self.actor_body_gyro:
            self.actor_gyro_z = float(gyro[2])
            if not math.isfinite(self.actor_gyro_z):
                raise ValueError('Body gyro Actor requires an enabled, finite gyro Z sensor')
        self.height = float(self.position[2]-self.cfg['floor_z'])
        if not np.isfinite(np.concatenate((self.position, self.velocity, self.gravity, self.gyro_value))).all():
            raise RuntimeError(f'Nonfinite robot feedback: position={self.position}, velocity={self.velocity}, gravity={self.gravity}, gyro={self.gyro_value}')
