"""The only S4 module that directly accesses Webots devices and physics APIs."""

import math
from typing import Sequence

from action_adapter import ActionAdapter
from contracts import (
    ACTION_JOINT_ORDER,
    ALL_LEG_JOINT_ORDER,
    CanonicalAction,
    OBSERVATION_JOINT_ORDER,
    RawRobotState,
)


def absolute_heading(orientation):
    """Heading of torso local +X in the world XY plane (row-major rotation)."""
    if not all(math.isfinite(float(v)) for v in orientation):
        raise ValueError('Non-finite world orientation')
    return math.atan2(orientation[3], orientation[0])


class WebotsRuntime:
    def __init__(self, supervisor, time_step_ms: int = 40, *, sensor_warmup_steps=5,
                 shoulder_position=1.1, motor_velocity_scale=0.8, joint_limit_margin=1e-3,
                 reset_mode='legacy', foot_contact_threshold_n=1.0):
        self.supervisor = supervisor
        self.time_step_ms = int(time_step_ms)
        self.sensor_warmup_steps = int(sensor_warmup_steps)
        self.shoulder_position = float(shoulder_position)
        self.motor_velocity_scale = float(motor_velocity_scale)
        self.joint_limit_margin = float(joint_limit_margin)
        self.reset_mode = reset_mode
        self.foot_contact_threshold_n = float(foot_contact_threshold_n)
        self._has_reset = False
        from sensor_progress import ContactOdometry
        self.progress_odometry = ContactOdometry()

    def initialize(self) -> None:
        self.right_shoulder = self.supervisor.getDevice("RShoulderPitch")
        self.left_shoulder = self.supervisor.getDevice("LShoulderPitch")
        self.right_shoulder.setPosition(self.shoulder_position)
        self.left_shoulder.setPosition(self.shoulder_position)

        self.motors = {
            name: self.supervisor.getDevice(name) for name in ALL_LEG_JOINT_ORDER
        }
        self.sensors = {}
        for name in ALL_LEG_JOINT_ORDER:
            sensor = self.supervisor.getDevice(f"{name}S")
            sensor.enable(self.time_step_ms)
            self.sensors[name] = sensor

        from com_estimator import NaoCOMEstimator
        self.com_estimator = NaoCOMEstimator()
        for name in self.com_estimator.joint_names:
            if name not in self.sensors:
                sensor = self.supervisor.getDevice(name + 'S')
                if sensor is None:
                    raise RuntimeError('Missing COM joint encoder: ' + name)
                sensor.enable(self.time_step_ms)
                self.sensors[name] = sensor

        self.accelerometer = self.supervisor.getDevice("accelerometer")
        self.imu = self.supervisor.getDevice("inertial unit")
        # Static configuration only: the R2022b IMU mounts at Rx(1.5708).
        # Enable its pitch channel, then undo mounting rotation for COM.
        imu_node = self.supervisor.getFromDef("NAO").getFromProtoDef("INERTIAL_UNIT")
        if imu_node is None:
            raise RuntimeError("NAO IMU configuration node missing")
        imu_node.getField("yAxis").setSFBool(True)
        self.gyro = self.supervisor.getDevice("gyro")
        self.accelerometer.enable(self.time_step_ms)
        self.imu.enable(self.time_step_ms)
        self.gyro.enable(self.time_step_ms)

        self.foot_sensors = []
        for name in ('LFsr', 'RFsr'):
            sensor = self.supervisor.getDevice(name)
            if sensor is None:
                raise RuntimeError('Missing foot force sensor: ' + name)
            sensor.enable(self.time_step_ms)
            self.foot_sensors.append(sensor)
        for _ in range(self.sensor_warmup_steps):
            self.step()

        self.nao = self.supervisor.getFromDef("NAO")
        if self.nao is None:
            raise RuntimeError("Webots world is missing DEF NAO")
        self.translation = self.nao.getField("translation")
        self.rotation = self.nao.getField("rotation")
        self.action_adapter = ActionAdapter.from_motors(self.motors)
        self._capture_initial_state()
        print(f"[runtime] initialized {len(self.motors)} leg motors")

    def _capture_initial_state(self) -> None:
        self.initial_translation = tuple(float(v) for v in self.translation.getSFVec3f())
        self.initial_rotation = tuple(float(v) for v in self.rotation.getSFRotation())
        self.initial_joint_positions = tuple(
            float(self.sensors[name].getValue()) for name in ALL_LEG_JOINT_ORDER
        )

    def step(self) -> int:
        return self.supervisor.step(self.time_step_ms)

    def get_time(self) -> float:
        return float(self.supervisor.getTime())

    def get_position(self):
        return tuple(float(v) for v in self.translation.getSFVec3f())

    def reset_progress_estimator(self):
        self.progress_odometry.reset()

    def read_state(self) -> RawRobotState:
        roll, pitch, imu_yaw = self.imu.getRollPitchYaw()
        yaw = imu_yaw
        gyro_x, gyro_y, _ = self.gyro.getValues()
        contacts = tuple(float(sensor.getValues()[2] > self.foot_contact_threshold_n)
                         for sensor in self.foot_sensors)
        angles = {name: float(self.sensors[name].getValue()) for name in self.com_estimator.joint_names}
        estimated_com, com_valid = self.com_estimator.estimate(
            angles,
            roll - 1.5708, pitch, contacts)
        _, soles = self.com_estimator.kinematics(angles)
        estimated_forward, odometry_valid = self.progress_odometry.observe(
            soles, roll - 1.5708, pitch, yaw, contacts)
        return RawRobotState(
            world_heading=absolute_heading(self.nao.getOrientation()),
            roll=float(roll),
            yaw=float(yaw),
            gyro_x=float(gyro_x),
            gyro_y=float(gyro_y),
            center_of_mass_height=float(self.translation.getSFVec3f()[2]),
            joint_positions=tuple(
                float(self.sensors[name].getValue())
                for name in OBSERVATION_JOINT_ORDER
            ),
            position=self.get_position(),
            simulation_time=self.get_time(),
            foot_contacts=contacts,
            estimated_com=estimated_com, com_valid=com_valid,
            estimated_forward=estimated_forward, odometry_valid=odometry_valid,
        )

    def apply_action(self, candidate: Sequence[float]) -> CanonicalAction:
        canonical_action = self.action_adapter.adapt(candidate)
        for name, target in zip(ACTION_JOINT_ORDER, canonical_action.values):
            self.motors[name].setPosition(target)
        return canonical_action

    def _set_default_motor_velocity(self, scale: float = 0.8) -> None:
        for motor in self.motors.values():
            motor.setVelocity(float(motor.getMaxVelocity()) * float(scale))

    def hard_reset(self, settle_steps: int = 64, margin=None) -> RawRobotState:
        if self.reset_mode == 'world_reload' and self._has_reset:
            # The owner must persist learner/diagnostic state before the controller exits.
            self.before_world_reload()
            self.supervisor.worldReload()
            while self.step() != -1:
                pass
            import os
            os._exit(0)  # Do not mark the continuing run failed/completed in the old wrapper.
        self._has_reset = True
        margin = self.joint_limit_margin if margin is None else float(margin)
        node = self.supervisor.getSelf()
        if node is not None:
            node.resetPhysics()
        self.translation.setSFVec3f(list(self.initial_translation))
        self.rotation.setSFRotation(list(self.initial_rotation))
        self._set_default_motor_velocity(scale=self.motor_velocity_scale)
        for name, initial_position in zip(
            ALL_LEG_JOINT_ORDER, self.initial_joint_positions
        ):
            motor = self.motors[name]
            lower = float(motor.getMinPosition()) + margin
            upper = float(motor.getMaxPosition()) - margin
            motor.setPosition(min(max(initial_position, lower), upper))
        for _ in range(int(settle_steps)):
            self.step()
        return self.read_state()
