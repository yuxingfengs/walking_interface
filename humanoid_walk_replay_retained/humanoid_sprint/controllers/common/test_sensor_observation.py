import unittest
from types import SimpleNamespace
from dataclasses import replace
from contracts import OBSERVATION_FIELDS, OBSERVATION_JOINT_ORDER
from tasks.forward_walk_heading_v3 import ForwardWalkHeadingV3
from tasks.forward_walk_v1 import ForwardWalkTaskConfig
from webots_platform.webots_runtime import WebotsRuntime
from test_unified_env import raw_state


class SensorObservationTests(unittest.TestCase):
    def test_runtime_uses_imu_yaw_without_reading_true_rotation(self):
        runtime = WebotsRuntime(SimpleNamespace(getTime=lambda: 1.))
        runtime.imu = SimpleNamespace(getRollPitchYaw=lambda: (.1, .2, -.3))
        def forbidden():
            raise AssertionError('Privileged rotation was read')
        runtime.rotation = SimpleNamespace(getSFRotation=forbidden)
        runtime.translation = SimpleNamespace(getSFVec3f=lambda: (1., 2., .7))
        runtime.gyro = SimpleNamespace(getValues=lambda: (.4, .5, .6))
        from com_estimator import NaoCOMEstimator
        runtime.com_estimator = NaoCOMEstimator()
        runtime.sensors = {name: SimpleNamespace(getValue=lambda: .01) for name in runtime.com_estimator.joint_names}
        runtime.foot_sensors = [SimpleNamespace(getValues=lambda: (0., 0., 2.)),
                                SimpleNamespace(getValues=lambda: (0., 0., .5))]
        original_estimate = runtime.com_estimator.estimate
        def check_mount(angles, roll, pitch, contacts):
            self.assertAlmostEqual(roll, .1 - 1.5708)
            self.assertAlmostEqual(pitch, .2)
            return original_estimate(angles, roll, pitch, contacts)
        runtime.com_estimator.estimate = check_mount
        state = runtime.read_state()
        self.assertEqual(state.yaw, -.3)
        self.assertEqual(state.roll, .1)
        runtime.foot_sensors = [SimpleNamespace(getValues=lambda: (0., 0., 2.)),
                                SimpleNamespace(getValues=lambda: (0., 0., .5))]
        self.assertEqual(runtime.read_state().foot_contacts, (1., 0.))

    def test_actor_critic_observation_independent_of_world_position_and_height(self):
        task = ForwardWalkHeadingV3(ForwardWalkTaskConfig())
        state = raw_state()
        task.reset(state)
        a = task.observe(state, 0)
        b = task.observe(replace(state, position=(100., -300., .1), center_of_mass_height=.1), 0)
        self.assertEqual(a, b)
        self.assertEqual(a.values[OBSERVATION_FIELDS.index('cm_norm')], 0.)
        self.assertEqual(len(a.values), 24)
        # Ground-truth termination still belongs to the environment feedback.
        self.assertTrue(task.termination(replace(state, center_of_mass_height=.1))[0])

    def test_world_height_is_always_excluded(self):
        task = ForwardWalkHeadingV3(ForwardWalkTaskConfig())
        state = raw_state()
        task.reset(state)
        self.assertEqual(task.observe(state, 0), task.observe(replace(state, center_of_mass_height=.1), 0))
