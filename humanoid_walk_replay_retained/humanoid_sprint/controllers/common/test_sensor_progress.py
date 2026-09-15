import unittest
import numpy as np
from sensor_progress import ContactOdometry, ForwardProgressWatchdog


class SensorProgressTests(unittest.TestCase):
    def test_stationary_reverse_and_oscillation_timeout(self):
        for positions in ([0.]*100, [-.01*i for i in range(100)],
                          [.04 if i%2 else -.04 for i in range(100)]):
            guard=ForwardProgressWatchdog(5,.02)
            for x in positions:
                guard.observe(x)
            self.assertGreaterEqual(guard.wait_steps,90)

    def test_forward_progress_resets_but_revisiting_old_peak_does_not(self):
        guard=ForwardProgressWatchdog(1,.02)
        guard.observe(.03)
        self.assertEqual(guard.wait_steps,0)
        for x in [-.1,0.,.03]*20:
            guard.observe(x)
        self.assertEqual(guard.wait_steps,60)
        guard.observe(.06)
        self.assertEqual(guard.wait_steps,0)

    def test_invalid_samples_do_not_pause_timeout(self):
        guard=ForwardProgressWatchdog(35,.02)
        for _ in range(700):
            guard.observe(float('nan'),False)
        self.assertEqual(guard.wait_steps,700)

    def test_forward_leg_motion_and_support_switch(self):
        odom=ContactOdometry()
        feet=np.array([[0,.05,-.3],[0,-.05,-.3]])
        odom.observe(feet,0,0,0,(1,0))
        p,valid=odom.observe(feet+[-.01,0,0],0,0,0,(1,0))
        self.assertTrue(valid);self.assertAlmostEqual(p,.01)
        p,valid=odom.observe(feet,0,0,0,(0,1))
        self.assertFalse(valid);self.assertAlmostEqual(p,.01)
        p,valid=odom.observe(feet+[-.01,0,0],0,0,0,(0,1))
        self.assertTrue(valid);self.assertAlmostEqual(p,.02)

    def test_turn_without_translation_and_missing_contact_no_false_progress(self):
        from com_estimator import rotation
        odom=ContactOdometry()
        feet=np.array([[.02,.05,-.3],[.02,-.05,-.3]])
        odom.observe(feet,0,0,0,(1,1))
        yaw=.3
        # Same feet in space expressed in a rotated torso frame.
        new=feet@rotation((0,0,1),yaw)
        p,valid=odom.observe(new,0,0,yaw,(1,1))
        self.assertAlmostEqual(p,0.,places=10)
        p,valid=odom.observe(feet,0,0,0,(0,0))
        self.assertFalse(valid);self.assertAlmostEqual(p,0.,places=10)
        odom.reset();self.assertEqual(odom.forward,0.)

    def test_gross_jump_rejected(self):
        odom=ContactOdometry()
        feet=np.array([[0,.05,-.3],[0,-.05,-.3]])
        odom.observe(feet,0,0,0,(1,1))
        p,valid=odom.observe(feet+[-1,0,0],0,0,0,(1,1))
        self.assertFalse(valid);self.assertEqual(p,0.)
