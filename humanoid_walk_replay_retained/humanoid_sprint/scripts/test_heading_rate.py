import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'controllers'),str(ROOT/'controllers/common')]
from contracts import RawRobotState,contract_metadata,validate_contract_metadata
from tasks.heading_rate import HeadingRateTask
from tasks.heading_correction import HeadingCorrectionConfig


def raw(yaw,time):
    return RawRobotState(roll=0.,yaw=0.,gyro_x=0.,gyro_y=0.,center_of_mass_height=.7,
                         joint_positions=(0.,)*12,world_heading=yaw,simulation_time=time)


class HeadingRateTests(unittest.TestCase):
    def task(self):
        return HeadingRateTask(HeadingCorrectionConfig(heading_reward_weight=.1))

    def test_crossing_and_signed_actual_rate(self):
        task=self.task();task.observe(raw(math.radians(179),1.),0)
        obs=task.observe(raw(math.radians(-179),1.04),1)
        self.assertAlmostEqual(obs.values[26],math.radians(2)/.04)
        self.assertEqual(len(obs.values),27)
        self.assertAlmostEqual(obs.values[24]**2+obs.values[25]**2,1.)

    def test_repeat_observation_and_reset(self):
        task=self.task();task.observe(raw(.1,1.),0)
        value=task.observe(raw(.11,1.04),1)
        self.assertEqual(value,task.observe(raw(.11,1.04),1))
        task.reset(raw(-2.,0.))
        self.assertEqual(task.observe(raw(-1.99,.04),0).values[26],0.)

    def test_invalid_time(self):
        task=self.task();task.observe(raw(0.,1.),0)
        with self.assertRaises(ValueError):task.observe(raw(.1,1.),0)
        with self.assertRaises(ValueError):task.observe(raw(0.,.9),0)
        with self.assertRaises(ValueError):task.observe(raw(float('nan'),2.),0)

    def test_reward_direction_fall_and_rate_independence(self):
        task=self.task()
        for error in [-.2,0.,.2]:
            for rate in [-10.,0.,10.]:
                values=(0.,)*24+(math.sin(error),math.cos(error),rate)
                step=SimpleNamespace(next_observation=SimpleNamespace(values=values),distance_moved=.003,info={'fell':False})
                self.assertAlmostEqual(task.step_reward_from_transition(step,None),.03-.1*abs(error))
                step.info['fell']=True
                self.assertEqual(task.step_reward_from_transition(step,None),-3.)

    def test_offline_causal_feature_agreement(self):
        task=self.task();yaw=np.array([.08,.079,.083,.082])
        outputs=[task.observe(raw(float(h),i*.04),i).values for i,h in enumerate(yaw)]
        offline=np.r_[0.,np.arctan2(np.sin(np.diff(yaw)),np.cos(np.diff(yaw)))/.04]
        np.testing.assert_allclose(np.array(outputs)[:,26],offline,atol=1e-12)

    def test_old_model_contract_rejected(self):
        with self.assertRaises(ValueError):
            validate_contract_metadata(contract_metadata(state_dim=25),artifact='old',expected_state_dim=27)

    def test_actor_critic_replay_update_and_checkpoint(self):
        import tempfile,torch
        from configs.runtime import load_config,algorithm_settings,task_settings
        from algorithms import load_algorithm
        from tasks import load_task
        from contracts import Transition,CanonicalAction
        cfg=load_config();task=load_task(cfg['environment']['task_module'],cfg['environment']['task_class'],task_settings(cfg,'train'))
        alg=load_algorithm(cfg['training']['algorithm_module'],cfg['training']['algorithm_class'],algorithm_settings(cfg,'cpu',task.task_id))
        for i in range(64):
            obs=task.observe(raw(.08+i*.0001,i*.04),i)
            action=CanonicalAction(tuple(alg.act(obs).tolist()))
            alg.observe(Transition(obs,action,.01,obs,False,False))
        result=alg.update()
        self.assertTrue(math.isfinite(result.critic_loss))
        self.assertEqual(alg.actor.net[0].in_features,27)
        with tempfile.TemporaryDirectory() as folder:
            path=str(Path(folder)/'test.pth');alg.save_checkpoint(path)
            alg.load_checkpoint(path)


if __name__=='__main__':unittest.main()
