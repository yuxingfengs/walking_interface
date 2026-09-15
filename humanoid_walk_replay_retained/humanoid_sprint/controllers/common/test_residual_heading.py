"""Integration checks for frozen BC, residual learning and sensor-only heading."""
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import torch
from configs.runtime import load_config, algorithm_settings, project_path, create_actor, validate
from algorithms.ddpg_residual import ResidualDDPGAlgorithm
from checkpointing import load_actor_weights
from contracts import HeadingObservation, CanonicalAction, Transition
from tasks.heading_correction import HeadingCorrectionTask, HeadingCorrectionConfig
from test_unified_env import raw_state

class ResidualHeadingTests(unittest.TestCase):
    def model(self):
        import json
        config=validate(json.loads(project_path('experiments/walk_bc_residual_heading_140k/design/after_config.json').read_text(encoding='utf-8')))
        settings=algorithm_settings(config,'cpu','forward_walk_v1')
        settings.update(batch_size=4,critic_warmup_updates=0)
        model=ResidualDDPGAlgorithm(**settings)
        initial=project_path(config['initialization']['bc_checkpoint'])
        self.assertTrue(model.load_actor_initialization(initial,reset_optimizer=True))
        return config,settings,model,initial

    def test_updates_checkpoint_and_bc_reset_preserve_frozen_base(self):
        config,settings,model,initial=self.model()
        states=torch.randn(8,25)*.1
        before={k:v.clone() for k,v in model.actor.base.state_dict().items()}
        initial_actions=model.actor(states).detach().clone()
        self.assertTrue(torch.equal(initial_actions,model.actor.base(states[:,:24])))
        for state in states:
            obs=HeadingObservation(tuple(state.tolist()))
            action=model.actor(state).squeeze(0).detach()
            model.observe(Transition(obs,CanonicalAction(tuple(action.tolist())),.035,obs,False,False))
        for _ in range(3): model.update()
        self.assertGreater(model.actor_update_count,0)
        self.assertFalse(torch.equal(initial_actions,model.actor(states)))
        for k,v in before.items():
            self.assertTrue(torch.equal(v,model.actor.base.state_dict()[k]))
            self.assertTrue(torch.equal(v,model.actor_target.base.state_dict()[k]))
        self.assertTrue(all(p.grad is None for p in model.actor.base.parameters()))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'trained.pth'
            model.save_checkpoint(path)
            restored=ResidualDDPGAlgorithm(**settings)
            restored.load_checkpoint(path)
            evaluation=create_actor(config)
            load_actor_weights(evaluation,path,torch.device('cpu'))
            self.assertTrue(torch.equal(model.actor(states),evaluation(states)))
            self.assertTrue(torch.equal(model.actor(states),restored.actor(states)))
            from world_reload_state import save_continuation, load_continuation
            snapshot=Path(folder)/'effective_config.json'
            snapshot.write_text('{"run_id":"residual-test"}',encoding='utf-8')
            save_continuation(snapshot,model,{'env_steps':8})
            continued=load_continuation(snapshot)['algorithm']
            self.assertEqual(model.experience_size,continued.experience_size)
            self.assertTrue(torch.equal(model.actor(states),continued.actor(states)))
            self.assertTrue(all(not p.requires_grad for p in continued.actor.base.parameters()))
        model.load_actor_initialization(initial,reset_optimizer=True)
        self.assertTrue(torch.equal(initial_actions,model.actor(states)))
        optimized={id(p) for group in model.actor_optimizer.param_groups for p in group['params']}
        self.assertTrue(all(id(p) not in optimized for p in model.actor.base.parameters()))

    def test_extreme_noise_is_bounded_and_full_action_goes_to_replay(self):
        _,_,model,_=self.model()
        obs=HeadingObservation(tuple([.01]*25))
        base=model.actor.base_action(obs.values).squeeze(0).detach()
        for sign in (-1,1):
            action=model.limit_exploration(obs,base+sign*100)
            self.assertLessEqual(float((action-base).abs().max()),.030001)
            model.observe(Transition(obs,CanonicalAction(tuple(action.tolist())),0.,obs,False,False))
            self.assertTrue(torch.equal(torch.tensor(model.replay_buffer.buffer[-1].action.values),action))

    def test_heading_is_wrapped_sensor_error_and_reward_is_symmetric(self):
        task=HeadingCorrectionTask(HeadingCorrectionConfig(target_heading_rad=.1,heading_reward_weight=.1))
        raw=raw_state();task.reset(raw)
        obs=task.observe(replace(raw,yaw=.3),0)
        self.assertAlmostEqual(obs.values[24],.2)
        self.assertEqual(obs,task.observe(replace(raw,yaw=.3,position=(99.,88.,.1),center_of_mass_height=.1),0))
        self.assertAlmostEqual(task.observe(replace(raw,yaw=.3+2*math.pi),0).values[24],.2)
        for yaw in (-.1,.3):
            step=SimpleNamespace(distance_moved=.0035,next_observation=task.observe(replace(raw,yaw=yaw),0),info={'fell':False})
            self.assertAlmostEqual(task.step_reward_from_transition(step,None),.015)
            step.info['fell']=True
            self.assertEqual(task.step_reward_from_transition(step,None),-3.)

    def test_collector_stores_action_after_exploration_limit(self):
        from continuous_training import ContinuousCollector
        from test_continuous_training import FakeEnv, FakeAlgorithm, Noise
        algorithm=FakeAlgorithm()
        seen=[]
        def limit(obs,action):
            seen.append(action)
            return [.012]*10
        algorithm.limit_exploration=limit
        training=dict(total_env_steps=3,update_every_steps=100,log_every_steps=35,
                      min_buffer_for_train=320,actor_lr=1e-5)
        init=dict(stall_window_steps=35,stall_timeout_steps=700,
                  stall_min_forward_progress_m=.02,consecutive_falls=5,
                  bc_reset_max_env_steps=0,reset_to_bc_on_failure=False)
        ContinuousCollector(FakeEnv(),algorithm,Noise(),training,init).run()
        self.assertEqual(len(seen),3)
        self.assertTrue(all(t.action.values==tuple([.012]*10) for t in algorithm.all_transitions))

    def test_diagnostic_rejects_standing_and_heading_oscillation(self):
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
        from validate_heading_correction import summarize
        rows=[dict(x=0.,y=0.,heading_error_rad=.1) for _ in range(35)]
        rows += [dict(x=0.,y=0.,heading_error_rad=0.) for _ in range(35)]
        self.assertFalse(summarize(rows,False)['passed'])
        for i,row in enumerate(rows): row['x']=i*.01
        self.assertTrue(summarize(rows,False)['passed'])
        self.assertFalse(summarize(rows,True)['passed'])
        for i,row in enumerate(rows[35:]): row['heading_error_rad']=.2*(-1)**i
        self.assertFalse(summarize(rows,False)['passed'])

    def test_incompatible_task_algorithm_rejected(self):
        config=load_config();config['training']['algorithm_module']='algorithms.ddpg_residual';config['environment']['task_module']='tasks.forward_walk_v1'
        with self.assertRaises(ValueError): validate(config)

if __name__=='__main__': unittest.main()
