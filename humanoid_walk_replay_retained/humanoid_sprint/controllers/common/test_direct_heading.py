"""Direct Actor initialization, real updates, restore and evaluation checks."""
import json
from pathlib import Path
import tempfile
import unittest
import torch
from configs.runtime import load_config,project_path,algorithm_settings,create_actor
from algorithms.ddpg import DDPGAlgorithm
from checkpointing import load_actor_weights
from contracts import HeadingObservation,CanonicalAction,Transition
from models import BoundedActor

class DirectHeadingTests(unittest.TestCase):
    def test_bc_equivalence_trainability_update_restore_and_eval(self):
        torch.manual_seed(7)
        config=load_config()
        settings=algorithm_settings(config,'cpu','forward_walk_v1')
        settings.update(batch_size=8,critic_warmup_updates=0)
        model=DDPGAlgorithm(**settings)
        initial=project_path(config['initialization']['bc_checkpoint'])
        model.load_actor_initialization(initial,reset_optimizer=True)
        old=BoundedActor(24,256,10)
        load_actor_weights(old,project_path('experiments/walk_sensor_com_freshbc_v2_140k/design/actor_bc_sensor24.pth'),torch.device('cpu'))
        states=torch.randn(32,25)*.2
        expected=old(states[:,:24]).detach()
        torch.testing.assert_close(model.actor(states),expected,rtol=1e-5,atol=1e-6)
        self.assertFalse(hasattr(model.actor,'base'))
        self.assertFalse(hasattr(model,'limit_exploration'))
        self.assertTrue(all(p.requires_grad for p in model.actor.parameters()))
        optimized={id(p) for g in model.actor_optimizer.param_groups for p in g['params']}
        self.assertEqual(optimized,{id(p) for p in model.actor.parameters()})
        before={k:v.clone() for k,v in model.actor.named_parameters()}
        for state in states:
            obs=HeadingObservation(tuple(state.tolist()))
            action=model.actor(state).squeeze(0).detach()
            model.observe(Transition(obs,CanonicalAction(tuple(action.tolist())),.03,obs,False,False))
        for _ in range(3): model.update()
        for k,p in model.actor.named_parameters(): self.assertFalse(torch.equal(before[k],p),k)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'direct.pth';model.save_checkpoint(path)
            evaluation=create_actor(config);load_actor_weights(evaluation,path,torch.device('cpu'))
            self.assertTrue(torch.equal(model.actor(states),evaluation(states)))
            restored=DDPGAlgorithm(**settings);restored.load_checkpoint(path)
            self.assertTrue(torch.equal(model.actor(states),restored.actor(states)))
            from world_reload_state import save_continuation,load_continuation
            snapshot=Path(folder)/'effective_config.json';snapshot.write_text('{"run_id":"direct-test"}',encoding='utf-8')
            save_continuation(snapshot,model,{'env_steps':32})
            continued=load_continuation(snapshot)['algorithm']
            self.assertEqual(continued.experience_size,32)
            self.assertTrue(torch.equal(model.actor(states),continued.actor(states)))
        model.load_actor_initialization(initial,reset_optimizer=True)
        torch.testing.assert_close(model.actor(states),expected,rtol=1e-5,atol=1e-6)

    def test_direct_action_can_exceed_old_residual_limit(self):
        actor=create_actor(load_config())
        with torch.no_grad():
            for p in actor.parameters(): p.zero_()
            before=actor(torch.zeros(25)).clone()
            actor.net[-1].bias.add_(.1)
            after=actor(torch.zeros(25))
        self.assertGreater(float((after-before).abs().max()),.03)

if __name__=='__main__':unittest.main()
