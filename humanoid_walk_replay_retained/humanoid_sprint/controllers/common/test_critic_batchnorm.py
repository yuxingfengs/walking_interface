"""Check fixed value-function semantics and real Actor/Critic optimization."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

import test_critic_warmup
from configs.runtime import load_config, validate
from world_reload_state import save_continuation, load_continuation


class CriticBatchNormTests(unittest.TestCase):
    def algorithm(self, warmup=0):
        model = test_critic_warmup.CriticWarmupTests().algorithm(warmup)
        model.manage_critic_batchnorm = True
        model.set_evaluation_mode()
        return model

    def test_target_forward_is_batch_independent_and_does_not_mutate(self):
        model = self.algorithm()
        model.update()
        before = copy.deepcopy(model.critic_target.state_dict())
        state, action, _, _, _ = model.replay_buffer.sample_batch(4)
        other = state.clone(); other[1:] = other[1:] * 10 + 20
        with torch.no_grad():
            a = model.critic_target(state, action)[0]
            b = model.critic_target(other, action)[0]
        torch.testing.assert_close(a, b)
        for k, v in before.items():
            self.assertTrue(torch.equal(v, model.critic_target.state_dict()[k]))

    def test_actor_backward_flows_without_changing_critic(self):
        model = self.algorithm()
        model.critic.train()
        before = copy.deepcopy(model.critic.state_dict())
        actor = copy.deepcopy(model.actor.state_dict())
        states, _, _, _, _ = model.replay_buffer.sample_batch(4)
        with model._fixed_critic():
            self.assertFalse(model.critic.training)
            self.assertFalse(any(p.requires_grad for p in model.critic.parameters()))
            actions = model.actor(states)
            actions.retain_grad()
            loss = -model.critic(states, actions).mean()
            model.actor_optimizer.zero_grad()
            loss.backward()
            self.assertGreater(float(actions.grad.abs().sum()), 0)
            self.assertTrue(all(p.grad is None for p in model.critic.parameters()))
            model.actor_optimizer.step()
        self.assertTrue(model.critic.training)
        self.assertTrue(all(p.requires_grad for p in model.critic.parameters()))
        self.assertTrue(any(not torch.equal(v, model.actor.state_dict()[k]) for k,v in actor.items()))
        for k,v in before.items():
            self.assertTrue(torch.equal(v, model.critic.state_dict()[k]))

    def test_real_update_only_updates_online_bn_for_critic_training(self):
        model = self.algorithm()
        critic = copy.deepcopy(model.critic.state_dict())
        actor = copy.deepcopy(model.actor.state_dict())
        target_states = []
        def before(module, args):
            self.assertFalse(module.training)
            target_states.append(copy.deepcopy(module.state_dict()))
        def after(module, args, output):
            for k,v in target_states[-1].items():
                self.assertTrue(torch.equal(v, module.state_dict()[k]))
        h1=model.critic_target.register_forward_pre_hook(before)
        h2=model.critic_target.register_forward_hook(after)
        model.update()
        h1.remove();h2.remove()
        self.assertEqual(int(model.critic.bn1.num_batches_tracked), 2)
        self.assertEqual(int(model.critic.bn2.num_batches_tracked), 2)
        self.assertTrue(any(not torch.equal(v, model.critic.state_dict()[k]) for k,v in critic.items()))
        self.assertTrue(any(not torch.equal(v, model.actor.state_dict()[k]) for k,v in actor.items()))

    def test_warmup_and_soft_update(self):
        model = self.algorithm(warmup=2)
        before = copy.deepcopy(model.actor.state_dict())
        model.update()
        for k,v in before.items():
            self.assertTrue(torch.equal(v, model.actor.state_dict()[k]))
        with torch.no_grad():
            model.critic.bn1.running_mean.fill_(2)
            model.critic_target.bn1.running_mean.fill_(0)
        model.soft_update(model.critic, model.critic_target)
        torch.testing.assert_close(model.critic_target.bn1.running_mean,
                                   torch.full_like(model.critic_target.bn1.running_mean, 2*model.tau))

    def test_context_restores_modes_and_flags_after_error(self):
        model = self.algorithm()
        first = next(model.critic.parameters());first.requires_grad_(False)
        model.critic.train();model.critic.bn1.eval()
        with self.assertRaises(RuntimeError):
            with model._fixed_critic():
                raise RuntimeError('test')
        self.assertTrue(model.critic.training)
        self.assertFalse(model.critic.bn1.training)
        self.assertFalse(first.requires_grad)

    def test_checkpoint_and_world_reload_restore_managed_modes(self):
        model=self.algorithm();model.update()
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            ckpt=folder/'model.pth';model.save_checkpoint(ckpt)
            restored=self.algorithm();restored.load_checkpoint(ckpt)
            self.assertFalse(restored.critic_target.training)
            self.assertFalse(restored.critic.training)
            legacy=test_critic_warmup.CriticWarmupTests().algorithm(0)
            with self.assertRaisesRegex(ValueError, 'BatchNorm'):
                legacy.load_checkpoint(ckpt)
            snapshot=folder/'effective_config.json'
            snapshot.write_text(json.dumps(dict(run_id='bn-test')))
            save_continuation(snapshot, model, dict(global_step=1))
            restored=load_continuation(snapshot)['algorithm']
            restored.set_evaluation_mode()
            self.assertFalse(restored.critic_target.training)
            count=int(restored.critic.bn1.num_batches_tracked)
            restored.update()
            self.assertEqual(int(restored.critic.bn1.num_batches_tracked), count+2)

    def test_old_config_defaults_to_legacy(self):
        config=load_config();config['training'].pop('manage_critic_batchnorm')
        config['training'].pop('sync_bn_running_stats')
        self.assertFalse(validate(config)['training']['manage_critic_batchnorm'])

    def test_bn_stats_copy_but_parameters_still_soft_update(self):
        model=self.algorithm();model.sync_bn_running_stats=True
        with torch.no_grad():
            for p in model.critic.parameters(): p.fill_(2)
            for p in model.critic_target.parameters(): p.fill_(0)
            model.critic.bn1.running_mean.fill_(3)
            model.critic.bn1.running_var.fill_(4)
            model.critic.bn1.num_batches_tracked.fill_(7)
        model.soft_update(model.critic,model.critic_target)
        for p in model.critic_target.parameters():
            torch.testing.assert_close(p,torch.full_like(p,2*model.tau))
        for name,buffer in model.critic.named_buffers():
            self.assertTrue(torch.equal(buffer,dict(model.critic_target.named_buffers())[name]))

    def test_stats_stay_synced_after_updates_and_reload(self):
        model=self.algorithm(warmup=1);model.sync_bn_running_stats=True
        model.update();model.update()
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);checkpoint=folder/'model.pth';model.save_checkpoint(checkpoint)
            other=self.algorithm(warmup=1)
            with self.assertRaisesRegex(ValueError,'statistics sync'):
                other.load_checkpoint(checkpoint)
            other.sync_bn_running_stats=True;other.load_checkpoint(checkpoint)
            self.assertFalse(other.critic_target.training)
            snapshot=folder/'effective_config.json'
            snapshot.write_text(json.dumps(dict(run_id='bn-sync')))
            save_continuation(snapshot,model,dict(global_step=2))
            other=load_continuation(snapshot)['algorithm']
            self.assertTrue(other.sync_bn_running_stats)
            other.set_evaluation_mode();other.update()
            for name,buffer in other.critic.named_buffers():
                self.assertTrue(torch.equal(buffer,dict(other.critic_target.named_buffers())[name]))
            state=copy.deepcopy(other.critic_target.state_dict())
            s,a,_,_,_=other.replay_buffer.sample_batch(4)
            with torch.no_grad(): other.critic_target(s,a)
            for k,v in state.items():
                self.assertTrue(torch.equal(v,other.critic_target.state_dict()[k]))

    def test_stats_sync_config_requires_managed_modes(self):
        config=load_config();config['training']['sync_bn_running_stats']=True
        config['training']['manage_critic_batchnorm']=False
        with self.assertRaisesRegex(ValueError,'managed Critic'):
            validate(config)
