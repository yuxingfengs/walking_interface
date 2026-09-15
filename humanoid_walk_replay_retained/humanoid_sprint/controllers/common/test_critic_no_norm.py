"""No-normalization ablation: real updates, batch independence, artifact isolation."""
import copy
from pathlib import Path
import tempfile
import unittest

import torch
from algorithms.ddpg import DDPGAlgorithm
from algorithms.ddpg_no_norm import DDPGNoNormAlgorithm
from contracts import ActorObservation, CanonicalAction, Transition


class NoNormTests(unittest.TestCase):
    def model(self, cls=DDPGNoNormAlgorithm):
        torch.manual_seed(17)
        model = cls(state_dim=18, action_dim=10, hidden_dim=32, actor_lr=1e-5,
                    critic_lr=1e-5, device='cpu', gamma=.98, tau=.0001,
                    batch_size=4, buffer_size=16, manage_critic_batchnorm=True,
                    sync_bn_running_stats=False, critic_warmup_updates=1)
        for _ in range(8):
            s = ActorObservation(tuple(torch.randn(18).tolist()))
            ns = ActorObservation(tuple(torch.randn(18).tolist()))
            a = CanonicalAction(tuple(model.act(s).tolist()))
            model.observe(Transition(s, a, .1, ns, False, False))
        return model

    def test_batch_independent_and_real_optimization(self):
        m = self.model()
        self.assertFalse(any(isinstance(x, (torch.nn.BatchNorm1d, torch.nn.LayerNorm)) for x in m.critic.modules()))
        s, a, *_ = m.replay_buffer.sample_batch(4)
        m.critic.train()
        q = m.critic(s, a)[0]
        changed = s.clone(); changed[1:] += 20
        torch.testing.assert_close(q, m.critic(changed, a)[0])
        m.critic.eval(); torch.testing.assert_close(q, m.critic(s, a)[0])
        old = copy.deepcopy(m.critic.state_dict())
        self.assertFalse(m.update().metrics['actor_updated'])
        self.assertTrue(any(not torch.equal(v, m.critic.state_dict()[k]) for k,v in old.items()))
        actor = copy.deepcopy(m.actor.state_dict())
        self.assertTrue(m.update().metrics['actor_updated'])
        self.assertTrue(any(not torch.equal(v, m.actor.state_dict()[k]) for k,v in actor.items()))

    def test_checkpoint_roundtrip_and_reject_cross_architecture(self):
        m = self.model(); m.update()
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'model.pth'; m.save_checkpoint(p)
            restored = self.model(); restored.load_checkpoint(p)
            for k,v in m.critic.state_dict().items():
                torch.testing.assert_close(v, restored.critic.state_dict()[k])
            with self.assertRaisesRegex(ValueError, 'Algorithm'):
                self.model(DDPGAlgorithm).load_checkpoint(p)
            self.model(DDPGAlgorithm).save_checkpoint(p)
            with self.assertRaisesRegex(ValueError, 'Algorithm'):
                self.model().load_checkpoint(p)


if __name__ == '__main__':
    unittest.main()
