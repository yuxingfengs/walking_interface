import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
import torch
from contracts import COMObservation, ContactObservation, CanonicalAction, Transition, validate_contract_metadata
from configs.runtime import load_config, algorithm_settings
from algorithms.ddpg import DDPGAlgorithm
from checkpointing import load_actor_weights, load_torch_artifact
from models import BoundedActor
from tasks.forward_walk_v1 import ForwardWalkTaskConfig
from tasks.forward_walk_heading_v3 import ForwardWalkHeadingV3
from test_unified_env import raw_state


class ContactTests(unittest.TestCase):
    def test_observation_appends_left_right_and_masks_height(self):
        task = ForwardWalkHeadingV3(ForwardWalkTaskConfig())
        state = replace(raw_state(), foot_contacts=(1., 0.))
        task.reset(state)
        obs = task.observe(state, 0)
        self.assertIsInstance(obs, COMObservation)
        self.assertEqual(obs.values[18:20], (1., 0.))
        self.assertEqual(obs.values[5], 0.)
        self.assertEqual(obs, task.observe(replace(state, position=(100., 200., .2), center_of_mass_height=.2), 0))

    def test_twenty_dim_training_checkpoint_and_eval_loading(self):
        config = load_config()
        config['training']['algorithm_module'] = 'algorithms.ddpg'
        settings = algorithm_settings(config, 'cpu', 'forward_walk_v1')
        settings.update(state_dim=20, batch_size=4)
        model = DDPGAlgorithm(**settings)
        for i in range(8):
            obs = ContactObservation(tuple([i*.001]*18+[float(i%2),float((i+1)%2)]))
            model.observe(Transition(obs, CanonicalAction(tuple([0.]*10)), .03, obs, False, False))
        model.update()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'checkpoint.pth'
            model.save_checkpoint(path)
            metadata = load_torch_artifact(path, 'cpu')['meta']
            validate_contract_metadata(metadata, artifact=str(path), expected_state_dim=20)
            with self.assertRaises(ValueError):
                validate_contract_metadata(metadata, artifact=str(path), expected_state_dim=18)
            restored = DDPGAlgorithm(**settings)
            restored.load_checkpoint(path)
            actor = BoundedActor(20,256,10)
            load_actor_weights(actor,path,torch.device('cpu'))
            with torch.no_grad():
                self.assertTrue(torch.equal(model.actor(torch.zeros(1,20)),actor(torch.zeros(1,20))))

    def test_com_training_checkpoint_and_eval_loading(self):
        config = load_config()
        config['training']['algorithm_module'] = 'algorithms.ddpg'
        settings = algorithm_settings(config, 'cpu', 'forward_walk_v1')
        settings.update(state_dim=24, batch_size=4)
        model = DDPGAlgorithm(**settings)
        for i in range(8):
            obs = COMObservation(tuple([i*.001]*18+[float(i%2),float((i+1)%2)]+[.01, 0., .3, 1.]))
            model.observe(Transition(obs, CanonicalAction(tuple([0.]*10)), .03, obs, False, False))
        model.update()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'checkpoint.pth'
            model.save_checkpoint(path)
            metadata = load_torch_artifact(path, 'cpu')['meta']
            validate_contract_metadata(metadata, artifact=str(path), expected_state_dim=24)
            with self.assertRaises(ValueError):
                validate_contract_metadata(metadata, artifact=str(path), expected_state_dim=18)
            restored = DDPGAlgorithm(**settings)
            restored.load_checkpoint(path)
            actor = BoundedActor(24,256,10)
            load_actor_weights(actor,path,torch.device('cpu'))
            with torch.no_grad():
                self.assertTrue(torch.equal(model.actor(torch.zeros(1,24)),actor(torch.zeros(1,24))))
