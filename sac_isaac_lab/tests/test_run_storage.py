from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'runner'))
from run_storage import exclusive_writer, restore_csv, selected_checkpoint, training_output


class TrainingOutputTests(unittest.TestCase):
    def test_automatic_continue_follows_only_the_matching_import(self):
        import json
        import torch
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root/'train_old/checkpoints/latest.pth'
            model = root/'config_a/train_model'
            recovery = model/'checkpoints/recovery_latest.pth'
            old.parent.mkdir(parents=True)
            recovery.parent.mkdir(parents=True)
            torch.save({'env_steps': 100}, old)
            torch.save({'env_steps': 120}, recovery)
            self.assertEqual(selected_checkpoint(model, old, follow_latest=True), old)
            (model/'imported_checkpoint.json').write_text(json.dumps({'source': str(old)}))
            self.assertEqual(selected_checkpoint(model, old), old)
            self.assertEqual(selected_checkpoint(model, old, follow_latest=True), recovery)
            newer = recovery.with_name('step_00000130.pth')
            torch.save({'env_steps': 130}, newer)
            self.assertEqual(selected_checkpoint(model, old, follow_latest=True), newer)
            torch.save({'env_steps': 130}, recovery)
            self.assertEqual(selected_checkpoint(model, old, follow_latest=True), recovery)

    def test_start_continue_and_changed_config(self):
        cfg = dict(initialization={'sha256': 'bc'}, isaac={}, source_webots={}, seed=1, sac={},
                   training=dict(total_env_steps=100, batch_size=8, replay_size=100, checkpoint_every=10))
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with training_output(root, cfg, None) as (first, _):
                with self.assertRaises(RuntimeError), exclusive_writer(first):
                    pass
            with training_output(root, cfg, None) as (second, _):
                self.assertEqual(first.parent, second.parent)
                self.assertNotEqual(first, second)
            checkpoint = first/'checkpoints'/'step_00000002.pth'
            with training_output(root, cfg, checkpoint) as (model, session):
                self.assertEqual(model, first)
                csv = model/'training.csv'
                prefix = b'step,updates\r\n1,0\r\n2,0\r\n'
                tail = b'3,0\r\n4,0\r\n'
                csv.write_bytes(prefix + tail)
                restore_csv(checkpoint, model, len(prefix), 2, session)
                self.assertEqual(csv.read_bytes(), prefix)
                self.assertEqual((session/'training_after_checkpoint.rows').read_bytes(), tail)
            cfg['training']['total_env_steps'] += 1
            with training_output(root, cfg, checkpoint) as (model, _):
                self.assertEqual(model, first)
            with training_output(root, cfg, None) as (fresh, _):
                self.assertNotEqual(fresh, first)
