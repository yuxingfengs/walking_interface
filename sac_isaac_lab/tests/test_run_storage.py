from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'runner'))
from run_storage import exclusive_writer, restore_csv, training_output


class TrainingOutputTests(unittest.TestCase):
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
