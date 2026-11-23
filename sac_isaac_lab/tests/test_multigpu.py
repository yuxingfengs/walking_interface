"""Hardware-independent regression for the multi-process collector and recovery protocol."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'runner'))
from configuration import load_configuration
from multigpu import local_curriculum, train
from multigpu_worker import simulate

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE/'training'))
from basewalk.fixed_cycle import online_stages


class FakeEnv:
    gravity = [0., 0., -1.]

    def reset(self, command):
        self.phase = 0.
        self.command = command
        return np.zeros(55, np.float32)

    def step(self, action):
        command = self.command
        self.phase = (self.phase + 0.125) % 1.
        stages = self.persistent_fixed_cycle_stages
        stages.advance_phase(0., self.phase)
        self.command = stages.command(0, 0)
        fell = stages.step % 23 == 0
        info = dict(command=command, actual_velocity=command, motion_phase='walk' if any(command[:2]) else 'turn',
                    reward_terms={'fall': -5. if fell else 0., 'tracking_yaw': 0.},
                    reward_velocity=command, executed_action=action)
        return np.zeros(55, np.float32), -5. if fell else 0., fell, False, info


def fake_worker(pipe, rank, cfg, folder, step):
    class Env(FakeEnv):
        def step(self, action):
            if rank == 0 and folder.name == 'first' and self.persistent_fixed_cycle_stages.step == 167:
                raise RuntimeError('injected stop')
            return super().step(action)
    try:
        simulate(pipe, Env(), online_stages(cfg, {'step': step}))
    except RuntimeError as err:
        pipe.send({'worker_error': str(err)})
    finally:
        pipe.close()


class MultiGPURecovery(unittest.TestCase):
    def test_uneven_tail_and_interrupted_cycle(self):
        import torch
        cfg = load_configuration(experiment='omni_step')
        cfg['multi_gpu'] = {'collectors': 3}
        cfg['training'].update(total_env_steps=503, batch_size=8, checkpoint_every=500, replay_size=1024)
        cfg['sac']['learning_starts'] = 495
        local_curriculum(cfg, 3)
        with tempfile.TemporaryDirectory() as tmp:
            first, second = [Path(tmp)/name for name in ('first', 'second')]
            first.mkdir()
            second.mkdir()
            with self.assertRaisesRegex(RuntimeError, 'injected stop'):
                train(cfg, first, 3, 'cpu', 90, worker_fn=fake_worker)
            checkpoint = first/'checkpoints/recovery_latest.pth'
            saved = torch.load(checkpoint, weights_only=True)
            self.assertEqual(saved['env_steps'], 501)
            self.assertTrue(any(saved['recovery']['collectors']['pending']))
            result = train(cfg, second, 3, 'cpu', 90, checkpoint, fake_worker)
            self.assertEqual(result['worker_steps'], [168, 168, 167])
            self.assertEqual(result['updates'], saved['updates'] + 2)
            final = torch.load(second/'checkpoints/recovery_latest.pth', weights_only=True)
            self.assertEqual(final['recovery']['count'], 503)
            self.assertFalse(any(final['recovery']['collectors']['pending']))
            steps = []
            for path in (first, second):
                with (path/'training.csv').open() as stream:
                    steps.extend(int(row['step']) for row in csv.DictReader(stream))
            self.assertEqual(sorted(steps), list(range(1, 504)))
            session = first/'sessions'/'resume'
            session.mkdir(parents=True)
            result = train(cfg, first, 3, 'cpu', 90, checkpoint, fake_worker,
                           session=session, worker_folder=session)
            with (first/'training.csv').open() as stream:
                steps = [int(row['step']) for row in csv.DictReader(stream)]
            self.assertEqual(result['step'], 503)
            self.assertEqual(sorted(steps), list(range(1, 504)))
            cfg['training']['total_env_steps'] = 506
            extension = first/'sessions'/'extension'
            extension.mkdir()
            result = train(cfg, first, 3, 'cpu', 90, checkpoint, fake_worker,
                           session=extension, worker_folder=extension)
            self.assertEqual(result['step'], 506)
            self.assertEqual(torch.load(checkpoint, weights_only=True)['recovery']['count'], 506)
            with (first/'training.csv').open() as stream:
                self.assertEqual(sorted(int(row['step']) for row in csv.DictReader(stream)), list(range(1, 507)))
            from scripts.tracking_report import write_report
            (first/'result.json').write_text(json.dumps(result), encoding='utf-8')
            (first/'effective_config.json').write_text(json.dumps(cfg), encoding='utf-8')
            report = write_report(first)
            self.assertEqual(report, first/'report/index.html')
            self.assertIn('"total":506', report.read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
