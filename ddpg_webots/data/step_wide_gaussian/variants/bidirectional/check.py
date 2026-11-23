"""Offline schedule, reward symmetry and physical-data coverage audit; no fitting."""
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

FOLDER = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER.parent / 'runtime'))
sys.path.insert(0, str(FOLDER.parent / 'runtime' / 'training'))
from project import training_configuration
from basewalk.configuration import training_config
from basewalk.fixed_cycle import FixedCycleStages
from basewalk.reward_dispatch import implementation


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    cfg = training_config(training_configuration(FOLDER / 'config.json'))
    reward = implementation(cfg)
    training = cfg['training']
    assert training['online_turn_only'] and training['online_fixed_turn_sign'] is None
    assert training['online_fixed_cycle_limit'] == 20
    assert training['online_turn_rate_rad_s'] == .110
    assert training['online_count_fall_as_attempt'] is True
    assert training['total_env_steps'] == 20000
    previous = json.loads((OUT / 'previous_config.json').read_text(encoding='utf-8'))
    current = json.loads((FOLDER / 'config.json').read_text(encoding='utf-8'))
    expected = copy.deepcopy(previous)
    expected['training_config']['training']['online_fixed_turn_sign'] = None
    expected['name'] = current['name']
    expected['status'] = current['status']
    assert expected == current, 'Unexpected configuration change'
    def stages(state=None):
        return FixedCycleStages(cfg, cycle_limit=20, state=state,
                                count_fall_as_attempt=True, turn_only=True,
                                fixed_turn_sign=None)
    stage = stages()
    commands = []
    # Reconstruct after every attempt to exercise reload/reset persistence.
    for index in range(80):
        commands.append(float(stage.command(0, 0)[2]))
        # Include falls, with and without a simultaneous phase wrap.
        stage.advance_phase(.9, .1 if index % 3 else .95, fell=index % 3 == 0)
        stage = stages(stage.state())
    np.testing.assert_allclose(commands, np.repeat([.11, -.11, .11, -.11], 20))
    assert stage.state() == dict(stage='turn', cycles=0, turn_sign=1.)
    stage.advance_phase(.1, .2)
    assert stage.cycles == 0
    stage.advance_phase(.9, .1, fell=True)
    assert stage.cycles == 1
    max_error = 0.
    for velocity in np.linspace(-1., 1., 81):
        args = ([0., 0., .11], [0., 0., velocity], [0., 0.], [0., 0., -1.], 0, cfg)
        positive = reward.motion_reward(*args)[0]
        negative = reward.motion_reward([0., 0., -.11], [0., 0., -velocity],
                                       *args[2:])[0]
        max_error = max(max_error, abs(positive - negative))
        assert abs(positive - 2 * np.exp(-((velocity - .11) / .70)**2)) < 1e-12
    assert max_error < 1e-12
    expert_path = cfg['three_class_online']['dataset']
    rollout_path = cfg['bc_rollout']['source']['path']
    expert = torch.load(expert_path, map_location='cpu', weights_only=False)
    rollout = torch.load(rollout_path, map_location='cpu', weights_only=False)
    def coverage(obs, modes):
        obs, modes = np.asarray(obs), np.asarray(modes).reshape(-1)
        turn = modes == 0
        return dict(total=len(obs), turn=int(turn.sum()),
                    positive=int((turn & (obs[:, 7] > 1e-6)).sum()),
                    negative=int((turn & (obs[:, 7] < -1e-6)).sum()))
    observations = np.concatenate([e['arrays']['observations'] for e in expert['episodes']])
    modes = np.concatenate([e['arrays']['modes'] for e in expert['episodes']])
    report = dict(optimizer_updates=0, simulation_started=False,
                  schedule_attempts=80, reload_after_each_attempt=True,
                  falls_count_once=True, mirrored_reward_max_error=max_error,
                  configuration_changes=['name', 'status', 'training.online_fixed_turn_sign'],
                  sources=dict(expert=dict(path=expert_path, sha256=sha(expert_path),
                                          coverage=coverage(observations, modes)),
                               rollout=dict(path=rollout_path, sha256=sha(rollout_path),
                                            coverage=coverage(rollout['arrays']['obs'], rollout['arrays']['modes']))),
                  decision='Reuse unchanged physical data, initial Actor and compatible pretrained Critic; retain original generation provenance.')
    (OUT / 'offline_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
