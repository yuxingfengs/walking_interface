"""Resume an Isaac omni policy at a fresh episode on its original command clock."""
import csv
import json
from pathlib import Path

import torch

from .sac import CONTRACT, SAC


def replay_arrays(replay):
    names = ['obs', 'next_obs', 'actions', 'rewards', 'terminal']
    if replay.feature_dim:
        names.extend(['critic_features', 'next_critic_features'])
    return names


def recovery_state(agent):
    replay = agent.replay
    state = dict(count=replay.count, cursor=replay.cursor, torch_rng=torch.get_rng_state(),
                 agent_rng=agent.rng.bit_generator.state, replay_rng=replay.rng.bit_generator.state)
    state['arrays'] = {name: torch.from_numpy(getattr(replay, name)[:replay.count])
                       for name in replay_arrays(replay)}
    if agent.log_alpha.is_cuda:
        state['cuda_rng'] = torch.cuda.get_rng_state(agent.log_alpha.device)
    return state


def save_recovery(agent, folder, step):
    agent.save(folder/'checkpoints'/'recovery_latest.pth', step, recovery=recovery_state(agent))


def restore(checkpoint, cfg, folder):
    path = Path(checkpoint).resolve(strict=True)
    parent = path.parent.parent
    if path.parent.name != 'checkpoints' or parent.parent != folder.parent:
        raise ValueError('Choose a checkpoint from an Isaac omni training run')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    old = saved['config']
    if saved['kind'] != 'sac_turn_checkpoint_v1' or saved['contract'] != CONTRACT:
        raise ValueError('Incompatible SAC checkpoint')
    # A longer run can continue the same task; compare all resolved task parameters.
    target = cfg['training']['total_env_steps']
    if target > old['training']['total_env_steps']:
        old = dict(old, training=dict(old['training'], total_env_steps=target),
                   budget_source=dict(old['budget_source'], effective_budget=target,
                                      sha256=cfg['budget_source']['sha256']))
    ignored = {'reward_module_path', 'source_hashes', 'initialization'}
    # Compute placement can change across machines without changing the task.
    if 'isaac' in old and 'isaac' in cfg:
        old_isaac = dict(old['isaac'])
        for key in ('device', 'learner_device'):
            if key in cfg['isaac']:
                old_isaac[key] = cfg['isaac'][key]
            else:
                old_isaac.pop(key, None)
        old = dict(old, isaac=old_isaac)
    if ({k: v for k, v in old.items() if k not in ignored} !=
            {k: v for k, v in cfg.items() if k not in ignored} or
            {k: v for k, v in old['initialization'].items() if k != 'actor_path'} !=
            {k: v for k, v in cfg['initialization'].items() if k != 'actor_path'}):
        raise ValueError('Checkpoint and current omni training configurations differ')
    step = saved['env_steps']
    if not 0 < step < cfg['training']['total_env_steps']:
        raise ValueError('Checkpoint outside the remaining training budget')
    agent = SAC(cfg)
    restore_agent(agent, saved)
    recovery = saved.get('recovery')

    falls = 0
    with (parent/'training.csv').open(newline='') as src, (folder/'training.csv').open('x', newline='') as dst:
        rows = csv.DictReader(src)
        writer = csv.DictWriter(dst, rows.fieldnames)
        writer.writeheader()
        for index in range(1, step + 1):
            row = next(rows, None)
            if row is None or int(row['step']) != index:
                raise ValueError(f'Missing historical training row {index}')
            writer.writerow(row)
            falls += int(row['fell'].lower() == 'true')
    if int(row['updates']) != agent.updates:
        raise ValueError('Checkpoint and CSV update counts differ')
    state = dict(next_step=step + 1, episode=int(row['episode']) + 1, falls=falls, stages={'step': step})
    metadata = dict(checkpoint=str(path), restored_steps=step, updates=agent.updates,
                    replay='restored' if recovery is not None else 'empty',
                    replay_count=agent.replay.count,
                    rng='restored' if recovery is not None else 'reinitialized',
                    physics='new_episode', state=state)
    (folder/'resume.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return dict(agent=agent, state=state)


def restore_agent(agent, saved):
    for key in ('actor', 'critic', 'critic_target', 'actor_optimizer', 'critic_optimizer', 'alpha_optimizer'):
        getattr(agent, key).load_state_dict(saved[key])
    with torch.no_grad():
        agent.log_alpha.copy_(saved['log_alpha'])
    agent.updates = saved['updates']
    recovery = saved.get('recovery')
    if recovery is not None:
        replay = agent.replay
        for name in replay_arrays(replay):
            getattr(replay, name)[:recovery['count']] = recovery['arrays'][name].numpy()
        replay.count, replay.cursor = recovery['count'], recovery['cursor']
        replay.rng.bit_generator.state = recovery['replay_rng']
        agent.rng.bit_generator.state = recovery['agent_rng']
        torch.set_rng_state(recovery['torch_rng'])
        if agent.log_alpha.is_cuda and 'cuda_rng' in recovery:
            torch.cuda.set_rng_state(recovery['cuda_rng'], agent.log_alpha.device)
