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


def save_recovery(agent, folder, step, stages=None):
    recovery = recovery_state(agent)
    if stages is not None:
        recovery['stages'] = stages
    agent.save(folder/'checkpoints'/'recovery_latest.pth', step, recovery=recovery)


def relocated_path(value, project_root):
    from configuration import relocated_experiment_path
    return str(relocated_experiment_path(value, project_root))


def normalize_legacy_walk_config(old, cfg):
    """Match checkpoints predating the experiment-owned random forward schedule."""
    if old.get('experiment_name') != 'walk_lateral_random_vx':
        return old
    commands = old['training']['walk_commands']
    if 'random_vx_range' in commands:
        if commands['random_vx_range'] != cfg['environment']['command_ranges'][0]:
            return old
        commands = commands.copy()
        del commands['random_vx_range']
        old = dict(old, training=dict(old['training'], walk_commands=commands))

    # Only the location of the budget changed. The caller still compares the
    # resolved commands, initialization, physics, optimizer and step budget.
    budget = old['budget_source']
    current = cfg['budget_source']
    if Path(current['path']).name == 'training_config.json' and budget['field'] == 'total_env_steps':
        root = Path(cfg['project_layout']['project_root'])
        prior_path = Path(relocated_path(budget['path'], root))
        experiment = Path(cfg['project_layout']['experiment_folder'])
        if prior_path == experiment/'config.json':
            old = dict(old, budget_source=dict(budget, path=current['path'], sha256=current['sha256']))
    return old


def restore(checkpoint, cfg, folder, session=None):
    path = Path(checkpoint).resolve(strict=True)
    parent = path.parent.parent
    project_root = Path(cfg['project_layout']['project_root'])
    runs_root = Path(cfg['project_layout']['runs_root'])
    origin = Path(relocated_path(parent.parent, project_root))
    if path.parent.name != 'checkpoints' or not (
            origin == runs_root or (origin.parent == runs_root and origin.name.startswith('config_'))):
        raise ValueError('Choose a checkpoint from an Isaac omni training run')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    old = normalize_legacy_walk_config(saved['config'], cfg)
    if saved['kind'] != 'sac_turn_checkpoint_v1' or saved['contract'] != CONTRACT:
        raise ValueError('Incompatible SAC checkpoint')
    # Normalize stored locations from before the directory migration, retaining every task check.
    old = dict(old)
    old_layout = dict(old['project_layout'])
    for key in ('project_root', 'config_path', 'experiment_folder', 'runs_root', 'reports_root'):
        old_layout[key] = relocated_path(old_layout[key], project_root)
    old['project_layout'] = old_layout
    old['budget_source'] = dict(old['budget_source'], path=relocated_path(old['budget_source']['path'], project_root))
    if 'initial_sac_checkpoint' in old['training']:
        old['training'] = dict(old['training'], initial_sac_checkpoint=relocated_path(
            old['training']['initial_sac_checkpoint'], project_root))
    # A longer run can continue the same task; compare all resolved task parameters.
    target = cfg['training']['total_env_steps']
    if target > old['training']['total_env_steps']:
        if cfg['isaac']['experiment_id'] == '004':
            old['budget_source'] = dict(old['budget_source'], source_webots_budget=cfg['budget_source']['source_webots_budget'])
        old = dict(old, training=dict(old['training'], total_env_steps=target),
                   budget_source=dict(old['budget_source'], effective_budget=target,
                                      sha256=cfg['budget_source']['sha256']))
    ignored = {'reward_module_path', 'source_hashes', 'initialization'}
    # Compute placement can change across machines without changing the task.
    if 'isaac' in old and 'isaac' in cfg:
        old_isaac = dict(old['isaac'])
        for key in ('asset', 'urdf_source', 'nao_meshes_path', 'standing_ground'):
            if key in old_isaac:
                old_isaac[key] = relocated_path(old_isaac[key], project_root)
        for key in ('device', 'learner_device', 'isaac_python', 'isaac_lab_path', 'isaac_sim_path'):
            if key in cfg['isaac']:
                old_isaac[key] = cfg['isaac'][key]
            else:
                old_isaac.pop(key, None)
        old = dict(old, isaac=old_isaac)
    if 'source_webots' in old:
        webots = dict(old['source_webots'])
        webots['world_template'] = relocated_path(webots['world_template'], project_root)
        webots['python'] = cfg['source_webots']['python']
        old['source_webots'] = webots
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

    if cfg['isaac']['experiment_id'] == '004' and 'stages' not in (recovery or {}):
        raise ValueError('Choose a recovery checkpoint containing the command schedule')
    source_csv = parent/'training.csv'
    falls = 0
    with source_csv.open(newline='', encoding='utf-8-sig') as src:
        rows = csv.DictReader(iter(src.readline, ''))
        for index in range(1, step + 1):
            row = next(rows, None)
            if row is None or int(row['step']) != index:
                raise ValueError(f'Missing historical training row {index}')
            falls += int(row['fell'].lower() == 'true')
        offset = src.tell()
    if int(row['updates']) != agent.updates:
        raise ValueError('Checkpoint and CSV update counts differ')
    from run_storage import restore_csv
    restore_csv(path, folder, offset, step, session)
    state = dict(next_step=step + 1, episode=int(row['episode']) + 1, falls=falls,
                 stages=(recovery or {}).get('stages', {'step': step}))
    metadata = dict(checkpoint=str(path), restored_steps=step, updates=agent.updates,
                    replay='restored' if recovery is not None else 'empty',
                    replay_count=agent.replay.count,
                    rng='restored' if recovery is not None else 'reinitialized',
                    physics='new_episode', state=state)
    record = json.dumps(metadata, indent=2)
    if session is not None:
        (session/'resume.json').write_text(record, encoding='utf-8')
    (folder/'resume.json').write_text(record, encoding='utf-8')
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
