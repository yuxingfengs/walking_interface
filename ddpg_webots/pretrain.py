"""Explicit offline Critic entry. Nothing is trained unless this entry is started without --check."""
import argparse
import copy
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from project import CODE, ROOT, digest, resolve_entry_config, training_configuration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', nargs='?')
    parser.add_argument('--config', type=Path, default=ROOT/'config.json')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--no-register', action='store_true', help='Save checkpoint without changing shared config')
    parser.add_argument('--result-file', type=Path)
    args = parser.parse_args()
    args.config, args.experiment = resolve_entry_config(args.config, args.experiment)
    from backend_dispatch import dispatch
    result = dispatch(args.config, args.experiment, mode='pretrain')
    if result is not None:
        return result
    sys.path.insert(0, str(CODE/'training'))
    import numpy as np
    import torch
    from basewalk.core import CONTRACT, critic_extra_dim, CRITIC_REWARD_MEAN_FIELDS
    from basewalk.network_inputs import observation_dim
    from basewalk.learning import create_algorithm, Replay
    from basewalk.bc_rollout import compatible_actor_identities, verify_frozen, check_rollout_critic
    from basewalk.expert_pipeline import atomic_save
    from basewalk.three_class_online import validate_online_experts
    from basewalk.configuration import training_config
    from basewalk.reward_dispatch import implementation
    cfg = training_config(training_configuration(args.config, args.experiment))
    implementation(cfg)
    if cfg.get('offline_walk_critic'):
        from basewalk.offline_walk_critic import pretrain
        return pretrain(cfg, check=args.check, no_register=args.no_register, result_file=args.result_file)
    if cfg['training'].get('online_fifo_only'):
        print('This experiment learns its Critic online; offline pretraining is not used.')
        print('No optimizer update, simulation or collection was started.')
        return
    if cfg.get('algorithm', 'ddpg') == 'sac':
        print('SAC trains its Actor, twin Critics, and entropy temperature online; offline DDPG Critic pretraining is not used.')
        print('No optimizer updates, simulation, or data collection were started.')
        return
    actor = Path(cfg['initialization']['bc_checkpoint'])
    if digest(actor) != cfg['initialization']['bc_sha256']:
        raise ValueError('BC identity mismatch')
    expert = cfg['three_class_online']
    if expert['dataset_sha256'] == 'GYRO_Z_MISSING':
        parser.exit(2, 'Pretraining prerequisite missing: compatible experts with measured gyro Z.\n')
    if digest(expert['dataset']) != expert['dataset_sha256']:
        raise ValueError('Online expert dataset hash mismatch')
    count = validate_online_experts(torch.load(expert['dataset'], map_location='cpu', weights_only=False), cfg)
    source = cfg['bc_rollout']['source']
    if source.get('sha256') == 'NOT_COLLECTED':
        parser.exit(2,'Pretraining prerequisite missing: collect a new frozen-policy rollout first with collect_bc.py. Old-policy rollouts cannot be relabeled.\n')
    path = Path(source['path'])
    if digest(path) != source['sha256']:
        raise ValueError('Frozen rollout hash mismatch')
    data = torch.load(path, map_location='cpu', weights_only=False)
    expected_source=data.get('reward_source_sha256',data.get('config',{}).get('reward_source_sha256'))
    if expected_source is None:
        expected_source=cfg.get('migration',{}).get('imported_reward_source_sha256')
    if expected_source!=cfg['reward_source_sha256']:
        raise ValueError('Rollout Reward implementation mismatch; derive compatible rollout rewards first')
    actor_identities = compatible_actor_identities(cfg)
    rollout_actor = data.get('actor_sha256')
    if (data.get('kind') != 'frozen_bc_rollout_v31' or data['contract'] != CONTRACT or
        not data['actor_frozen_verified'] or data['optimizer_updates'] != 0 or
        rollout_actor not in actor_identities or data['identity'] != actor_identities[rollout_actor]):
        raise ValueError('Frozen rollout/BC/Reward mismatch; derive compatible data before pretraining')
    a = data['arrays']; n = len(a['obs'])
    obs_dim = observation_dim(cfg)
    for key, width in [('obs', obs_dim), ('next_obs', obs_dim), ('actions', 12), ('rewards', 1), ('terminal', 1)]:
        if a[key].shape != (n, width) or not torch.isfinite(a[key]).all():
            raise ValueError('Invalid frozen rollout '+key)
    if n < cfg['training']['batch_size'] or not torch.all((a['terminal'] == 0) | (a['terminal'] == 1)):
        raise ValueError('Invalid rollout size/terminal masks')
    dim = critic_extra_dim(cfg)
    if dim:
        from basewalk.reward_history import sequence_context
        context = sequence_context(a, cfg)
        for key in ('critic_features', 'next_critic_features'):
            if (key not in a or tuple(a[key].shape) != (n,dim)
                    or not np.allclose(a[key].numpy(), context[key], rtol=1e-5, atol=1e-7)):
                raise ValueError('Rollout Critic history mismatch: '+key)
    print('Expert transitions:', count, '; frozen rollout:', n, '; Critic budget:', cfg['bc_rollout']['critic_updates'])
    if args.check:
        print('CHECK PASSED. No optimizer update, simulation or collection.')
        return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid4().hex[:6]
    folder = Path(cfg['project_layout']['runs_root'])/('critic_pretrain_'+stamp)
    folder.mkdir(parents=True, exist_ok=False)
    frozen_reward = folder/'source/experiment_reward.py'
    frozen_reward.parent.mkdir()
    shutil.copyfile(cfg['reward_module_path'], frozen_reward)
    if digest(frozen_reward) != cfg['reward_source_sha256']:
        raise ValueError('Reward changed while snapshotting')
    cfg['reward_module_path'] = str(frozen_reward.resolve())
    shutil.copyfile(cfg['project_layout']['config_path'], folder/'project_config.json')
    (folder/'effective_config.json').write_text(json.dumps(dict(config=cfg,kind='critic_pretrain'),indent=2),encoding='utf-8')
    runtime = copy.deepcopy(cfg); runtime.pop('three_class_online', None)
    runtime['bc_rollout']['collection_mode'] = False
    agent = create_algorithm(runtime); agent.replay = Replay(n, cfg['seed'], dim, obs_dim)
    for key in ('obs', 'actions', 'rewards', 'next_obs', 'terminal'):
        getattr(agent.replay, key)[:] = a[key].numpy()
    if dim:
        for key in ('critic_features', 'next_critic_features'):
            getattr(agent.replay, key)[:] = a[key].numpy()
    agent.replay.count = n
    agent.actor.requires_grad_(False); agent.actor_target.requires_grad_(False)
    before = copy.deepcopy(agent.actor.state_dict())
    with (folder/'critic_metrics.jsonl').open('w',encoding='utf-8') as stream:
        for step in range(cfg['bc_rollout']['critic_updates']):
            metrics = agent.update(critic_only=True, update_actor=False)
            if step == 0 or (step+1) % 100 == 0:
                stream.write(json.dumps(dict(step=step+1,**metrics))+'\n'); stream.flush()
                print('Critic', step+1, 'loss', metrics['critic_loss'], flush=True)
    verify_frozen(agent, before)
    payload = dict(kind='support_state_critic_v28',contract=CONTRACT,config=cfg,
        training_source='frozen_bc_rollout_v31',rollout_path=str(path),rollout_identity=data['identity'],
        rollout_actor_sha256=rollout_actor,
        actor_sha256=digest(actor),dataset_sha256=digest(path),
        expert_replay_dataset_sha256=expert['dataset_sha256'],actor_frozen_verified=True,
        updates=agent.updates,critic=agent.critic.state_dict(),critic_target=agent.critic_target.state_dict(),
        critic_optimizer=agent.critic_optimizer.state_dict())
    if dim:
        payload['critic_contract'] = dict(extra_fields=list(CRITIC_REWARD_MEAN_FIELDS),
            input_dim=obs_dim+dim+12, window_samples=cfg['velocity_tracking']['window_samples'])
    from basewalk.network_inputs import compact, critic_input_dim
    if compact(cfg):
        payload.setdefault('critic_contract',{}).update(input_dim=critic_input_dim(cfg,dim),physical_packed_dim=obs_dim+dim+12,
            extra_feature_dim=dim,network_inputs=cfg['network_inputs'])
    check_rollout_critic(payload,cfg,actor_identities)
    output=folder/'critic.pth'; atomic_save(payload,output)
    if args.result_file:
        args.result_file.write_text(json.dumps(dict(experiment=cfg['experiment'],
            critic=str(output.resolve()),critic_sha256=digest(output),
            updates=agent.updates,actor_frozen_verified=True),indent=2),encoding='utf-8')
    if args.no_register:
        print('Critic saved:',output,'; shared config not changed.',flush=True)
        return
    config_path=Path(cfg['project_layout']['config_path'])
    if digest(config_path)!=cfg['project_layout']['config_sha256']:
        raise ValueError('Root config changed during pretraining; checkpoint kept, root config not overwritten')
    document=json.loads(config_path.read_text(encoding='utf-8-sig'))
    target=document.get('experiment_overrides',{}).get(cfg['experiment'],document['training_config'])
    target.setdefault('three_class_online',{}).update(critic=str(output),critic_sha256=digest(output))
    temporary=config_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(document,ensure_ascii=False,indent=2),encoding='utf-8');temporary.replace(config_path)
    print('Critic saved:',output,'; online training was not started.')


if __name__ == '__main__':
    main()
