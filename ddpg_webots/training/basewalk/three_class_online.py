"""V28 online FIFO plus immutable experts classified by source task."""
import numpy as np
import torch
from .core import critic_extra_dim,CONTRACT
from .network_inputs import uses_body_gyro_observation, observation_dim
from .expert_pipeline import digest
from .learning import Replay
from .fixed_replay import quotas

CATEGORIES=('turn','stand','turn_stand_walk')


def expert_config_value(config,key):
    """Historical command schedules are provenance, not relabeled target-policy rollouts."""
    value=config[key]
    ignored={'turn_stand_walk':('angle_tolerance_deg','stand_angle_tolerance_deg'),
             'direction_training':('heading_tolerance_deg','goal_distance_m','goal_measure'),
             'stationary_reward':('angle_tolerance_deg',)}
    return {k:v for k,v in value.items() if k not in ignored[key]} if key in ignored else value


def validate_online_experts(data,cfg):
    from .turn_stand_walk import step_reward_inputs,revalue
    from .support_dataset import validate_arrays
    if data.get('contract') != CONTRACT:
        raise ValueError('V34 requires the 50D State contract; migrate old expert data explicitly')
    if (uses_body_gyro_observation(cfg)
            and not uses_body_gyro_observation(data['config'])):
        raise ValueError('Expert transitions require measured gyro Z at both ends')
    if data['config'].get('network_inputs') != cfg.get('network_inputs'):
        raise ValueError('Expert network input contract changed')
    if data['config'].get('reward_velocity_source_sha256') != cfg.get('reward_velocity_source_sha256'):
        raise ValueError('Expert Reward velocity implementation changed')
    if (cfg.get('reward',{}).get('biped_tracking',{}).get('velocity_source') == 'causal_mean'
            and data['config'].get('reward_history_sha256') != cfg.get('reward_history_sha256')):
        raise ValueError('Online expert Reward history implementation changed')
    for key in ('support_observation','physical_gait','contact_expert','turn_stand_walk',
                'stationary_reward','environment','direction_training','velocity_objective',
                'reward','staged_navigation','velocity_tracking','critic_observation',
                'actor_type','actor_hidden','critic_hidden'):
        if expert_config_value(data['config'],key)!=expert_config_value(cfg,key):
            raise ValueError('Online expert contract changed: '+key)
    for episode in data['episodes']:
        a=episode['arrays'];validate_arrays(a,cfg)
        if np.any(a['terminals']):raise ValueError('Failed expert trajectory')
        if cfg['reward'].get('biped_tracking'):
            from .reward_dispatch import revalue_arrays
            current=revalue_arrays(a,cfg)
            if cfg.get('critic_observation') == 'actor_plus_reward_velocity_mean':
                from .reward_history import sequence_context
                context = sequence_context(a,cfg)
                for name in ('critic_features', 'next_critic_features'):
                    if (np.asarray(a.get(name)).shape != context[name].shape
                            or not np.allclose(a[name], context[name], rtol=1e-5, atol=1e-7)):
                        raise ValueError('Online expert causal Critic context changed: '+name)
        else:
            current=revalue(step_reward_inputs(a),a['modes'],cfg)
        if not np.allclose(current,episode['revalued_rewards'],rtol=1e-5,atol=1e-7):
            raise ValueError('Online expert Reward changed; derive a new Reward dataset')
    return sum(len(e['arrays']['modes']) for e in data['episodes'])


def initialize(agent,cfg):
    spec=cfg['three_class_online']
    outcome=cfg['outcome_replay']
    if outcome.get('expert_success_fraction') != .2 or not outcome.get('protected_success_experts'):
        raise ValueError('V34 requires protected experts in 20% of success samples')
    if outcome['success_fraction'] != .7 or outcome['failure_fraction'] != .3:
        raise ValueError('V34 sampling contract is success70/failure30')
    if not cfg['training'].get('phase_fall_backfill') or cfg.get('initial_gait_phase',0.) != 0.:
        raise ValueError('V34 requires cycle finalization and reset phase zero')
    if cfg.get('outcome_replay',{}).get('capacity_fractions') != [0.7,0.3]:
        raise ValueError('V34 requires success/failure 70/30 capacity')
    for name in ('actor','dataset','critic'):
        if name == 'dataset' and spec.get('dataset_sha256') == 'GYRO_Z_MISSING':
            raise ValueError('Collect compatible expert transitions with measured gyro Z first')
        if name == 'critic' and spec.get('critic_sha256') == 'NOT_PRETRAINED':
            raise ValueError('New Reward/Critic is NOT_PRETRAINED. Run Pretrain Critic.cmd explicitly, then Check Training.cmd.')
        if digest(spec[name])!=spec[name+'_sha256']:raise ValueError('V28 artifact hash mismatch: '+name)
    if cfg['initialization'].get('bc_checkpoint')!=spec['actor']:
        raise ValueError('Online Actor path must match V28 BC')
    data=torch.load(spec['dataset'],map_location='cpu',weights_only=False)
    saved=torch.load(spec['critic'],map_location='cpu',weights_only=False)
    expected_reward_source=saved.get('config',{}).get('reward_source_sha256')
    if expected_reward_source is None and spec['critic_sha256']==cfg.get('migration',{}).get('source_critic_sha256'):
        expected_reward_source=cfg['migration'].get('imported_reward_source_sha256')
    if expected_reward_source != cfg['reward_source_sha256']:
        raise ValueError('Critic Reward implementation mismatch; prepare compatible data and pretrain Critic explicitly')
    if any(a.get('contract')!=CONTRACT for a in (data,saved)):
        raise ValueError('V28 requires freshly recorded 50D support State artifacts')
    if data.get('kind')!='support_state_expert_v28' or saved.get('kind')!='support_state_critic_v28':
        raise ValueError('Expected genuine three-class V28 artifacts')
    from .bc_rollout import check_rollout_critic, compatible_actor_identities
    actor_identities = compatible_actor_identities(cfg)
    if (saved.get('expert_replay_dataset_sha256') != spec['dataset_sha256']
            or spec['actor_sha256'] != cfg['initialization']['bc_sha256']
            or data.get('actor_sha256') not in actor_identities):
        raise ValueError('Dataset/Critic/BC identity mismatch')
    check_rollout_critic(saved, cfg, actor_identities)
    if saved['config'].get('fixed_cycle_commands')!=cfg.get('fixed_cycle_commands'):
        raise ValueError('Critic command scheduler mismatch')
    if not saved.get('actor_frozen_verified'):raise ValueError('Missing fixed-policy Critic provenance')
    validate_online_experts(data,cfg)
    for source_name, source in (('Online expert dataset',data['config']),('Frozen-BC Critic',saved['config'])):
        for key in ('support_observation','physical_gait','contact_expert','turn_stand_walk','stationary_reward','environment','direction_training','velocity_objective','reward','staged_navigation',
                    'velocity_tracking','critic_observation','actor_type','actor_hidden','critic_hidden'):
            matches = (expert_config_value(source,key)==expert_config_value(cfg,key)
                       if source_name=='Online expert dataset' else source[key]==cfg[key])
            if not matches:
                remedy = ('Prepare a compatible online expert dataset; this is not a Critic pretraining failure'
                          if source_name == 'Online expert dataset' else 'Pretrain Critic with the current configuration')
                raise ValueError(f'{source_name} configuration mismatch: {key}. {remedy}. '
                                 f'Saved={source[key]!r}; current={cfg[key]!r}')
        for key in ('gamma','tau','critic_lr','phase_fall_backfill'):
            if source['training'][key]!=cfg['training'][key]:raise ValueError('Pretraining mismatch: '+key)
        for key in ('control_time_step_ms','basic_time_step_ms'):
            if source['webots'][key]!=cfg['webots'][key]:raise ValueError('Control timing mismatch')
    groups=[[] for _ in CATEGORIES];dim=critic_extra_dim(cfg)
    width = observation_dim(cfg)
    for episode in data['episodes']:
        category=episode['category'];a=episode['arrays'];rewards=episode['revalued_rewards'];group=groups[CATEGORIES.index(category)]
        n=len(rewards)
        from .support_dataset import validate_arrays
        validate_arrays(a,cfg)
        expected={'observations':(n,width),'next_observations':(n,width),'executed_actions':(n,12),
                  'critic_features':(n,dim),'next_critic_features':(n,dim),'terminals':(n,),'modes':(n,)}
        for key,shape in expected.items():
            if np.asarray(a[key]).shape!=shape or not np.isfinite(a[key]).all():raise ValueError('Invalid expert '+key)
        if np.any(a['terminals']) or not np.isfinite(rewards).all():raise ValueError('Failed/nonfinite experts')
        if set(a['modes'].tolist())!={'turn':{0},'stand':{1},'turn_stand_walk':{0,1,2}}[category]:
            raise ValueError('Source class coverage mismatch')
        for i,r in enumerate(rewards):
            group.append((a['observations'][i],a['executed_actions'][i],r,a['next_observations'][i],False,
                          a['critic_features'][i],a['next_critic_features'][i],int(a['modes'][i])))
    from .outcome_replay import OutcomeReplay
    high = outcome.get('high_score')
    replay_type = OutcomeReplay
    options = {}
    if high:
        if not cfg['training'].get('online_turn_only') or cfg['training'].get('online_fixed_turn_sign') != 1.0:
            raise ValueError('High-score Replay requires fixed positive Turn-only training')
        from .high_score_replay import HighScoreReplay, HighScoreTransitionReplay
        ranking = high.get('ranking', 'cycle_reward')
        if ranking not in ('cycle_reward', 'cycle_speed_only', 'cycle_reward_only', 'transition_reward'):
            raise ValueError('Unknown high-score Replay ranking')
        replay_type = HighScoreTransitionReplay if ranking == 'transition_reward' else HighScoreReplay
        options = dict(spec=high, target_turn_rate=cfg['training']['online_turn_rate_rad_s'])
    agent.replay = replay_type(cfg['training']['replay_size'], cfg['seed'], dim,
        expert_success_fraction=outcome['expert_success_fraction'], obs_dim=width, **options)
    agent.replay.seed_experts(data['episodes'], turn_only=outcome.get('expert_turn_only', False),
                            turn_sign=outcome.get('expert_turn_sign'))
    if high:
        agent.replay.seed_high_score(high['source'], high['sha256'])
    agent.critic.load_state_dict(saved['critic'],strict=True)
    agent.critic_target.load_state_dict(saved['critic_target'],strict=True)
    agent.critic_optimizer.load_state_dict(saved['critic_optimizer'])
    agent.pretrained_critic_updates=saved['updates']
    agent.reusable_expert_source=dict(spec,source_counts=dict(zip(CATEGORIES,[len(g) for g in groups])))

