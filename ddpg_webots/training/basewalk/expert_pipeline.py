"""Reusable successful BC trajectories and fixed-policy Critic initialization."""
import copy
import hashlib
import json
import shutil
from datetime import datetime, timezone
from uuid import uuid4
from pathlib import Path
import numpy as np
import torch
from .core import CONTRACT, OBS_DIM, critic_extra_dim


def signature(cfg):
    return dict(contract=CONTRACT, reward_revision='walk_body_forward_turn_tracking_v19', velocity_objective=cfg['velocity_objective'], staged_navigation=cfg['staged_navigation'],
        **({'stationary_reward': cfg['stationary_reward']} if cfg.get('stationary_reward') else {}),
        reward=cfg['reward'], environment=cfg['environment'],
        direction=cfg['direction_training'], velocity_tracking=cfg['velocity_tracking'],
        actor_type=cfg['actor_type'], actor_hidden=cfg['actor_hidden'],
        critic_hidden=cfg['critic_hidden'], critic_observation=cfg['critic_observation'],
        bc_sha256=cfg['initialization']['bc_sha256'],
        control_time_step_ms=cfg['webots']['control_time_step_ms'],
        basic_time_step_ms=cfg['webots']['basic_time_step_ms'],
        phase_fall_backfill=cfg['training'].get('phase_fall_backfill', False),
        stand_tolerance_deg=cfg['training']['success_expert_replay']['stand_tolerance_deg'])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)


class SuccessfulCollection:
    """Transient failed episodes are discarded; only successful episodes persist."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.episodes = []
        self.attempts = {}
        self.next_index = 0
        self.pending = []
        self.metadata = []
        destination = Path(cfg['expert_pipeline']['dataset'])
        existing = destination if destination.exists() else destination.with_suffix('.partial.pth')
        if existing.exists():
            saved = torch.load(existing, map_location='cpu', weights_only=False)
            if saved.get('signature') != signature(cfg):
                raise ValueError('Existing collection has a different task/reward contract')
            self.episodes = saved['episodes']
            self.attempts = saved.get('attempts',{})

    @property
    def count(self):
        return sum(len(e['transitions']) for e in self.episodes)

    def record_step(self, info, source_phase, next_phase):
        self.metadata.append(copy.deepcopy(dict(source_phase=source_phase, next_phase=next_phase,
            reward_terms=info.get('reward_terms', {}), position=info['position'],
            heading_error=info['heading_error'], fell=info['fell'] if 'fell' in info else False)))

    def append_transition(self, obs, action, reward, nxt, terminal, mode=None,
            critic_features=None, next_critic_features=None):
        self.pending.append(copy.deepcopy((mode, obs, action, reward, nxt, terminal,
                                          critic_features, next_critic_features)))

    def finish_episode(self, success, angle):
        task=self.cfg['expert_pipeline']['collection_task']
        key=f'{task}:{float(angle):g}'
        self.attempts[key]=self.attempts.get(key,0)+1
        quota = self.cfg['expert_pipeline']['successes_per_angle']
        accepted = bool(success and self.pending and
                        sum(e['angle'] == angle and e.get('task') == self.cfg['expert_pipeline']['collection_task'] for e in self.episodes) < quota)
        counts = [0, 0, 0]
        if accepted:
            if len(self.pending) != len(self.metadata):
                raise ValueError('Collection metadata and transitions disagree')
            for row in self.pending:
                counts[row[0]] += 1
            self.episodes.append(dict(angle=float(angle), success=True, task=self.cfg['expert_pipeline']['collection_task'],
                transitions=self.pending, metadata=self.metadata))
        self.pending, self.metadata = [], []
        return dict(admitted=accepted, added_turn=counts[0], added_stand=counts[1], added_walk=counts[2])

    def remaining_angles(self, task):
        angles=[0] if task=='stand' else self.cfg['direction_training']['angles_deg']
        quota=self.cfg['expert_pipeline']['successes_per_angle']
        cap=self.cfg['expert_pipeline'].get('max_attempts_per_angle',3)
        return [a for a in angles if sum(e['angle']==a and e.get('task')==task for e in self.episodes)<quota
                and self.attempts.get(f'{task}:{float(a):g}',0)<cap]

    def next_task(self):
        task=self.cfg['expert_pipeline']['collection_task']
        remaining=self.remaining_angles(task)
        if not remaining:raise RuntimeError('Collection attempts finished')
        angle=remaining[self.next_index % len(remaining)]
        self.next_index+=1
        return angle,[self.cfg['direction_training']['speed_m_s'],0.,0.]

    def task_complete(self, task):
        return not self.remaining_angles(task)

    def complete(self):
        return self.task_complete(self.cfg['expert_pipeline']['collection_task'])

    def dataset_complete(self):
        return {0, 2}.issubset({row[0] for e in self.episodes for row in e['transitions']})

    def export(self):
        pipeline = self.cfg['expert_pipeline']
        payload = dict(kind='successful_navigation_dataset_v1', signature=signature(self.cfg),
            config=self.cfg, complete=self.dataset_complete(), episodes=self.episodes, attempts=self.attempts,
            coverage_policy='available_successful_angles_turn_walk')
        # Partial successes remain inspectable but cannot initialize training.
        destination = Path(pipeline['dataset'])
        if not payload['complete']:
            destination = destination.with_suffix('.partial.pth')
        atomic_save(payload, destination)
        report = dict(complete=payload['complete'], transitions=self.count, attempts=self.attempts,
            covered_navigation_angles=sorted({e['angle'] for e in self.episodes if e.get('task')=='navigation'}),
            missing_categories=[name for i,name in enumerate(('turn','stand','walk')) if i != 1 and not any(row[0]==i for e in self.episodes for row in e['transitions'])],
            successes_by_angle={str(a): sum(e['angle'] == a for e in self.episodes)
                                for a in self.cfg['direction_training']['angles_deg']})
        destination.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


def read_dataset(path, cfg):
    data = torch.load(path, map_location='cpu', weights_only=False)
    if (data.get('kind') != 'successful_navigation_dataset_v1' or not data.get('complete')
            or data.get('signature') != signature(cfg)):
        raise ValueError('Expert dataset is incomplete or incompatible with current config')
    angles, groups = set(), [[], [], []]
    for episode in data['episodes']:
        if (not episode['success'] or not episode['transitions']
                or len(episode['metadata']) != len(episode['transitions'])
                or any(m['fell'] for m in episode['metadata'])):
            raise ValueError('Only complete successful episodes are accepted')
        if episode.get('task') == 'navigation':
            angles.add(episode['angle'])
        for row in episode['transitions']:
            mode, obs, action, reward, nxt, terminal, features, next_features = row
            if mode not in (0, 1, 2) or terminal not in (True, False, 0, 1):
                raise ValueError('Invalid expert transition labels')
            for value, shape in ((obs,(OBS_DIM,)), (nxt,(OBS_DIM,)), (action,(12,)),
                                 (features,(critic_extra_dim(cfg),)),
                                 (next_features,(critic_extra_dim(cfg),)), (reward,())):
                array = np.asarray(value)
                if array.shape != shape or not np.isfinite(array).all():
                    raise ValueError('Invalid expert transition shape or nonfinite data')
            groups[mode].append(row)
    if not groups[0] or not groups[2]:
        raise ValueError('Expert dataset requires nonempty Turn and Walk categories; full angle coverage is not required')
    return groups


class ExpertOnlySampler:
    def __init__(self, experts, ratios, seed):
        self.experts=experts
        self.ratios=np.asarray(ratios)/sum(ratios)
        self.rng=np.random.default_rng(seed)
        self.last_sample_counts={}

    def sample(self,n,**kwargs):
        from .fixed_replay import quotas
        counts=quotas(self.ratios,n,self.rng)
        self.last_sample_counts=dict(zip(('expert_turn','expert_stand','expert_walk'),counts))
        return self.experts.sample(*counts)


def initialize(agent, cfg):
    pipeline = cfg.get('expert_pipeline')
    if not pipeline:
        return
    mode = pipeline['mode']
    if mode == 'collect':
        agent.collector = SuccessfulCollection(cfg)
        agent.replay = None
        return
    groups = read_dataset(pipeline['dataset'], cfg)
    from .fixed_replay import FixedExpertStore, OnlineExpertReplay
    from .learning import TurnStandWarmupReplay
    spec = cfg['training']['success_expert_replay']
    capacities = [int(x*cfg['training']['replay_size']) for x in spec['capacity_ratios']]
    experts = FixedExpertStore(groups, capacities[1:], cfg['seed']+2, critic_extra_dim(cfg))
    if mode == 'pretrain':
        agent.replay = ExpertOnlySampler(experts,spec['ratios'][1:],cfg['seed'])
        return
    agent.replay = OnlineExpertReplay(int(capacities[0]), experts, spec['ratios'], cfg['seed']+1, critic_extra_dim(cfg), spec['online_turn_capacity'], spec['online_turn_fraction'])
    saved = torch.load(pipeline['critic_checkpoint'], map_location='cpu', weights_only=False)
    if (saved.get('kind') != 'expert_critic_v1' or saved.get('signature') != signature(cfg)
            or saved.get('dataset_sha256') != digest(pipeline['dataset'])
            or saved.get('gamma') != cfg['training']['gamma']
            or saved.get('critic_lr') != cfg['training']['critic_lr']
            or saved.get('tau') != cfg['training']['tau']):
        raise ValueError('Pretrained Critic does not match dataset/config')
    agent.critic.load_state_dict(saved['critic'], strict=True)
    agent.critic_target.load_state_dict(saved['critic_target'], strict=True)
    agent.critic_optimizer.load_state_dict(saved['critic_optimizer'])
    agent.pretrained_critic_updates = saved['updates']
    agent.reusable_expert_source = dict(dataset=str(pipeline['dataset']),
        dataset_sha256=saved['dataset_sha256'], critic_checkpoint=str(pipeline['critic_checkpoint']),
        critic_checkpoint_sha256=digest(pipeline['critic_checkpoint']))
    # Online update counters and Actor optimizer start fresh; Actor remains the BC policy.


def pretrain(cfg, updates):
    from .learning import create_algorithm
    cfg = copy.deepcopy(cfg)
    cfg['expert_pipeline']['mode'] = 'pretrain'
    if type(updates) is not int or updates < 1:
        raise ValueError('Pretraining updates must be positive')
    output = Path(cfg['expert_pipeline']['critic_checkpoint'])
    if cfg.get('experiment_scoped_assets'):
        from .experiment_layout import ROOT
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid4().hex[:8]
        output = ROOT/'experiments'/cfg['experiment']/'pretrain'/('run_' + stamp)/'critic.pth'
        cfg['expert_pipeline']['critic_checkpoint'] = str(output)
    agent = create_algorithm(cfg)
    before = copy.deepcopy(agent.actor.state_dict())
    output.parent.mkdir(parents=True, exist_ok=True)
    attempt = output.parent if cfg.get('experiment_scoped_assets') else output.parent / 'pretrain_history' / (
        output.stem + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid4().hex[:8])
    attempt.mkdir(parents=True, exist_ok=True)
    metrics_path = output.with_suffix('.metrics.jsonl')
    for previous in (output, metrics_path):
        if previous.exists():
            shutil.copy2(previous, attempt / ('previous_' + previous.name))
    print(f'Critic pretraining: {updates} updates; history: {attempt}', flush=True)
    # Keep the last usable checkpoint and its matching log until this attempt succeeds.
    (attempt/'effective_config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    attempt_metrics = attempt / metrics_path.name
    with attempt_metrics.open('x', encoding='utf-8') as log:
        for index in range(updates):
            metrics = agent.update(critic_only=True, update_actor=False)
            if index == 0 or (index+1) % 100 == 0 or index+1 == updates:
                log.write(json.dumps(metrics, allow_nan=False)+'\n')
    if any(not torch.equal(value, agent.actor.state_dict()[key]) for key, value in before.items()):
        raise RuntimeError('Actor changed during Critic pretraining')
    atomic_save(dict(kind='expert_critic_v1', signature=signature(cfg), config=cfg,
        dataset_sha256=digest(cfg['expert_pipeline']['dataset']), updates=agent.updates,
        gamma=cfg['training']['gamma'], tau=cfg['training']['tau'], critic_lr=cfg['training']['critic_lr'],
        critic=agent.critic.state_dict(), critic_target=agent.critic_target.state_dict(),
        critic_optimizer=agent.critic_optimizer.state_dict()), output)
    if attempt_metrics != metrics_path:
        shutil.copy2(attempt_metrics, metrics_path)
    if cfg.get('experiment_scoped_assets'):
        from .experiment_layout import publish
        publish(cfg, output)
    return output


def migrate_stationary_dataset(source, cfg):
    """Recompute successful V11 trajectories without generating robot transitions."""
    import math
    from .stationary_reward import NetSpeedWindow, stationary_terms
    output = Path(cfg['expert_pipeline']['dataset'])
    if output.exists():
        raise FileExistsError('Migration refuses to overwrite an existing dataset')
    original = torch.load(source, map_location='cpu', weights_only=False)
    old_cfg = original['config']
    read_dataset(source, old_cfg)
    if old_cfg.get('stationary_reward') or not cfg.get('stationary_reward'):
        raise ValueError('Migration requires legacy V11 data and a V12 destination')
    old_sig, new_sig = signature(old_cfg), signature(cfg)
    new_sig.pop('stationary_reward')
    new_sig['reward_revision'] = old_sig['reward_revision']
    new_sig['critic_observation'] = old_sig['critic_observation']
    if new_sig != old_sig:
        raise ValueError('Migration permits only stationary reward and Critic feature changes')
    data = copy.deepcopy(original)
    dt = cfg['webots']['control_time_step_ms']/1000.
    for episode in data['episodes']:
        window = NetSpeedWindow(cfg['stationary_reward']['window_samples'], dt)
        old_error = math.radians(episode['angle'])
        transitions = []
        for row, meta in zip(episode['transitions'], episode['metadata']):
            mode, obs, action, reward, nxt, terminal, raw, next_raw = row
            features = np.concatenate((raw, [window.speed, window.fill])).astype(np.float32)
            window.advance(meta['position'])
            next_features = np.concatenate((next_raw, [window.speed, window.fill])).astype(np.float32)
            old_terms = meta['reward_terms']
            # Legacy stored the squared yaw error; its magnitude is sufficient here.
            yaw_error = math.sqrt(max(0., -old_terms['turn_yaw_rate_error'] /
                                      (old_cfg['reward']['turn_yaw_rate_error']*dt)))
            terms = stationary_terms(old_error, meta['heading_error'], yaw_error,
                                     window.speed, window.ready, cfg, dt)
            total = sum(terms.values())
            reward = float(reward)-old_terms['turn_total']+total
            meta['original_reward_terms'] = copy.deepcopy(old_terms)
            meta['reward_terms'].update(terms, turn_total=total, turn_weighted=total)
            transitions.append((mode, obs, action, reward, nxt, terminal, features, next_features))
            old_error = meta['heading_error']
        episode['transitions'] = transitions
    data.update(config=copy.deepcopy(cfg), signature=signature(cfg),
                migration=dict(source=str(Path(source).resolve()), source_sha256=digest(source),
                               method='recompute_v12_from_positions_and_reward_terms'))
    atomic_save(data, output)
    read_dataset(output, cfg)
    return output
