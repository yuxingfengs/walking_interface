"""Step-based collection independent of logging windows and simulator APIs."""
from sensor_progress import ForwardProgressWatchdog
from contracts import ACTION_JOINT_ORDER, OBSERVATION_FIELDS


class ContinuousCollector:
    def __init__(self, env, algorithm, exploration, training, initialization,
                 emit=lambda *args, **kwargs: None, metadata=None):
        self.env, self.algorithm, self.exploration = env, algorithm, exploration
        self.training, self.initialization, self.emit = training, initialization, emit
        state = (metadata or {}).get('continuous_training', {})
        if metadata and not state:
            raise ValueError('Cannot resume legacy rollout training as continuous training')
        self.env_steps = int(state.get('env_steps', 0))
        self.steps_since_update = int(state.get('steps_since_update', 0))
        self.update_count = int((metadata or {}).get('global_step', 0))
        self.total_falls = int(state.get('total_falls', 0))
        self.bc_reset_count = int(state.get('bc_reset_count', 0))
        self.falls_without_goal = int(state.get('falls_without_goal', 0))
        self.progress = ForwardProgressWatchdog(initialization['stall_window_steps'],
            initialization.get('stall_min_forward_progress_m', .02))
        self.odometry_valid = False
        self.estimated_forward = 0.
        self.stalled_steps = 0
        self.window_reward = float(state.get('window_reward', 0))
        self.window_steps = int(state.get('window_steps', 0))

    def apply_learning_rate_stage(self):
        switch = self.training.get('lr_switch_env_steps', 0)
        if not switch:
            return
        # env_steps counts completed transitions; stage 2 starts at switch + 1.
        suffix = '_after_switch' if self.env_steps > switch else ''
        for name in ('actor', 'critic'):
            optimizer = getattr(self.algorithm, name + '_optimizer')
            for group in optimizer.param_groups:
                group['lr'] = self.training[name + '_lr' + suffix]

    def metadata(self):
        return dict(global_step=self.update_count, continuous_training=dict(
            env_steps=self.env_steps, steps_since_update=self.steps_since_update,
            total_falls=self.total_falls, bc_reset_count=self.bc_reset_count,
            falls_without_goal=self.falls_without_goal,
            window_reward=self.window_reward, window_steps=self.window_steps),
            exploration=self.exploration.state_dict(), sampling_mode='continuous',
            reward_version=getattr(self.env.task, 'reward_version', 'continuous_step_v1'))

    def exploration_action(self, observation):
        action = self.algorithm.act(observation)
        switch = self.training.get('noise_switch_env_steps', 0)
        if not switch:
            return self.exploration.apply(action)
        # env_steps counts completed transitions; the next action starts stage 2.
        scale = (self.training['action_noise_std_after_switch'] if self.env_steps >= switch
                 else self.training['action_noise_std_rad'])
        return self.exploration.apply(action, std_rad=scale)

    def reset(self):
        observation = self.env.reset(settle_steps=self.initialization.get('settle_steps', 64))
        # Measure the actual settled joints, not an artificial zero action.
        previous = tuple(observation.values[OBSERVATION_FIELDS.index(name)]
                         for name in ACTION_JOINT_ORDER)
        self.progress.reset()
        self.odometry_valid = False
        self.estimated_forward = 0.
        self.stalled_steps = 0
        return observation, previous

    def run(self):
        t, init = self.training, self.initialization
        observation, previous = self.reset()
        episode_step = 0
        self.emit('start', self)
        # total_env_steps is the cumulative budget, including resumed runs.
        while self.env_steps < t['total_env_steps']:
            self.algorithm.set_evaluation_mode()
            action = self.exploration_action(observation)
            if hasattr(self.algorithm, "limit_exploration"):
                action = self.algorithm.limit_exploration(observation, action)
            step = self.env.step(action, episode_step)
            transition = self.env.step_transition(step, previous)
            self.algorithm.observe(transition)
            observation, previous = step.next_observation, step.action.values
            self.env_steps += 1
            self.apply_learning_rate_stage()
            self.steps_since_update += 1
            episode_step += 1
            self.window_steps += 1
            self.window_reward += transition.reward
            fell, goal = step.info['fell'], step.info['reached_goal']
            self.odometry_valid = step.info["odometry_valid"]
            self.estimated_forward = step.info["estimated_forward"]
            self.stalled_steps = self.progress.observe(
                step.info['estimated_forward'], step.info['odometry_valid'])
            if fell:
                self.total_falls += 1
                self.falls_without_goal += 1
            elif goal:
                self.falls_without_goal = 0
            reason = None
            if self.falls_without_goal >= init['consecutive_falls']:
                reason = 'consecutive_falls'
            elif self.stalled_steps >= init['stall_timeout_steps']:
                reason = 'stall_timeout_steps'
            cutoff = init['bc_reset_max_env_steps']
            bc_allowed = cutoff == 0 or self.env_steps <= cutoff
            bc_applied = False
            if reason and init['reset_to_bc_on_failure'] and bc_allowed:
                bc_applied = self.algorithm.load_actor_initialization(
                    init['bc_checkpoint'], reset_optimizer=True,
                    actor_lr=(t['actor_lr_after_switch']
                              if t.get('lr_switch_env_steps', 0) and self.env_steps > t['lr_switch_env_steps']
                              else t['actor_lr']))
                if bc_applied:
                    if init.get('clear_replay_on_bc_reset', True):
                        self.algorithm.clear_experience()
                    self.steps_since_update = 0
                    self.bc_reset_count += 1
            if fell or goal or reason:
                if reason:
                    self.falls_without_goal = 0
                self.emit('reset', self, reason=reason or ('fall' if fell else 'goal'),
                          fell=fell, reached_goal=goal, bc_reset_applied=bc_applied,
                          bc_reset_allowed=bc_allowed, x_value=step.info['position'][0])
            # A BC recovery starts a new collection interval. Physical resets do not.
            warming_up = (getattr(self.algorithm, 'critic_warmup_completed', 0)
                          < getattr(self.algorithm, 'critic_warmup_updates', 0))
            update_interval = (t.get('critic_warmup_update_every_steps', t['update_every_steps'])
                               if warming_up else t['update_every_steps'])
            if (not bc_applied and self.steps_since_update >= update_interval
                    and self.algorithm.experience_size >= max(t['min_buffer_for_train'], self.algorithm.batch_size)):
                update = self.algorithm.update()
                self.steps_since_update = 0
                self.update_count += 1
                self.emit('update', self, update=update)
            if self.window_steps == t['log_every_steps'] or self.env_steps == t['total_env_steps']:
                self.emit('window', self, reward=self.window_reward, seq_len=self.window_steps)
                self.window_reward, self.window_steps = 0.0, 0
            if (fell or goal or reason) and self.env_steps < t['total_env_steps']:
                observation, previous = self.reset()
                episode_step = 0
        self.emit('end', self)


def train_continuous(host, config, *, resume_checkpoint_path=None, resume_metadata=None,
                     use_bc_initialization=False, **unused):
    """Wire the simulator-independent collector to existing logs and checkpoints."""
    from pathlib import Path
    from configs.runtime import project_path
    from experiment_paths import ExperimentPaths
    from exploration import GaussianActionNoise
    from helperfunction import _prepare_logging, _append_metric, _save_meta

    paths = ExperimentPaths(config['experiment']['name'], config['experiment']['seed'])
    paths.ensure_directories()
    host.resume_checkpoint_path = resume_checkpoint_path
    _prepare_logging(host, str(paths.train_log_dir), resume=resume_checkpoint_path is not None)
    host.global_step = int((resume_metadata or {}).get('global_step', 0))
    initialization = dict(config['initialization'], settle_steps=config['environment']['settle_steps'])
    initialization['bc_checkpoint'] = str(project_path(initialization['bc_checkpoint']))
    exploration = GaussianActionNoise(config['training']['action_noise_std_rad'], host.seed)
    if resume_checkpoint_path:
        exploration.load_state_dict((resume_metadata or {}).get('exploration'))
    saved_step = (resume_metadata or {}).get('last_eval_checkpoint_step')

    def save_candidate(collector):
        nonlocal saved_step
        if host.algorithm.actor_update_count == 0 or saved_step == collector.update_count:
            return
        path = paths.eval_checkpoint(collector.update_count)
        host.algorithm.save_checkpoint(str(path), metadata=collector.metadata())
        saved_step = collector.update_count
        print(f'[eval-ckpt] env_steps={collector.env_steps} saved -> {path}')

    def emit(kind, collector, **values):
        host.global_step = collector.update_count
        host.env_steps = collector.env_steps
        row = dict(type='session_event', event_name=kind, env_steps=collector.env_steps,
                   global_step=collector.update_count, sampling_mode='continuous',
                   reward_version=getattr(host.task, 'reward_version', 'continuous_step_v1'))
        if kind == 'update':
            update = values.pop('update')
            row.update(update.metrics)
            t = config['training']
            row['action_noise_std_rad'] = (t['action_noise_std_after_switch']
                if t.get('noise_switch_env_steps', 0) and collector.env_steps > t['noise_switch_env_steps']
                else t['action_noise_std_rad'])
            row.update(type='update', actor_loss=update.actor_loss, critic_loss=update.critic_loss,
                       buffer_len=host.algorithm.experience_size)
            every = config['training']['checkpoint_every_updates']
            if every and collector.update_count % every == 0:
                save_candidate(collector)
        elif kind == 'window':
            row.update(type='progress')
        elif kind == 'reset':
            row.update(bc_reset_count=collector.bc_reset_count)
            print(f"[reset] env_steps={collector.env_steps} reason={values['reason']} "
                  f"BC={values['bc_reset_applied']}")
        elif kind == 'start':
            row.update(event_name='session_start', total_env_steps=config['training']['total_env_steps'],
                       update_every_steps=config['training']['update_every_steps'],
                       critic_warmup_update_every_steps=config['training'].get(
                           'critic_warmup_update_every_steps', config['training']['update_every_steps']),
                       use_bc_initialization=use_bc_initialization)
        elif kind == 'end':
            row.update(event_name='session_end', total_falls=collector.total_falls)
            if host.algorithm.actor_update_count > 0:
                host.algorithm.save_checkpoint(str(paths.resume_checkpoint), metadata=collector.metadata())
            save_candidate(collector)
            _save_meta(host)
        row.update(odometry_valid=collector.odometry_valid,
                   odometry_forward_m=collector.estimated_forward,
                   estimated_forward_m=collector.progress.filtered_forward,
                   progress_anchor_m=collector.progress.anchor, stalled_steps=collector.stalled_steps,
                   stall_detector='contact_odometry_forward_progress_v1')
        row.update(values)
        if hasattr(host.task, 'yaw_rate'):
            row.update(heading_error_rad=host.task.heading_error,
                       yaw_rate_rad_s=host.task.yaw_rate,
                       heading_penalty=-host.task.config.heading_reward_weight*abs(host.task.heading_error))
        _append_metric(host, row)

    host.task.configure(stop_x=config['training']['stop_x_m'])
    collector = ContinuousCollector(host.env, host.algorithm, exploration, config['training'],
                                    initialization, emit, resume_metadata)
    if initialization.get('training_reset_mode') == 'world_reload':
        import os
        from configs.runtime import SNAPSHOT_ENV
        from world_reload_state import save_continuation
        def before_reload():
            metadata = collector.metadata()
            metadata['last_eval_checkpoint_step'] = saved_step
            save_continuation(os.environ[SNAPSHOT_ENV], host.algorithm, metadata)
            emit('world_reload', collector, replay_size=host.algorithm.experience_size)
        host.runtime.before_world_reload = before_reload
    collector.run()
