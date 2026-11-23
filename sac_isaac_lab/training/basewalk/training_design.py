"""Opt-in stage-A command sampling and causal reward velocity averaging."""
from collections import deque
import math
import numpy as np


def turn_weight_bucket(turn_weight, boundaries):
    """Return the low/mixed/high turn-weight Replay bucket and label."""
    if (not isinstance(turn_weight, (int, float)) or isinstance(turn_weight, bool)
            or not math.isfinite(turn_weight) or not 0 <= turn_weight <= 1):
        raise ValueError('turn_weight must be finite within [0, 1]')
    values = np.asarray(boundaries, dtype=float)
    if (values.shape != (2,) or not np.isfinite(values).all() or
            not 0 < values[0] < values[1] < 1):
        raise ValueError('Replay boundaries must contain two ordered probabilities')
    index = int(np.searchsorted(values, turn_weight, side='right'))
    return index, ('low_turn', 'mixed', 'high_turn')[index]


def turn_stand_bucket(heading_error, tolerance_deg):
    """Label a Turn-only source State; aligned stabilization is Stand."""
    if (not isinstance(heading_error,(int,float)) or isinstance(heading_error,bool)
            or not math.isfinite(heading_error)):
        raise ValueError('heading_error must be finite')
    if (not isinstance(tolerance_deg,(int,float)) or isinstance(tolerance_deg,bool)
            or not math.isfinite(tolerance_deg) or tolerance_deg<=0):
        raise ValueError('stand tolerance must be positive')
    wrapped=math.atan2(math.sin(heading_error),math.cos(heading_error))
    return (1,'stand') if abs(wrapped)<=math.radians(tolerance_deg) else (0,'turn')


class BalancedDirections:
    """Cover every direction once, shuffled or in configured order."""
    def __init__(self, config, seed):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.pending = []
        self.stages = config.get('curriculum_stages_deg') or [list(config['angles_deg'])]
        self.stage = 0
        self.successes = {float(a): 0 for stage in self.stages for a in stage}
        focus = config.get('late_focus_sampling') or {}
        self.focus = dict(focus)
        window = int(focus.get('window', 1))
        self.recent_outcomes = {
            float(a): deque(maxlen=window) for stage in self.stages for a in stage}
        self.last_weights = {float(a): 1 for a in self.active_angles}
        self.focus_active = False

    @property
    def active_angles(self):
        return self.stages[self.stage]

    def _focus_weight(self, angle):
        history = self.recent_outcomes[float(angle)]
        if not history:
            return 1
        weights = self.focus['weights_by_success']
        successes = sum(history)
        return int(weights[min(successes, 2)])

    def next(self, step=None):
        start = self.focus.get('start_step', float('inf'))
        focus_now = bool(self.focus.get('enabled') and step is not None and step >= start)
        # Start a fresh weighted bag at the activation boundary instead of
        # finishing a potentially long uniform bag created before start_step.
        if focus_now and not self.focus_active:
            self.pending = []
        if not self.pending:
            self.focus_active = focus_now
            self.last_weights = {float(a): (self._focus_weight(a) if self.focus_active else 1)
                                 for a in self.active_angles}
            bag = [a for a in self.active_angles
                   for _ in range(self.last_weights[float(a)])]
            self.pending = (list(reversed(bag)) if self.config.get('sampling_order') == 'ordered'
                            else self.rng.permutation(bag).tolist())
        angle = self.pending.pop()
        speed = self.config['speed_m_s']
        if self.config.get('navigation_mode') == 'turn_only':
            return angle, [0., 0., 0.]
        if self.config.get('navigation_mode') in ('turn_then_forward', 'blended_turn_walk'):
            return angle, [speed, 0., 0.]
        theta = math.radians(angle)
        return angle, [speed*math.cos(theta), speed*math.sin(theta), 0.]

    def record_outcome(self, angle, strict_success):
        self.recent_outcomes[float(angle)].append(bool(strict_success))

    def reset_focus_history(self):
        for history in self.recent_outcomes.values():
            history.clear()
        self.pending = []
        self.last_weights = {float(a): 1 for a in self.active_angles}
        self.focus_active = False

    def sampling_state(self, angle):
        history = list(self.recent_outcomes[float(angle)])
        return dict(focus_active=self.focus_active,
                    sampling_weight=self.last_weights.get(float(angle), 1),
                    recent_strict_results=history,
                    recent_strict_successes=sum(history))

    def record_strict_success(self, angle):
        key = float(angle)
        self.successes[key] = self.successes.get(key, 0) + 1
        required = self.config.get('curriculum_successes_per_angle', 0)
        if required and self.stage + 1 < len(self.stages) and all(
                self.successes.get(float(a), 0) >= required for a in self.active_angles):
            self.stage += 1
            self.pending = []
            return True
        return False


def direction_progress(position, origin, initial_yaw, angle):
    """Signed progress relative to the commanded ray at episode reset."""
    theta = initial_yaw + math.radians(angle)
    delta = np.asarray(position)[:2] - np.asarray(origin)[:2]
    along = delta[0]*math.cos(theta) + delta[1]*math.sin(theta)
    cross = -delta[0]*math.sin(theta) + delta[1]*math.cos(theta)
    return float(along), float(cross)


def turn_goal_status(position, origin, heading_error, hold_steps,
                     max_translation, config):
    """Track sustained heading accuracy without allowing translational escape."""
    position = np.asarray(position, dtype=float)
    origin = np.asarray(origin, dtype=float)
    if (position.shape[0] < 2 or origin.shape[0] < 2 or
            not np.isfinite(position[:2]).all() or not np.isfinite(origin[:2]).all() or
            not math.isfinite(float(heading_error))):
        raise ValueError('Turn goal requires finite position and heading error')
    radial = float(np.linalg.norm(position[:2]-origin[:2]))
    maximum = max(float(max_translation), radial)
    error_deg = abs(math.degrees(math.atan2(
        math.sin(float(heading_error)), math.cos(float(heading_error)))))
    within = (error_deg <= config['heading_tolerance_deg'] and
              maximum <= config['turn_max_translation_m'])
    hold = int(hold_steps)+1 if within else 0
    return hold >= config['turn_success_hold_steps'], hold, maximum


class TrackingAverage:
    """Average planar velocity and source commands over the same causal window.

    Startup uses available samples, without zero padding. Yaw stays instantaneous.
    Call reset at every episode boundary; no future samples or Actor inputs.
    """
    def __init__(self, samples=1):
        self.samples = samples
        self.reset()

    def reset(self):
        self.history = deque(maxlen=self.samples)

    def advance(self, command, velocity):
        command, velocity = np.asarray(command).copy(), np.asarray(velocity).copy()
        self.history.append(np.array([command[:2], velocity[:2]], dtype=float))
        mean = np.mean(self.history, axis=0)
        command[:2], velocity[:2] = mean
        return command, velocity


class TurnRecoveryMonitor:
    """Detect a sustained large-error turn that is stalled or mostly rotating away."""
    def __init__(self, config):
        self.config = dict(config or {})
        self.samples = deque(maxlen=self.config.get('window_steps', 1))
        self.cooldown_until = 0

    def reset_window(self):
        self.samples.clear()

    def mark_recovery(self, step):
        self.cooldown_until = step + self.config.get('cooldown_steps', 0)
        self.reset_window()

    def advance(self, step, critic_only, turning, heading_error, yaw_rate):
        """Return recovery diagnostics after one eligible transition, otherwise None."""
        if (not self.config.get('enabled', False) or critic_only or not turning or
                step < self.cooldown_until):
            self.reset_window()
            return None
        error = abs(math.degrees(float(heading_error)))
        correct = float(heading_error) * float(yaw_rate) > 0.
        self.samples.append((error, correct))
        if len(self.samples) < self.samples.maxlen:
            return None
        comparison_samples = self.config.get('comparison_samples')
        if comparison_samples is not None:
            start_error = sum(sample[0] for sample in list(self.samples)[:comparison_samples]) / comparison_samples
            end_error = sum(sample[0] for sample in list(self.samples)[-comparison_samples:]) / comparison_samples
            worsening = end_error - start_error
            if (end_error <= self.config['min_heading_error_deg'] or
                    worsening < self.config['min_worsening_deg']):
                return None
            return dict(reason='turn_divergence', heading_error_deg=end_error,
                        window_start_mean_error_deg=start_error,
                        window_end_mean_error_deg=end_error,
                        worsening_deg=worsening,
                        turn_window_steps=len(self.samples), turn_diverging=True)
        improvement = self.samples[0][0] - error
        correct_fraction = sum(sample[1] for sample in self.samples) / len(self.samples)
        if error <= self.config['min_heading_error_deg']:
            return None
        stalled = improvement < self.config['min_improvement_deg']
        wrong_way = correct_fraction < self.config['min_correct_yaw_fraction']
        if not (stalled or wrong_way):
            return None
        return dict(heading_error_deg=error, improvement_deg=improvement,
                    correct_yaw_fraction=correct_fraction,
                    turn_window_steps=len(self.samples), turn_stalled=stalled,
                    turn_wrong_way=wrong_way)


def update_schedule(training, step, warmup_completed):
    if 'critic_warmup_env_steps' in training:
        critic_only = step <= training['critic_warmup_env_steps']
    else:
        critic_only = warmup_completed < training.get('critic_warmup_updates', 0)
    interval = training.get('critic_warmup_update_every', training['update_every']) if critic_only else training['update_every']
    return critic_only, interval


def validate_design(config):
    t = config['training']
    actor_interval = t.get('actor_update_every', t['update_every'])
    if (type(actor_interval) is not int or actor_interval < t['update_every']
        or actor_interval % t['update_every'] != 0):
        raise ValueError('actor_update_every must be a positive integer multiple of update_every')
    if 'critic_warmup_env_steps' in t:
        if type(t['critic_warmup_env_steps']) is not int or t['critic_warmup_env_steps'] < 0:
            raise ValueError('Invalid critic_warmup_env_steps')
        interval = t.get('critic_warmup_update_every', t['update_every'])
        if type(interval) is not int or interval < 1:
            raise ValueError('Invalid critic_warmup_update_every')
    if 'velocity_tracking' in config:
        tracking = config['velocity_tracking']
        if tracking['mode'] not in ('causal_planar_mean', 'causal_command_mean','command_ema','instantaneous'):
            raise ValueError('Unknown velocity_tracking.mode')
        if tracking['mode'] in ('causal_planar_mean','causal_command_mean') and (type(tracking.get('window_samples')) is not int or tracking['window_samples'] < 1):
            raise ValueError('Invalid velocity_tracking.window_samples')
    if 'direction_training' not in config:
        return
    d = config['direction_training']
    if d.get('sampling_order', 'shuffled') not in ('ordered', 'shuffled'):
        raise ValueError('Invalid direction_training.sampling_order')
    if 'critic_warmup_env_steps' not in t:
        raise ValueError('Direction training requires critic_warmup_env_steps')
    if d['stage'] != 'A' or d['mode'] not in ('balanced_fixed_episode','balanced_random_episode','ordered_anchor_episode') or 'straight' in config:
        raise ValueError('Only balanced direction stage A is implemented')
    if d['mode'] == 'balanced_random_episode':
        if d.get('sampling_order') != 'shuffled' or d.get('retry_same_angle') or d.get('promote_success_to_bc'):
            raise ValueError('Balanced random episodes require shuffled sampling without retry or Actor handoff')
        stages = d.get('curriculum_stages_deg')
        if stages is not None:
            required = d.get('curriculum_successes_per_angle')
            if (not isinstance(stages,list) or not stages or any(not isinstance(s,list) or not s for s in stages)
                or type(required) is not int or required < 1):
                raise ValueError('Invalid balanced random curriculum')
            if stages[0] != d['angles_deg'] or any(not set(stages[i-1]).issubset(stages[i]) for i in range(1,len(stages))):
                raise ValueError('Curriculum stages must start at angles_deg and expand monotonically')
    if d['mode'] == 'ordered_anchor_episode':
        if d.get('sampling_order') != 'ordered' or d.get('retry_same_angle') or d.get('promote_success_to_bc'):
            raise ValueError('Ordered anchor episodes require ordered sampling without retry or Actor handoff')
    angles = np.asarray(d['angles_deg'], dtype=float)
    duplicate_invalid = d['mode'] != 'ordered_anchor_episode' and len(set(angles)) != len(angles)
    if angles.ndim != 1 or not len(angles) or not np.isfinite(angles).all() or duplicate_invalid:
        raise ValueError('Invalid direction_training.angles_deg')
    if not math.isfinite(d['speed_m_s']) or d['speed_m_s'] <= 0:
        raise ValueError('Invalid direction_training.speed_m_s')
    if d.get('navigation_mode', 'body_fixed_translation') not in ('body_fixed_translation', 'turn_then_forward', 'turn_hold_forward', 'blended_turn_walk', 'turn_only'):
        raise ValueError('Invalid direction_training.navigation_mode')
    commands = (np.zeros((len(angles), 3)) if d.get('navigation_mode') == 'turn_only' else
                np.column_stack((np.full(len(angles), d['speed_m_s']), np.zeros((len(angles), 2))))
                if d.get('navigation_mode') in ('turn_then_forward', 'turn_hold_forward', 'blended_turn_walk') else
                np.column_stack((d['speed_m_s']*np.cos(np.radians(angles)),
                                 d['speed_m_s']*np.sin(np.radians(angles)), np.zeros(len(angles)))))
    limits = np.asarray(config['environment']['command_ranges'])
    if np.any(commands < limits[:, 0]) or np.any(commands > limits[:, 1]):
        raise ValueError('Direction training commands exceed environment ranges')
    if not math.isfinite(d['warmup_episode_seconds']) or d['warmup_episode_seconds'] <= 0:
        raise ValueError('Invalid warmup_episode_seconds')
    if d.get('navigation_mode') in ('turn_then_forward', 'turn_hold_forward'):
        tolerance = config['environment'].get('turn_forward_tolerance_deg')
        if not math.isfinite(tolerance) or not 0 < tolerance < 90:
            raise ValueError('Invalid environment.turn_forward_tolerance_deg')
        enter = config['environment'].get('turn_to_walk_tolerance_deg', tolerance)
        leave = config['environment'].get('walk_to_turn_tolerance_deg', enter)
        if (not all(math.isfinite(value) for value in (enter, leave)) or
                not 0 < enter <= leave < 90):
            raise ValueError('Invalid turn/walk hysteresis tolerances')
        if d.get('navigation_mode') == 'turn_hold_forward':
            hold = config['environment'].get('turn_hold_seconds')
            if (not isinstance(hold, (int, float)) or isinstance(hold, bool) or
                    not math.isfinite(hold) or hold <= 0):
                raise ValueError('Invalid environment.turn_hold_seconds')
    if d.get('navigation_mode') == 'blended_turn_walk':
        center = config['environment'].get('heading_blend_center_deg')
        width = config['environment'].get('heading_blend_width_deg')
        if (not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) for value in (center, width)) or
                not 0 < center < 90 or width <= 0):
            raise ValueError('Invalid continuous turn/walk heading blend')
    if d.get('navigation_mode') == 'turn_only':
        if (type(d.get('turn_success_hold_steps')) is not int or
                d['turn_success_hold_steps'] < 1):
            raise ValueError('Invalid direction_training.turn_success_hold_steps')
        for key in ('heading_tolerance_deg', 'turn_max_translation_m'):
            value = d.get(key)
            if (not isinstance(value, (int, float)) or isinstance(value, bool) or
                    not math.isfinite(value) or value <= 0):
                raise ValueError('Invalid direction_training.'+key)
    focus = d.get('late_focus_sampling')
    if focus:
        if type(focus.get('enabled')) is not bool:
            raise ValueError('late_focus_sampling.enabled must be boolean')
        if type(focus.get('start_step')) is not int or not 1 <= focus['start_step'] <= config['training']['total_env_steps']:
            raise ValueError('Invalid late_focus_sampling.start_step')
        if type(focus.get('window')) is not int or focus['window'] != 3:
            raise ValueError('late_focus_sampling.window must be 3')
        weights = focus.get('weights_by_success')
        if (not isinstance(weights, list) or len(weights) != 3 or
                any(type(value) is not int or value < 1 for value in weights) or
                not weights[0] >= weights[1] >= weights[2]):
            raise ValueError('Invalid late_focus_sampling.weights_by_success')
        if type(focus.get('reset_on_bc_recovery')) is not bool:
            raise ValueError('late_focus_sampling.reset_on_bc_recovery must be boolean')
        if d.get('sampling_order') != 'shuffled':
            raise ValueError('late_focus_sampling requires shuffled sampling')
    init = config['initialization']
    if init.get('reset_to_bc_on_failure'):
        if not init.get('use_bc', True) or not init.get('bc_checkpoint'):
            raise ValueError('BC recovery requires a BC checkpoint')
        if type(init.get('consecutive_falls')) is not int or init['consecutive_falls'] < 1:
            raise ValueError('BC recovery requires positive consecutive_falls')
        if init.get('clear_replay_on_bc_reset') or init.get('stall_timeout_steps', 0) != 0:
            raise ValueError('Direction BC recovery retains Replay and does not use translation stall')
        turn_reset = init.get('turn_bc_reset', {})
        if turn_reset.get('enabled'):
            if d.get('navigation_mode') not in ('turn_then_forward', 'blended_turn_walk', 'turn_only'):
                raise ValueError('Turn BC recovery requires heading-controlled navigation')
            for key in ('window_steps', 'cooldown_steps'):
                if type(turn_reset.get(key)) is not int or turn_reset[key] < 1:
                    raise ValueError('Invalid initialization.turn_bc_reset.' + key)
            numeric_keys = ['min_heading_error_deg']
            if 'comparison_samples' in turn_reset:
                numeric_keys.append('min_worsening_deg')
                samples = turn_reset.get('comparison_samples')
                if (type(samples) is not int or samples < 1 or
                        2*samples > turn_reset['window_steps']):
                    raise ValueError('Invalid initialization.turn_bc_reset.comparison_samples')
            else:
                numeric_keys.append('min_improvement_deg')
            for key in numeric_keys:
                value = turn_reset.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError('Invalid initialization.turn_bc_reset.' + key)
            if 'comparison_samples' not in turn_reset:
                fraction = turn_reset.get('min_correct_yaw_fraction')
                if not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
                    raise ValueError('Invalid initialization.turn_bc_reset.min_correct_yaw_fraction')
    if 'goal_distance_m' in d:
        for key in ('goal_distance_m', 'cross_track_tolerance_m', 'heading_tolerance_deg', 'velocity_tolerance_m_s'):
            if not math.isfinite(d[key]) or d[key] <= 0:
                raise ValueError('Invalid direction_training.' + key)
        if not isinstance(d.get('goal_as_terminal', False), bool):
            raise ValueError('Invalid direction_training.goal_as_terminal')
    if d.get('retry_same_angle'):
        if 'goal_distance_m' not in d or type(d.get('pause_after_failed_angles')) is not int or d['pause_after_failed_angles'] < 1:
            raise ValueError('Angle retries require a goal and positive pause_after_failed_angles')
        if type(d.get('successes_to_advance', 1)) is not int or d.get('successes_to_advance', 1) < 1:
            raise ValueError('Angle retries require positive successes_to_advance')
    if 'steps_per_angle' in d:
        if not d.get('retry_same_angle') or type(d['steps_per_angle']) is not int or d['steps_per_angle'] < 1:
            raise ValueError('Invalid steps_per_angle')
        warmup = d.get('critic_warmup_steps_per_angle')
        if type(warmup) is not int or not 0 <= warmup < d['steps_per_angle']:
            raise ValueError('Invalid critic_warmup_steps_per_angle')
        if t['learning_starts'] > warmup or t['batch_size'] > warmup:
            raise ValueError('Replay startup must fit within per-angle Critic warmup')
