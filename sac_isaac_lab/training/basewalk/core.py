"""Versioned interfaces, command generation and reward (no Webots dependency)."""
import copy
import math
import numpy as np

JOINTS = tuple(side + joint for side in ('L', 'R') for joint in
               ('HipYawPitch', 'HipRoll', 'HipPitch', 'KneePitch', 'AnklePitch', 'AnkleRoll'))
DEFAULT_POSE = np.array([0., 0., -0.25, 0.5, -0.25, 0.] * 2, dtype=np.float32)
MINIMUM = np.array([-1.14529, -.379472, -1.77378, -.092346, -1.189516, -.39788,
                    -1.14529, -.738274, -1.77378, -.092346, -1.186448, -.768992], dtype=np.float32)
MAXIMUM = np.array([.740718, .790477, .48409, 2.112528, .922747, .769001,
                    .740718, .414754, .48409, 2.112528, .932056, .397935], dtype=np.float32)
FIELDS = (tuple('gravity_' + x for x in 'xyz') + tuple('gyro_' + x for x in 'xy') +
          ('command_vx', 'command_vy', 'command_wz', 'phase_cos', 'phase_sin') +
          tuple('position_' + j for j in JOINTS) + tuple('velocity_' + j for j in JOINTS) +
          tuple('previous_action_' + j for j in JOINTS) +
          ('yaw_rate', 'body_vx_m_s', 'body_vy_m_s', 'support_load_difference'))
OBS_DIM = len(FIELDS)
CRITIC_VELOCITY_FIELDS = ('body_vx_m_s', 'body_vy_m_s')
CRITIC_REWARD_MEAN_FIELDS = ('reward_mean_vx_m_s', 'reward_mean_vy_m_s', 'reward_mean_wz_rad_s')


def critic_extra_dim(config):
    if config.get('critic_observation') == 'actor_plus_reward_velocity_mean':
        return 3
    return {'actor_plus_turn_boundary': 3, 'actor_plus_navigation': 7, 'actor': 0, 'actor_plus_body_velocity': 2, 'actor_plus_body_velocity_net_speed': 4}.get(config.get('critic_observation', 'actor'), 0)


def critic_velocity_features(velocity):
    # Causal body-frame velocity at the same instant as the Actor observation.
    return vector(velocity, 3, 'Critic body velocity')[:2].copy()


CONTRACT = dict(version='state50_no_walk_flag_v34_v1', observation_fields=list(FIELDS),
                correction_inputs='gravity_gyro_xy_phase_joints_previous_action_actual_yaw_rate',
                action_joints=list(JOINTS), action_semantics='normalized_default_pose_offset',
                hip_coupling='mean_of_indices_0_and_6', default_pose=DEFAULT_POSE.tolist(),
                minimum=MINIMUM.tolist(), maximum=MAXIMUM.tolist(),
                actor_sensors='full_axis_IMU_gyro_xy_encoders_actual_velocity_and_filtered_foot_load_difference',
                support_semantics='(normal_left-normal_right)/(normal_left+normal_right+epsilon)',
                previous_action_semantics='previous_executed_normalized_action_after_clipping_and_hip_coupling; zero_at_reset',
                reward_velocity='causal_world_position_difference_to_body_xy_wrapped_heading_rate')


def vector(value, size, name):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f'{name} must contain {size} finite values')
    return result


def project_action(action):
    """The same projection must precede execution, Replay storage and Q evaluation."""
    a = np.clip(vector(action, 12, 'action'), -1., 1.).copy()
    a[0] = a[6] = (a[0] + a[6]) * .5
    return a


def joint_targets(action, scale, lower=MINIMUM, upper=MAXIMUM):
    lower, upper = vector(lower, 12, 'minimum'), vector(upper, 12, 'maximum')
    if scale <= 0 or np.any(lower >= upper):
        raise ValueError('Invalid action scale or joint limits')
    q = np.clip(DEFAULT_POSE + scale * project_action(action), lower, upper)
    low, high = max(lower[0], lower[6]), min(upper[0], upper[6])
    if low > high:
        raise ValueError('HipYawPitch limits do not intersect')
    q[0] = q[6] = np.clip((q[0] + q[6]) * .5, low, high)
    return q, (q - DEFAULT_POSE) / scale


def projected_gravity(imu_roll_pitch):
    # R2022b InertialUnit mounts at Rx(1.5708); remove that local rotation.
    roll, pitch = vector(imu_roll_pitch, 2, 'IMU roll/pitch')
    roll = float(roll) - 1.5708
    return np.array([math.sin(pitch), -math.sin(roll)*math.cos(pitch),
                     -math.cos(roll)*math.cos(pitch)], dtype=np.float32)


def observation(gravity, gyro, command, phase, position, velocity, previous_action, cfg, *, yaw_rate, body_velocity, support_load_difference,
                heading_error=0.):
    # A memoryless Actor needs a changing phase signal to produce a periodic
    # in-place gait for the zero velocity command. Historical experiments keep
    # the old masked-phase behavior unless this versioned preprocessing option
    # is explicitly enabled.
    active = float(cfg.get('phase_runs_at_zero_command', False) or
                   np.linalg.norm(command) > 1e-5)
    error = math.atan2(math.sin(float(heading_error)), math.cos(float(heading_error)))
    command = vector(command, 3, 'command').copy()
    obs = np.concatenate((vector(gravity, 3, 'gravity'), vector(gyro, 2, 'gyro_xy'),
        command, [active*math.cos(2*math.pi*phase), active*math.sin(2*math.pi*phase)],
        vector(position, 12, 'position') - DEFAULT_POSE,
        vector(velocity, 12, 'velocity') * cfg['joint_velocity_scale'],
        vector(previous_action, 12, 'previous action'),
        [yaw_rate*cfg['yaw_rate_scale']],
        vector(body_velocity, 2, 'Actor body velocity'),[support_load_difference]))
    if not -1.<=support_load_difference<=1.:raise ValueError('Support difference outside [-1,1]')
    return vector(obs, OBS_DIM, 'observation')


class Commands:
    def __init__(self, cfg, rng):
        self.cfg, self.rng = cfg, rng
        self.target = np.zeros(3, dtype=np.float32)
        self.current = self.target.copy()
        self.remaining = 0.

    def reset(self, fixed=None):
        self.target[:] = 0.
        self.current[:] = 0.
        self.remaining = 0.
        if fixed is not None:
            self.set_target(fixed)

    def set_target(self, value):
        value = vector(value, 3, 'command')
        limits = np.asarray(self.cfg['command_ranges'])
        if np.any(value < limits[:, 0]) or np.any(value > limits[:, 1]):
            raise ValueError('Command exceeds configured training ranges')
        self.target[:] = value

    def advance(self, dt, randomize=False):
        if randomize:
            self.remaining -= dt
            if self.remaining <= 0:
                limits = np.asarray(self.cfg['command_ranges'])
                self.target[:] = self.rng.uniform(limits[:, 0], limits[:, 1])
                if self.rng.random() < self.cfg['stand_probability']:
                    self.target[:] = 0.
                self.remaining = self.rng.uniform(*self.cfg['command_resampling_seconds'])
        limit = np.asarray(self.cfg['command_slew_per_second']) * dt
        self.current += np.clip(self.target-self.current, -limit, limit)
        return self.current.copy()


def reward(command, velocity, gravity, height, action, previous_action, joint_velocity,
           fell, cfg, dt, *, walking=None):
    error = vector(velocity, 3, 'velocity') - vector(command, 3, 'command')
    r = cfg['reward']
    terms = {
        'tracking_x': r['tracking_x']*math.exp(-float(error[0]**2)/r.get('tracking_sigma_x', r['tracking_sigma_linear'])),
        'tracking_y': r['tracking_y']*math.exp(-float(error[1]**2)/r.get('tracking_sigma_y', r['tracking_sigma_linear'])),
        'tracking_yaw': r['tracking_yaw']*math.exp(-float(error[2]**2)/r['tracking_sigma_yaw']),
        'survival': r['survival'],
        'orientation': -r['orientation']*float(np.square(gravity[:2]).sum()),
        'height': -r['height']*(height-cfg['environment']['base_height_target'])**2,
        'action_rate': -r['action_rate']*float(np.square(action-previous_action).sum()),
        'joint_velocity': -r['joint_velocity']*float(np.square(joint_velocity).sum()),
    }
    terms = {k: v*dt for k, v in terms.items()}
    if walking is True and r.get('walk_tracking_as_penalty', False):
        # Retain the smooth exponential tracking shape while making correct
        # tracking zero and every tracking error negative. Navigation progress
        # is then the main positive walk objective; heading improvement remains
        # a smaller signed transient shaping term.
        for key in ('tracking_x', 'tracking_y', 'tracking_yaw'):
            terms[key] -= r[key]*dt
        if r.get('walk_disable_survival', False):
            terms['survival'] = 0.
    terms['turn_yaw_rate_error'] = 0.
    if walking is False and r.get('turn_disable_positive_base_terms', False):
        # Turn has its own objective. Positive locomotion/survival terms otherwise
        # let a stable translating policy outscore a policy that actually turns.
        for key in ('tracking_x', 'tracking_y', 'tracking_yaw', 'survival'):
            terms[key] = 0.
        # Turn-only staged navigation owns the yaw-rate penalty and its
        # expert-scaled target. Do not also track the unscaled base command.
        staged_turn_only = (cfg['environment'].get('navigation_mode') == 'turn_only'
                            and cfg['environment'].get('reward_mode') == 'turn_walk')
        if not staged_turn_only:
            terms['turn_yaw_rate_error'] = (
                -r.get('turn_yaw_rate_error', 0.)*float(error[2]**2)*dt)
    terms['fall'] = -r['fall_penalty'] if fell else 0.
    return float(sum(terms.values())), terms


def progress_reward(previous_position, position, target_heading,
                    previous_heading, heading, cfg):
    """Potential differences for target-ray displacement and heading alignment."""
    previous_position = vector(previous_position, 3, 'previous position')
    position = vector(position, 3, 'position')
    values = (target_heading, previous_heading, heading)
    if not all(math.isfinite(value) for value in values):
        raise ValueError('Progress reward requires finite headings')
    direction = np.array([math.cos(target_heading), math.sin(target_heading)])
    forward_delta = float(direction @ (position[:2]-previous_position[:2]))
    previous_error = abs(math.atan2(math.sin(target_heading-previous_heading),
                                    math.cos(target_heading-previous_heading)))
    current_error = abs(math.atan2(math.sin(target_heading-heading),
                                   math.cos(target_heading-heading)))
    heading_delta = previous_error-current_error
    weights = cfg['reward']
    terms = {
        'forward_progress': weights.get('forward_progress', 0.)*forward_delta,
        'heading_progress': weights.get('heading_progress', 0.)*heading_delta,
    }
    return float(sum(terms.values())), terms


def turn_navigation_reward(previous_position, position, target_heading,
                           previous_heading, heading, cfg, dt,
                           desired_yaw_rate=0., actual_yaw_rate=0., actual_linear_velocity=None,
                           net_speed=None, window_ready=False, fell=False):
    """Reward heading improvement and penalize large error and translation."""
    previous_position = vector(previous_position, 3, 'previous position')
    position = vector(position, 3, 'position')
    values = (target_heading, previous_heading, heading, dt,
              desired_yaw_rate, actual_yaw_rate)
    if not all(math.isfinite(value) for value in values) or dt <= 0:
        raise ValueError('Turn reward requires finite headings and positive dt')
    previous_error = abs(math.atan2(math.sin(target_heading-previous_heading),
                                    math.cos(target_heading-previous_heading)))
    current_error = abs(math.atan2(math.sin(target_heading-heading),
                                   math.cos(target_heading-heading)))
    if cfg.get('stationary_reward'):
        from .stationary_reward import stationary_terms
        terms = stationary_terms(previous_error, current_error, actual_yaw_rate-desired_yaw_rate,
                                 net_speed, window_ready, cfg, dt, fell)
        return float(sum(terms.values())), terms
    distance = float(np.linalg.norm(position[:2]-previous_position[:2]))
    weights = cfg['reward']
    velocity_weight = weights.get('turn_linear_velocity_penalty', 0.)
    velocity_penalty = 0.
    if velocity_weight:
        velocity_xy = vector(actual_linear_velocity, 2, 'Turn actual body vx/vy')
        velocity_penalty = -velocity_weight*float(np.square(velocity_xy).sum())*dt
    terms = {
        'turn_translation': -weights.get('turn_translation_penalty', 0.)*distance,
        'turn_linear_velocity': velocity_penalty,
        'turn_heading_progress': weights.get('turn_heading_progress', 0.)*(previous_error-current_error),
        'turn_heading_error': -weights.get('turn_heading_error', 0.)*(current_error/math.pi)**2*dt,
        'turn_yaw_rate_error': -weights.get('turn_yaw_rate_error', 0.)*
            (float(actual_yaw_rate)-float(desired_yaw_rate))**2*dt,
        'walk_forward_progress': 0.,
        'walk_heading_progress': 0.,
        'walk_heading_error': 0.,
        'walk_cross_track': 0.,
    }
    return float(sum(terms.values())), terms


def walk_navigation_reward(previous_position, position, target_heading,
                           previous_heading, heading, cfg, dt=1., origin=None):
    """Reward signed target-ray progress and weaker heading correction."""
    previous_position = vector(previous_position, 3, 'previous position')
    position = vector(position, 3, 'position')
    values = (target_heading, previous_heading, heading)
    if not all(math.isfinite(value) for value in values):
        raise ValueError('Navigation reward requires finite headings')
    delta = position[:2]-previous_position[:2]
    direction = np.array([math.cos(target_heading), math.sin(target_heading)])
    previous_error = abs(math.atan2(math.sin(target_heading-previous_heading),
                                    math.cos(target_heading-previous_heading)))
    current_error = abs(math.atan2(math.sin(target_heading-heading),
                                   math.cos(target_heading-heading)))
    weights = cfg['reward']
    direction_cfg = cfg.get('direction_training', {})
    heading_tolerance = math.radians(direction_cfg.get('heading_tolerance_deg', 1.))
    cross_tolerance = direction_cfg.get('cross_track_tolerance_m', 1.)
    origin = previous_position if origin is None else vector(origin, 3, 'episode origin')
    origin_delta = position[:2]-origin[:2]
    cross_track = float(-origin_delta[0]*math.sin(target_heading)
                        + origin_delta[1]*math.cos(target_heading))
    terms = {
        'turn_translation': 0.,
        'turn_heading_progress': 0.,
        'turn_heading_error': 0.,
        'walk_forward_progress': weights.get('walk_forward_progress', 0.)*float(direction @ delta),
        'walk_heading_progress': weights.get('walk_heading_progress', 0.)*(previous_error-current_error),
        'walk_heading_error': -weights.get('walk_heading_error', 0.)*
            (current_error/heading_tolerance)**2*dt,
        'walk_cross_track': -weights.get('walk_cross_track', 0.)*
            (cross_track/cross_tolerance)**2*dt,
    }
    return float(sum(terms.values())), terms


def blended_navigation_reward(previous_position, position, target_heading,
                              previous_heading, heading, turn_weight, walk_weight,
                              tracking_x, tracking_y, cfg, dt=1.):
    """Continuously blend task-aligned turn and walk objectives."""
    previous_position = vector(previous_position, 3, 'previous position')
    position = vector(position, 3, 'position')
    values = (target_heading, previous_heading, heading, turn_weight, walk_weight,
              tracking_x, tracking_y, dt)
    if (not all(math.isfinite(value) for value in values) or dt <= 0 or
            turn_weight < 0 or walk_weight < 0 or
            not math.isclose(turn_weight+walk_weight, 1., rel_tol=0., abs_tol=1e-6)):
        raise ValueError('Blended reward requires finite normalized weights')
    delta = position[:2]-previous_position[:2]
    direction = np.array([math.cos(target_heading), math.sin(target_heading)])
    previous_error = abs(math.atan2(math.sin(target_heading-previous_heading),
                                    math.cos(target_heading-previous_heading)))
    current_error = abs(math.atan2(math.sin(target_heading-heading),
                                   math.cos(target_heading-heading)))
    heading_progress = previous_error-current_error
    weights = cfg['reward']
    terms = {
        'turn_translation': -weights.get('turn_translation_penalty', 0.)*
            float(np.linalg.norm(delta)),
        'turn_heading_progress': weights.get('turn_heading_progress', 0.)*heading_progress,
        'turn_heading_error': -weights.get('turn_heading_error', 0.)*
            (current_error/math.pi)**2*dt,
        'turn_yaw_rate_error': 0.,
        'walk_forward_progress': weights.get('walk_forward_progress', 0.)*
            float(direction @ delta),
        'walk_heading_progress': weights.get('walk_heading_progress', 0.)*heading_progress,
        'walk_heading_error': 0.,
        'walk_cross_track': 0.,
        'turn_weight': float(turn_weight),
        'walk_weight': float(walk_weight),
    }
    turn_total = (terms['turn_translation'] + terms['turn_heading_progress'] +
                  terms['turn_heading_error'])
    walk_total = (terms['walk_forward_progress'] + terms['walk_heading_progress'] +
                  float(tracking_x) + float(tracking_y))
    # V17 optionally gates translation faster than heading correction. The
    # default exponent of one preserves historical V16 behavior.
    translation_power = float(weights.get('turn_translation_weight_power', 1.))
    turn_weighted = (float(turn_weight)**translation_power*terms['turn_translation'] +
                     float(turn_weight)*(terms['turn_heading_progress'] +
                                         terms['turn_heading_error']))
    terms.update(
        turn_total=turn_total,
        walk_total=walk_total,
        turn_weighted=turn_weighted,
        walk_weighted=float(walk_weight)*walk_total,
    )
    return terms['turn_weighted']+terms['walk_weighted'], terms


def staged_navigation_reward(previous_position, position, target_heading,
                             previous_heading, heading, walking, cfg, dt=1., origin=None,
                             desired_yaw_rate=0., actual_yaw_rate=0., actual_linear_velocity=None,
                           net_speed=None, window_ready=False, fell=False):
    """Dispatch to the reward function selected by the existing motion gate."""
    if walking:
        return walk_navigation_reward(previous_position, position, target_heading,
                                      previous_heading, heading, cfg, dt, origin)
    return turn_navigation_reward(previous_position, position, target_heading,
                                  previous_heading, heading, cfg, dt,
                                  desired_yaw_rate=desired_yaw_rate,
                                  actual_yaw_rate=actual_yaw_rate,
                                  actual_linear_velocity=actual_linear_velocity,
                                  net_speed=net_speed, window_ready=window_ready, fell=fell)


def validate(config):
    c = copy.deepcopy(config)
    from .network_inputs import (REVISION, ACTOR_NO_LINEAR_VELOCITY, ACTOR_NO_LINEAR_OR_SUPPORT,
                                 ACTOR_NO_SUPPORT, ACTOR_BODY_GYRO, SHARED_VELOCITY_EMA, SHARED_GYRO_EMA, SHARED_GYRO_RAW, SHARED_YAW_ONLY, uses_body_gyro_observation)
    if c.get('actor_observation') not in (None, ACTOR_NO_LINEAR_VELOCITY, ACTOR_NO_LINEAR_OR_SUPPORT,
                                         ACTOR_NO_SUPPORT, ACTOR_BODY_GYRO, SHARED_VELOCITY_EMA, SHARED_GYRO_EMA, SHARED_GYRO_RAW, SHARED_YAW_ONLY):
        raise ValueError('Unsupported Actor observation revision')
    if c.get('actor_observation') and (c.get('network_inputs') != REVISION or c.get('actor_type') != 'concat_mlp'):
        raise ValueError('Actor observation ablation requires compact concat Actor')
    if uses_body_gyro_observation(c) and not c.get('physical_gait', {}).get('imu_full_axes'):
        raise ValueError('Body gyro Actor requires the local NAO model')
    if c.get('actor_observation') == SHARED_VELOCITY_EMA and not c['reward']['biped_tracking'].get('turn_yaw_lowpass_tau_s', 0) > 0:
        raise ValueError('Shared State requires the causal Turn yaw EMA')
    if c.get('network_inputs') not in (None, REVISION):
        raise ValueError('Unsupported network input revision')
    if c.get('network_inputs') == REVISION and (c.get('actor_type') != 'concat_mlp'
            or not c.get('fixed_cycle_commands',{}).get('enabled')):
        raise ValueError('Compact inputs require fixed Turn/Walk commands and concat Actor')
    if c.get('critic_observation') not in ('actor', 'actor_plus_reward_velocity_mean'):
        raise ValueError('Unsupported Actor/Critic State contract')
    if c.get('critic_observation') == 'actor_plus_reward_velocity_mean':
        tracking = c.get('velocity_tracking', {})
        if (c.get('reward', {}).get('biped_tracking', {}).get('revision') != 'causal_velocity_widths_v1'
                or tracking.get('mode') != 'causal_command_mean'
                or type(tracking.get('window_samples')) is not int or tracking['window_samples'] < 1
                or tracking.get('reset_on_command_change') is not True):
            raise ValueError('Causal Critic requires command-segment Reward history')
    support = c.get('support_observation', {})
    if (support.get('version') != 'normal_load_difference_v1'
            or not 0 < support.get('filter_alpha', 0) <= 1
            or not math.isfinite(support.get('epsilon', 0))
            or support.get('epsilon', 0) <= 0):
        raise ValueError('V28 requires valid current support observation settings')
    if not c.get('contact_expert', {}).get('enabled') or not c.get('physical_gait', {}).get('imu_full_axes'):
        raise ValueError('V28 requires foot sensors and full-axis IMU')
    if c.get('cycle_tracking_reward') and not c['training'].get('phase_fall_backfill'):
        raise ValueError('Cycle Reward requires the phase finalization buffer')
    if c.get('algorithm', 'ddpg') not in ('ddpg', 'sac'):
        raise ValueError('Supported algorithms are DDPG and SAC')
    if c.get('actor_type') not in ('film_gait_correction','concat_mlp'):
        raise ValueError('Unknown Actor architecture')
    if c['actor_type']=='film_gait_correction':
        if not c.get('correction_hidden') or any(type(v) is not int or v <= 0 for v in c['correction_hidden']):
            raise ValueError('Invalid correction_hidden')
        if not math.isfinite(c['correction_scale']) or not 0 < c['correction_scale'] <= 1:
            raise ValueError('Invalid correction_scale')
    yaw_source = c['environment'].get('yaw_source')
    if yaw_source not in ('supervisor_truth', 'integrated_yaw_rate'):
        raise ValueError('Unknown yaw_source')
    if yaw_source == 'integrated_yaw_rate':
        if c['environment'].get('yaw_rate_source') not in ('supervisor_delta_sim_only', 'gyro_z'):
            raise ValueError('Integrated heading requires an explicit yaw_rate_source')
        if not math.isfinite(c['environment'].get('yaw_rate_bias', 0.)):
            raise ValueError('Invalid yaw_rate_bias')
    if not math.isfinite(c['environment']['yaw_rate_scale']) or c['environment']['yaw_rate_scale'] <= 0:
        raise ValueError('Invalid yaw_rate_scale')
    if type(c['environment'].get('phase_runs_at_zero_command', False)) is not bool:
        raise ValueError('Invalid phase_runs_at_zero_command')
    for key in ('heading_kp', 'heading_max_rate'):
        if not math.isfinite(c['environment'][key]) or c['environment'][key] <= 0:
            raise ValueError('Invalid '+key)
    for key in ('forward_progress', 'heading_progress', 'turn_translation_penalty', 'turn_linear_velocity_penalty',
                'turn_heading_progress', 'turn_heading_error',
                'turn_yaw_rate_error', 'walk_forward_progress', 'walk_heading_progress',
                'walk_heading_error', 'walk_cross_track'):
        value = c['reward'].get(key, 0.)
        if not math.isfinite(value) or value < 0:
            raise ValueError('Invalid reward.'+key)
    if not isinstance(c['reward'].get('turn_disable_positive_base_terms', False), bool):
        raise ValueError('Invalid reward.turn_disable_positive_base_terms')
    for key in ('walk_tracking_as_penalty', 'walk_disable_survival'):
        if not isinstance(c['reward'].get(key, False), bool):
            raise ValueError('Invalid reward.'+key)
    yaw_limits = c['environment']['command_ranges'][2]
    if c['environment']['heading_max_rate'] > min(-yaw_limits[0], yaw_limits[1]):
        raise ValueError('Heading rate limit exceeds command range')
    for role in ('actor_hidden', 'critic_hidden'):
        if not c[role] or any(type(v) is not int or v <= 0 for v in c[role]):
            raise ValueError('Hidden sizes must be positive integers')
    w, e, t = c['webots'], c['environment'], c['training']
    if c.get('critic_observation', 'actor') not in ('actor_plus_turn_boundary', 'actor_plus_navigation', 'actor', 'actor_plus_body_velocity', 'actor_plus_body_velocity_net_speed', 'actor_plus_reward_velocity_mean'):
        raise ValueError('Invalid critic_observation')
    if critic_extra_dim(c) and ((t.get('warmup_expert_replay', {}).get('mode') != 'turn_stand'
                                and not t.get('success_expert_replay')
                                and c.get('three_class_pipeline', {}).get('task') != 'turn_stand_walk_three_class')
                                or t.get('stratified_replay')):
        raise ValueError('Privileged Critic velocity requires online-collected Turn/Stand Replay')
    if w['control_time_step_ms'] <= 0 or w['basic_time_step_ms'] <= 0 or w['control_time_step_ms'] % w['basic_time_step_ms']:
        raise ValueError('Control interval must be a positive multiple of physics interval')
    limits = np.asarray(e['command_ranges'])
    if limits.shape != (3, 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] > 0) or np.any(limits[:, 1] < 0) or np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError('Command ranges must be finite, increasing and include zero')
    for key in ('action_scale', 'episode_seconds', 'base_height_target', 'gait_frequency_hz', 'command_timeout_seconds'):
        if not math.isfinite(e[key]) or e[key] <= 0:
            raise ValueError('Invalid environment.' + key)
    if not 0 < e['velocity_filter'] <= 1 or not 0 <= e['stand_probability'] <= 1:
        raise ValueError('Invalid filter or probability')
    if type(e.get('disable_time_limit', False)) is not bool:
        raise ValueError('environment.disable_time_limit must be boolean')
    if (e['fall_height'] <= 0 or e['fall_height'] >= e['base_height_target'] or
        not 0 < e['fall_tilt_rad'] < math.pi/2 or not 0 < e['motor_velocity_scale'] <= 1 or
        e['joint_margin'] < 0 or type(e['settle_steps']) is not int or e['settle_steps'] < 1):
        raise ValueError('Invalid termination/reset/motor configuration')
    slew = vector(e['command_slew_per_second'], 3, 'command slew')
    resampling = vector(e['command_resampling_seconds'], 2, 'resampling interval')
    if np.any(slew <= 0) or not 0 < resampling[0] <= resampling[1]:
        raise ValueError('Invalid command timing')
    for key in ('actor_lr', 'critic_lr', 'batch_size', 'replay_size', 'update_every', 'checkpoint_every', 'total_env_steps'):
        if not math.isfinite(t[key]) or t[key] <= 0:
            raise ValueError('Invalid training.' + key)
    if t['learning_starts'] < t['batch_size'] or t['replay_size'] < t['batch_size'] or not 0 < t['tau'] <= 1 or not 0 <= t['gamma'] <= 1:
        raise ValueError('Invalid Replay/target parameters')
    for key in ('batch_size', 'replay_size', 'learning_starts', 'total_env_steps', 'checkpoint_every', 'update_every'):
        if type(t[key]) is not int:
            raise ValueError('training.'+key+' must be an integer')
    if not math.isfinite(t['exploration_std']) or t['exploration_std'] < 0:
        raise ValueError('Invalid exploration noise')
    coefficient = t.get('actor_bc_regularization', 0.)
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError('Invalid actor_bc_regularization')
    stratified=t.get('stratified_replay')
    if stratified:
        ratios=np.asarray(stratified.get('ratios'),dtype=float)
        if (ratios.shape!=(4,) or not np.isfinite(ratios).all() or np.any(ratios<=0)
                or not math.isclose(float(ratios.sum()),1.,rel_tol=0.,abs_tol=1e-9)):
            raise ValueError('stratified_replay ratios must be four positive values summing to one')
        if not stratified.get('expert_dataset') or not stratified.get('expert_sha256'):
            raise ValueError('stratified_replay requires an identified expert dataset')
    freeze_only=t.get('expert_replay_actor_freeze_only',False)
    if type(freeze_only) is not bool or (freeze_only and not stratified):
        raise ValueError('expert_replay_actor_freeze_only requires stratified Replay')
    warmup_expert=t.get('warmup_expert_replay')
    if warmup_expert:
        ratios=np.asarray(warmup_expert.get('ratios'),dtype=float)
        boundaries=warmup_expert.get('turn_weight_boundaries')
        replay_mode=warmup_expert.get('mode')
        expected_shape=(6,) if boundaries is not None else (4,)
        if (stratified or freeze_only or ratios.shape!=expected_shape or not np.isfinite(ratios).all()
                or np.any(ratios<=0) or not math.isclose(float(ratios.sum()),1.,abs_tol=1e-9)):
            raise ValueError('warmup_expert_replay has invalid positive ratios or external Expert Replay')
        if replay_mode not in (None,'turn_stand'):
            raise ValueError('Invalid warmup_expert_replay mode')
        if replay_mode=='turn_stand':
            warmup_ratios = np.asarray(warmup_expert.get('warmup_ratios', [.9, .1]), dtype=float)
            if (warmup_ratios.shape != (2,) or not np.isfinite(warmup_ratios).all()
                    or np.any(warmup_ratios <= 0)
                    or not math.isclose(float(warmup_ratios.sum()), 1., abs_tol=1e-9)):
                raise ValueError('Turn/Stand warmup_ratios must be two positive values summing to one')
            tolerance=warmup_expert.get('stand_tolerance_deg')
            if (boundaries is not None or not isinstance(tolerance,(int,float)) or
                    isinstance(tolerance,bool) or not math.isfinite(tolerance) or tolerance<=0):
                raise ValueError('Turn/Stand Replay requires a positive stand tolerance')
            if warmup_expert.get('ratio_labels') != [
                    'online_turn','online_stand','expert_turn','expert_stand']:
                raise ValueError('Invalid Turn/Stand Replay ratio labels')
        if boundaries is not None:
            boundaries=np.asarray(boundaries,dtype=float)
            labels=warmup_expert.get('turn_weight_labels')
            if (boundaries.shape!=(2,) or not np.isfinite(boundaries).all() or
                    not 0 < boundaries[0] < boundaries[1] < 1 or
                    labels != ['low_turn','mixed','high_turn']):
                raise ValueError('Invalid three-bucket turn-weight Replay definition')
    if coefficient and not c.get('initialization', {}).get('use_bc'):
        raise ValueError('actor_bc_regularization requires BC initialization')
    successful = t.get('success_expert_replay')
    if successful:
        if warmup_expert or stratified or freeze_only:
            raise ValueError('Successful Expert Replay cannot be combined with another Replay mode')
        for key in ('ratios', 'capacity_ratios') + (('late_ratios',) if 'late_ratios' in successful else ()):
            values = np.asarray(successful.get(key), dtype=float)
            if (values.shape != ((4,) if c.get('staged_navigation') else (3,)) or not np.isfinite(values).all() or np.any(values < 0) or (key == 'capacity_ratios' and np.any(values == 0))
                    or not math.isclose(float(values.sum()), 1., abs_tol=1e-9)):
                raise ValueError('Invalid success_expert_replay fractions: require sum one, nonnegative sampling and positive capacities')
        if 'late_ratios' in successful and (type(successful.get('switch_step')) is not int or successful['switch_step'] < 1):
            raise ValueError('Replay schedule requires positive integer switch_step')
        tolerance = successful.get('stand_tolerance_deg')
        if (not isinstance(tolerance, (int,float)) or isinstance(tolerance,bool)
                or not math.isfinite(tolerance) or tolerance <= 0):
            raise ValueError('Successful Expert Replay requires a positive stand tolerance')
        if (e.get('navigation_mode') != ('turn_then_forward' if c.get('staged_navigation') else 'turn_only') or
                c.get('direction_training', {}).get('navigation_mode') != ('turn_then_forward' if c.get('staged_navigation') else 'turn_only') or
                not c.get('initialization', {}).get('use_bc') or
                (t.get('critic_warmup_env_steps', 0) <= 0 and not c.get('expert_pipeline')) or
                c.get('direction_training', {}).get('steps_per_angle')):
            raise ValueError('Successful Expert Replay requires three-category expert pipeline with blended navigation')
        if t['replay_size'] < 4 or any(t['replay_size']*v < 1 for v in successful['capacity_ratios']):
            raise ValueError('Successful Expert Replay capacity is too small')
    if c.get('staged_navigation'):
        if any(not math.isfinite(c['staged_navigation'][k]) or c['staged_navigation'][k] < 0 for k in ('forward_progress_weight','speed_error_weight','lateral_speed_weight')):
            raise ValueError('Invalid linear navigation reward weights')
    stationary = c.get('stationary_reward')
    if stationary:
        if e.get('navigation_mode') != ('turn_then_forward' if c.get('staged_navigation') else 'turn_only'):
            raise ValueError('Stationary reward requires configured navigation mode')
        if type(stationary.get('window_samples')) is not int or stationary['window_samples'] < 2:
            raise ValueError('Stationary window requires at least two positions')
        for key in ('angle_tolerance_deg','speed_tolerance_m_s','yaw_tolerance_rad_s','hold_bonus','progress_per_degree','progress_limit'):
            if not math.isfinite(stationary[key]) or stationary[key] <= 0:
                raise ValueError('Invalid stationary_reward.'+key)
        if stationary['progress_limit'] >= stationary['hold_bonus']:
            raise ValueError('Hold bonus must exceed progress limit')
    elif c.get('critic_observation') == 'actor_plus_body_velocity_net_speed':
        raise ValueError('Net-speed Critic requires stationary_reward')
    pipeline = c.get('expert_pipeline')
    if pipeline:
        if pipeline.get('mode') not in ('collect', 'pretrain', 'train') or not successful:
            raise ValueError('Expert pipeline requires successful Replay and a valid mode')
        if not pipeline.get('dataset') or not pipeline.get('critic_checkpoint'):
            raise ValueError('Expert pipeline requires dataset and critic_checkpoint paths')
        if type(pipeline.get('successes_per_angle')) is not int or pipeline['successes_per_angle'] < 1:
            raise ValueError('successes_per_angle must be positive')
        if pipeline['mode'] != 'collect' and t.get('critic_warmup_env_steps') != 0:
            raise ValueError('Reusable expert training must disable the online frozen phase')
        if pipeline['mode'] == 'collect' and t.get('critic_warmup_env_steps', 0) < t['total_env_steps']:
            raise ValueError('Collection must keep Actor frozen for the entire budget')
        if c['initialization'].get('actor_checkpoint') or c['direction_training'].get('promote_success_to_bc'):
            raise ValueError('Expert pipeline requires the registered fixed BC Actor')
    evaluation = c['evaluation']
    if type(evaluation['episodes_per_command']) is not int or evaluation['episodes_per_command'] < 1:
        raise ValueError('Invalid evaluation episodes')
    for key in ('duration_seconds', 'linear_tolerance', 'yaw_tolerance', 'straight_goal_distance', 'straight_timeout_seconds'):
        if not math.isfinite(evaluation[key]) or evaluation[key] <= 0:
            raise ValueError('Invalid evaluation.'+key)
    if not 0 < evaluation['minimum_tracking_fraction'] <= 1 or not 0 <= evaluation['ignore_initial_seconds'] < evaluation['duration_seconds']:
        raise ValueError('Invalid evaluation success definition')
    for key, value in c['reward'].items():
        if key == 'biped_tracking':
            if not isinstance(value, dict):
                raise ValueError('Invalid reward.biped_tracking')
            if value.get('revision') not in ('velocity_stability_tilt_v1', 'velocity_stability_filtered_v1', 'causal_velocity_widths_v1','velocity_widths_variants_v1') or any(
                    name in value for name in ('reference', 'periodic_coefficient', 'periodic_weight')):
                raise ValueError('Unsupported biped Reward revision/reference')
            if value.get('revision') == 'velocity_stability_filtered_v1':
                if value.get('velocity_source') != 'filtered':
                    raise ValueError('Filtered Reward requires filtered velocity_source')
                alpha = c['environment'].get('velocity_filter')
                if type(alpha) not in (int, float) or not math.isfinite(alpha) or not 0 < alpha <= 1:
                    raise ValueError('Filtered Reward requires 0 < environment.velocity_filter <= 1')
            elif value.get('revision') in ('causal_velocity_widths_v1','velocity_widths_variants_v1'):
                widths = value.get('tracking_widths')
                sources=('causal_mean',) if value['revision']=='causal_velocity_widths_v1' else ('command_ema','instantaneous')
                if (value.get('velocity_source') not in sources
                        or c.get('critic_observation') not in ('actor','actor_plus_reward_velocity_mean')
                        or not isinstance(widths, list) or len(widths) != 3
                        or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in widths)):
                    raise ValueError('Causal Reward requires three positive tracking_widths')
                tracking = c.get('velocity_tracking', {})
                if value['revision']=='velocity_widths_variants_v1':
                    if (c.get('critic_observation')!='actor' or tracking.get('mode')!=value['velocity_source']
                            or tracking.get('reset_on_task_change') is not True or 'window_samples' in tracking):
                        raise ValueError('Velocity variants require no Critic means and no rolling window')
                elif (tracking.get('mode') != 'causal_command_mean'
                        or type(tracking.get('window_samples')) is not int or tracking['window_samples'] < 1
                        or tracking.get('reset_on_command_change') is not True):
                    raise ValueError('Causal Reward requires command-segment velocity history independently of Critic inputs')
            elif value.get('velocity_source', 'instantaneous') != 'instantaneous':
                raise ValueError('Legacy Reward must use instantaneous velocity')
            positive = () if value.get('revision') in ('causal_velocity_widths_v1','velocity_widths_variants_v1') else ('velocity_coefficient',)
            nonnegative = ('yaw_weight', 'velocity_weight', 'stability_coefficient',
                           'stability_weight', 'orientation_coefficient')
            for name in positive + nonnegative:
                number = value.get(name)
                if (type(number) not in (int, float) or not math.isfinite(number)
                        or number < 0 or (name in positive and number == 0)):
                    raise ValueError('Invalid reward.biped_tracking.' + name)
            if not any(value[name] > 0 for name in ('velocity_weight', 'stability_weight')):
                raise ValueError('Biped Reward requires a nonzero term weight')
            continue
        if key == 'command_calibration':
            if not isinstance(value,dict) or set(value.get('stages',{})) != {'turn','stand','walk'}:
                raise ValueError('Invalid reward.command_calibration')
            for params in value['stages'].values():
                scale,offset=params.get('scale'),params.get('offset')
                if not isinstance(scale,(int,float)) or not math.isfinite(scale) or scale<=0 or not isinstance(offset,(int,float)) or not math.isfinite(offset):
                    raise ValueError('Invalid calibration scale/offset')
            continue
        if not math.isfinite(value) or value < 0 or ('sigma' in key and value == 0):
            raise ValueError('Invalid reward.' + key)
    bounded_keys=('task_reward_bound','turn_translation_error_scale','turn_yaw_error_scale','stand_translation_error_scale','stand_yaw_error_scale','walk_speed_reference')
    # Historical BC weights have no bounded Reward fields; their State/Action
    # contract is checked by policy loading, independently of current Reward.
    if any(key in c['reward'] for key in bounded_keys):
        for key in bounded_keys:
            if c['reward'].get(key,0)<=0:raise ValueError('Expected positive reward.'+key)
    joint_targets(np.zeros(12), e['action_scale'])
    if 'straight' in c:
        cmd = vector(c['straight']['command'], 3, 'straight.command')
        if 'directional' in c:
            if cmd[0] <= 0 or cmd[2] != 0 or np.any(cmd < limits[:, 0]) or np.any(cmd > limits[:, 1]):
                raise ValueError('Directional sine requires supported planar translation and zero yaw command')
        elif cmd[0] <= 0 or np.any(cmd[1:] != 0) or cmd[0] > limits[0, 1]:
            raise ValueError('Straight experiment requires supported positive vx and zero vy/wz')
        init = c['initialization']
        if init.get('clear_replay_on_bc_reset') is not False:
            raise ValueError('Approved straight design retains Replay on BC recovery')
        for key in ('consecutive_falls', 'stall_timeout_steps'):
            if type(init[key]) is not int or init[key] < 1:
                raise ValueError('Invalid initialization.'+key)
        for key in ('critic_warmup_updates', 'critic_warmup_update_every', 'switch_steps'):
            if type(t[key]) is not int or t[key] < 1:
                raise ValueError('Invalid training.'+key)
        if not math.isfinite(t['actor_lr_after_switch']) or t['actor_lr_after_switch'] <= 0 or not math.isfinite(t['exploration_std_after_switch']) or t['exploration_std_after_switch'] < 0:
            raise ValueError('Invalid fine-tuning schedule')
    from .training_design import validate_design
    validate_design(c)
    return c
