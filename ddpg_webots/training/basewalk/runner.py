"""Controller entry for training, independent fixed-policy evaluation and live commands."""
import csv
import gzip
import copy
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
from collections import deque
import numpy as np
import torch
from .core import CONTRACT, vector, critic_extra_dim, critic_velocity_features
from .environment import WebotsBaseWalk
from .learning import create_algorithm
from .phase_fall import PhaseFallBuffer
from .transition_logging import TransitionWriter
from .stationary_reward import critic_features as current_critic_features
from .training_design import (BalancedDirections, TurnRecoveryMonitor,
                              update_schedule, direction_progress, turn_goal_status)
from .events import emit
from .direction_status import direction_metrics, format_direction, format_reward_breakdown
from .world_reload import WorldReloadRequested, save_continuation, load_continuation, reload_world
from .angle_bc import bc_source, remember_success, finish_angle



def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    # Windows readers can briefly deny replacement of an open destination.
    # Retain the complete temporary document and retry the atomic replacement.
    for attempt in range(9):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 8:
                raise
            time.sleep(min(.025 * 2**attempt, .4))


def bellman_terminal(physical_terminal, reached_goal, direction):
    """Mask bootstrap at physical falls and opt-in task-goal boundaries."""
    return bool(physical_terminal or (reached_goal and direction and
                direction.get('goal_as_terminal', False)))


def run_train(env, config, folder, smoke, *, checkpoint_resume=False):
    world_reload = config['environment'].get('reset_mode', 'node') == 'world_reload'
    continuation = load_continuation(folder, config) if world_reload else None
    checkpoint_start = config['training'].get('checkpoint_start')
    if (not continuation and checkpoint_resume and checkpoint_start and
            Path(checkpoint_start['path']).parent.parent == folder):
        continuation = load_continuation(folder, config, checkpoint=checkpoint_start)
    if not continuation and not checkpoint_resume and (folder/'training.csv').exists():
        raise RuntimeError('Existing training log without continuation; refusing to overwrite this run')
    agent = continuation['agent'] if continuation else create_algorithm(config)
    archive_cycle = getattr(agent.replay, 'add_high_score_cycle', None)
    if config['training'].get('save_best_reward_cycle', False):
        from .best_cycle import BestRewardCycle
        archive_cycle = BestRewardCycle(folder, agent, archive_cycle)
    checkpoint_step = 0
    if checkpoint_start and not continuation:
        path = Path(checkpoint_start['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != checkpoint_start['sha256']:
            raise ValueError('Branch checkpoint hash mismatch')
        saved = torch.load(path, map_location='cpu', weights_only=False)
        checkpoint_step = saved['env_steps']
        if saved['contract'] != CONTRACT or checkpoint_step != checkpoint_start['step']:
            raise ValueError('Branch checkpoint contract or step mismatch')
        if not 0 < checkpoint_step < config['training']['total_env_steps']:
            raise ValueError('Branch checkpoint must precede the training budget')
        for key in ('actor', 'critic', 'actor_target', 'critic_target',
                    'actor_optimizer', 'critic_optimizer'):
            getattr(agent, key).load_state_dict(saved[key])
        agent.updates = saved['updates']
        agent.warmup_completed = saved['critic_warmup_completed']
        replay_source = checkpoint_start.get('reconstructed_replay')
        if replay_source:
            replay_path = Path(replay_source['path'])
            if hashlib.sha256(replay_path.read_bytes()).hexdigest() != replay_source['sha256']:
                raise ValueError('Reconstructed Replay hash mismatch')
            payload = torch.load(replay_path, map_location='cpu', weights_only=False)
            if (payload['source_checkpoint_sha256'] != checkpoint_start['sha256']
                    or payload['source_step'] != checkpoint_step
                    or payload['reward_sha256'] != config['reward_source_sha256']):
                raise ValueError('Reconstructed Replay provenance mismatch')
            agent.replay = payload['replay']
            # Callback must reference the restored pool, not the initial empty one.
            archive_cycle = getattr(agent.replay, 'add_high_score_cycle', None)
            if config['training'].get('save_best_reward_cycle', False):
                from .best_cycle import BestRewardCycle
                archive_cycle = BestRewardCycle(folder, agent, archive_cycle)
    collecting = config.get('expert_pipeline', {}).get('mode') == 'collect'
    managed_replay = bool(config.get('expert_pipeline') or config.get('three_class_online'))
    insert_transition = (agent.collector.append_transition if collecting else
                         agent.replay.add_online if managed_replay else agent.replay.add)
    fixed = config.get('straight', {}).get('command')
    direction = config.get('direction_training')
    if not collecting and config['training'].get('online_angles_deg') is not None:
        online_angles = config['training']['online_angles_deg']
        if not online_angles or not set(online_angles).issubset(direction['angles_deg']):
            raise ValueError('online_angles_deg must be a nonempty subset of task angles')
        direction = dict(direction, angles_deg=list(online_angles))
    from .failure_retry import FailureRetry
    stand_collection = collecting and config['expert_pipeline'].get('collection_task') == 'stand'
    clock_only_walk = bool(config['training'].get('walk_commands')) and not collecting
    if clock_only_walk:
        direction = None
        sampler = None
    else:
        sampler = FailureRetry(dict(direction, angles_deg=[0]) if stand_collection else direction, config['seed'])
    angle = 0. if clock_only_walk else None
    command = fixed
    retry_angles = bool(direction and direction.get('retry_same_angle'))
    step_blocks = retry_angles and 'steps_per_angle' in direction
    promote_bc = bool(step_blocks and direction.get('promote_success_to_bc', False))
    block_goals, block_successes, block_frozen_goals = 0, 0, 0
    block_steps, block_falls, completed_blocks, failed_blocks = 0, 0, 0, 0
    paused = False
    episode_duration = config['environment']['episode_seconds']
    displayed_angle_steps = direction.get('steps_per_angle') if direction else None
    if (direction and displayed_angle_steps is None and
            not config['environment'].get('disable_time_limit', False)):
        displayed_angle_steps = int(round(
            episode_duration*1000/config['webots']['control_time_step_ms']))
    fixed_cycles=config.get('fixed_cycle_commands',{}).get('enabled',False)
    from .fixed_cycle import FixedCycleStages, TurnFallWalkReset
    fixed_cycle_online = fixed_cycles and not collecting and not config.get('bc_rollout', {}).get('collection_mode')
    online_cycle_limit = config['training'].get('online_fixed_cycle_limit') if fixed_cycle_online else None
    persistent_stages = None
    if online_cycle_limit is not None:
        saved_stages = (continuation['state']['fixed_cycle_state'] if continuation else
                        (checkpoint_start or {}).get('fixed_cycle_state'))
        stage_type = FixedCycleStages
        if config['training'].get('gradual_turn', {}).get('enabled'):
            from .gradual_turn import GradualTurnStages
            stage_type = GradualTurnStages
        if config['training'].get('turn_and_walk', {}).get('enabled'):
            from .turn_and_walk import TurnAndWalkStages
            stage_type = TurnAndWalkStages
        if config['training'].get('walk_commands'):
            from .walk_commands import WalkCommands
            stage_type = WalkCommands
        persistent_stages = stage_type(
            config, cycle_limit=online_cycle_limit, state=saved_stages,
            count_fall_as_attempt=config['training'].get('online_count_fall_as_attempt', False),
            turn_only=config['training'].get('online_turn_only', False),
            fixed_turn_sign=config['training'].get('online_fixed_turn_sign'),
        )
        env.persistent_fixed_cycle_stages = persistent_stages
    reset_threshold = (config['training'].get('turn_fall_walk_reset_threshold', 0)
                       if fixed_cycle_online and persistent_stages is None else 0)
    turn_fall_walk_reset = TurnFallWalkReset(reset_threshold,
        continuation['state'].get('turn_fall_walk_reset') if continuation else None)
    episodes, falls = 0, 0
    if checkpoint_resume and not continuation:
        history = json.loads((folder/'resume.json').read_text(encoding='utf-8'))
        if history['step'] != checkpoint_step:
            raise ValueError('Checkpoint and prepared history disagree')
        episodes, falls = history['completed_episodes'], history['falls']
    recoveries, consecutive_falls = 0, 0
    recovery_log = []
    turn_recovery = TurnRecoveryMonitor(config.get('initialization', {}).get('turn_bc_reset'))
    start_step, keep_angle = checkpoint_step + 1, False
    episode_started_critic_only = False
    reward_console_window = deque(maxlen=100)
    turn_hold_steps, max_turn_translation = 0, 0.
    if continuation:
        saved = continuation['state']
        sampler, angle, command = saved['sampler'], saved['angle'], saved['command']
        block_goals, block_successes, block_frozen_goals = saved['block_counts']
        block_steps, block_falls, completed_blocks, failed_blocks = saved['blocks']
        episodes, falls, recoveries, consecutive_falls = saved['counts']
        recovery_log = saved['recovery_log']
        turn_recovery = saved.get('turn_recovery_monitor', turn_recovery)
        start_step, keep_angle = saved['next_step'], saved['keep_angle']
        if hasattr(env, 'commands'):
            env.commands.rng.bit_generator.state = saved['command_rng']
        last_restore_step = recovery_log[-1]['step'] if recovery_log else 'none'
        print(f'[world_reload] restored step={start_step-1} task_angle={angle:+g}deg '
              f'updates={agent.updates} reload_reason={saved.get("last_boundary_reason", "unknown")} '
              f'previous_bc_restore_step={last_restore_step} (history)', flush=True)

    from .turn_boundary import TurnBoundary
    turn_boundary = TurnBoundary(config["training"].get("turn_reset_distance_m", 0.))

    from .walk_distance import WalkDistance
    walk_distance=WalkDistance(direction.get('goal_distance_m',2.)) if not fixed_cycles and direction and direction.get('goal_measure')=='walk_path_cycle_end' else None

    def reset_training(step, keep_angle=False):
        nonlocal angle, command, episode_started_critic_only
        nonlocal turn_hold_steps, max_turn_translation
        turn_boundary.reset()
        if walk_distance is not None:walk_distance.reset()
        old_angle = angle
        # Reset only display history; Replay stays on the existing agent.
        reward_console_window.clear()
        turn_hold_steps, max_turn_translation = 0, 0.
        if sampler:
            if not keep_angle:
                angle, command = agent.collector.next_task() if collecting else sampler.next(step)
            warmup = step <= config['training']['critic_warmup_env_steps']
            episode_started_critic_only = warmup
            env.cfg['disable_time_limit'] = bool(
                step_blocks or config['environment'].get('disable_time_limit', False))
            env.cfg['episode_seconds'] = (episode_duration if step_blocks else (episode_duration - block_steps*config['webots']['control_time_step_ms']/1000.
                if retry_angles else (min(episode_duration, direction['warmup_episode_seconds']) if warmup else episode_duration))
            )
        if sampler:
            env.direction_target_deg = angle
        if fixed_cycles and persistent_stages is None:
            env.fixed_cycle_start_stage = turn_fall_walk_reset.next_start()
        obs = env.reset(command=command, randomize=fixed is None and sampler is None)
        if reset_threshold:
            with (folder/'turn_fall_walk_resets.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(step=step, start_stage=env.fixed_cycle_start_stage,
                    threshold=reset_threshold, **turn_fall_walk_reset.state()))+'\n')
        if sampler:
            changed = old_angle is not None and old_angle != angle
            pause = config.get('monitoring', {}).get('direction_pause_seconds', 0.) if changed else 0.
            yaw = getattr(env, 'yaw', 0.)
            state = dict(step=step-1, direction_deg=angle, angle_step=block_steps, gait_phase=float(env.phase),
                bc_source=bc_source(agent),
                angle_steps=displayed_angle_steps, phase='reset_pause' if pause else 'ready',
                training_phase='expert_collection' if collecting else ('critic_warmup' if warmup else 'joint'),
                pause_seconds=pause, **direction_metrics(
                    env.position, env.position, yaw, yaw, angle,
                    angle if direction.get('navigation_mode') in ('turn_then_forward', 'blended_turn_walk', 'turn_only') else 0.))
            state.update(sampler.sampling_state(angle))
            write_json(folder/'direction_status.json', state)
            if old_angle is None or changed:
                with (folder/'direction_events.jsonl').open('a', encoding='utf-8') as event_handle:
                    event_handle.write(json.dumps(state, allow_nan=False)+'\n')
                focus_text = (f" focus_weight={state['sampling_weight']} "
                    f"recent_strict={state['recent_strict_results']}"
                    if state.get('focus_active') else '')
                if fixed_cycles:
                    limit = env.stages.limit
                    if config['training'].get('walk_commands'):
                        print(f'[task] Walk only; each command holds {limit} control steps',flush=True)
                    else:
                        print(f'[task] Turn({limit} cycles) -> Walk({limit} cycles); Turn alternates left/right', flush=True)
                else:
                    print(f'[task] angle={angle:+g}deg angle_step={block_steps}/{displayed_angle_steps if displayed_angle_steps is not None else "?"}'
                          f'{focus_text}', flush=True)
            if pause:
                # Synchronous Webots waits for the next step: no physics, Replay or optimizer advances.
                time.sleep(pause)
                state.update(phase='ready', pause_seconds=0.)
                write_json(folder/'direction_status.json', state)
        return obs
    if fixed is not None and agent.bc_state is None and config.get('initialization',{}).get('use_bc',True):
        raise ValueError('Straight DDPG requires verified BC initialization')
    obs = reset_training(start_step, keep_angle=keep_angle)
    if continuation:
        # Consume only after successful setup. A crash must not silently replay old CSV rows.
        (folder/'world_reload_state.pth').unlink(missing_ok=True)
    phase_buffer = PhaseFallBuffer(config['reward']['fall_penalty'], config) if config['training'].get('phase_fall_backfill', False) else None
    stalled = 0
    init = config.get('initialization', {})
    progress_origin = env.position[0]
    episode_start = env.position.copy()
    initial_yaw = getattr(env, 'yaw', 0.)
    max_heading = 0.
    final_heading_error = 0.
    velocity_errors = []
    attempt_interval = config['training'].get('joint_update_every_turn_attempts')
    actor_attempt_interval = config['training'].get('actor_update_every_turn_attempts', attempt_interval)
    gait_attempt_interval = config['training'].get('joint_update_every_gait_attempts')
    attempt_phases = ('turn',)
    if config['training'].get('turn_and_walk', {}).get('enabled') or config['training'].get('walk_commands'):
        attempt_phases = ('turn', 'walk')
    if gait_attempt_interval is not None:
        if attempt_interval is not None or config['training'].get('actor_update_every_turn_attempts') is not None:
            raise ValueError('Choose either synchronous gait attempt updates or Turn attempt updates')
        if not fixed_cycle_online or persistent_stages is None or config['training'].get('online_turn_only'):
            raise ValueError('Synchronous gait attempt updates require online Turn/Walk cycle scheduling')
        if not persistent_stages.count_fall_as_attempt:
            raise ValueError('Synchronous gait attempt updates must count falls as attempts')
        attempt_interval = actor_attempt_interval = gait_attempt_interval
        attempt_phases = ('turn', 'walk')
    if attempt_interval is not None and (
            type(attempt_interval) is not int or attempt_interval < 1
            or type(actor_attempt_interval) is not int or actor_attempt_interval < 1
            or actor_attempt_interval % attempt_interval
            or persistent_stages is None
            or not persistent_stages.count_fall_as_attempt
            or not config['training'].get('phase_fall_backfill')
            or collecting or config.get('bc_rollout', {}).get('collection_mode')
            or config['training'].get('freeze_online_actor')):
        raise ValueError('Attempt updates require a valid online cadence and cycle Replay')
    joint_attempts = (continuation['state'].get('joint_attempts', 0) if continuation else
                      (checkpoint_start or {}).get('joint_attempts', 0))
    episode_steps, episode_reward = 0, 0.
    log_fields = ['step', 'reward', 'fell', 'episode_end', 'motion_phase', 'vx_cmd', 'vy_cmd', 'wz_cmd',
                  'vx', 'vy', 'wz', 'actor_loss', 'actor_rl_loss', 'actor_bc_loss',
                  'critic_loss', 'updates',
                  'critic_only', 'actor_lr', 'exploration_std', 'bc_recoveries', 'critic_lr',
                  'q_mean', 'target_q_mean', 'td_abs_mean', 'critic_grad_norm', 'actor_grad_norm',
                  'direction_deg', 'reward_vx', 'reward_vy', 'reward_vx_cmd', 'reward_vy_cmd', 'angle_step', 'actor_updated',
                  'heading_error', 'actor_yaw_rate']
    if config.get('algorithm', 'ddpg') == 'sac':
        log_fields += ['entropy_loss', 'policy_entropy', 'entropy_coefficient']
    log_fields += ['reward_turn_translation', 'reward_turn_linear_velocity', 'reward_walk_forward_progress',
                   'reward_walk_heading_progress', 'reward_turn_heading_progress',
                   'reward_turn_heading_error', 'reward_turn_yaw_rate_error',
                   'reward_walk_heading_error', 'reward_walk_cross_track']
    log_fields += ['reward_tracking_x', 'reward_tracking_y', 'reward_tracking_yaw',
                   'reward_survival', 'reward_orientation', 'reward_height',
                   'reward_action_rate', 'reward_joint_velocity', 'reward_fall',
                   'bellman_terminal']
    log_fields += ['turn_weight', 'walk_weight', 'reward_turn_total',
                   'reward_walk_total', 'reward_turn_weighted', 'reward_walk_weighted',
                   'source_motion_class', 'replay_destination']
    log_fields += ['replay_online_mixed','replay_expert_turn','replay_expert_stand','replay_expert_walk',
                   'replay_success','replay_failure','success_fifo_count','failure_fifo_count','online_fifo_count','online_turn_fifo_count','replay_online_turn','replay_expert_turn_stand_walk']
    log_fields += ['q_executed','walk_path_m','reward_stand_excess_translation','reward_command_calibration_offset','reward_turn_hold','stand_hold_steps','next_motion_phase']
    log_fields += ['reward_walk_speed_error','reward_walk_lateral_speed','reward_walk_yaw_rate_error','reward_stand_total']
    log_fields += ['reward_velocity_tracking','reward_stability',
                   'replay_expert_success','replay_online_success','expert_success_count']
    log_fields += ['gait_phase_before', 'gait_phase_after', 'cycle_end', 'cycle_complete',
                   'cycle_dt', 'world_dx', 'world_dy', 'yaw_delta', 'target_world_dx',
                   'target_world_dy', 'target_yaw_delta', 'reward_before_backfill',
                   'reward_fall_before_backfill', 'reward_gravity_xy_sq', 'reward_gyro_xy_sq']
    log_fields += ['diagnostic_episode_id', 'source_observation', 'executed_action',
                   'next_observation', 'reward_gravity', 'reward_gyro_xy']
    if config['training'].get('walk_commands'):
        log_fields += ['reward_tracking_xy']
    log_fields += ['actual_vx', 'actual_vy', 'actual_wz', 'reward_wz', 'reward_wz_cmd']
    if config['reward'].get('biped_tracking', {}).get('turn_world_z_regularization'):
        log_fields += ['reward_ankle_soft_limit', 'reward_hip_deviation']
    if config['reward'].get('biped_tracking', {}).get('turn_reverse_foot_yaw_weight'):
        log_fields += ['reward_reverse_foot_yaw']
    if config['reward'].get('biped_tracking', {}).get('phase_foot', {}).get('kind') == 'phase_position_minimum_v1':
        log_fields += ['reward_phase_foot_position', 'reward_foot_left_rear',
                       'reward_foot_left_front', 'reward_foot_right_rear', 'reward_foot_right_front']
    if config['reward'].get('biped_tracking', {}).get('turn_heading_window_seconds'):
        log_fields += ['reward_turn_heading', 'turn_window_heading_error']
    if (config['reward'].get('biped_tracking', {}).get('turn_yaw_error_progress_weight') or config['reward'].get('biped_tracking', {}).get('turn_yaw_error_cost_weight') or config['reward'].get('biped_tracking', {}).get('turn_acceleration_weight')):
        log_fields.append('reward_turn_speed_delta')
    if hasattr(agent.replay, 'pool_size_log_field'):
        log_fields += ['replay_high_score', agent.replay.pool_size_log_field]
    if config.get('landing_alignment'):
        log_fields += ['landing_gate', 'landing_target_yaw_error_deg']
    if getattr(env, 'actor_body_gyro', False):
        log_fields += ['sensor_gyro_z']
    privileged_critic = bool(critic_extra_dim(config))
    if privileged_critic:
        log_fields += ['critic_body_vx', 'critic_body_vy', 'next_critic_body_vx', 'next_critic_body_vy']
    if config.get('critic_observation') == 'actor_plus_reward_velocity_mean':
        log_fields += ['critic_reward_mean_vx', 'critic_reward_mean_vy', 'critic_reward_mean_wz',
                       'next_critic_reward_mean_vx', 'next_critic_reward_mean_vy', 'next_critic_reward_mean_wz']
    if config['reward'].get('biped_tracking',{}).get('velocity_source') in ('causal_mean','command_ema','instantaneous'):
        log_fields += ['reward_history_count']
    if critic_extra_dim(config) == 4:
        log_fields += ['critic_net_speed', 'critic_window_fill', 'next_critic_net_speed', 'next_critic_window_fill']
    mode = 'a' if continuation or checkpoint_resume else 'w'
    csv_exists = (folder/'training.csv').is_file()
    episodes_exist = (folder/'training_episodes.csv').is_file()
    csv_fields = log_fields
    if mode == 'a' and csv_exists:
        with (folder/'training.csv').open(newline='', encoding='utf-8-sig') as existing_log:
            csv_fields = next(csv.reader(existing_log))
    with (folder/'training.csv').open(mode, newline='', encoding='utf-8') as handle, \
         (folder/'training_episodes.csv').open(mode, newline='', encoding='utf-8') as episode_handle, \
         gzip.open(folder/'transition_trace.jsonl.gz', mode+'t', encoding='utf-8') as trace_handle:
        writer = TransitionWriter(handle, trace_handle, csv_fields, config)
        if not csv_exists:
            writer.writeheader()
        episode_writer = csv.DictWriter(episode_handle, fieldnames=['episode', 'end_step', 'steps', 'elapsed_s',
            'reward', 'end_reason', 'arrival_time_s', 'signed_lateral_m', 'absolute_lateral_m',
            'net_forward_m', 'max_planar_translation_m', 'fell', 'reached_goal', 'bc_recovery', 'direction_deg',
            'along_track_m', 'cross_track_m', 'max_heading_deg', 'mean_velocity_error_m_s', 'success', 'critic_only_at_end', 'success_excluded_reason'])
        if not episodes_exist:
            episode_writer.writeheader()
        for step in range(start_step, config['training']['total_env_steps']+1):
            agent.schedule(step)
            action = agent.act(obs, explore=True)
            # Capture s_t BEFORE physics advances; s_(t+1) comes from the next feedback.
            critic_features = current_critic_features(env, config) if privileged_critic else None
            source_phase = float(env.phase)
            diagnostic_source_observation = json.dumps(np.asarray(obs).tolist(), allow_nan=False)
            walk_before=np.asarray(env.position).copy()
            turn_boundary.begin(bool(float(obs[5]) <= 1e-8 and abs(float(obs[7])) > 1e-8), env.position)
            nxt, value, terminal, truncated, info = env.step(action, randomize=fixed is None and sampler is None)
            if reset_threshold:
                turn_fall_walk_reset.observe(info['motion_phase'], terminal)
            from .action_value import executed_action_q
            q_executed=executed_action_q(agent.critic,obs,info['executed_action'],critic_features)
            turn_limit = (not fixed_cycles and not collecting and not terminal and turn_boundary.exceeded(info["position"]))
            if turn_limit:
                truncated = True
            if collecting:
                agent.collector.record_step(info, source_phase, float(env.phase))
            critic_replay_args, critic_log = {}, {}
            if privileged_critic:
                next_features = current_critic_features(env, config)
                critic_replay_args = dict(critic_features=critic_features, next_critic_features=next_features)
                critic_log = dict(critic_body_vx=float(obs[47]), critic_body_vy=float(obs[48]),
                                  next_critic_body_vx=float(nxt[47]), next_critic_body_vy=float(nxt[48]))
                if config.get('critic_observation') == 'actor_plus_reward_velocity_mean':
                    for i, axis in enumerate(('vx', 'vy', 'wz')):
                        critic_log['critic_reward_mean_'+axis] = float(critic_features[i])
                        critic_log['next_critic_reward_mean_'+axis] = float(next_features[i])
            if critic_extra_dim(config) == 4:
                critic_log.update(critic_net_speed=float(critic_features[2]), critic_window_fill=float(critic_features[3]),
                    next_critic_net_speed=float(next_features[2]), next_critic_window_fill=float(next_features[3]),
                    reward_turn_hold=info['reward_terms'].get('turn_hold', 0.))
            block_steps += 1
            block_falls += int(terminal)
            block_timeout = (block_steps >= direction['steps_per_angle'] if step_blocks else
                retry_angles and block_steps*config['webots']['control_time_step_ms']/1000. + 1e-8 >= episode_duration)
            if step_blocks:
                truncated = False
            if block_timeout:
                truncated = True
            reached = not fixed_cycles and fixed is not None and info['position'][0]-env.start_position[0] >= config['evaluation']['straight_goal_distance']
            along, cross = (direction_progress(info['position'], episode_start, initial_yaw, angle)
                            if sampler else (None, None))
            turn_only = stand_collection
            if turn_only:
                goal_direction = sampler.goal_config(direction, angle)
                reached, turn_hold_steps, max_turn_translation = turn_goal_status(
                    info['position'], episode_start, info['heading_error'],
                    turn_hold_steps, max_turn_translation, goal_direction)
            elif not fixed_cycles and direction and 'goal_distance_m' in direction:
                reached = (walk_distance.advance(info.get('motion_phase'),walk_before,info['position'],source_phase,float(env.phase))
                           if walk_distance is not None else along >= direction['goal_distance_m'])
            if direction and ('goal_distance_m' in direction or turn_only):
                yaw_error = info['yaw'] - initial_yaw
                max_heading = max(max_heading, abs(float(np.degrees(np.arctan2(np.sin(yaw_error), np.cos(yaw_error))))))
                # Lightweight test/robot adapters may only expose yaw. In that case
                # derive the same wrapped error relative to the episode heading.
                heading_error = info.get('heading_error', info['yaw']-initial_yaw)
                final_heading_error = abs(math.degrees(math.atan2(
                    math.sin(float(heading_error)), math.cos(float(heading_error)))))
                if (episode_steps+1)*env.dt >= 5.:
                    velocity_errors.append(np.asarray(info['velocity'])[:2]-np.asarray(info['command'])[:2])
            if fixed_cycles:reached=False
            if turn_limit:
                reached = False
                truncated = True
            if reached:
                truncated = True  # Task boundary; retain Bellman bootstrap and stay inside the arena.
            goal_actor = copy.deepcopy(agent.actor.state_dict()) if promote_bc and reached and not terminal else None
            t = config['training']
            critic_only, interval = update_schedule(t, step, agent.warmup_completed)
            if step_blocks:
                critic_only = block_steps <= direction['critic_warmup_steps_per_angle']
                interval = t.get('critic_warmup_update_every', t['update_every']) if critic_only else t['update_every']
            replay_terminal = bool(turn_limit or bellman_terminal(terminal, reached, direction))
            potential_weight = config['reward'].get('biped_tracking', {}).get('turn_yaw_potential_weight', 0.)
            if potential_weight and replay_terminal and info.get('replay_mode') == 0:
                target = float(info['command'][2])
                after_error = abs(float(info['reward_velocity'][2]) - target) / abs(target)
                correction = potential_weight * t['gamma'] * after_error
                value += correction
                terms = info['reward_terms']
                for key in ('turn_speed_delta', 'turn_potential', 'turn_total', 'turn_weighted'):
                    terms[key] += correction
            episode_steps += 1
            episode_reward += value
            reward_terms = info.get('reward_terms', {})
            if 'turn_weight' in reward_terms:
                reward_console_window.append([
                    float(reward_terms.get('turn_weighted', 0.)),
                    float(reward_terms.get('walk_weighted', 0.)),
                    float(reward_terms.get('fall', 0.)),
                    float(value), float(reward_terms.get('stand_total',0.))])
            replay_terminal = bool(turn_limit or bellman_terminal(terminal, reached, direction))
            replay_args = dict(expert=critic_only and bool(t.get('warmup_expert_replay') or t.get('success_expert_replay')),
                               mode=info.get('replay_mode'), **critic_replay_args)
            if managed_replay:
                replay_args = dict(critic_replay_args)
                replay_args['mode'] = info.get('replay_mode')
            if phase_buffer is None:
                insert_transition(obs, info['executed_action'], value, nxt, replay_terminal, **replay_args)
            if not managed_replay and t.get('success_expert_replay') and not critic_only and not agent.replay.expert_ready():
                raise RuntimeError('Frozen BC collection produced insufficient successful Turn/Stand expert data; refusing to replace expert quotas with failures')
            metrics = dict(actor_loss=None, critic_loss=None, updates=agent.updates)
            if (attempt_interval is None and not config.get('bc_rollout', {}).get('collection_mode') and not collecting and step >= t['learning_starts'] and
                    agent.replay.ready(t['batch_size'],critic_only=critic_only) and
                    step % interval == 0):
                try:
                    # Keep the warmup stage distinct from delayed Actor updates.
                    # Use global environment steps so episode resets/BC recovery do not restart the cadence.
                    actor_due = (not t.get('freeze_online_actor', False) and
                                 step % t.get('actor_update_every', t['update_every']) == 0)
                    metrics = agent.update(critic_only=critic_only, update_actor=actor_due)
                except (FloatingPointError, RuntimeError) as exc:
                    write_json(folder/'training_failure.json', dict(step=step, updates=agent.updates,
                        replay_count=(agent.collector.count if collecting else agent.replay.count), error=str(exc)))
                    raise
            log_row = dict(step=step, q_executed=q_executed, reward=value, fell=terminal, episode_end=terminal or truncated,
                diagnostic_episode_id=episodes,
                actual_vx=info['actual_velocity'][0], actual_vy=info['actual_velocity'][1], actual_wz=info['actual_velocity'][2],
                reward_wz=info['reward_velocity'][2], reward_wz_cmd=info['reward_command'][2],
                source_observation=diagnostic_source_observation,
                executed_action=json.dumps(np.asarray(info['executed_action']).tolist(), allow_nan=False),
                next_observation=json.dumps(np.asarray(nxt).tolist(), allow_nan=False),
                reward_gravity=json.dumps(np.asarray(env.gravity).tolist(), allow_nan=False),
                reward_gyro_xy=json.dumps(np.asarray(env.gyro_value).tolist(), allow_nan=False),
                gait_phase_before=source_phase, gait_phase_after=float(env.phase),
                cycle_end=terminal or truncated or step == t['total_env_steps'] or float(env.phase) < source_phase,
                cycle_complete=float(env.phase) < source_phase,
                cycle_dt=info['cycle_dt'], world_dx=info['cycle_actual'][0],
                world_dy=info['cycle_actual'][1], yaw_delta=info['cycle_actual'][2],
                target_world_dx=info['cycle_target'][0], target_world_dy=info['cycle_target'][1],
                target_yaw_delta=info['cycle_target'][2], reward_before_backfill=value,
                reward_fall_before_backfill=info.get('reward_terms', {}).get('fall', 0.),
                reward_gravity_xy_sq=float(env.gravity[:2] @ env.gravity[:2]),
                reward_gyro_xy_sq=float(env.gyro_value @ env.gyro_value),
                **critic_log,
                motion_phase=info.get('motion_phase'),
                heading_error=info.get('heading_error'), actor_yaw_rate=info.get('actor_yaw_rate'),
                vx_cmd=info['command'][0], vy_cmd=info['command'][1], wz_cmd=info['command'][2],
                vx=info['velocity'][0], vy=info['velocity'][1], wz=info['velocity'][2],
                critic_only=critic_only, actor_updated=metrics['actor_loss'] is not None,
                actor_lr=agent.actor_optimizer.param_groups[0]['lr'],
                critic_lr=agent.critic_optimizer.param_groups[0]['lr'],
                exploration_std=agent.noise_std, bc_recoveries=recoveries, direction_deg=angle, angle_step=block_steps,
                reward_vx=info.get('reward_velocity', info['velocity'])[0],
                reward_vy=info.get('reward_velocity', info['velocity'])[1],
                reward_vx_cmd=info.get('reward_command', info['command'])[0],
                reward_vy_cmd=info.get('reward_command', info['command'])[1],
                reward_turn_translation=info.get('reward_terms', {}).get('turn_translation'),
                reward_turn_linear_velocity=info.get('reward_terms', {}).get('turn_linear_velocity'),
                reward_turn_heading_progress=info.get('reward_terms', {}).get('turn_heading_progress'),
                reward_turn_heading_error=info.get('reward_terms', {}).get('turn_heading_error'),
                reward_turn_yaw_rate_error=info.get('reward_terms', {}).get('turn_yaw_rate_error'),
                reward_walk_forward_progress=info.get('reward_terms', {}).get('walk_forward_progress'),
                reward_walk_heading_progress=info.get('reward_terms', {}).get('walk_heading_progress'),
                reward_walk_heading_error=info.get('reward_terms', {}).get('walk_heading_error'),
                reward_walk_cross_track=info.get('reward_terms', {}).get('walk_cross_track'),
                reward_tracking_x=info.get('reward_terms', {}).get('tracking_x'),
                reward_tracking_y=info.get('reward_terms', {}).get('tracking_y'),
                reward_tracking_yaw=info.get('reward_terms', {}).get('tracking_yaw'),
                reward_survival=info.get('reward_terms', {}).get('survival'),
                reward_orientation=info.get('reward_terms', {}).get('orientation'),
                reward_height=info.get('reward_terms', {}).get('height'),
                reward_action_rate=info.get('reward_terms', {}).get('action_rate'),
                reward_joint_velocity=info.get('reward_terms', {}).get('joint_velocity'),
                reward_fall=info.get('reward_terms', {}).get('fall'),
                reward_velocity_tracking=info.get('reward_terms', {}).get('velocity_tracking'),
                reward_stability=info.get('reward_terms', {}).get('stability'),
                turn_weight=info.get('turn_weight'), walk_weight=info.get('walk_weight'),
                reward_turn_total=info.get('reward_terms', {}).get('turn_total'),
                reward_walk_total=info.get('reward_terms', {}).get('walk_total'),
                reward_turn_weighted=info.get('reward_terms', {}).get('turn_weighted'),
                reward_walk_weighted=info.get('reward_terms', {}).get('walk_weighted'),
                walk_path_m=walk_distance.distance if walk_distance is not None else None,reward_stand_excess_translation=reward_terms.get('stand_excess_translation',0.),reward_command_calibration_offset=reward_terms.get('command_calibration_offset',0.),reward_turn_hold=reward_terms.get('turn_hold',0.),stand_hold_steps=info.get('hold_steps',0),next_motion_phase=info.get('next_stage'),
                source_motion_class=info.get('replay_bucket'), replay_destination='expert_collection' if collecting else ('online_fifo' if t.get('online_fifo_only') else 'pending_cycle_outcome' if hasattr(agent.replay,'success') else 'bc_rollout'),
                replay_success=agent.replay.last_sample_counts.get('success'),
                replay_expert_success=agent.replay.last_sample_counts.get('expert_success'),
                replay_online_success=agent.replay.last_sample_counts.get('online_success'),
                expert_success_count=getattr(agent.replay,'expert_count',0),
                replay_failure=agent.replay.last_sample_counts.get('failure'),
                success_fifo_count=getattr(getattr(agent.replay,'success',None),'count',None),
                failure_fifo_count=getattr(getattr(agent.replay,'failure',None),'count',None),
                online_fifo_count=0 if collecting else agent.replay.online.count-getattr(agent.replay,'expert_count',0),
                online_turn_fifo_count=0 if collecting else getattr(getattr(agent.replay,'online_turn',None),'count',0),
                bellman_terminal=replay_terminal,
                reward_walk_speed_error=reward_terms.get('walk_speed_error',0.),
                reward_walk_lateral_speed=reward_terms.get('walk_lateral_speed',0.),
                reward_walk_yaw_rate_error=reward_terms.get('walk_yaw_rate_error',0.),
                reward_stand_total=reward_terms.get('stand_total',0.),
                **metrics)
            if 'reward_history_count' in info:
                log_row['reward_history_count'] = info['reward_history_count']
            if t.get('walk_commands'):
                log_row['reward_tracking_xy'] = reward_terms['tracking_xy']
            if config['reward'].get('biped_tracking', {}).get('turn_world_z_regularization'):
                log_row.update(reward_ankle_soft_limit=reward_terms.get('ankle_soft_limit'),
                               reward_hip_deviation=reward_terms.get('hip_deviation'))
            if config['reward'].get('biped_tracking', {}).get('turn_reverse_foot_yaw_weight'):
                log_row['reward_reverse_foot_yaw'] = reward_terms.get('reverse_foot_yaw')
            if config['reward'].get('biped_tracking', {}).get('phase_foot', {}).get('kind') == 'phase_position_minimum_v1':
                for key in ('phase_foot_position', 'foot_left_rear', 'foot_left_front',
                            'foot_right_rear', 'foot_right_front'):
                    log_row['reward_'+key] = reward_terms.get(key)
            if config['reward'].get('biped_tracking', {}).get('turn_heading_window_seconds'):
                log_row.update(reward_turn_heading=reward_terms.get('turn_heading'),
                               turn_window_heading_error=info.get('turn_heading_error'))
            if (config['reward'].get('biped_tracking', {}).get('turn_yaw_error_progress_weight') or config['reward'].get('biped_tracking', {}).get('turn_yaw_error_cost_weight') or config['reward'].get('biped_tracking', {}).get('turn_acceleration_weight')):
                log_row['reward_turn_speed_delta'] = reward_terms.get('turn_speed_delta')
            if config.get('landing_alignment'):
                log_row.update(landing_gate=info['landing_gate'], landing_target_yaw_error_deg=info['landing_target_yaw_error_deg'])
            log_row.update(info['report_diagnostics'])
            if phase_buffer is None:
                writer.writerow(log_row)
            else:
                log_row['_cycle_actual'] = info['cycle_actual']
                log_row['_cycle_target'] = info['cycle_target']
                log_row['_cycle_dt'] = info['cycle_dt']
                phase_buffer.append(obs, info['executed_action'], value, nxt, replay_terminal,
                    replay_args, log_row, reward_console_window[-1] if 'turn_weight' in reward_terms else None)
                if terminal or truncated or step == t['total_env_steps'] or float(env.phase) < source_phase:
                    complete_cycle = env.phase < source_phase
                    after_replay = None
                    if (attempt_interval is not None and (complete_cycle or terminal)
                            and log_row['motion_phase'] in attempt_phases):
                        joint_attempts += 1
                        if joint_attempts % attempt_interval == 0:
                            def update_joint(row):
                                if step < t['learning_starts'] or not agent.replay.ready(t['batch_size'], critic_only=False):
                                    return
                                actor_due = joint_attempts % actor_attempt_interval == 0
                                row.update(agent.update(critic_only=False, update_actor=actor_due))
                                row['actor_updated'] = actor_due
                                if hasattr(agent.replay, 'pool_size_log_field'):
                                    sampled = agent.replay.last_sample_counts
                                    row.update(replay_high_score=sampled['high_score'],
                                               replay_online_success=sampled['online_success'],
                                               replay_expert_success=sampled['expert_success'],
                                               replay_failure=sampled['failure'])
                                    row[agent.replay.pool_size_log_field] = len(agent.replay.high_selected)
                            after_replay = update_joint
                    episode_reward += phase_buffer.flush(
                        insert_transition, writer, fell=terminal, complete_cycle=complete_cycle,
                        after_replay=after_replay, archive_cycle=archive_cycle)
            if step % 100 == 0:
                handle.flush()
                trace_handle.flush()
            obs = nxt
            stalled += 1
            if info['position'][0]-progress_origin >= init.get('stall_min_forward_progress_m', .02):
                stalled = 0
                progress_origin = info['position'][0]
            if terminal:
                consecutive_falls += 1
            elif truncated:
                consecutive_falls = 0
            reason = None
            recovery_details = None
            focus_reset_on_recovery = False
            if init.get('reset_to_bc_on_failure'):
                if consecutive_falls >= init['consecutive_falls']:
                    reason = 'consecutive_falls'
                elif init.get('stall_timeout_steps', 0) > 0 and stalled >= init['stall_timeout_steps']:
                    reason = 'stalled'
                elif (not terminal and not truncated and direction and
                      direction.get('navigation_mode') in ('turn_then_forward', 'blended_turn_walk', 'turn_only')):
                    recovery_details = turn_recovery.advance(
                        step, critic_only, info.get('replay_mode', int(abs(float(info['command'][0])) > 1e-8)) == 0,
                        info.get('heading_error', 0.), info.get('actor_yaw_rate', 0.))
                    if recovery_details:
                        reason = recovery_details.get('reason', 'turn_stall')
            if recovery_details is None and (terminal or truncated):
                turn_recovery.reset_window()
            if reason:
                if phase_buffer is not None:
                    episode_reward += phase_buffer.flush(insert_transition, writer, fell=False)
                focus_reset_on_recovery = bool(sampler and
                    direction.get('late_focus_sampling', {}).get('reset_on_bc_recovery', False))
                restored_source = bc_source(agent)
                agent.restore_bc()
                turn_recovery.mark_recovery(step)
                recoveries += 1
                recovery_record = dict(step=step, reason=reason, replay_count=(agent.collector.count if collecting else agent.replay.count),
                    bc_source=restored_source,
                    direction_deg=angle, angle_step=block_steps, consecutive_falls=consecutive_falls,
                    late_focus_history_reset=focus_reset_on_recovery)
                if recovery_details:
                    recovery_record.update(recovery_details)
                recovery_log.append(recovery_record)
                write_json(folder/'recoveries.json', recovery_log)
                if restored_source.get('kind') == 'training_success':
                    restored = f"angle={restored_source['angle_deg']:+g}deg step={restored_source['step']} file={restored_source['checkpoint']}"
                else:
                    restored = f"{restored_source.get('kind', 'unknown')} file={restored_source.get('checkpoint')}"
                if recovery_details and 'worsening_deg' in recovery_details:
                    diagnostics = (f" start_error={recovery_details['window_start_mean_error_deg']:.1f}deg"
                        f" end_error={recovery_details['window_end_mean_error_deg']:.1f}deg"
                        f" worsening={recovery_details['worsening_deg']:+.1f}deg")
                elif recovery_details:
                    diagnostics = (f" error={recovery_details['heading_error_deg']:.1f}deg"
                        f" improvement={recovery_details['improvement_deg']:+.1f}deg"
                        f" correct_yaw={100*recovery_details['correct_yaw_fraction']:.0f}%")
                else:
                    diagnostics = ''
                print(f'[bc-recovery] step={step} task_angle={angle:+g}deg '
                      f'reason={reason}{diagnostics} restored={restored}', flush=True)
                consecutive_falls = 0
            if terminal or truncated or reason or step == t['total_env_steps']:
                displacement = np.asarray(info['position'])-episode_start
                elapsed = episode_steps*config['webots']['control_time_step_ms']/1000.
                end_reason = 'fall' if terminal else ('bc_recovery' if reason else ('goal' if reached else ('angle_steps' if step_blocks and block_timeout else ('time_limit' if truncated else 'budget_end'))))
                if turn_limit:
                    end_reason = 'turn_translation_limit'
                    print(f'[reset] global_step={step} reason={end_reason} distance={turn_boundary.distance:.3f}m angle={angle:+g}', flush=True)
                mean_error = float(np.linalg.norm(np.mean(velocity_errors, axis=0))) if velocity_errors else None
                frozen_goal = bool(step_blocks and critic_only and reached and not terminal)
                success = bool(reached and not terminal and not frozen_goal)
                if turn_only:
                    heading_metric = final_heading_error
                    success = bool(success and
                        max_turn_translation <= direction['turn_max_translation_m'] and
                        heading_metric <= goal_direction['heading_tolerance_deg'])
                elif direction and 'goal_distance_m' in direction:
                    heading_metric = (final_heading_error if direction.get('navigation_mode') in ('turn_then_forward', 'blended_turn_walk')
                                      else max_heading)
                    success = bool(success and abs(cross) <= direction['cross_track_tolerance_m']
                        and heading_metric <= direction['heading_tolerance_deg']
                        and mean_error is not None and mean_error <= direction['velocity_tolerance_m_s'])
                if collecting or (not managed_replay and t.get('success_expert_replay')):
                    admission = (agent.collector if collecting else agent.replay).finish_episode(success and not reason, angle)
                    if collecting:
                        agent.collector.export()
                    with (folder/'expert_replay_events.jsonl').open('a', encoding='utf-8') as events:
                        events.write(json.dumps(dict(step=step, angle=angle, end_reason=end_reason,
                                                     success=success, **admission), allow_nan=False)+'\n')
                if sampler and not episode_started_critic_only:
                    sampler.record_outcome(angle, success and not reason)
                if sampler and (reason or focus_reset_on_recovery):
                    sampler.reset_focus_history()
                if promote_bc and success:
                    remember_success(agent, goal_actor, folder, dict(angle_deg=angle, step=step,
                        episode=episodes, angle_step=block_steps, arrival_time_s=elapsed,
                        signed_lateral_m=float(displacement[1]), absolute_lateral_m=float(abs(displacement[1])),
                        net_forward_m=float(displacement[0]), along_track_m=along, cross_track_m=cross,
                        max_heading_deg=max_heading, mean_velocity_error_m_s=mean_error,
                        goal_distance_m=direction.get('goal_distance_m'),
                        turn_max_translation_m=direction.get('turn_max_translation_m')))
                episode_writer.writerow(dict(episode=episodes, end_step=step, steps=episode_steps,
                    elapsed_s=elapsed, reward=episode_reward, end_reason=end_reason,
                    arrival_time_s=elapsed if reached and not terminal else None,
                    signed_lateral_m=float(displacement[1]), absolute_lateral_m=float(abs(displacement[1])),
                    net_forward_m=float(displacement[0]),
                    max_planar_translation_m=max_turn_translation if turn_only else None,
                    fell=terminal, reached_goal=bool(reached and not terminal),
                    bc_recovery=reason or '', direction_deg=angle, along_track_m=along, cross_track_m=cross,
                    max_heading_deg=heading_metric if direction and ('goal_distance_m' in direction or turn_only) else None,
                    mean_velocity_error_m_s=mean_error, success=success, critic_only_at_end=critic_only,
                    success_excluded_reason='actor_frozen' if frozen_goal else ''))
                episode_handle.flush()
                block_goals += int(reached and not terminal)
                block_successes += int(success)
                block_frozen_goals += int(frozen_goal)
            success_target = (direction or {}).get('successes_to_advance', 1)
            enough_successes = step_blocks and block_successes >= success_target
            block_done = retry_angles and ((not step_blocks and bool(reached and not terminal)) or enough_successes or block_timeout or step == t['total_env_steps'])
            if block_done:
                budget_only = not (block_timeout or enough_successes) if step_blocks else not (reached and not terminal) and not block_timeout
                angle_success = block_successes > 0 if step_blocks else success
                completed_blocks += int(not budget_only)
                if not budget_only:
                    # A reached goal remains acceptable for sweep progression even when
                    # strict cross-track/yaw/velocity criteria reject BC promotion.
                    fall_failed = block_goals == 0 and block_falls > 0
                    failed_blocks = failed_blocks + 1 if fall_failed else 0
                paused = not budget_only and failed_blocks >= direction['pause_after_failed_angles']
                block_result = dict(block=completed_blocks, angle_deg=angle, end_step=step,
                    simulated_seconds=block_steps*config['webots']['control_time_step_ms']/1000.,
                    falls=block_falls, reached_goal=block_goals > 0, success=angle_success,
                    angle_steps=block_steps, goal_count=block_goals, successful_attempts=block_successes,
                    frozen_goal_count=block_frozen_goals,
                    end_reason='budget_end' if budget_only else ('success_target' if enough_successes else ('angle_steps' if step_blocks else ('goal' if reached and not terminal else 'time_limit'))),
                    fall_failed=fall_failed if not budget_only else False,
                    consecutive_fall_failed_angles=failed_blocks,
                    consecutive_failed_angles=failed_blocks, along_track_m=along, cross_track_m=cross)
                with (folder/'angle_results.jsonl').open('a',encoding='utf-8') as out:
                    out.write(json.dumps(block_result)+'\n')
                emit(folder,config,'angle_finished',f'角度 {angle:g}° 训练结束',
                     f"原因：{block_result['end_reason']}；到达终点：{'是' if block_result['reached_goal'] else '否'}；达标：{'是' if angle_success else '否'}；步数 {block_steps}；到达 {block_goals} 次（冻结期 {block_frozen_goals} 次，不计成功）；成功 {block_successes} 次；跌倒 {block_falls} 次。",**block_result)
                if promote_bc and not budget_only:
                    finish_angle(agent, folder, angle, step)
            if collecting and agent.collector.complete():
                paused = True
            if terminal or truncated or reason:
                episodes += 1
                falls += int(terminal)
            if step % t['checkpoint_every'] == 0 or step == t['total_env_steps'] or paused:
                checkpoint = folder/'checkpoints'/f'step_{step:08d}.pth'
                agent.save(checkpoint, step)
                print(f'[checkpoint] {checkpoint.name} updates={agent.updates} falls={falls}', flush=True)
            if step % 100 == 0 or step == t['total_env_steps'] or paused:
                handle.flush()
                if sampler:
                    motion_phase = info.get(
                        'motion_phase', 'walk' if abs(float(info['command'][0])) > 1e-8 else 'turn')
                    state = dict(step=step, direction_deg=angle, angle_step=block_steps, gait_phase=float(env.phase),
                        bc_source=bc_source(agent), angle_steps=displayed_angle_steps,
                        phase='paused' if paused else motion_phase,
                        training_phase='expert_collection' if collecting else ('critic_warmup' if critic_only else 'joint'),
                        pause_seconds=0., reward=float(value), **direction_metrics(
                            info['position'], episode_start, initial_yaw,
                            info.get('yaw', initial_yaw), angle,
                            angle if direction.get('navigation_mode') in ('turn_then_forward', 'blended_turn_walk', 'turn_only') else 0.))
                    state['instant_reward'] = float(value)
                    state['instant_reward_terms'] = dict(reward_terms)
                    if reward_console_window:
                        reward_average = np.mean(
                            np.asarray(reward_console_window, dtype=float), axis=0)
                        state['reward'] = float(reward_average[3])
                        state['reward_window_steps'] = len(reward_console_window)
                        state['reward_terms'] = dict(
                            turn_weight=info.get('turn_weight', 0.),
                            walk_weight=info.get('walk_weight', 0.),
                            turn_weighted=float(reward_average[0]),
                            walk_weighted=float(reward_average[1]),
                            fall=float(reward_average[2]), stand_total=float(reward_average[4]))
                    else:
                        state['reward_terms'] = reward_terms
                    state['source_motion_class'] = info.get('replay_bucket')
                    state['replay_destination'] = 'expert_collection' if collecting else ('online_fifo' if t.get('online_fifo_only') else 'pending_cycle_outcome' if hasattr(agent.replay,'success') else 'bc_rollout')
                    state['online_fifo_count'] = 0 if collecting else agent.replay.online.count
                    state['online_turn_fifo_count'] = 0 if collecting else getattr(getattr(agent.replay,'online_turn',None),'count',0)
                    state['replay_online_turn'] = 0 if collecting else agent.replay.last_sample_counts.get('online_turn', 0)
                    state['last_sample_counts'] = {} if collecting else dict(agent.replay.last_sample_counts)
                    state.update(sampler.sampling_state(angle))
                    if fixed_cycles:
                        state.update(fixed_cycle_commands=True,phase=env.stages.stage,
                            active_command=env.stages.command(0.,0.).tolist(),
                            command_cycles=env.stages.cycles,command_cycle_limit=env.stages.limit)
                    write_json(folder/'direction_status.json', state)
                    print(format_direction(state), flush=True)
                    breakdown = format_reward_breakdown(state)
                    if breakdown:
                        print(breakdown, flush=True)
                write_json(folder/'progress.json', dict(step=step, total_steps=t['total_env_steps'],
                    updates=agent.updates, falls=falls, episodes=episodes, bc_recoveries=recoveries,
                    stage='paused' if paused else ('critic_warmup' if critic_only else 'joint'), updated_at=time.time(),
                    completed_angle_blocks=completed_blocks, consecutive_failed_angles=failed_blocks,
                    actor_lr=agent.actor_optimizer.param_groups[0]['lr'],
                    critic_lr=agent.critic_optimizer.param_groups[0]['lr'], exploration_std=agent.noise_std))
            final_step = (step == t['total_env_steps'] and not collecting and not smoke
                          and not config.get('bc_rollout', {}).get('collection_mode'))
            if terminal or truncated or reason or final_step:
                if not paused and (step < t['total_env_steps'] or final_step):
                    # Online failures must pass through FailureRetry.next(): a
                    # different random task first, then the pending failed task.
                    # In particular Turn drift resets must not bypass the sampler.
                    keep_angle = (turn_limit or (retry_angles and not block_done)) if collecting else False
                    if block_done:
                        consecutive_falls = 0
                    if block_done or not retry_angles:
                        block_steps, block_falls = 0, 0
                        block_goals, block_successes, block_frozen_goals = 0, 0, 0
                    if world_reload or final_step:
                        handle.flush()
                        episode_handle.flush()
                        state = dict(next_step=step+1, keep_angle=keep_angle,
                            sampler=sampler, angle=angle, command=command,
                            block_counts=(block_goals, block_successes, block_frozen_goals),
                            blocks=(block_steps, block_falls, completed_blocks, failed_blocks),
                            counts=(episodes, falls, recoveries, consecutive_falls), recovery_log=recovery_log,
                            turn_recovery_monitor=turn_recovery,
                            turn_fall_walk_reset=turn_fall_walk_reset.state(),
                            last_boundary_reason=end_reason, joint_attempts=joint_attempts,
                            command_rng=env.commands.rng.bit_generator.state if hasattr(env, 'commands') else None)
                        if persistent_stages is not None:
                            state['fixed_cycle_state'] = persistent_stages.state()
                        if final_step:
                            if not (terminal or truncated or reason):
                                state['counts'] = (episodes + 1, falls, recoveries, consecutive_falls)
                            state['checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                            save_continuation(checkpoint.parent, config, agent, state,
                                              filename=f'continuation_{step:08d}.pth')
                            break
                        save_continuation(folder, config, agent, state)
                        with (folder/'world_reload_events.jsonl').open('a', encoding='utf-8') as out:
                            out.write(json.dumps(dict(step=step, reason=end_reason, keep_angle=keep_angle,
                                replay_count=(agent.collector.count if collecting else agent.replay.count), updates=agent.updates))+'\n')
                        raise WorldReloadRequested()
                    obs = reset_training(step+1, keep_angle=keep_angle)
                    progress_origin, stalled = env.position[0], 0
                    episode_start = env.position.copy()
                    initial_yaw = getattr(env, 'yaw', 0.)
                    max_heading, velocity_errors = 0., []
                    episode_steps, episode_reward = 0, 0.
            if paused:
                break
    if collecting:
        complete = agent.collector.complete()
        agent.collector.export()
        return dict(status='completed' if complete else 'paused', kind='expert_collection',
                    stop_reason='collection_attempts_finished' if complete else 'collection_budget_exhausted',
                    env_steps=step, updates=agent.updates, dataset_complete=agent.collector.dataset_complete(),
                    dataset=config['expert_pipeline']['dataset'], independent_evaluation='pending',
                    policy_improvement_claim=False)
    if config.get('bc_rollout', {}).get('collection_mode'):
        from .bc_rollout import export_collection
        export_collection(agent, config, folder, step)
        return dict(status='paused' if paused else 'completed',kind='frozen_bc_rollout',
                    stop_reason='collection_paused' if paused else 'budget_end',
                    env_steps=step,updates=agent.updates,episodes=episodes,falls=falls,
                    independent_evaluation='pending',policy_improvement_claim=False)
    emit(folder,config,'training_paused' if paused else 'training_completed',
         '训练已暂停' if paused else '训练已结束',
         f"步数 {step}；连续全程摔倒且未到终点的角度 {failed_blocks}。" + ('已保存Checkpoint，等待处理，不自动重启。' if paused else '达到训练步数上限，已保存Checkpoint。'),
         step=step, consecutive_failed_angles=failed_blocks)
    return dict(status='paused' if paused else 'completed', kind='interface_smoke' if smoke else 'training',
                stop_reason='consecutive_fall_failed_angles' if paused else 'budget_end',
                completed_angle_blocks=completed_blocks, consecutive_failed_angles=failed_blocks,
                env_steps=step, updates=agent.updates, episodes=episodes, falls=falls, bc_recoveries=recoveries,
                independent_evaluation='pending', policy_improvement_claim=False)


def main(snapshot):
    from controller import Supervisor
    payload = json.loads(Path(snapshot).read_text(encoding='utf-8'))
    folder = Path(payload.get('output_run', Path(snapshot).parent))
    robot = Supervisor()
    write_json(folder/'status.json', dict(status='running'))
    try:
        config, options = payload['config'], payload['options']
        if options['mode'] != 'train':
            raise ValueError('This controller supports direction training only')
        env = WebotsBaseWalk(robot, config)
        try:
            result = run_train(env, config, folder, options['smoke'], checkpoint_resume='output_run' in payload)
        except WorldReloadRequested:
            write_json(folder/'status.json', dict(status='running', phase='world_reload'))
            print('[world_reload] state saved; reloading world', flush=True)
            reload_world(robot, config['webots']['control_time_step_ms'])
            return
        result.update(contract=CONTRACT['version'], checkpoint_sha256=payload.get('checkpoint_sha256'),
                      artifact_purpose=config.get('artifact_purpose', 'research'), source_training_steps=None)
        write_json(folder/'result.json', result)
        write_json(folder/'status.json', dict(status=result['status'], reason=result.get('stop_reason')))
        robot.simulationQuit(0)
    except BaseException as exc:
        emit(folder,payload.get('config',{}),'training_failed','训练异常停止',str(exc))
        write_json(folder/'status.json', dict(status='failed', error=str(exc), traceback=traceback.format_exc()))
        print(traceback.format_exc(), flush=True)
        robot.simulationQuit(1)
