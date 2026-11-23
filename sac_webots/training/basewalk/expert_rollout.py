"""Fresh-world expert validation, 48D collection and frozen BC evaluation."""
import csv
import hashlib
import json
import math
from pathlib import Path
import traceback
import numpy as np
import torch
from .core import DEFAULT_POSE, MINIMUM, MAXIMUM, CONTRACT, joint_targets
from .network_inputs import observation_dim
from .sine_expert import expert_targets, turn_then_forward_targets, three_stage_targets


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rollout_success(kind, fell, tracking, arrival, cross, elapsed, seconds, dt,
                    cross_track_tolerance_m):
    """Apply the effective experiment tolerance to rollout acceptance."""
    completion = (arrival is not None and abs(cross) <= cross_track_tolerance_m
                  if kind == 'eval' else elapsed >= seconds-dt/2)
    return bool(not fell and tracking and completion)


def rollout(env, job, actor, folder):
    env.direction_target_deg = job['angle']
    if job.get('fixed_command') is not None:
        env.expert_fixed_command=np.asarray(job['fixed_command'],dtype=np.float32)
    three_stage = job.get('expert_type') == 'turn_hold_forward'
    if job.get('expert_type') == 'physical_navigation_51d':
        env.cfg['initial_gait_phase'] = float(job.get('initial_phase', 0.))
    if three_stage:
        env.cfg['turn_hold_seconds'] = float(job['hold_seconds'])
        env.cfg['initial_gait_phase'] = float(job['initial_phase'])
    obs = env.reset(command=job['command'])
    contact_trace = []
    teacher=None
    if actor is None:
        from .sine_support import SineSupport
        teacher=SineSupport(env.config['physical_gait'],env.config['contact_expert'],
            env.foot_history,env.q,env.lower,env.upper,env.cfg['action_scale'])
        write(folder/'force_calibration.json',teacher.calibration)
    turn_arrival = None
    accumulated_yaw = 0.
    max_excursion = 0.
    start = env.position.copy()
    theta = env.yaw + math.radians(job['angle'])
    direction = np.asarray([math.cos(theta), math.sin(theta)], dtype=float)
    env.cfg['episode_seconds'] = job['seconds']
    xs, ys, next_xs, rewards, terminals, modes, velocities, headings = [], [], [], [], [], [], [], []
    stage_counts = dict(turn=0, hold=0, stand=0, walk=0)
    stage_positions = dict()
    hold_max_translation = 0.
    filtered_expert_target = None
    last_expert_stage = None
    transition_source = None
    transition_started_at = 0.
    action_noise_std = float(job.get('expert_action_noise_std', 0.))
    action_noise_rng = np.random.default_rng(int(job.get('noise_seed', 0)))
    critic_states, next_critic_states, executed_actions = [], [], []
    cycle_actual, cycle_target, cycle_dt, cycle_ends = [], [], [], []
    recorded_reward_inputs=[]
    support_loads=[];next_support_loads=[]
    stand_target=None
    stand_started=0.
    arrival = None
    arrival_metrics = None
    reported_stage = None
    with (folder/'trajectory.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['time_s','x','y','z','yaw','filtered_vx','filtered_vy','filtered_yaw_rate','wz_cmd','fell','actor_yaw_rate','motion_stage'])
        for step_index in range(round(job['seconds']/env.dt)):
            if actor is None:
                if teacher is not None:
                    from .physical_validation import torso_roll_pitch
                    source_command=env._command()
                    target=teacher.update(source_command,env.q,env.foot_forces,
                        torso_roll_pitch(env.imu.getRollPitchYaw()[:2]),env.gyro_value,env.dt)
                elif job.get('expert_type') == 'physical_navigation_51d':
                    source_command = env._command()
                    stage = 'walk' if source_command[0] > 1e-8 else ('turn' if abs(source_command[2])>1e-8 else 'hold')
                    target = three_stage_targets(env.phase, env.elapsed, job['parameters'],
                                                 stage, source_command, job.get('expert_rate_reference',env.config.get('physical_expert_control',{}).get('yaw_rate_reference',env.cfg['heading_max_rate'])))
                elif three_stage:
                    source_command = np.array([obs[5], 0., obs[7]], dtype=np.float32)
                    raw_target = three_stage_targets(
                        env.phase, env.elapsed, job['parameters'], env.motion_stage,
                        source_command, env.cfg['heading_max_rate'])
                    if last_expert_stage is None:
                        filtered_expert_target = raw_target
                        last_expert_stage = env.motion_stage
                    elif env.motion_stage != last_expert_stage:
                        transition_source = filtered_expert_target.copy()
                        transition_started_at = env.elapsed
                        last_expert_stage = env.motion_stage
                    if transition_source is not None:
                        blend = min(1., (env.elapsed-transition_started_at)/.4)
                        filtered_expert_target = ((1.-blend)*transition_source+
                                                  blend*raw_target)
                        if blend >= 1.:
                            transition_source = None
                    else:
                        filtered_expert_target = raw_target
                    # Only command boundaries are blended; the periodic target
                    # is otherwise preserved. Recorded actions equal execution.
                    target = filtered_expert_target
                else:
                        # In the 48D contract index 6 is heading_error_sin,
                        # not command_vy. The physical expert still receives
                        # vy=0, so reconstruct the three-command vector using
                        # the unchanged vx/wz indices.
                        source_command = np.array([obs[5], 0., obs[7]], dtype=np.float32)
                        target = (turn_then_forward_targets(env.phase, env.elapsed, job['parameters'], source_command,
                                                        env.cfg['heading_max_rate'])
                              if job.get('expert_type') == 'turn_then_forward' else
                              expert_targets(env.phase, env.elapsed, job['parameters']))
                if teacher is None and abs(source_command[0])+abs(source_command[2])<1e-8:
                    if stand_target is None:
                        stand_target=DEFAULT_POSE+env.cfg['action_scale']*env.last_action
                        stand_started=env.elapsed
                    alpha=min(1.,(env.elapsed-stand_started)/.4)
                    alpha=alpha*alpha*(3.-2.*alpha)
                    target=(1.-alpha)*stand_target+alpha*DEFAULT_POSE
                else:stand_target=None
                if job.get('balance_pitch_gain') is not None:
                    from .expert_balance import stabilize
                    target=stabilize(target,env.gravity,env.gyro_value,job)
                if hasattr(env, 'contact_controller'):
                    target = env.contact_controller.stabilize(target, env.gyro_value)
                clean_action = (target-DEFAULT_POSE)/env.cfg['action_scale']
                _, supervision_action = joint_targets(
                    clean_action, env.cfg['action_scale'],
                    getattr(env,'lower',MINIMUM),getattr(env,'upper',MAXIMUM))
                action = clean_action.copy()
                if action_noise_std:
                    noise=action_noise_rng.normal(0., action_noise_std, size=action.shape)
                    noise[6]=noise[0]
                    action += noise
            else:
                with torch.no_grad():
                    action = actor(torch.from_numpy(obs).unsqueeze(0))[0].numpy()
                supervision_action = None
            from .stationary_reward import critic_features
            critic_states.append(critic_features(env, env.config))
            phase_before = float(env.phase)
            before = obs.copy()
            support_loads.append(env.support_feedback.loads.copy())
            expected_heading_error = math.atan2(
                math.sin(env.heading_reference-env.yaw),
                math.cos(env.heading_reference-env.yaw))
            obs, step_reward, fell, truncated, info = env.step(action)
            next_support_loads.append(env.support_feedback.loads.copy())
            if not np.allclose(obs[34:46],info['executed_action'],atol=1e-6):
                raise ValueError('Next State previous Action is not the executed Action')
            if (not np.isclose(before[5], info['command'][0], atol=1e-7) or
                    not np.allclose(before[5:8], info['command'], atol=1e-6)):
                raise ValueError('Source Observation command mismatch')
            if hasattr(env, 'contact_controller'):
                contact_trace.append(dict(time_s=env.elapsed,phase_before=phase_before,phase_after=float(env.phase),
                    force_raw=env.foot_forces.tolist(), **env.contact_controller.diagnostics()))
            recorded_reward_inputs.append(info['reward_inputs'])
            next_critic_states.append(critic_features(env, env.config))
            executed_actions.append(info["executed_action"].copy())
            xs.append(before)
            # DART-style collection executes a perturbed expert action but
            # labels the resulting visited State with the clean expert action.
            ys.append((supervision_action if supervision_action is not None
                       else info['executed_action']).copy())
            next_xs.append(obs.copy())
            rewards.append(step_reward)
            cycle_actual.append(info['cycle_actual'])
            cycle_target.append(info['cycle_target'])
            cycle_dt.append(info['cycle_dt'])
            cycle_ends.append(float(env.phase)<phase_before)
            terminals.append(fell)
            stage = info.get('motion_phase',
                             'turn' if abs(float(before[5])) < 1e-7 else 'walk')
            if three_stage and stage != reported_stage:
                event=dict(angle=job['angle'],episode=job['episode'],step=step_index,
                           elapsed_s=env.elapsed,previous=reported_stage or 'start',current=stage)
                with (folder.parent/'stage_events.jsonl').open('a',encoding='utf-8') as events:
                    events.write(json.dumps(event,ensure_ascii=False)+'\n')
                reported_stage=stage
            mode = {'turn': 0, 'hold': 1, 'stand':1, 'walk': 2}.get(stage,
                    0 if abs(float(before[5])) < 1e-7 else 2)
            modes.append(mode)
            if stage in stage_counts:
                stage_counts[stage] += 1
                stage_positions.setdefault(stage, [env.position.copy(), env.position.copy()])[1] = env.position.copy()
                if stage == 'hold':
                    hold_max_translation = max(
                        hold_max_translation,
                        float(np.linalg.norm(env.position[:2]-stage_positions['hold'][0][:2])))
            headings.append(abs(math.degrees(info['heading_error'])))
            if env.elapsed >= 5:
                velocities.append(info['velocity'])
            displacement = env.position-start
            along = float(direction @ displacement[:2])
            cross = float(-direction[1]*displacement[0]+direction[0]*displacement[1])
            if arrival is None and along >= env.config['direction_training']['goal_distance_m'] and not fell:
                arrival = env.elapsed
                arrival_metrics = dict(arrival_time_s=arrival, net_forward_m=float(displacement[0]),
                    signed_lateral_m=float(displacement[1]), absolute_lateral_m=float(abs(displacement[1])),
                    along_track_m=along, cross_track_m=cross)
            accumulated_yaw += float(info['cycle_actual'][2])
            max_excursion = max(max_excursion,float(np.linalg.norm(displacement[:2])))
            if job.get('stop_at_turn_deg') is not None and not fell:
                if abs(math.degrees(accumulated_yaw)-job['stop_at_turn_deg']) <= job.get('turn_tolerance_deg',3.):
                    turn_arrival = env.elapsed
            writer.writerow([env.elapsed,*env.position,env.yaw,*info['velocity'],before[7],fell,info['actor_yaw_rate'],stage])
            if turn_arrival is not None or fell or truncated or (job['kind']=='eval' and arrival is not None):
                break
    velocity = np.mean(velocities,axis=0) if velocities else np.zeros(3)
    actual_angle = math.degrees(math.atan2(velocity[1],velocity[0]))
    # velocity is in the robot body frame. After turn-then-forward alignment,
    # body-forward tracking is 0 degrees; the world progress ray is still +90.
    heading_controlled = env.cfg.get('navigation_mode') in ('turn_then_forward', 'turn_hold_forward', 'blended_turn_walk', 'turn_only')
    expected_velocity_angle = 0. if heading_controlled else job['angle']
    angle_error = (actual_angle-expected_velocity_angle+180)%360-180
    speed = float(np.linalg.norm(velocity[:2]))
    heading_metric = headings[-1] if heading_controlled else max(headings)
    tracking = abs(angle_error)<=2 and abs(speed-np.linalg.norm(job['command'][:2]))<=.008 and heading_metric<=5
    hold_translation = hold_max_translation if 'hold' in stage_positions else float('inf')
    hold_net_speed = float('inf')
    if 'hold' in stage_positions and stage_counts['hold']:
        hold_net_distance=float(np.linalg.norm(
            stage_positions['hold'][1][:2]-stage_positions['hold'][0][:2]))
        hold_net_speed=hold_net_distance/(stage_counts['hold']*env.dt)
    walk_progress = 0.
    if 'walk' in stage_positions:
        walk_delta = stage_positions['walk'][1]-stage_positions['walk'][0]
        walk_progress = float(direction @ walk_delta[:2])
    if three_stage:
        stage_coverage = (stage_counts['walk'] > 0 and
                          (job['angle'] == 0 or stage_counts['turn'] > 0) and
                          stage_counts['hold'] > 0)
        if job['kind'] == 'eval':
            success = bool(not fell and stage_coverage and arrival is not None and
                           abs(cross) <= env.config['direction_training']['cross_track_tolerance_m'] and
                           heading_metric <= env.config['direction_training']['heading_tolerance_deg'])
        else:
            hold_ok = (hold_translation <= env.cfg['expert_hold_translation_tolerance_m'] and
                       hold_net_speed <= env.cfg['expert_hold_speed_tolerance_m_s'])
            success = bool(not fell and stage_coverage and hold_ok and
                           walk_progress >= env.cfg['expert_min_walk_progress_m'] and
                           heading_metric <= env.config['direction_training']['heading_tolerance_deg'] and
                           env.elapsed >= job['seconds']-env.dt/2)
    else:
        success = rollout_success(job['kind'], fell, tracking, arrival, cross, env.elapsed,
                                  job['seconds'], env.dt,
                                  env.config['direction_training']['cross_track_tolerance_m'])
    if job.get('expert_type') == 'physical_navigation_51d':
        # Demonstration quality does not require every angle or speed tracking.
        success = bool(not fell and stage_counts['walk'] > 0 and
                       (arrival is not None if job['kind'] == 'eval' else
                        walk_progress >= 2. and env.elapsed >= job['seconds']-env.dt/2))
    if job.get('fixed_command') is not None:
        complete_duration=env.elapsed>=job['seconds']-env.dt/2
        if job['category'] in ('turn_left','turn_right'):
            signed_yaw=sum(v[2] for v in cycle_actual)*np.sign(job['fixed_command'][2])
            minimum=env.config.get('physical_expert_control',{}).get('minimum_turn_rate_fraction',.5)*abs(job['fixed_command'][2])*env.elapsed
            max_translation=max((math.hypot(v[0],v[1]) for v in np.cumsum(np.asarray(cycle_actual)[:,:2],axis=0)),default=0.)
            success=bool(not fell and complete_duration and signed_yaw>=minimum and max_translation<=env.config.get('physical_expert_control',{}).get('maximum_turn_translation_m',.3))
        else:
            success=bool(not fell and (arrival is not None if job['kind']=='eval' else complete_duration and walk_progress>=2.))
    if job.get('stop_at_turn_deg') is not None:
        success=bool(not fell and turn_arrival is not None and max_excursion<=env.config.get('physical_expert_control',{}).get('maximum_turn_translation_m',.3))
    if job.get('category')=='turn_stand_walk':
        success=success and stage_counts['stand']>=env.config['turn_stand_walk']['stand_steps']
    if job.get('category')=='stand':
        success=bool(not fell and env.elapsed>=job['seconds']-env.dt/2 and stage_counts['stand']==len(xs)
            and max_excursion<=job.get('stand_max_translation_m',.3)
            and abs(math.degrees(accumulated_yaw))<=job.get('stand_max_yaw_deg',10.))
    row = dict(turn_arrival_time_s=turn_arrival,achieved_turn_deg=math.degrees(accumulated_yaw),max_excursion_m=max_excursion,**job, elapsed_s=env.elapsed, arrival_time_s=arrival, arrival_metrics=arrival_metrics,
        net_forward_m=float(displacement[0]),signed_lateral_m=float(displacement[1]),
        absolute_lateral_m=float(abs(displacement[1])),along_track_m=along,cross_track_m=cross,
        direction_error_deg=angle_error,speed_m_s=speed,max_heading_deg=heading_metric,
        success=success,fell=bool(fell),samples=len(xs),contract=CONTRACT,
        stage_counts=stage_counts,hold_translation_m=hold_translation,
        hold_net_speed_m_s=hold_net_speed,walk_progress_m=walk_progress)
    # Each episode is a separate immutable source; validation runs never enter BC.
    if job['kind']=='collect':
        from .cycle_tracking import trajectory_rewards
        from .turn_stand_walk import revalue
        if env.config['reward'].get('biped_tracking'):
            from .reward_dispatch import revalue_arrays
            tracking = revalue_arrays(dict(observations=xs,next_observations=next_xs,modes=modes),env.config)
        else:
            tracking = revalue(recorded_reward_inputs,modes,env.config)
        # Failed demonstrations are rejected downstream; preserve immediate fall flags here.
        rewards = tracking-np.asarray(terminals,dtype=float)*env.config['reward']['fall_penalty']
        x,y=np.asarray(xs,np.float32),np.asarray(ys,np.float32)
        if x.shape!=(len(y),observation_dim(env.config)) or y.shape!=(len(x),12) or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Invalid demonstration arrays')
        np.savez_compressed(folder/'demonstrations.npz', support_normal_loads=np.asarray(support_loads),next_support_normal_loads=np.asarray(next_support_loads),reward_inputs=np.asarray(recorded_reward_inputs),observations=x,actions=y,executed_actions=np.asarray(executed_actions,np.float32),
            next_observations=np.asarray(next_xs,np.float32),rewards=np.asarray(rewards,np.float32),
            terminals=np.asarray(terminals,np.float32),modes=np.asarray(modes,np.int8),
            critic_features=np.asarray(critic_states),next_critic_features=np.asarray(next_critic_states),
            cycle_actual=np.asarray(cycle_actual),cycle_target=np.asarray(cycle_target),cycle_dt=np.asarray(cycle_dt),cycle_ends=np.asarray(cycle_ends))
        row['dataset_sha256']=digest(folder/'demonstrations.npz')
    if contact_trace:
        write(folder/'contact_trace.json', contact_trace)
    write(folder/'result.json',row)
    return row


def main(snapshot):
    from controller import Supervisor
    from .environment import WebotsBaseWalk
    from .learning import load_policy
    torch.set_num_threads(1)
    snapshot=Path(snapshot)
    folder=snapshot.parent
    payload=read(snapshot)
    robot=Supervisor()
    try:
        progress=read(folder/'progress.json') if (folder/'progress.json').exists() else dict(cursor=0,rows=[])
        jobs=payload['options']['jobs']
        index=progress['cursor']
        if index>=len(jobs):
            write(folder/'status.json',dict(status='completed',episodes=len(jobs)))
            robot.simulationQuit(0)
            return
        write(folder/'status.json',dict(status='running',episode=index,total=len(jobs)))
        actor=None
        checkpoint=payload['options']['checkpoint']
        if checkpoint:
            if digest(checkpoint)!=payload['checkpoint_sha256']:
                raise ValueError('Checkpoint changed after launch')
            actor,_,_=load_policy(checkpoint)
        episode=folder/f'episode_{index:03d}'
        episode.mkdir(exist_ok=False)
        env=WebotsBaseWalk(robot,payload['config'])
        row=rollout(env,jobs[index],actor,episode)
        row['folder']=str(episode)
        progress['rows'].append(row)
        progress['cursor']+=1
        write(folder/'progress.json',progress)
        print(f"[expert] {index+1}/{len(jobs)} angle={row['angle']} success={row['success']} fall={row['fell']} heading={row['max_heading_deg']:.3f}",flush=True)
        if progress['cursor']==len(jobs):
            write(folder/'status.json',dict(status='completed',episodes=len(jobs)))
            robot.simulationQuit(0)
        else:
            robot.worldReload()
            robot.step(env.ms)
    except Exception as error:
        write(folder/'status.json',dict(status='failed',error=str(error),traceback=traceback.format_exc()))
        traceback.print_exc()
        robot.simulationQuit(1)

