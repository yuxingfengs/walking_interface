"""Physical-only validation. Never exports usable BC/Critic training assets."""
import csv,json,math,traceback
from pathlib import Path
import numpy as np
from .core import DEFAULT_POSE, joint_targets, projected_gravity
from .sine_support import SineSupport
from .footstep_ik import forward,rotation


def write(path,data):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def torso_roll_pitch(raw_imu):
    """Use the same R2022b IMU mounting compensation as the environment."""
    gravity=projected_gravity(raw_imu)
    return np.array([math.atan2(-float(gravity[1]),-float(gravity[2])),
        math.atan2(float(gravity[0]),math.hypot(float(gravity[1]),float(gravity[2])))])


def run_episode(env,job,folder):
    env.expert_fixed_command=np.asarray(job['schedule'][0]['command'],np.float32)
    env.direction_target_deg=0
    obs=env.reset(command=env.expert_fixed_command)
    # Supervisor poses are used for kinematic validation/diagnostics only.
    # The teacher below receives encoders, IMU and aggregate foot forces.
    nodes=[env.node.getFromProtoDef(name) for name in ('LFsr','RFsr')]
    if any(n is None for n in nodes):raise RuntimeError('Cannot audit NAO LFsr/RFsr geometry')
    origin=np.asarray(env.node.getPosition());torso=np.asarray(env.node.getOrientation()).reshape(3,3)
    observed=[]
    for node in nodes:
        r=np.asarray(node.getOrientation()).reshape(3,3)
        pose=np.eye(4);pose[:3,3]=torso.T@(np.asarray(node.getPosition())+r@np.array([0,0,-.005])-origin)
        pose[:3,:3]=torso.T@r@rotation((0,0,1),-3.1416)
        observed.append(pose)
    expected=forward(env.q)
    poserr=float(np.max(np.linalg.norm(expected[:,:3,3]-np.asarray(observed)[:,:3,3],axis=1)))
    roterr=float(max(np.arccos(np.clip((np.trace(a[:3,:3].T@b[:3,:3])-1)/2,-1,1)) for a,b in zip(expected,observed)))
    write(folder/'kinematics_audit.json',dict(position_error_m=poserr,orientation_error_rad=roterr,
        expected=expected.tolist(),observed=np.asarray(observed).tolist()))
    if poserr>.003 or roterr>.03:raise RuntimeError('FK/model mismatch; see kinematics_audit.json')
    gait=SineSupport(env.config['physical_gait'],env.config['contact_expert'],env.foot_history,
                      env.q,env.lower,env.upper,env.cfg['action_scale'])
    write(folder/'calibration.json',gait.calibration)
    start=env.position.copy();start_yaw=env.truth_yaw
    rng=np.random.default_rng(job['seed']);noise=job['noise']
    data=[];trace=[];stop_reason='duration_completed';fell=False;yaw=0.;previous_yaw=env.truth_yaw
    last_steps=0;max_drift=0.;counts={};events=0;max_slip=0.;support_anchor=None;previous_support=None
    for i in range(round(job['seconds']/env.dt)):
        item=next(s for s in reversed(job['schedule']) if env.elapsed>=s['at_s'])
        command=np.asarray(item['command'],np.float32)
        env.expert_fixed_command=command
        obs=env._observation()
        raw_imu=np.asarray(env.imu.getRollPitchYaw()[:2],dtype=float)
        tilt=torso_roll_pitch(raw_imu)
        truth_rotation=np.asarray(env.node.getOrientation()).reshape(3,3)
        truth_tilt=[math.atan2(truth_rotation[2,1],truth_rotation[2,2]),
                    math.atan2(-truth_rotation[2,0],math.hypot(truth_rotation[2,1],truth_rotation[2,2]))]
        try:
            target=gait.update(command,env.q,env.foot_forces,
                tilt,env.gyro_value,env.dt)
        except (ValueError,RuntimeError) as error:
            stop_reason=str(error);trace.append(dict(time_s=env.elapsed,
                imu_raw_roll_pitch=raw_imu.tolist(),torso_roll_pitch=tilt.tolist(),
                diagnostic_truth_roll_pitch=truth_tilt,**gait.diagnostics()));break
        action=(target-DEFAULT_POSE)/env.cfg['action_scale']
        clean=action.copy()
        if noise:
            perturb=rng.normal(0.,noise,12)
            perturb[6]=perturb[0]  # Shared HipYawPitch cannot receive independent noise.
            action+=perturb
        env.phase=gait.phase
        nxt,reward,fell,truncated,info=env.step(action,randomize=False)
        delta=math.atan2(math.sin(env.truth_yaw-previous_yaw),math.cos(env.truth_yaw-previous_yaw))
        yaw+=delta;previous_yaw=env.truth_yaw
        drift=float(np.linalg.norm(env.position[:2]-start[:2]));max_drift=max(max_drift,drift)
        foot_position=np.asarray(nodes[gait.support].getPosition())
        if previous_support!=gait.support:
            support_anchor=foot_position.copy();previous_support=gait.support
        slip=float(np.linalg.norm(foot_position[:2]-support_anchor[:2]));max_slip=max(max_slip,slip)
        counts[item['label']]=counts.get(item['label'],0)+1
        data.append((obs,clean,info['executed_action'],nxt,info['reward_inputs'],fell))
        trace.append(dict(time_s=env.elapsed,segment=item['label'],command=command.tolist(),
            position=env.position.tolist(),yaw_rad=env.truth_yaw,velocity=env.velocity.tolist(),
            q=env.q.tolist(),imu_raw_roll_pitch=raw_imu.tolist(),torso_roll_pitch=tilt.tolist(),
            diagnostic_truth_roll_pitch=truth_tilt,
            support_slip_m=slip,fell=bool(fell),**gait.diagnostics()))
        if len(gait.events)>events:
            events=len(gait.events)
        last_steps=gait.steps
        if fell:stop_reason='fall';break
        if job.get('turn_goal_deg') is not None and yaw*np.sign(job['turn_goal_deg'])>=math.radians(abs(job['turn_goal_deg'])-3.):
            stop_reason='turn_goal_reached';break
    delta=env.position-start
    c,s=math.cos(start_yaw),math.sin(start_yaw)
    net_forward=float(c*delta[0]+s*delta[1]);lateral=float(-s*delta[0]+c*delta[1])
    complete=(stop_reason=='duration_completed' and env.elapsed>=job['seconds']-env.dt/2) or stop_reason=='turn_goal_reached'
    duration=max(env.elapsed,env.dt)
    result=dict(task=job['name'],noise=noise,completed=complete,stop_reason=stop_reason,
        fell=bool(fell),elapsed_s=env.elapsed,steps=last_steps,stage_counts=counts,
        net_forward_m=net_forward,signed_lateral_m=lateral,absolute_lateral_m=abs(lateral),
        max_excursion_m=max_drift,achieved_turn_deg=math.degrees(yaw),mean_wz=yaw/duration,
        mean_vx=net_forward/duration,max_support_slip_m=max_slip,arrival_time_s=None,success=False,
        training_eligible=False,limitations='Physical probe only; no global pose is supplied to teacher. No geometric goal-time comparison.')
    if job['name']=='step':
        quality=max_drift<=.15 and abs(math.degrees(yaw))<=10
    elif job['name'].startswith('turn'):
        goal=job['schedule'][0]['command'][2]
        quality=(stop_reason=='turn_goal_reached' if job.get('turn_goal_deg') is not None
                 else yaw*goal>0 and .4*abs(goal)<=abs(yaw/duration)<=1.6*abs(goal)) and max_drift<=.3
    elif job['name']=='walk':
        goal=job['schedule'][0]['command'][0]
        quality=net_forward>=.4*goal*duration and abs(lateral)<=.15 and abs(math.degrees(yaw))<=10
    else:
        # Sequence acceptance requires actual stepping in every timed segment.
        quality=all(any(t['segment']==seg['label'] and t['steps']>
            min(r['steps'] for r in trace if r.get('segment')==seg['label']) for t in trace if 'segment' in t)
            for seg in job['schedule'] if any(r.get('segment')==seg['label'] for r in trace)) and len(counts)==len(job['schedule'])
    result['success']=bool(complete and not fell and last_steps>=4 and quality)
    write(folder/'trace.json',trace);write(folder/'result.json',result)
    if data:
        np.savez_compressed(folder/'diagnostic_transitions.npz',
            observations=np.asarray([r[0] for r in data]),clean_actions=np.asarray([r[1] for r in data]),
            executed_actions=np.asarray([r[2] for r in data]),next_observations=np.asarray([r[3] for r in data]),
            reward_inputs=np.asarray([r[4] for r in data]),fell=np.asarray([r[5] for r in data]))
    return result


def main(snapshot):
    from controller import Supervisor
    from .environment import WebotsBaseWalk
    snapshot=Path(snapshot);folder=snapshot.parent
    payload=json.loads(snapshot.read_text(encoding='utf-8'));robot=Supervisor()
    try:
        path=folder/'progress.json'
        progress=json.loads(path.read_text()) if path.exists() else dict(cursor=0,rows=[])
        index=progress['cursor'];jobs=payload['options']['jobs']
        write(folder/'status.json',dict(status='running',episode=index))
        episode=folder/f'episode_{index:03d}';episode.mkdir()
        env=WebotsBaseWalk(robot,payload['config'])
        row=run_episode(env,jobs[index],episode);row['folder']=str(episode)
        progress['rows'].append(row);progress['cursor']+=1;write(path,progress)
        print('[physical]',row['task'],row['success'],row['stop_reason'],flush=True)
        if progress['cursor']==len(jobs):
            write(folder/'status.json',dict(status='completed',episodes=len(jobs),
                successes=sum(r['success'] for r in progress['rows'])))
            robot.simulationQuit(0)
        else:
            robot.worldReload();robot.step(env.ms)
    except Exception as error:
        write(folder/'status.json',dict(status='failed',error=str(error),traceback=traceback.format_exc()))
        traceback.print_exc();robot.simulationQuit(1)
