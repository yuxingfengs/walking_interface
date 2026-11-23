"""Online SAC control loop for the fixed Turn task."""
import csv
import json
import math
from pathlib import Path
import traceback

from .fixed_cycle import online_stages
from .phase_fall import PhaseFallBuffer
from .sac import SAC
from .world_reload import load_continuation, save_continuation, reload_world


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


LOG_FIELDS = ('step episode reward reward_fall reward_tracking_yaw wz_cmd body_gyro_z '
              'actual_wz x y yaw fell truncated actor_loss critic_loss entropy_coefficient updates '
              'reward_tracking_x actual_vx actual_vy yaw_delta cycle_dt phase_before phase_after '
              'reward_tracking_y reward_yaw_rate reward_orientation reward_stability '
              'tilt_rad body_gyro_x body_gyro_y vx_cmd vy_cmd motion_phase').split()


def training_row(step, episode, value, fell, truncated, info, env, agent, phase):
    terms = info['reward_terms']
    return dict(step=step, episode=episode, reward=value, reward_fall=terms['fall'],
        reward_tracking_yaw=terms['tracking_yaw'], wz_cmd=info['command'][2],
        vx_cmd=info['command'][0], vy_cmd=info['command'][1], motion_phase=info['motion_phase'],
        body_gyro_z=float(env.actor_gyro_z), actual_wz=info['actual_velocity'][2],
        x=info['position'][0], y=info['position'][1], yaw=info['yaw'],
        fell=fell, truncated=truncated, updates=agent.updates,
        reward_tracking_x=terms['tracking_x'], reward_tracking_y=terms['tracking_y'],
        reward_orientation=terms['orientation'], reward_stability=terms['stability'],
        tilt_rad=math.atan2(math.hypot(*env.gravity[:2]), -env.gravity[2]),
        body_gyro_x=float(env.gyro_value[0]), body_gyro_y=float(env.gyro_value[1]),
        reward_yaw_rate=info['reward_velocity'][2], actual_vx=info['actual_velocity'][0],
        actual_vy=info['actual_velocity'][1], yaw_delta=info['cycle_actual'][2],
        cycle_dt=info['cycle_dt'], phase_before=phase, phase_after=env.phase)


def run(env, cfg, folder, *, resume=None):
    t = cfg['training']
    reload_on_end = cfg['environment']['reset_mode'] == 'world_reload'
    if reload_on_end:
        resume = load_continuation(folder, cfg)
    agent = resume['agent'] if resume else SAC(cfg)
    previous = resume['state'] if resume else {}
    stages = online_stages(cfg, previous.get('stages'))
    env.persistent_fixed_cycle_stages = stages
    obs = env.reset(command=stages.command(0, 0))
    falls = previous.get('falls', 0)
    episode = previous.get('episode', 1)
    total = t['total_env_steps']
    pending = PhaseFallBuffer(cfg['reward']['fall_penalty'], cfg)
    recovery_step = previous.get('next_step', 1) - 1
    fields = LOG_FIELDS
    with (folder/'training.csv').open('a' if resume else 'x', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        if not resume:
            writer.writeheader()
        for step in range(previous.get('next_step', 1), total + 1):
            agent.schedule(step)
            phase = env.phase
            nxt, value, fell, truncated, info = env.step(agent.act(obs, explore=True))
            row = training_row(step, episode, value, fell, truncated, info, env, agent, phase)
            transition = (obs, info['executed_action'], value, nxt, fell)
            if t['phase_fall_backfill']:
                pending.append(*transition, {}, row, None)
            else:
                agent.replay.add(*transition)
            if step >= agent.learning_starts and agent.replay.ready(t['batch_size']):
                metrics = agent.update()
                row.update({key: metrics[key] for key in
                            ('actor_loss','critic_loss','entropy_coefficient','updates')})
            if t['phase_fall_backfill']:
                if fell or truncated or env.phase < phase or step == total:
                    pending.flush(agent.replay, writer, fell=fell, complete_cycle=env.phase < phase)
                    stream.flush()
            else:
                writer.writerow(row)
            falls += int(fell)
            if step < total and step - recovery_step >= 500 and (fell or truncated):
                from .omni_resume import save_recovery
                stream.flush()
                save_recovery(agent, folder, step, stages.state())
                recovery_step = step
            if step % t['checkpoint_every'] == 0 or step == total:
                agent.save(folder/'checkpoints'/f'step_{step:08d}.pth', step)
            obs = nxt
            if (fell or truncated) and step < total:
                episode += 1
                if reload_on_end:
                    save_continuation(folder, cfg, agent, dict(next_step=step+1,
                        falls=falls, episode=episode, stages=stages.state()))
                    return None
                obs = env.reset(command=stages.command(0, 0))
    from .omni_resume import save_recovery
    save_recovery(agent, folder, total, stages.state())
    return dict(status='completed', env_steps=total, updates=agent.updates, falls=falls)

