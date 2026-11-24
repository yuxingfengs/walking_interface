"""Independent walk trajectories sharing one PhysX articulation view and one SAC learner."""
import csv
import numpy as np
from basewalk.phase_fall import PhaseFallBuffer
from basewalk.runner import LOG_FIELDS, training_row
from basewalk.sac import SAC
from batched_environment import PhysicsTrace, ResetDiagnostics, WalkBatch


def check_batch(env, cfg, steps, folder):
    agent = SAC(cfg)
    samples = falls = 0
    max_hip = 0.
    paths = []
    for trial in range(cfg['isaac']['check_resets']):
        batch = WalkBatch(env, [(1 + trial) * cfg['training']['walk_commands']['hold_steps']] * env.count, folder)
        counts = [0] * env.count
        while min(counts) < steps:
            active = [i for i in sorted(batch.obs) if counts[i] < steps]
            ends = []
            for i, prev, nxt, reward, fell, trunc, info, phase in batch.tick(agent, active, False):
                lane = batch.lanes[i]
                np.testing.assert_array_equal(prev[5:8], info['command'])
                np.testing.assert_array_equal(nxt[5:8], lane.command)
                hip = abs(float(lane.q[0] - lane.q[6]))
                if hip > cfg['isaac']['hip_tolerance_rad'] or not np.isfinite(reward):
                    raise ValueError(f'Robot {i} failed physical feedback check')
                max_hip = max(max_hip, hip)
                counts[i] += 1
                samples += 1
                falls += int(fell)
                if fell or trunc:
                    ends.append(i)
            if ends and cfg['isaac'].get('experiment_variant') == 'walk_lateral_random_vx':
                break
            batch.reset(ends)
        paths.append(counts)
    return dict(status='simulation_check_passed', num_envs=env.count, steps=samples,
                steps_by_robot=paths, falls=falls, max_measured_hip_difference=max_hip,
                optimizer_updates=0, training_started=False)


class OrderedRows:
    def __init__(self, writer, next_step):
        self.writer = writer
        self.next_step = next_step
        self.ready = {}

    def writerow(self, row):
        self.ready[row['step']] = row
        while self.next_step in self.ready:
            self.writer.writerow(self.ready.pop(self.next_step))
            self.next_step += 1


def run_batch(env, cfg, folder, *, resume=None):
    agent = resume['agent'] if resume else SAC(cfg)
    state = resume['state'] if resume else {}
    step = state.get('next_step', 1) - 1
    falls = state.get('falls', 0)
    clocks = [0] * env.count
    next_episode = 1
    if resume:
        with (folder/'training.csv').open(newline='') as stream:
            for row in csv.DictReader(stream):
                clocks[int(row['env_id'])] += 1
                next_episode = max(next_episode, int(row['episode']) + 1)
    batch = WalkBatch(env, clocks, folder, mode='train')
    episodes = list(range(next_episode, next_episode + env.count))
    next_episode += env.count
    train = cfg['training']
    total = train['total_env_steps']
    pending = [PhaseFallBuffer(cfg['reward']['fall_penalty'], cfg) for _ in clocks]
    recovery_step = step
    unpublished = []
    with (folder/'training.csv').open('a' if resume else 'x', newline='') as stream:
        writer = csv.DictWriter(stream, LOG_FIELDS + ['env_id', 'env_step'])
        if not resume:
            writer.writeheader()
        rows = OrderedRows(writer, step + 1)
        while step < total:
            agent.schedule(step + 1)
            active = sorted(batch.obs)[:total-step]
            ends = []
            for i, prev, nxt, reward, fell, trunc, info, phase in batch.tick(agent, active, True):
                lane = batch.lanes[i]
                step += 1
                row = training_row(step, episodes[i], reward, fell, trunc, info, lane, agent, phase)
                row.update(env_id=i, env_step=lane.persistent_fixed_cycle_stages.step)
                pending[i].append(prev, info['executed_action'], reward, nxt, fell, {}, row, None)
                if step >= agent.learning_starts and agent.replay.ready(train['batch_size']):
                    metrics = agent.update()
                    row.update({k: metrics[k] for k in ('actor_loss','critic_loss','entropy_coefficient','updates')})
                if fell or trunc or lane.phase < phase:
                    pending[i].flush(agent.replay, rows, fell=fell, complete_cycle=lane.phase < phase)
                falls += int(fell)
                if fell or trunc:
                    ends.append(i)
                    episodes[i] = next_episode
                    next_episode += 1
                if step % train['checkpoint_every'] == 0 or step == total:
                    path = folder/'checkpoints'/f'step_{step:08d}.pth'
                    staged = path.with_suffix('.pending')
                    agent.save(staged, step)
                    unpublished.append((step, staged, path))
                if step - recovery_step >= 500 and (fell or trunc):
                    from basewalk.omni_resume import recovery_state
                    path = folder/'checkpoints'/'recovery_latest.pth'
                    staged = path.with_name(f'recovery_{step}.pending')
                    agent.save(staged, step, recovery=recovery_state(agent))
                    unpublished.append((step, staged, path))
                    recovery_step = step
            for i in batch.interrupted:
                pending[i].flush(agent.replay, rows)
                episodes[i] = next_episode
                next_episode += 1
            if step < total:
                batch.reset(ends)
            stream.flush()
            while unpublished and unpublished[0][0] < rows.next_step:
                _, staged, path = unpublished.pop(0)
                staged.replace(path)
        for buffer in pending:
            buffer.flush(agent.replay, rows)
        assert rows.next_step == total + 1
        stream.flush()
        for _, staged, path in unpublished:
            staged.replace(path)
    from basewalk.omni_resume import save_recovery
    save_recovery(agent, folder, total)
    return dict(status='completed', env_steps=total, updates=agent.updates, falls=falls, num_envs=env.count)
