"""Batched PhysX stepping, per-lane reset retries and diagnostic traces."""
import copy
import json
import math
import time
import numpy as np
import torch
from isaaclab.utils.math import quat_apply_inverse
from basewalk.core import joint_targets
from basewalk.fixed_cycle import online_stages
from basewalk.support_observation import SupportObservation
from environment import host


class PhysicsTrace:
    """Keep two control ticks on-device. Decode just the failed lane when reporting."""
    def __init__(self, env):
        self.env = env
        self.tick = 0
        self.frames = []
        self.previous = []
        self.reset_context = {}
        self.awaiting_reset_sample = set()
        self.after_reset = {}

    def begin(self):
        self.tick += 1
        self.previous, self.frames = self.frames, []

    def capture(self, substep, drive_velocity):
        env = self.env
        data = env.robot.data
        fields = dict(root_xyz_w=data.root_link_pos_w, root_quat_wxyz=data.root_link_quat_w,
                      gravity_b=data.projected_gravity_b, angular_velocity_b=data.root_ang_vel_b,
                      joint_rad=data.joint_pos, joint_rad_s=data.joint_vel,
                      target_rad=env.targets)
        if drive_velocity is not None:
            fields['drive_target_rad_s'] = drive_velocity
        if substep:
            for name, foot in zip(env.settings['feet'], env.feet):
                fields[name + '_force_xyz_w_N'] = foot.data.net_forces_w[:, 0]
        layout = [(key, value.shape[1]) for key, value in fields.items()]
        self.frames.append((self.tick, substep, layout, torch.cat(list(fields.values()), dim=1)))
        if substep == 0 and self.awaiting_reset_sample:
            for i in self.awaiting_reset_sample:
                self.after_reset[i] = self.frames[-1]
            self.awaiting_reset_sample.clear()

    def mark_reset(self, indices):
        for i in indices:
            self.reset_context[i] = (self.tick, self.frames[-1] if self.frames else None)
        self.awaiting_reset_sample.update(indices)

    @staticmethod
    def decode(frame, i):
        tick, substep, layout, snapshot = frame
        values = iter(host(snapshot[i]).tolist())
        fields = {}
        bad = {}
        for key, width in layout:
            vector = [next(values) for _ in range(width)]
            invalid = [j for j, v in enumerate(vector) if not math.isfinite(v)]
            if invalid:
                bad[key] = invalid
            fields[key] = [v if math.isfinite(v) else str(v) for v in vector]
        return dict(tick=tick, substep=substep, values=fields, nonfinite=bad)

    def report(self, i):
        frames = [self.decode(f, i) for f in self.previous + self.frames]
        reset_tick, before = self.reset_context[i]
        first = next((dict(tick=f['tick'], substep=f['substep'], fields=f['nonfinite'])
                      for f in frames if f['tick'] > reset_tick and f['nonfinite']), None)
        return dict(physics_dt_s=self.env.settings['physics_dt'], tick=self.tick,
                    last_reset_after_tick=reset_tick, first_nonfinite_in_window=first,
                    joint_names=list(self.env.robot.joint_names),
                    drive_joint_names=[self.env.robot.joint_names[j] for j in getattr(self.env, 'servo_ids', [])],
                    before_reset=self.decode(before, i) if before else None,
                    after_reset_before_physics=self.decode(self.after_reset[i], i), frames=frames)


class ResetDiagnostics:
    """Preserve the onset of divergence and comparable successful settles."""
    def __init__(self, trace, folder, mode):
        self.trace = trace
        self.folder = folder/'reset_diagnostics'
        self.mode = mode
        self.samples = {}
        self.controls = set()

    def start(self, indices, retrying):
        for i in indices:
            context = 'retry' if i in retrying else ('initial' if not self.trace.frames else 'episode_end')
            self.samples[i] = dict(context=context, batch_ids=list(indices), blocks=[],
                                   first_alert=None, elapsed_ms=[])

    def collect(self, elapsed_ms):
        if not self.samples:
            return
        # Post-physics frames have one consistent layout, including both contact sensors.
        frames = self.trace.frames[1:]
        layout = frames[0][2]
        ids = list(self.samples)
        values = host(torch.stack([f[3] for f in frames])[:, ids])
        offset = 0
        fields = {}
        for name, width in layout:
            fields[name] = slice(offset, offset + width)
            offset += width
        for column, i in enumerate(ids):
            sample = self.samples[i]
            block = values[:, column].copy()
            sample['blocks'].append(block)
            sample['elapsed_ms'].append(elapsed_ms)
            sample['layout'] = layout
            if sample['first_alert'] is not None:
                continue
            speed = abs(block[:, fields['joint_rad_s']])
            height = block[:, fields['root_xyz_w']][:, 2]
            if 'initial_z' not in sample:
                sample['initial_z'] = float(self.trace.after_reset[i][3][i, 2].item())
            z0 = sample['initial_z']
            alert = (speed > 50.).any(axis=1) | (abs(height - z0) > .25) | ~np.isfinite(block).all(axis=1)
            hits = np.flatnonzero(alert)
            if hits.size:
                j = int(hits[0])
                sample['first_alert'] = dict(tick=self.trace.tick, substep=j + 1,
                    reset_substep=(len(sample['blocks']) - 1) * len(block) + j + 1,
                    fast_joints=[self.trace.env.robot.joint_names[k] for k in np.flatnonzero(speed[j] > 50.)],
                    height_deviation=bool(abs(height[j]-z0) > .25),
                    nonfinite=not bool(np.isfinite(block[j]).all()))
                self.save(i, 'alert')

    def save(self, i, outcome):
        sample = self.samples[i]
        reset_tick, before = self.trace.reset_context[i]
        meta = dict(mode=self.mode, env_id=i, outcome=outcome, seed=self.trace.env.cfg['seed'],
                    reset_after_tick=reset_tick, physics_dt_s=self.trace.env.settings['physics_dt'],
                    context=sample['context'], simultaneous_reset_ids=sample['batch_ids'],
                    first_alert=sample['first_alert'], joint_names=list(self.trace.env.robot.joint_names),
                    drive_joint_names=[self.trace.env.robot.joint_names[k] for k in getattr(self.trace.env,'servo_ids',[])],
                    columns=sample['layout'], before_reset=self.trace.decode(before, i) if before else None,
                    after_reset_before_physics=self.trace.decode(self.trace.after_reset[i], i))
        self.folder.mkdir(exist_ok=True)
        path = self.folder/f'{self.mode}_{time.time_ns()}_lane{i}_{outcome}.npz'
        np.savez_compressed(path, metadata=json.dumps(meta, allow_nan=False),
                            substeps=np.concatenate(sample['blocks']), physics_batch_wall_ms=sample['elapsed_ms'])

    def finish(self, i, outcome):
        if i not in self.samples:
            return
        sample = self.samples[i]
        key = (i, sample['context'])
        if outcome != 'success' or sample['first_alert'] or key not in self.controls:
            self.save(i, outcome)
        if outcome == 'success' and not sample['first_alert']:
            self.controls.add(key)
        del self.samples[i]


class WalkBatch:
    def __init__(self, env, clocks, folder, mode='check'):
        self.env = env
        self.origins = (env.scene.env_origins if env.count > 1 else
                        env.robot.data.default_root_state.new_zeros((1, 3)))
        self.lanes = [copy.copy(env) for _ in clocks]
        self.history = {}
        self.obs = {}
        self.folder = folder
        self.retry_at = {}
        self.recovering = set()
        self.interrupted = []
        self.trace = PhysicsTrace(env)
        self.diagnostics = ResetDiagnostics(self.trace, folder, mode)
        for i, (lane, clock) in enumerate(zip(self.lanes, clocks)):
            lane.yaw_filter = copy.deepcopy(env.yaw_filter)
            lane.persistent_fixed_cycle_stages = online_stages(env.cfg, {'step': clock, 'robot_index': i})
        self.reset(list(range(env.count)))

    def isolate(self, i, reason, **details):
        settling = i in self.history
        self.diagnostics.finish(i, 'fault')
        lane = self.lanes[i]
        event = dict(time_unix=time.time(), env_id=i, reason=reason,
                     stage='settling' if settling else 'walking',
                     settle_steps_done=len(self.history.get(i, [])),
                     env_step=lane.persistent_fixed_cycle_stages.step,
                     command=np.asarray(lane.command).tolist(), gait_phase=float(lane.phase),
                     physics_trace=self.trace.report(i), **details)
        with (self.folder/'lane_faults.jsonl').open('a', encoding='utf-8') as log:
            log.write(json.dumps(event, allow_nan=False) + '\n')
        print(f'Robot {i}: {reason}; retrying reset in 120 seconds', flush=True)
        self.history.pop(i, None)
        self.obs.pop(i, None)
        self.env.targets[i] = self.env.robot.data.default_joint_pos[i]
        self.retry_at[i] = time.monotonic() + 120.
        self.recovering.discard(i)
        if not settling:
            self.interrupted.append(i)

    def reset(self, indices):
        if not indices:
            return
        env = self.env
        robot = env.robot
        self.trace.mark_reset(indices)
        self.diagnostics.start(indices, self.retry_at)
        ids = torch.tensor(indices, device=env.sim.device)
        root = robot.data.default_root_state[ids].clone()
        root[:, :3] += self.origins[ids]
        robot.write_root_pose_to_sim(root[:, :7], env_ids=ids)
        robot.write_root_velocity_to_sim(torch.zeros_like(root[:, 7:]), env_ids=ids)
        q = robot.data.default_joint_pos[ids]
        env.targets[ids] = q
        robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
        robot.reset(ids)
        for foot in env.feet:
            foot.reset(ids)
        for i in indices:
            lane = self.lanes[i]
            lane.command = lane.persistent_fixed_cycle_stages.command(0, 0)
            lane.phase = float(env.env.get('initial_gait_phase', 0.))
            lane.elapsed = 0.
            lane.velocity = np.zeros(3, np.float32)
            lane.last = np.zeros(12, np.float32)
            lane.qvel = np.zeros(12, np.float32)
            lane.yaw_filter.reset()
            self.history[i] = []
            self.obs.pop(i, None)

    def tick(self, agent, active, explore):
        env = self.env
        self.interrupted.clear()
        if len(self.retry_at) == env.count:
            time.sleep(max(0., min(self.retry_at.values()) - time.monotonic()))
        due = [i for i, deadline in self.retry_at.items() if deadline <= time.monotonic()]
        self.reset(due)
        for i in due:
            del self.retry_at[i]
            self.recovering.add(i)
        data = env.robot.data
        old_pos = data.root_link_pos_w.clone()
        old_local = {i: self.lanes[i].pos.copy() for i in active}
        old_q = {i: self.lanes[i].q for i in active}
        old_yaw = {i: self.lanes[i].yaw for i in active}
        if active:
            actions = agent.act_batch(np.stack([self.obs[i] for i in active]), explore)
            if not np.isfinite(actions).all():
                raise FloatingPointError('SAC actor returned nonfinite actions before PhysX step')
            pairs = [joint_targets(a, env.env['action_scale'], env.lower, env.upper) for a in actions]
            joint_pos, executed = zip(*pairs)
            ids = torch.tensor(active, device=env.sim.device)
            env.targets[ids[:, None], env.ids] = env.targets.new_tensor(np.stack(joint_pos))
        self.trace.begin()
        physics_start = time.perf_counter()
        forces = host(env._advance_physics(self.trace.capture))
        self.diagnostics.collect((time.perf_counter() - physics_start) * 1000.)
        local = data.root_link_pos_w - self.origins
        quat = data.root_link_quat_w
        linear = quat_apply_inverse(quat, (data.root_link_pos_w - old_pos) / env.dt)
        states = host(torch.cat([local, quat, data.projected_gravity_b, data.root_ang_vel_b,
                                 data.joint_pos[:, env.ids], linear], dim=1))
        valid = np.isfinite(states).all(axis=1) & np.isfinite(forces).reshape(env.count, -1).all(axis=1)
        for i, lane in enumerate(self.lanes):
            if i in self.retry_at:
                continue
            if not valid[i]:
                previous = host(old_pos[i])
                self.isolate(i, 'nonfinite PhysX feedback',
                             state_columns=np.flatnonzero(~np.isfinite(states[i])).tolist(),
                             force_indices=np.argwhere(~np.isfinite(forces[i])).tolist(),
                             previous_position=[float(v) if np.isfinite(v) else None for v in previous],
                             finite_targets=bool(torch.isfinite(env.targets[i]).all()))
                continue
            lane.pos, rotation, lane.gravity, lane.gyro, lane.q, _ = np.split(states[i], [3, 7, 10, 13, 25])
            w, x, y, z = rotation
            lane.yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
            lane.gyro_value = lane.gyro[:2]
            lane.actor_gyro_z = float(lane.gyro[2])
        for i in list(self.history):
            history = self.history[i]
            history.append(forces[i])
            if len(history) == env.env['settle_steps']:
                lane = self.lanes[i]
                if lane._fell():
                    self.isolate(i, 'fell during settle', height=float(lane.pos[2]),
                                 gravity_z=float(lane.gravity[2]))
                    continue
                lane.support = SupportObservation(env.cfg['support_observation'], history)
                self.obs[i] = lane._obs()
                self.diagnostics.finish(i, 'success')
                del self.history[i]
                if i in self.recovering:
                    self.recovering.remove(i)
                    with (self.folder/'lane_faults.jsonl').open('a', encoding='utf-8') as log:
                        log.write(json.dumps(dict(time_unix=time.time(), env_id=i, event='recovered')) + '\n')
        transitions = []
        for i, action in zip(active, executed if active else ()):
            if i in self.retry_at:
                continue
            lane = self.lanes[i]
            lane.support.advance(forces[i])
            delta = lane.yaw - old_yaw[i]
            wz = math.atan2(math.sin(delta), math.cos(delta)) / env.dt
            phases = lane.phase
            prev = self.obs[i]
            nxt, reward, fell, trunc, info = lane.finish_step(action, old_local[i], old_q[i], states[i,-3:], wz)
            self.obs[i] = nxt
            transitions.append((i, prev, nxt, reward, fell, trunc, info, phases))
        return transitions
