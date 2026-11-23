"""Build the standalone five-chart Webots Turn report from finalized SAC rows."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import fmean


def group_four(points):
    grouped = []
    for start in range(0, len(points), 4):
        group = points[start:start + 4]
        point = [group[-1][0]]
        for column in zip(*(row[1:] for row in group)):
            values = [v for v in column if v is not None]
            point.append(fmean(values) if values else None)
        grouped.append(point)
    return grouped


def build_data(folder):
    snapshot = json.loads((folder/'effective_config.json').read_text(encoding='utf-8'))
    cfg = snapshot.get('config', snapshot)
    result = json.loads((folder/'result.json').read_text(encoding='utf-8'))
    with (folder/'training.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if result['status'] != 'completed' or not rows or len(rows) != result['env_steps']:
        raise ValueError('A completed run and its full training.csv are required')
    turn_only = cfg['training'].get('online_turn_only', True)
    bidirectional = turn_only and cfg['training'].get('online_fixed_turn_sign', 1) is None
    if not turn_only or bidirectional:
        if __package__:
            from .walk_turn_report import mixed_data, direction_summary
        else:
            from walk_turn_report import mixed_data, direction_summary
        data = mixed_data(rows, cfg)
        if bidirectional and 'yaw_ramp' not in cfg['training']:
            data['directions'] = direction_summary(rows)
        return data
    env = cfg['environment']
    legacy = 'phase_before' not in rows[0]
    initial = env.get('initial_gait_phase', 0.) % 1.
    if legacy and not (env['phase_runs_at_zero_command'] and env['yaw_source'] == 'supervisor_truth'
                       and cfg['reward']['biped_tracking']['turn_xy_penalty_scale'] == 0):
        raise ValueError('Historical log lacks measurements for this configuration')
    steps, cycles = [], []
    cycle_durations, cycle_yaws, cycle_gyros = [], [], []
    previous_episode = previous_position = None
    window_start = max(1, len(rows) - 999)
    net_tail = gyro_tail = time_tail = cycle_tail_yaw = cycle_tail_time = 0.
    tail_cycle_count = 0
    for step, row in enumerate(rows, 1):
        if int(row['step']) != step:
            raise ValueError(f'Missing or unordered training row at step {step}')
        episode = int(row['episode'])
        if episode != previous_episode:
            cycle_durations.clear()
            cycle_yaws.clear()
            cycle_gyros.clear()
            previous_position = None
            phase = initial
            complete_start = initial == 0.
            cycle_start = step
        x, y, yaw = (float(row[k]) for k in ('x', 'y', 'yaw'))
        gyro = float(row['body_gyro_z'])
        if legacy:
            dt = cfg['webots']['control_time_step_ms'] / 1000.
            delta = float(row['actual_wz']) * dt
            before, phase = phase, (phase + dt * env['gait_frequency_hz']) % 1.
            vx = vy = None
            if previous_position is not None:
                dx, dy = (x - previous_position[0])/dt, (y - previous_position[1])/dt
                vx = math.cos(yaw)*dx + math.sin(yaw)*dy
                vy = -math.sin(yaw)*dx + math.cos(yaw)*dy
            x_reward = 0.
        else:
            dt, delta, before, phase, vx, vy, x_reward = (
                float(row[k]) for k in ('cycle_dt', 'yaw_delta', 'phase_before', 'phase_after',
                                          'actual_vx', 'actual_vy', 'reward_tracking_x'))
        rate = delta / dt
        steps.append([step, x_reward, vx, float(row['reward_tracking_yaw']), rate,
                      float(row['reward']), vy, gyro])
        if step >= window_start:
            net_tail += delta
            gyro_tail += gyro * dt
            time_tail += dt
        cycle_durations.append(dt)
        cycle_yaws.append(delta)
        cycle_gyros.append(gyro * dt)
        if phase < before:
            if complete_start and row['fell'].lower() not in ('true', '1'):
                duration, net = math.fsum(cycle_durations), math.fsum(cycle_yaws)
                cycles.append([len(cycles) + 1, net/duration, math.fsum(cycle_gyros)/duration])
                if cycle_start >= window_start:
                    cycle_tail_yaw += net
                    cycle_tail_time += duration
                    tail_cycle_count += 1
            cycle_durations.clear()
            cycle_yaws.clear()
            cycle_gyros.clear()
            cycle_start, complete_start = step + 1, True
        previous_episode, previous_position = episode, (x, y)
    tail = (f"第 {window_start:,}–{len(rows):,} 步：平均净航向转速 {net_tail/time_tail:+.4f} rad/s，"
            f"原始 gyro Z 均值 {gyro_tail/time_tail:+.4f} rad/s，"
            f"指令 {float(rows[-1]['wz_cmd']):+.3f} rad/s。")
    if tail_cycle_count:
        tail += f"窗口内 {tail_cycle_count} 个完整周期的时间加权净转速为 {cycle_tail_yaw/cycle_tail_time:+.4f} rad/s。"
    method = ("历史日志按每回合固定相位时钟重建周期。前进、横向速度由平面位移和航向估算，"
              "缺少上一步位置的点留空。X Reward 按冻结配置为零。"
              if legacy else "前进与横向速度、X Reward、航向变化和周期相位均来自逐步训练记录。")
    return dict(experiment=f"SAC {cfg['experiment']} · 转向训练报告", samples=len(rows),
                cycle_count=len(cycles), steps=group_four(steps), cycles=group_four(cycles),
                has_speed_delta=False, tail=tail, method=method)


def write_report(folder):
    folder = Path(folder)
    view = build_data(folder)
    if 'blocks' in view:
        if __package__:
            from .walk_turn_report import render_report
        else:
            from walk_turn_report import render_report
        return render_report(folder, view)
    payload = json.dumps(view, ensure_ascii=False, allow_nan=False, separators=(',', ':')).replace('<', r'\u003c')
    template = Path(__file__).with_name('turn_view.html').read_text(encoding='utf-8')
    output = folder/'report'/'index.html'
    output.parent.mkdir(exist_ok=True)
    temp = output.with_suffix('.tmp')
    temp.write_text(template.replace('__DATA__', payload, 1), encoding='utf-8')
    temp.replace(output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    print(write_report(parser.parse_args().run))
