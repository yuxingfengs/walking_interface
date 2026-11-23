"""Generate one immutable report; never import a trainer or start a simulation."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
import webbrowser
import difflib
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from project import resolve_entry_config

ROOT = Path(__file__).resolve().parent / 'data/runtime'
TERMS = ('x', 'y', 'yaw', 'tilt', 'gyro', 'fall')
LEGACY_TERMS = ('reward_tracking_x', 'reward_tracking_y', 'reward_tracking_yaw',
                'reward_orientation', 'reward_stability', 'reward_fall')
PHYSICAL = ('vx', 'vy', 'wz', 'roll', 'pitch', 'gyro_x', 'gyro_y', 'height',
            'forward', 'lateral', 'gx', 'gy', 'gz')
SENSORS = ('roll', 'pitch', 'gx', 'gy', 'gz', 'gyro_x', 'gyro_y', 'gyro_z', 'vx', 'vy', 'wz')
JOINT_NAMES = tuple(side+name for side in ('L','R') for name in
                    ('HipYawPitch','HipRoll','HipPitch','KneePitch','AnklePitch','AnkleRoll'))
JOINT_FIELDS = tuple(prefix+name for prefix in ('position_','velocity_') for name in JOINT_NAMES)
PHYSICAL += JOINT_FIELDS + ('left_contact_count','right_contact_count','left_contact','right_contact',
                           'left_foot_height','right_foot_height','support_load_difference','left_normal_load','right_normal_load')
SENSORS += JOINT_FIELDS + ('support_load_difference','left_normal_load','right_normal_load')
EVAL_METRICS = ('arrival_time_s', 'lateral_signed_m', 'lateral_abs_m', 'forward_m', 'success_rate', 'fall_rate')
REWARD_INPUT_FIELDS = ('vx','vy','wz','gx','gy','gz','gyro_x','gyro_y')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def numeric(value):
    if value in (None, ''):
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Nonfinite numeric value')
    return result


def flag(value):
    return str(value).lower() in ('true', '1')


def avg(values):
    return sum(values)/len(values) if values else None


def rms(values):
    return math.sqrt(avg([x*x for x in values])) if values else None


def identifier(value):
    value = str(value)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', value):
        raise ValueError('Experiment ID must contain only letters, digits, _ or -')
    return value


def experiment(root_config, selected):
    selected = identifier(selected)
    if root_config.get('experiment_id') and 'training_config' in root_config:
        if selected != root_config['experiment_id']:
            raise ValueError('Report experiment differs from the local config')
        path = Path(root_config['_config_path']).parent
        return path, root_config
    path = (ROOT/root_config['experiments_root']/selected).resolve()
    conf = read_json(path/'config.json')
    if conf['experiment_id'] != selected:
        raise ValueError('Experiment ID/config mismatch')
    return path, conf


def choose_run(path, conf, override=None):
    if override:
        run = Path(override).resolve()
    elif conf.get('current_run'):
        run = path/conf['runs_root']/identifier(conf['current_run'])
    else:
        runs = [p for p in (path/conf['runs_root']).glob('*') if p.is_dir() and
                any((p/name).exists() for name in ('report_trace.jsonl', 'report_trace.jsonl.gz', 'training.csv'))]
        if not runs:
            raise ValueError(f'实验 {conf["experiment_id"]} 尚无日志。请指定 --run 日志目录；本工具不会启动训练。')
        run = max(runs, key=lambda p: p.name)
    if not (run/'effective_config.json').is_file():
        raise ValueError(f'Missing saved run configuration: {run}/effective_config.json')
    return run.resolve()


def latest_completed_training(root_config):
    completed = []
    descriptors = ([Path(root_config['_config_path'])] if root_config.get('experiment_id')
                   and 'training_config' in root_config else
                   Path(root_config['experiments_root']).glob('*/config.json'))
    for descriptor in descriptors:
        experiment_id = root_config.get('experiment_id') or descriptor.parent.name
        folder, config = experiment(root_config, experiment_id)
        for run in (folder / config['runs_root']).glob('train_*'):
            result = run / 'result.json'
            if not result.is_file() or not (run / 'training.csv').is_file():
                continue
            if read_json(result).get('status') == 'completed':
                completed.append((run.name, experiment_id, run))
    if not completed:
        raise ValueError('本项目还没有完成的在线训练')
    _, experiment_id, run = max(completed)
    return experiment_id, run


def open_report(path):
    if os.name == 'nt':
        os.startfile(str(path.resolve()))
    elif not webbrowser.open(path.resolve().as_uri()):
        raise OSError(f'无法打开浏览器。请直接打开 {path}')


def saved_latest_view(root_config, selected, run, view_hash):
    folder, descriptor = experiment(root_config, selected)
    reports = folder / descriptor.get('reports_root', 'reports')
    for metadata_path in sorted(reports.glob('*/data.json'), reverse=True):
        page = metadata_path.with_name('index.html')
        if not page.is_file() or not metadata_path.with_name('REPORT.md').is_file():
            continue
        try:
            metadata = read_json(metadata_path)
        except (OSError, ValueError):
            continue
        if (metadata.get('run') == str(run) and metadata.get('view_sha256') == view_hash
                and metadata.get('report_options', {}).get('cycle_group_size', 4)
                == descriptor.get('report', {}).get('cycle_group_size', root_config['report'].get('cycle_group_size', 4))
                and metadata.get('report_range') == [None, None] and metadata.get('comparison') is None):
            return page
    return None

def input_rows(run):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parent/'training'))
    from basewalk.compact_logs import records
    for name in ('report_trace.jsonl', 'report_trace.jsonl.gz'):
        path = run/name
        if path.exists():
            opener = gzip.open if name.endswith('.gz') else open
            with opener(path, 'rt', encoding='utf-8-sig') as stream:
                yield from (json.loads(line) for line in stream if line.strip())
            return
    with ExitStack() as stack:
        stream = stack.enter_context((run/'training.csv').open(newline='', encoding='utf-8-sig'))
        trace_path = run/'transition_trace.jsonl.gz'
        trace = stack.enter_context(gzip.open(trace_path, 'rt', encoding='utf-8-sig')) if trace_path.exists() else None
        entries = records(trace) if trace else iter(())
        head = next(entries, None)
        for row in csv.DictReader(stream):
            step = int(row['step'])
            if head and int(head['step']) < step:
                raise ValueError('Trace/CSV step mismatch')
            if head and int(head['step']) == step:
                if abs(float(row['reward'])-float(head['reward'])) > 1e-6:
                    raise ValueError('Trace/CSV Reward mismatch')
                row.update(head)
                head = next(entries, None)
            yield row
        if head:
            raise ValueError('Unfinalized trace extends beyond CSV; use a closed log snapshot')


def vector(value):
    return json.loads(value) if isinstance(value, str) and value else value


def load_run(run, start=None, end=None):
    document = read_json(run/'effective_config.json')
    cfg = document.get('config', document)
    dt = cfg.get('webots', {}).get('control_time_step_ms', 40)/1000
    result, previous, episode, cycle = [], None, 0, 0
    source_profiles = {}
    for source in input_rows(run):
        step = int(source['step'])
        if end is not None and step > end:
            break
        if step % 10000 == 0:
            print(f'  已读取 {step} 步', flush=True)
        if previous is not None and step <= previous['step']:
            raise ValueError('Steps must strictly increase')
        ep = source.get('episode_id', source.get('diagnostic_episode_id'))
        boundary = previous is None or step != previous['step']+1 or previous['episode_end']
        boundary |= previous is not None and ep not in ('', None) and str(ep) != str(previous['episode_id'])
        if boundary and previous is not None:
            episode += 1
        stage = source.get('stage', source.get('motion_phase', 'unknown'))
        terms = [numeric(source.get('reward_'+k, source.get(old))) for k, old in zip(TERMS, LEGACY_TERMS)]
        if any(x is None for x in terms):
            raise ValueError(f'Missing six-term Reward at step {step}; see LOG_CONTRACT.md')
        total = numeric(source.get('reward'))
        regularization = {}
        params = cfg.get('reward', {}).get('biped_tracking', {})
        if stage == 'turn' and params.get('turn_world_z_regularization'):
            before_action = vector(source.get('source_observation'))
            executed_action = vector(source.get('executed_action'))
            if before_action is None or len(before_action) < 46 or executed_action is None or len(executed_action) != 12:
                raise ValueError(f'Missing executed Action history at step {step}')
            previous_action = before_action[34:46]
            regularization = {
                'action_rate': -params['turn_action_rate_weight'] * sum(
                    (current - prior) ** 2 for current, prior in zip(executed_action, previous_action)
                ),
                'ankle_soft_limit': numeric(source.get('reward_ankle_soft_limit')),
                'hip_deviation': numeric(source.get('reward_hip_deviation')),
            }
            if any(value is None for value in regularization.values()):
                raise ValueError(f'Missing Turn regularization at step {step}')
        if stage == 'turn' and params.get('turn_reverse_foot_yaw_weight'):
            foot_yaw = numeric(source.get('reward_reverse_foot_yaw'))
            if foot_yaw is None:
                raise ValueError(f'Missing reverse stance foot yaw Reward at step {step}')
            regularization['reverse_foot_yaw'] = foot_yaw
        if stage == 'turn' and params.get('phase_foot', {}).get('kind') == 'phase_position_minimum_v1':
            foot_position = numeric(source.get('reward_phase_foot_position'))
            if foot_position is None:
                raise ValueError(f'Missing phase-position Reward at step {step}')
            regularization['phase_foot_position'] = foot_position
        if stage == 'turn' and params.get('turn_heading_window_seconds'):
            heading = numeric(source.get('reward_turn_heading'))
            if heading is None:
                raise ValueError(f'Missing Turn heading Reward at step {step}')
            regularization['turn_heading'] = heading
        if stage == 'turn' and (params.get('turn_yaw_error_progress_weight') or params.get('turn_yaw_error_cost_weight') or params.get('turn_acceleration_weight')):
            speed_delta = numeric(source.get('reward_turn_speed_delta'))
            if speed_delta is None:
                raise ValueError(f'Missing Turn speed-delta Reward at step {step}')
            regularization['turn_speed_delta'] = speed_delta
        joint_xy = None
        term_sum = sum(terms)
        if stage == 'walk' and cfg.get('training', {}).get('walk_commands'):
            joint_xy = numeric(source.get('reward_tracking_xy'))
            if joint_xy is None:
                raise ValueError(f'Missing joint XY Walk reward at step {step}')
            # The logged planar term supports either joint Gaussian or additive legacy Walk.
            term_sum = joint_xy + sum(terms[2:])
        if total is None or abs(term_sum + sum(regularization.values()) - total) > 1e-5:
            raise ValueError(f'Reward terms do not sum to total at step {step}')
        truth = {k: numeric(source.get('global_'+k)) for k in PHYSICAL}
        sensors = {k: numeric(source.get('sensor_'+k)) for k in SENSORS}
        g = vector(source.get('reward_gravity'))
        gyro = vector(source.get('reward_gyro_xy'))
        inputs = {k: numeric(source.get('reward_input_'+k)) for k in REWARD_INPUT_FIELDS}
        for k, old in [('vx', 'reward_vx'), ('vy', 'reward_vy'), ('wz', 'reward_wz')]:
            if inputs[k] is None:
                inputs[k] = numeric(source.get(old))
        if inputs['wz'] is None:
            inputs['wz'] = numeric(source.get('actor_yaw_rate'))
        if g:
            for i, k in enumerate(('gx', 'gy', 'gz')):
                inputs[k] = float(g[i])
        if gyro:
            for i, k in enumerate(('gyro_x', 'gyro_y')):
                inputs[k] = float(gyro[i])
        before, after = numeric(source.get('gait_phase_before')), numeric(source.get('gait_phase_after'))
        wrapped = before is not None and after is not None and after < before
        if previous and (boundary or previous['cycle_end']):
            cycle += 1
        command = [numeric(source.get(k, source.get(old))) for k, old in
                   [('command_vx', 'reward_vx_cmd'), ('command_vy', 'reward_vy_cmd'), ('command_wz', 'wz_cmd')]]
        if any(x is None for x in command):
            raise ValueError(f'Missing command at step {step}')
        profile = vector(source.get('sensor_sources')) or {}
        profile_key = tuple(sorted(profile.items()))
        if profile_key not in source_profiles:
            source_profiles[profile_key] = profile
        row = dict(step=step, time_s=numeric(source.get('time_s')) if source.get('time_s') not in ('', None) else step*dt,
                   episode_id=str(ep) if ep not in ('', None) else str(episode), stage=stage,
                   cycle_id=str(source.get('cycle_id', cycle)), cycle_start=flag(source.get('cycle_start')) or before == 0 or
                   bool(previous and previous['wrapped'] and not boundary),
                   cycle_end=flag(source.get('cycle_end')) or flag(source.get('cycle_complete')) or wrapped,
                   wrapped=wrapped or flag(source.get('cycle_complete')),
                   episode_end=flag(source.get('episode_end')), boundary=boundary,
                   fell=flag(source.get('global_fell', source.get('fell'))), command=command,
                   global_fell=flag(source['global_fell']) if source.get('global_fell') not in ('', None) else None,
                   fall_source='global' if source.get('global_fell') not in ('', None) else 'legacy_detector',
                   terms=terms, regularization=regularization, reward=total, global_state=truth, sensors=sensors, inputs=inputs,
                   original_reward=numeric(source.get('reward_before_backfill')),
                   action=vector(source.get('executed_action')), observation=vector(source.get('next_observation')),
                   sensor_sources=source_profiles[profile_key],
                   truth_unavailable=vector(source.get('truth_unavailable')) or {},
                   foot_contact_points=vector(source.get('foot_contact_points')),
                   world_dx=numeric(source.get('world_dx')), world_dy=numeric(source.get('world_dy')),
                   actor_updated=flag(source.get('actor_updated')), bc_recoveries=int(source.get('bc_recoveries') or 0))
        row['reward_history_count'] = numeric(source.get('reward_history_count'))
        if joint_xy is not None:
            row['tracking_xy'] = joint_xy
            row['axis_terms_are_diagnostics'] = cfg.get('reward', {}).get('biped_tracking', {}).get('walk_reward_revision') != 'linear_forward_lateral_gaussian_v1'
        yaw_delta = numeric(source.get('yaw_delta'))
        cycle_dt = numeric(source.get('cycle_dt'))
        row['cycle_dt'] = cycle_dt
        row['heading_rate'] = yaw_delta / cycle_dt if yaw_delta is not None and cycle_dt is not None and cycle_dt > 0 else None
        row['critic_features'] = [numeric(source.get('critic_reward_mean_'+axis)) for axis in ('vx','vy','wz')]
        row['next_critic_features'] = [numeric(source.get('next_critic_reward_mean_'+axis)) for axis in ('vx','vy','wz')]
        # Never relabel legacy actual_vx or IMU feedback as independent global truth.
        if (start is None or step >= start) and (end is None or step <= end):
            if not result:
                row['boundary'] = True
            result.append(row)
        previous = row
    if not result:
        raise ValueError('No samples in requested interval')
    return result, cfg


def summary(rows):
    out = dict(samples=len(rows), detected_fall_steps=sum(r['fell'] for r in rows),
               mean_reward=avg([r['reward'] for r in rows]),
               term_mean=[avg([r['terms'][i] for r in rows]) for i in range(6)],
               term_sum=[sum(r['terms'][i] for r in rows) for i in range(6)],
               mean_command=[avg([r['command'][i] for r in rows]) for i in range(3)])
    for key in PHYSICAL:
        values = [r['global_state'][key] for r in rows if r['global_state'][key] is not None]
        out[key+'_coverage'] = len(values)
        out[key+'_mean'] = avg(values)
        out[key+'_rms'] = rms(values)
        out[key+'_peak'] = max(map(abs, values)) if values else None
    for i, key in enumerate(('vx', 'vy', 'wz')):
        out[key+'_error_rms'] = rms([r['global_state'][key]-r['command'][i] for r in rows if r['global_state'][key] is not None])
    for key in ('forward', 'lateral'):
        values = [r['global_state'][key] for r in rows]
        out[key+'_net'] = values[-1]-values[0] if all(x is not None for x in values) and not any(r['boundary'] for r in rows[1:]) else None
    out['crosses_boundary'] = any(r['boundary'] or r['step'] != rows[i-1]['step']+1 for i, r in enumerate(rows) if i)
    return out


def sensor_checks(rows):
    checks = []
    for key in SENSORS:
        pairs = [r for r in rows if r['sensors'][key] is not None and r['global_state']['wz' if key == 'gyro_z' else key] is not None
                 and r['sensor_sources'].get(key) not in ('global', 'supervisor', 'same_as_global')]
        if not pairs:
            continue
        errors = [r['sensors'][key]-r['global_state']['wz' if key == 'gyro_z' else key] for r in pairs]
        # Wrapped angular errors; radians are the only accepted input unit.
        if key in ('roll', 'pitch'):
            errors = [math.atan2(math.sin(x), math.cos(x)) for x in errors]
        checks.append(dict(field=key, samples=len(pairs), coverage=len(pairs)/len(rows),
                           bias=avg(errors), rmse=rms(errors), peak=max(map(abs, errors)),
                           interpretation='偏差提示；需进一步检查传感器、坐标变换和时间对齐'))
    return checks


def windows(rows, size, stride, tolerance):
    segments, group = [], []
    for row in rows:
        if group and (row['boundary'] or row['stage'] != group[-1]['stage'] or row['bc_recoveries'] != group[-1]['bc_recoveries'] or
                      any(abs(a-b) > tolerance for a, b in zip(row['command'], group[-1]['command']))):
            segments.append(group)
            group = []
        group.append(row)
    if group:
        segments.append(group)
    out = []
    for segment in segments:
        chunks = [('window', segment[i:i+size], len(segment[i:i+size]) == size) for i in range(0, len(segment), stride)]
        cycle_groups = []
        for row in segment:
            if cycle_groups and row['cycle_id'] == cycle_groups[-1][-1]['cycle_id']:
                cycle_groups[-1].append(row)
            else:
                cycle_groups.append([row])
        chunks += [('cycle', g, g[0]['cycle_start'] and g[-1]['wrapped'] and not any(r['fell'] for r in g)) for g in cycle_groups]
        for kind, chunk, complete in chunks:
            entry = dict(id=len(out)+1, kind=kind, stage=chunk[0]['stage'], start=chunk[0]['step'], end=chunk[-1]['step'],
                         complete=bool(complete), cycle_gyro_z=None, cycle_net_heading_rate=None, **summary(chunk))
            if kind == 'cycle' and complete:
                durations = [r.get('cycle_dt') for r in chunk]
                if all(dt is not None and dt > 0 for dt in durations):
                    duration = math.fsum(durations)
                    for field, values in (
                        ('cycle_gyro_z', [r['sensors']['gyro_z'] for r in chunk]),
                        ('cycle_net_heading_rate', [r.get('heading_rate') for r in chunk]),
                    ):
                        if all(v is not None for v in values):
                            entry[field] = math.fsum(v * dt for v, dt in zip(values, durations)) / duration
            out.append(entry)
    return out


def flatten(document, prefix=''):
    out = {}
    if isinstance(document, dict):
        for k, v in document.items():
            out.update(flatten(v, prefix+'.'+k if prefix else k))
    else:
        out[prefix] = document
    return out


def differences(a, b):
    a, b = flatten(a), flatten(b)
    return [dict(field=k, current=a.get(k), comparison=b.get(k)) for k in sorted(a.keys() | b.keys()) if a.get(k) != b.get(k)]


def evaluation(run):
    path = run/'report_evaluation.json'
    if not path.exists():
        return dict(status='missing', message='缺少独立 Evaluation；在线窗口不能替代成功率或到达时间')
    data = read_json(path)
    if not all(k in data for k in ('checkpoint', 'evaluation_id', 'conditions', 'metrics')):
        raise ValueError('Evaluation requires checkpoint, evaluation_id, conditions and metrics')
    if any(numeric(data['metrics'].get(k)) is None for k in EVAL_METRICS):
        raise ValueError('Incomplete independent Evaluation metrics')
    if abs(data['metrics']['lateral_signed_m']) > data['metrics']['lateral_abs_m']+1e-8:
        raise ValueError('Invalid absolute lateral displacement')
    for k in ('success_rate', 'fall_rate'):
        if not 0 <= float(data['metrics'][k]) <= 1:
            raise ValueError('Evaluation rates must be in [0,1]')
    return dict(status='available', **data)


def hash_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def build(root_config, selected, override=None, start=None, end=None):
    path, conf = experiment(root_config, selected)
    run = choose_run(path, conf, override)
    print(f'读取训练日志：{run.name}', flush=True)
    rows, cfg = load_run(run, start, end)
    print(f'计算摘要、传感器误差和周期窗口：{len(rows)} 步', flush=True)
    options = dict(root_config['report'], **conf.get('report', {}))
    frozen_reward = next((p for p in (run/'source/experiment_reward.py', run/'source/basewalk/biped_reward.py') if p.exists()), None)
    reward_identity = dict(sha256=hash_file(frozen_reward), file=str(frozen_reward),
                           source=frozen_reward.read_text(encoding='utf-8-sig')) if frozen_reward else None
    if reward_identity and cfg.get('reward_source_sha256') and cfg['reward_source_sha256'] != reward_identity['sha256']:
        raise ValueError('Saved Reward source hash mismatch')
    return dict(experiment=conf['experiment_id'], experiment_name=conf['name'], run=str(run), report_options=options,
                reward_implementation=reward_identity, sensor_fields=list(SENSORS),
                stages=sorted({r['stage'] for r in rows}),
                external_source=path.resolve() not in run.parents, source_experiment=cfg.get('experiment'),
                effective_config=cfg, summary=summary(rows), sensor_checks=sensor_checks(rows), rows=rows,
                windows=windows(rows, options['window_steps'], options['stride'], options['command_tolerance']),
                evaluation=evaluation(run),
                files={p.name:hash_file(p) for p in run.iterdir() if p.is_file() and p.name in
                       ('effective_config.json', 'training.csv', 'transition_trace.jsonl.gz', 'report_trace.jsonl',
                        'report_trace.jsonl.gz', 'report_evaluation.json')})


def compare(current, other):
    a,b=current.get('reward_implementation'),other.get('reward_implementation')
    source_diff=''.join(difflib.unified_diff(b['source'].splitlines(True),a['source'].splitlines(True),
                    fromfile='comparison_reward.py',tofile='current_reward.py')) if a and b and 'source' in a and 'source' in b else None
    return dict(experiment=other['experiment'], run=other['run'], summary=other['summary'],
                reward_source_diff=source_diff,
                reward_changes=differences({k:v for k,v in current['effective_config'].items() if 'reward' in k and k != 'reward_module_path'},
                                           {k:v for k,v in other['effective_config'].items() if 'reward' in k and k != 'reward_module_path'}),
                reward_implementation=dict(current=current.get('reward_implementation'),comparison=other.get('reward_implementation')),
                config_changes=differences(current['effective_config'], other['effective_config']),
                evaluation=other['evaluation'], files=other['files'],
                evaluation_conditions_match=current['evaluation'].get('conditions') == other['evaluation'].get('conditions')
                if current['evaluation']['status'] == other['evaluation']['status'] == 'available' else None,
                interpretation='在线行为对比存在探索、Actor 更新和预算差异；不单独证明 Reward 的因果效果')


def historical_reports(root_config):
    """List saved snapshots, including the former project-level report directory."""
    paths = set((ROOT/'reports').glob('*/data.json'))
    for folder in (ROOT/root_config['experiments_root']).glob('*'):
        descriptor = folder/'config.json'
        if descriptor.is_file():
            conf = read_json(descriptor)
            paths.update((folder/conf.get('reports_root', 'reports')).glob('*/data.json'))
    return sorted(paths, key=lambda p:p.parent.name, reverse=True)


def select_historical_report(root_config):
    reports = historical_reports(root_config)
    if not reports:
        print('尚无历史报告，本次不做对比。')
        return None
    print('\n是否需要与历史报告比较？输入序号或完整报告编号，直接回车不比较。')
    for index, path in enumerate(reports, 1):
        print(f'  {index}. {path.parent.name}  [{path.parent.parent}]')
    while True:
        try:
            answer = input('对比历史报告 [回车跳过]：').strip()
        except (EOFError, KeyboardInterrupt):
            print('\n未选择历史报告，本次不做对比。')
            return None
        if answer.lower() in ('', 'n', 'no', 'none', '不', '否'):
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(reports):
            return reports[int(answer)-1]
        matches = [path for path in reports if path.parent.name == answer]
        if len(matches) == 1:
            return matches[0]
        print('未找到该报告，请输入列表中的序号或报告编号；回车跳过。')


def load_historical_report(root_config, value):
    path = Path(value)
    if path.is_dir():
        path = path/'data.json'
    if not path.is_file():
        matches = [p for p in historical_reports(root_config) if p.parent.name == str(value)]
        if len(matches) != 1:
            raise ValueError('Historical report ID not found or ambiguous: '+str(value))
        path = matches[0]
    if path.stat().st_size > 64*1024*1024:
        raise ValueError('该历史报告使用过大的旧版数据格式，请先用新版重新生成后再对比')
    document = read_json(path)
    if not all(k in document for k in ('report_id','experiment','run','summary','effective_config','evaluation','files')):
        raise ValueError('Unsupported historical report snapshot')
    return document, path.resolve()


def write_csv(path, records):
    if not records:
        path.write_text('', encoding='utf-8')
        return
    opener = (lambda:gzip.open(path,'wt',newline='',encoding='utf-8-sig',compresslevel=1)) if path.suffix=='.gz' else (lambda:path.open('w',newline='',encoding='utf-8-sig'))
    with opener() as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows({k:json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k,v in r.items()} for r in records)


def visual_schema(rows):
    omitted = {'observation','foot_contact_points','sensor_sources','truth_unavailable'}
    nested = {'global_state':list(PHYSICAL),'sensors':list(SENSORS),'inputs':list(REWARD_INPUT_FIELDS)}
    fields = [k for k in rows[0] if k not in omitted and k not in nested]
    return dict(fields=fields,nested=nested,sensor_sources=rows[0]['sensor_sources'],
                truth_unavailable=rows[0]['truth_unavailable'],precision=6)


def visual_row(row, schema):
    def rounded(v):
        if isinstance(v,float):return round(v,6)
        if isinstance(v,list):return [rounded(x) for x in v]
        return v
    return [rounded(row.get(k)) for k in schema['fields']]+[
        [rounded(row[name].get(k)) for k in keys] for name,keys in schema['nested'].items()]+[
        row['sensor_sources'] if row['sensor_sources']!=schema['sensor_sources'] else None,
        row['truth_unavailable'] if row['truth_unavailable']!=schema['truth_unavailable'] else None]


def write_html(path, data, metadata):
    """Stream a compact column representation, without materializing a second full log."""
    template=(ROOT/'report/view.html').read_text(encoding='utf-8')
    before,after=template.split('__DATA__',1)
    schema=visual_schema(data['rows'])
    view={**metadata,'windows':data['windows'],'visual_schema':schema,'row_count':len(data['rows'])}
    def encoded(value):
        return json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':')).replace('<','\\u003c').replace('&','\\u0026')
    with path.open('w',encoding='utf-8') as stream:
        stream.write(before)
        stream.write(encoded(view)[:-1]+',"compact_rows":[')
        for index,row in enumerate(data['rows']):
            if index:stream.write(',')
            stream.write(encoded(visual_row(row,schema)))
        stream.write(']}')
        stream.write(after)


def write_turn_html(path, data, metadata):
    cycle_group_size = metadata.get('report_options', {}).get('cycle_group_size', 4)
    if type(cycle_group_size) is not int or cycle_group_size < 1:
        raise ValueError('Cycle group size must be a positive integer')
    def group_points(rows, x_value, columns, group_size=4):
        points = []
        for start in range(0, len(rows), group_size):
            group = rows[start:start+group_size]
            values = []
            for get_value in columns:
                samples = [get_value(row) for row in group]
                values.append(avg([value for value in samples if value is not None]))
            points.append([x_value(group[-1]), *values])
        return points

    steps = group_points(data['rows'], lambda r: r['step'], (
        lambda r: r['terms'][0],
        lambda r: r['global_state']['vx'],
        lambda r: r['terms'][2],
        lambda r: r['heading_rate'],
        lambda r: r['reward'],
        lambda r: r['global_state']['vy'],
        lambda r: r['sensors']['gyro_z'],
        lambda r: r['regularization'].get('turn_speed_delta'),
    ))
    completed = [w for w in data['windows'] if w['kind'] == 'cycle' and w['complete']]
    cycles = group_points(list(enumerate(completed, start=1)), lambda r: r[0], (
        lambda r: r[1]['cycle_net_heading_rate'],
        lambda r: r[1]['cycle_gyro_z'],
    ), group_size=cycle_group_size)
    view = dict(experiment=metadata['experiment_name'], samples=len(data['rows']),
                cycle_count=len(completed), steps=steps, cycles=cycles, cycle_group_size=cycle_group_size,
                has_speed_delta=any('turn_speed_delta' in row['regularization'] for row in data['rows']))
    payload = json.dumps(view, ensure_ascii=False, allow_nan=False, separators=(',', ':')).replace('<', '\\u003c')
    template = (ROOT/'report/turn_view.html').read_text(encoding='utf-8')
    path.write_text(template.replace('__DATA__', payload, 1), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', nargs='?', help='Experiment ID, default: current_experiment in config')
    parser.add_argument('comparison', nargs='?', help='Optional comparison experiment ID')
    parser.add_argument('--experiment', dest='experiment_option')
    parser.add_argument('--compare')
    parser.add_argument('--compare-report', help='Saved report ID, directory or data.json; compare its saved snapshot')
    parser.add_argument('--ask-compare', action='store_true', help='Ask whether to compare a saved historical report')
    parser.add_argument('--no-compare', action='store_true', help='Skip comparison without asking')
    parser.add_argument('--run', type=Path, help='Explicit saved run directory, including external historical data')
    parser.add_argument('--latest-completed', action='store_true', help='Use this project’s latest completed training run')
    parser.add_argument('--compare-run', type=Path)
    parser.add_argument('--config', type=Path, default=ROOT/'config.json')
    parser.add_argument('--start', type=int)
    parser.add_argument('--end', type=int)
    parser.add_argument('--open', action='store_true')
    args = parser.parse_args()
    args.config, args.experiment = resolve_entry_config(args.config, args.experiment)
    try:
        if args.experiment and args.experiment_option or args.comparison and args.compare:
            raise ValueError('Use positional IDs or named options, without duplicates')
        if args.start is not None and args.start < 1 or args.end is not None and args.end < 1 or args.start and args.end and args.start > args.end:
            raise ValueError('Invalid step interval')
        conf = read_json(args.config)
        conf['_config_path'] = str(args.config.resolve())
        conf['experiments_root'] = str((args.config.resolve().parent / conf['experiments_root']).resolve())
        if min(conf['report']['window_steps'], conf['report']['stride']) < 1:
            raise ValueError('Window size/stride must be positive')
        if numeric(conf['report']['command_tolerance']) < 0:
            raise ValueError('Command tolerance must be nonnegative')
        selected = args.experiment_option or args.experiment or conf.get('experiment_id') or conf['current_experiment']
        if args.latest_completed:
            if args.run or args.experiment or args.experiment_option:
                raise ValueError('--latest-completed cannot be combined with an explicit run or experiment')
            selected, args.run = latest_completed_training(conf)
        turn_view = conf['report'].get('view') == 'turn'
        template = ROOT/'report'/('turn_view.html' if turn_view else 'view.html')
        view_hash = hash_file(template)
        if args.latest_completed and args.no_compare and args.open and args.start is None and args.end is None:
            saved_view = saved_latest_view(conf, selected, args.run, view_hash)
            if saved_view:
                print(f'Report: {saved_view}', flush=True)
                open_report(saved_view)
                return saved_view.parent
        comparison = args.compare or args.comparison or None
        if args.compare_report and (comparison or args.compare_run):
            raise ValueError('Use --compare-report OR experiment/run comparison')
        if args.no_compare and (comparison or args.compare_report or args.compare_run):
            raise ValueError('--no-compare conflicts with explicit comparison')
        if args.compare_run and comparison is None:
            raise ValueError('--compare-run requires a comparison experiment ID')
        saved_report = args.compare_report
        if args.ask_compare and not args.no_compare and not comparison and not saved_report:
            saved_report = select_historical_report(conf)
        data = build(conf, selected, args.run, args.start, args.end)
        if saved_report:
            other, path = load_historical_report(conf, saved_report)
            data['comparison'] = compare(data, other)
            data['comparison'].update(report_id=other['report_id'],report_path=str(path),snapshot_sha256=hash_file(path))
            comparison = 'report:'+other['report_id']
        else:
            data['comparison'] = compare(data, build(conf, comparison, args.compare_run, args.start, args.end)) if comparison else None
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        report_id = f'{identifier(selected)}_{stamp}'
        data.update(report_id=report_id, created_at_utc=stamp, report_options=data.get('report_options', conf['report']),
                    fixed_baseline=dict(id='pre_s1_20260724_v1', status='pending',
                                        message='本工具尚未接入固定 baseline 的独立 Evaluation，不能判定训练改进'))
        experiment_folder, descriptor = experiment(conf, selected)
        output = experiment_folder/descriptor.get('reports_root', 'reports')/report_id
        output.mkdir(parents=True, exist_ok=False)
        print('导出完整精度数据与压缩页面，请稍候…',flush=True)
        metadata={k:v for k,v in data.items() if k not in ('rows','windows')}
        metadata.update(schema='compact_report_v2',record_count=len(data['rows']),window_count=len(data['windows']),
                        trajectory_archive='trajectory.jsonl.gz',csv_archive='trajectory.csv.gz',
                        visual_precision=6,full_precision_preserved=True,view_sha256=view_hash,
                        report_range=[args.start,args.end])
        with (output/'data.json').open('w',encoding='utf-8') as stream:
            json.dump(metadata,stream,ensure_ascii=False,indent=2,allow_nan=False)
        with gzip.open(output/'trajectory.jsonl.gz','wt',encoding='utf-8',compresslevel=1) as stream:
            for row in data['rows']:
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
        print('完整轨迹已归档，正在写入交互报告…',flush=True)
        if turn_view:
            write_turn_html(output/'index.html', data, metadata)
        else:
            write_html(output/'index.html',data,metadata)
        write_csv(output/'windows.csv', data['windows'])
        write_csv(output/'sensor_checks.csv', data['sensor_checks'])
        write_csv(output/'trajectory.csv.gz', data['rows'])
        (output/'REPORT.md').write_text('\n'.join(['# Train Reward Analyse', f'Report: {report_id}',
            f'Experiment: {selected}; source run: {data["run"]}; external source: {data["external_source"]}',
            f'Comparison: {comparison or "none"}', '## 1. State sensor checks', json.dumps(data['sensor_checks'], ensure_ascii=False),
            '## 2. Reward and global behavior',
            ('See index.html for five zoomable charts, sampled in groups of four.'
             if turn_view else 'See index.html for the component and total plots with shared cursor.'),
            '## 3. Optional experiment comparison', json.dumps(data['comparison'], ensure_ascii=False),
            '## 4. Continuous interval summary', json.dumps(data['summary'], ensure_ascii=False),
            '## 5. Cycles and windows', f'{len(data["windows"])} windows; index.html shows gyro Z per control step and mean rates per completed cycle.',
            'Missing global fields stay missing. Historical feedback is never relabeled global truth.',
            'Online windows are not independent Evaluation. Fixed baseline pre_s1_20260724_v1: pending.',
            'No training, simulation or Evaluation was started.']), encoding='utf-8')
        print(f'Report: {output / "index.html"}')
        if args.open:
            open_report(output/'index.html')
        return output
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f'报告生成失败：{exc}\n')


if __name__ == '__main__':
    main()
