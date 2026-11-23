# -*- coding: utf-8 -*-
"""Compare two saved reports. Standard library only; never imports training code."""
import argparse
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import statistics
import webbrowser

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def flag(value):
    return str(value).lower() in ('true', '1', '1.0')


def number(row, key, default=0.):
    value = row.get(key)
    return float(value) if value not in (None, '') else default


def report_info(path):
    path = Path(path).resolve()
    if path.is_dir(): path = path/'index.html'
    if path.name == 'data.json': path = path.with_name('index.html')
    if not path.is_file(): raise FileNotFoundError(path)
    sidecar = path.with_name('data.json')
    data = read(sidecar) if sidecar.exists() else {}
    if not data:
        match = re.search(r'<script[^>]*type=["\']application/json["\'][^>]*>(.*?)</script>',
                          path.read_text(encoding='utf-8'), re.S)
        if match: data = json.loads(match.group(1))
    source_name = re.split(r'[\\/]', str(data.get('run', '')))[-1]
    candidates = [path.parent.parent]
    if source_name: candidates.append(path.parent.parent.parent/'runs'/source_name)
    run = next((p for p in candidates if (p/'training.csv').exists()), None)
    if run is None: raise ValueError('Report has no matching saved training.csv: '+str(path))
    result = read(run/'result.json')
    if result.get('status') != 'completed': raise ValueError('Report source run is not completed')
    frozen = read(run/'effective_config.json')
    cfg = frozen.get('config', frozen)
    expected = data.get('files', {}).get('training.csv')
    if expected and hashlib.sha256((run/'training.csv').read_bytes()).hexdigest() != expected:
        raise ValueError('Report source training.csv hash mismatch')
    try:
        relative = run.relative_to(PROJECT)
        group = '/'.join(relative.parts[:2]) if relative.parts[0] == 'experiments' else relative.parts[0]
    except ValueError:
        group = 'external/'+run.parent.parent.name
    return dict(path=path, run=run, group=group, data=data, cfg=cfg)


def discover():
    patterns = ('*/reports/*/index.html', 'experiments/*/runs/*/report/index.html',
                'isaac_lab/runs/*/report/index.html', 'isaac_lab/reports/*/index.html')
    unique = {}
    errors = []
    for pattern in patterns:
        for path in PROJECT.glob(pattern):
            try: item = report_info(path)
            except (ValueError, OSError, KeyError) as error:
                errors.append(dict(report=str(path), error=str(error)))
                continue
            key = str(item['run'])
            # Regenerating the same run's report is not another training run.
            if key not in unique or str(item['path']) > str(unique[key]['path']): unique[key] = item
    return sorted(unique.values(), key=lambda x:x['run'].name, reverse=True), errors


def explicit(value):
    p = Path(value)
    return report_info(p if p.is_absolute() else HERE/p)


def choose(settings):
    pool, errors = discover()
    experiment = settings.get('experiment', 'auto')
    if experiment != 'auto': pool = [r for r in pool if r['group'] == experiment]
    current_value = settings['reports']['current']
    previous_value = settings['reports']['previous']
    current = (pool[0] if pool else None) if current_value == 'latest' else explicit(current_value)
    if previous_value != 'previous': previous = explicit(previous_value)
    elif current:
        earlier = [p for p in pool if p['run'] != current['run'] and
                   (not settings.get('same_experiment', True) or p['group'] == current['group']) and
                   p['run'].name < current['run'].name]
        previous = earlier[0] if earlier else None
    else: previous = None
    return current, previous, errors


def summarize(item, window):
    with (item['run']/'training.csv').open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    bounds = item['data'].get('report_range') or [None, None]
    rows = [r for r in rows if (bounds[0] is None or int(r['step']) >= bounds[0]) and
            (bounds[1] is None or int(r['step']) <= bounds[1])]
    if not rows: raise ValueError('Report range has no raw transitions')
    end = int(rows[-1]['step']); start = max(int(rows[0]['step']), end-window+1)
    selected = [r for r in rows if int(r['step']) >= start]
    pending = []; cycles = []
    dt_default = item['cfg'].get('webots', {}).get('control_time_step_ms', 40)/1000.
    for row in rows:
        pending.append(row)
        before = number(row, 'gait_phase_before', number(row, 'phase_before'))
        after = number(row, 'gait_phase_after', number(row, 'phase_after'))
        wrap = after < before
        fell = flag(row.get('fell'))
        ended = flag(row.get('cycle_end')) if 'cycle_end' in row else wrap or fell or flag(row.get('truncated'))
        if not ended: continue
        complete = flag(row.get('cycle_complete')) if 'cycle_complete' in row else wrap
        if complete and not any(flag(r.get('fell')) for r in pending) and int(pending[0]['step']) >= start:
            command = number(pending[0], 'wz_cmd')
            if command and all(r.get('motion_phase','turn') == 'turn' for r in pending):
                duration = sum(number(r,'cycle_dt',dt_default) for r in pending)
                angles = [number(r,'yaw_delta',number(r,'actual_wz')*number(r,'cycle_dt',dt_default)) for r in pending]
                speed = sum(angles)/duration
                direction = math.copysign(1, command)
                cycles.append(dict(speed=speed,error=abs(speed-command),
                    forward=sum(max(direction*a,0) for a in angles),
                    reverse=sum(max(-direction*a,0) for a in angles)))
        pending = []
    mean = lambda values: statistics.mean(values) if values else None
    return dict(window_start=start,window_end=end,env_steps=len(selected),full_turn_cycles=len(cycles),
        net_rate=mean([c['speed'] for c in cycles]),target_error=mean([c['error'] for c in cycles]),
        forward_angle=mean([c['forward'] for c in cycles]),reverse_angle=mean([c['reverse'] for c in cycles]),
        falls=sum(flag(r.get('fell')) for r in selected),mean_reward=mean([number(r,'reward') for r in selected]),
        gyro_z=mean([number(r,'sensor_gyro_z',number(r,'body_gyro_z')) for r in selected]),
        target=mean([number(r,'wz_cmd') for r in selected]))


def local(path):
    return Path(os.path.relpath(path,HERE)).as_posix()


def build(settings):
    current, previous, errors = choose(settings)
    result = dict(status='ready' if current and previous else 'pending_previous' if current else 'pending_current',
                  default_rule='latest completed run vs previous distinct completed run, same experiment',
                  discovery_errors=errors, notes=['Online report comparison, not independent frozen-policy Evaluation.'])
    for label,item in [('current',current),('previous',previous)]:
        if item: result[label] = dict(report=local(item['path']),run=local(item['run']),experiment=item['group'],
             metrics=summarize(item,settings.get('window_env_steps',5000)))
    if current and previous:
        result['delta'] = {k:result['current']['metrics'][k]-result['previous']['metrics'][k]
            for k in ['net_rate','target_error','forward_angle','reverse_angle','falls','mean_reward','gyro_z']
            if result['current']['metrics'][k] is not None and result['previous']['metrics'][k] is not None}
        if current['cfg'].get('reward') != previous['cfg'].get('reward'):
            result['notes'].append('Reward formulas differ: absolute Reward is not a performance ranking.')
        if current['group'] != previous['group']: result['notes'].append('Different experiments/backends; conditions may not be matched.')
    else: result['notes'].append('缺少上一轮独立训练的report，待比较；不会把同一次训练的重复报告冒充上一轮。')
    return result


def write(result):
    (HERE/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    labels=[('net_rate','实际净转速 rad/s'),('target_error','目标误差 rad/s'),('forward_angle','顺指令转角 rad/周期'),
            ('reverse_angle','反向转角 rad/周期'),('falls','窗口跌倒次数'),('mean_reward','每步平均总Reward'),('gyro_z','原始gyro Z均值 rad/s')]
    def fmt(v): return '—' if v is None else format(v,'.6f')
    table=[]
    for key,title in labels:
        table.append('<tr><th>'+html.escape(title)+'</th>'+''.join('<td>'+fmt(result.get(side,{}).get('metrics',{}).get(key))+'</td>' for side in ['previous','current'])+'<td>'+fmt(result.get('delta',{}).get(key))+'</td></tr>')
    sections=[]
    for side,title in [('previous','上一次report'),('current','当前report')]:
        value=result.get(side)
        if value:
            link=html.escape(value['report'],quote=True)
            metrics = value['metrics']
            window = 'step '+str(metrics['window_start'])+'–'+str(metrics['window_end'])
            sections.append('<section><h2>'+title+'</h2><a href="'+link+'">打开完整可放大报告</a>'
                '<p>'+html.escape(value['run'])+'; '+window+'; cycles='+str(metrics['full_turn_cycles'])+
                '</p><iframe src="'+link+'" title="'+title+'"></iframe></section>')
    page = ('<!doctype html><meta charset="utf-8"><title>Baseline报告比较</title>'
            '<style>body{font:16px system-ui;margin:32px;color:#213547}'
            'table{border-collapse:collapse}th,td{padding:10px;border:1px solid #ccd;text-align:right}'
            'section{margin-top:30px}iframe{width:100%;height:720px;border:1px solid #ccd}</style>'
            '<h1>Baseline：当前与上一次report</h1><p>状态：' + html.escape(result['status']) +
            '</p><p>' + html.escape(' '.join(result['notes'])) + '</p>'
            '<table><tr><th>末段指标</th><th>上一次</th><th>当前</th><th>当前−上一次</th></tr>' +
            ''.join(table) + '</table>' + ''.join(sections))
    (HERE/'comparison.html').write_text(page,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--open',action='store_true');args=parser.parse_args()
    result=build(read(HERE/'config.json'));write(result)
    print('Comparison: '+str(HERE/'comparison.html')+' ['+result['status']+']')
    if args.open:webbrowser.open((HERE/'comparison.html').as_uri())


if __name__=='__main__':main()
