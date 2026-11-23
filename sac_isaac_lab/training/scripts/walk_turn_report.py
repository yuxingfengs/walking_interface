"""Five training charts averaged over short, single-command windows."""
import math
from html import escape
from statistics import fmean


def mixed_data(rows, cfg):
    blocks, windows = [], {}
    ramp = 'yaw_ramp' in cfg['training']
    last = {}
    counts = {}
    for step, row in enumerate(rows, 1):
        if int(row['step']) != step:
            raise ValueError('Missing or unordered training row')
        task = row['motion_phase']
        counts[task] = counts.get(task, 0) + 1
        key = tuple(row[k] for k in ('episode', 'vx_cmd', 'vy_cmd', 'wz_cmd'))
        if ramp:
            key = (row['episode'], task)
        robot = row.get('env_id', 0)
        window = windows.setdefault(robot, [])
        if window and (key != last.get(robot) or len(window) == 40):
            blocks.append(summarize(window))
            window.clear()
        window.append(row)
        last[robot] = key
    blocks.extend(summarize(window) for window in windows.values())
    blocks.sort(key=lambda block: block['step'])
    return dict(experiment=cfg['experiment_name'], blocks=blocks, counts=counts)


def summarize(rows):
    last = rows[-1]
    point = {k: float(last[k]) for k in ('step', 'vx_cmd', 'vy_cmd', 'wz_cmd')}
    for k in ('actual_vx', 'actual_vy', 'actual_wz', 'reward', 'wz_cmd'):
        point[k] = fmean(float(r[k]) for r in rows)
    point['falls'] = sum(r['fell'].lower() in ('true', '1') for r in rows)
    return point


def direction_summary(rows):
    """Summarize complete gait cycles and all falls separately for each command."""
    result = []
    for label, start in (('Full run', 0), ('Last 5,000 steps', max(0, len(rows)-5000))):
        stats = {}
        elapsed = delta = 0.
        previous = None
        for row in rows[start:]:
            target = float(row['wz_cmd'])
            entry = stats.setdefault(target, dict(steps=0, falls=0, cycles=0, time=0., yaw=0., error=0.))
            fell = row['fell'].lower() in ('true', '1')
            entry['steps'] += 1
            entry['falls'] += int(fell)
            key = row['episode'], target
            before, after = float(row['phase_before']), float(row['phase_after'])
            if key != previous:
                elapsed = delta = 0.
                full = before == 0. or (previous is not None and wrapped)
            previous = key
            elapsed += float(row['cycle_dt'])
            delta += float(row['yaw_delta'])
            wrapped = after < before
            if wrapped or fell:
                if wrapped and full and not fell:
                    entry['cycles'] += 1
                    entry['time'] += elapsed
                    entry['yaw'] += delta
                    entry['error'] += abs(delta/elapsed-target)
                elapsed = delta = 0.
                full = not fell
        for target, entry in sorted(stats.items(), reverse=True):
            result.append((label, target, entry))
    return result


def chart(blocks, title, series):
    width, height = 800, 150
    values = [p[k] for p in blocks for k in series]
    low, high = min(0., min(values)), max(0., max(values))
    span = max(high-low, 1e-8)
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Nonfinite training chart data')
    lines = []
    for key, color in zip(series, ('#185d82', '#c45820')):
        points = ' '.join(f"{64 + (width-80)*p['step']/blocks[-1]['step']:.1f},"
                          f"{10 + (height-24)*(high-p[key])/span:.1f}" for p in blocks)
        lines.append(f'<polyline fill="none" stroke="{color}" points="{points}"/>')
    legend = ' · '.join(series)
    return (f'<h2>{title}</h2><p>{legend} (blue, orange)</p>'
            f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{title}">'
            f'<text y="19">{high:.3f}</text><text y="{height-10}">{low:.3f}</text>'
            + ''.join(lines) + '</svg>')


def render_report(folder, view):
    blocks = view['blocks']
    falls = 0
    for point in blocks:
        falls += point['falls']
        point['falls'] = falls
    panels = [('Body forward velocity (m/s)', ('vx_cmd', 'actual_vx')),
              ('Body lateral velocity (m/s)', ('vy_cmd', 'actual_vy')),
              ('Yaw rate (rad/s)', ('wz_cmd', 'actual_wz')),
              ('Mean reward per step', ('reward',)),
              ('Cumulative falls', ('falls',))]
    title = escape(view['experiment'])
    body = ''.join(chart(blocks, label, series) for label, series in panels)
    if 'directions' in view:
        headers = ('Window', 'Target rad/s', 'Steps', 'Falls', 'Complete cycles', 'Mean yaw rad/s', 'Cycle MAE rad/s')
        table = ['<table><tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr>']
        for label, target, stats in view['directions']:
            n = stats['cycles']
            mean = f"{stats['yaw']/stats['time']:+.4f}" if n else '—'
            mae = f"{stats['error']/n:.4f}" if n else '—'
            values = (label, f'{target:+.3f}', stats['steps'], stats['falls'], n, mean, mae)
            table.append('<tr>' + ''.join(f'<td>{v}</td>' for v in values) + '</tr>')
        body = ''.join(table) + '</table>' + body
    counts = ' · '.join(f'{escape(k)}: {n} steps' for k, n in view['counts'].items())
    path = folder/'report'/'index.html'
    path.parent.mkdir(exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(f'<!doctype html><meta charset="utf-8"><title>{title}</title>'
                    '<style>body{max-width:900px;margin:auto;font:16px system-ui}svg{width:100%}</style>'
                    f'<h1>{title}</h1><p>{counts}</p><p>Horizontal axis: training steps 1–{int(blocks[-1]["step"])}. '
                    'Command and response means cover up to 40 steps within each episode.</p>'+body, encoding='utf-8')
    temp.replace(path)
    return path
