"""Read-only training diagnostics and paired independent evaluation reports."""
import csv
import hashlib
import html
import io
import json
import math
from pathlib import Path
import statistics
import time
import uuid

from . import TRAINING_ROOT as ROOT


def write_text(path, text):
    path = Path(path)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


def read_json(path, default=None):
    path = Path(path)
    if not path.is_file():
        return default
    for attempt in range(9):
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except PermissionError:
            if attempt == 8:
                raise
            time.sleep(min(.025 * 2**attempt, .4))


def table(path):
    path = Path(path)
    if not path.exists():
        return [], None
    raw = path.read_bytes()
    # A live append may end in a partial row; never turn it into a false anomaly.
    complete = raw[:raw.rfind(b'\n')+1]
    return list(csv.DictReader(io.StringIO(complete.decode('utf-8-sig')))), hashlib.sha256(complete).hexdigest()


def number(value):
    if value is None or str(value).strip() == '':
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def analyze(rows, spike_factor=10.):
    updates, alerts, history = [], [], []
    previous_group, last_count = None, 0
    keys = ['actor_loss', 'critic_loss', 'q_mean', 'target_q_mean', 'td_abs_mean', 'actor_grad_norm', 'critic_grad_norm']
    for row in rows:
        count = int(row['updates'])
        changed = count > last_count
        last_count = count
        if not changed and not row.get('critic_loss'):
            continue
        step = int(row['step'])
        point = dict(step=step, updates=count, **{key:number(row.get(key)) for key in keys})
        for key in keys:
            if row.get(key, '').strip() and point[key] is None:
                alerts.append(dict(step=step, severity='error', kind='nonfinite_or_malformed', field=key))
        warmup = row.get('critic_only') == 'True'
        if point['critic_loss'] is None:
            alerts.append(dict(step=step, severity='error', kind='missing_critic_loss'))
        elif point['critic_loss'] < 0:
            alerts.append(dict(step=step, severity='error', kind='negative_critic_mse'))
        if not warmup and row.get('actor_updated') != 'False' and point['actor_loss'] is None:
            alerts.append(dict(step=step, severity='error', kind='missing_joint_actor_loss'))
        # Negative Actor loss is normal (-mean Q), unlike negative Critic MSE.
        group = (row.get('bc_recoveries'), warmup, row.get('actor_lr'), row.get('critic_lr'))
        if group != previous_group:
            history = []
        loss = point['critic_loss']
        if loss is not None and loss >= 0:
            if len(history) >= 5 and loss > spike_factor*max(statistics.median(history[-20:]), 1e-12):
                alerts.append(dict(step=step, severity='notice', kind='critic_loss_spike', value=loss))
            history.append(loss)
        else:
            history = []
        point.update(critic_rolling_mean=statistics.mean(history[-20:]) if history else None,
            stage='critic_warmup' if warmup else 'joint', actor_lr=number(row.get('actor_lr')),
            critic_lr=number(row.get('critic_lr')), bc_recoveries=int(row.get('bc_recoveries') or 0))
        updates.append(point)
        previous_group = group
    return dict(updates=updates, alerts=alerts, update_count=len(updates), latest_update=updates[-1] if updates else None)


def plots(data, episodes, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)
    updates = data['updates']
    for key, label in [('critic_loss', 'Critic MSE'), ('critic_rolling_mean', 'Critic rolling mean (20)')]:
        points = [p for p in updates if p[key] is not None]
        axes[0].plot([p['step'] for p in points], [p[key] for p in points], label=label, linewidth=1.)
    axes[0].set_ylabel('Critic loss'); axes[0].legend()
    for key, label in [('actor_loss', 'Actor loss (-Q)'), ('q_mean', 'Q mean'), ('target_q_mean', 'Target Q mean')]:
        points = [p for p in updates if p[key] is not None]
        if points: axes[1].plot([p['step'] for p in points], [p[key] for p in points], label=label, linewidth=1.)
    axes[1].set_ylabel('Actor / Q')
    if axes[1].lines: axes[1].legend()
    if episodes:
        axes[2].plot([int(p['end_step']) for p in episodes], [float(p['net_forward_m']) for p in episodes], label='Episode net forward (m)')
        axes[2].plot([int(p['end_step']) for p in episodes], [float(p['absolute_lateral_m']) for p in episodes], label='Episode absolute lateral (m)')
        axes[2].legend()
    else:
        axes[2].text(.5, .5, 'No episode displacement log available', ha='center', transform=axes[2].transAxes)
    axes[2].set_ylabel('Training episode displacement')
    previous = 0
    for point in updates:
        if point['bc_recoveries'] > previous:
            for axis in axes: axis.axvline(point['step'], color='gray', alpha=.4, linestyle='--')
            previous = point['bc_recoveries']
    for axis in axes:
        axis.set_xlabel('Environment steps'); axis.grid(alpha=.2)
        axis.set_xlim(0, max([p['step'] for p in updates]+[int(p['end_step']) for p in episodes]+[1]))
    temporary = out/('curves.'+uuid.uuid4().hex+'.png')
    fig.savefig(temporary, dpi=130)
    plt.close(fig)
    temporary.replace(out/'curves.png')


def evaluation_report(run):
    """All metrics remain paired per trial; no selecting the latest unrelated run."""
    run = Path(run)
    result = read_json(run/'result.json', {})
    if result.get('kind') != 'independent_straight_evaluation':
        return None
    payload = read_json(run/'effective_config.json')
    cfg = payload['config']
    checkpoint = Path(payload['options']['checkpoint'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != result['checkpoint_sha256']:
        raise ValueError('Evaluation checkpoint hash mismatch')
    def section(title, item):
        lines = ['', '## '+title, '']
        if not item:
            return lines+['待比较：缺少对应独立评估。']
        lines += ['| 试验 | 到达时间s | 带符号横移m | 绝对横移m | 净前进m | 到达 | 跌倒 |',
                  '|---|---:|---:|---:|---:|---|---|']
        for row in item['episodes']:
            arrival = f"{row['arrival_time_s']:.2f}" if row['arrival_time_s'] is not None else '未到达'
            lines.append(f"| {row['episode']+1} | {arrival} | {row['signed_lateral_m']:.6f} | {row['absolute_lateral_m']:.6f} | {row['net_forward_m']:.6f} | {row['success']} | {row['fell']} |")
        lines.append(f"\n到达率 {item['success_rate']:.1%}，跌倒率 {item['fall_rate']:.1%}。")
        return lines
    lines = ['# 独立直行 Evaluation', '', f'Checkpoint：`{checkpoint}`',
        f"SHA256：`{result['checkpoint_sha256']}`", '',
        f"用途：{cfg.get('artifact_purpose', 'research')}。interface_smoke仅用于接口验证，不是正式训练成果。", '',
        '到达只要求净前进10米且未跌倒，不含横移门槛；请同时看横向偏移。每行时间和位移来自同次试验。']
    lines += section('当前模型', result)
    proof = cfg.get('initialization', {}).get('bc_evaluation_file')
    bc_result = read_json(proof) if proof else None
    bc_hash = cfg.get('initialization', {}).get('bc_sha256')
    if bc_result is None and bc_hash:
        # Historical runs predate the named interface; find matching evidence by hash, not recency.
        for candidate in sorted(run.parent.glob('straight_eval_*/result.json')):
            item = read_json(candidate)
            if item.get('checkpoint_sha256') == bc_hash:
                bc_result = item
                break
    if bc_result is not None and bc_result.get('checkpoint_sha256') != bc_hash:
        raise ValueError('BC reference evaluation hash mismatch')
    lines += section('初始化 BC 参考', bc_result)
    previous = cfg.get('previous_evaluation_run')
    previous_path = ROOT/previous if previous else None
    prev = read_json(previous_path/'result.json') if previous_path else None
    lines += section('直接上一版：'+cfg['previous_experiment'], prev)
    if prev:
        previous_payload = read_json(previous_path/'effective_config.json')
        previous_config = previous_payload['config']
        if previous_config['experiment'] != cfg['previous_experiment']:
            raise ValueError('Previous evaluation belongs to a different experiment')
        previous_checkpoint = Path(previous_payload['options']['checkpoint'])
        if hashlib.sha256(previous_checkpoint.read_bytes()).hexdigest() != prev['checkpoint_sha256']:
            raise ValueError('Previous evaluation checkpoint hash mismatch')
        from .bc import model_contract
        equal_protocol = all(previous_config.get(k) == cfg.get(k) for k in ('webots', 'evaluation')) and model_contract(previous_config) == model_contract(cfg)
        lines += ['', f'上一版评估来源：`{previous_path}`。Webots与Evaluation配置完全一致：{equal_protocol}。',
            f"当前训练预算 {cfg['training']['total_env_steps']} 步，Checkpoint实际训练 {result.get('source_training_steps')} 步；上一版Checkpoint {prev.get('source_training_steps')} 步。预算不同或协议不匹配时不作等预算改善结论。"]
    baseline = ROOT/'baselines/pre_s1_20260724_v1/pre_s1_20260724.csv'
    if hashlib.sha256(baseline.read_bytes()).hexdigest() != '463cf1723419b789e84209efe3d326dbf16d20e7583a45f11d03cb5c18d6f482':
        raise ValueError('Fixed baseline changed')
    old, _ = table(baseline)
    lines += ['', '## 固定 baseline：pre_s1_20260724_v1', '',
        '仅作描述性比较：历史终点净前进约9.52米，本轮10米；起点、朝向、场地、reset和训练设计不同。保留全部7个原始配对记录。', '',
        '| Checkpoint | 到达时间s | 带符号横移m | 绝对横移m | 净前进m | 到达率 | 跌倒率 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in old:
        lines.append(f"| gs{row['global_step']} | {row['reach_time_mean']} | {row['lateral_mean']} | {abs(float(row['lateral_mean'])):.6f} | {row['forward_mean']} | {row['eval_success_rate']} | {row['fell_rate']} |")
    lines += ['', '训练loss下降不证明行走改善；相同初始条件的重复评估不代表扰动鲁棒性。']
    write_text(run/'EVALUATION.md', '\n'.join(lines)+'\n')
    return run/'EVALUATION.md'


def report(run):
    run = Path(run).resolve()
    payload = read_json(run/'effective_config.json')
    if payload is None: raise ValueError('Missing run effective_config.json')
    if payload.get('options', {}).get('mode') == 'straight_eval':
        return evaluation_report(run)
    cfg = payload['config']
    rows, checksum = table(run/'training.csv')
    episodes, episode_checksum = table(run/'training_episodes.csv')
    settings = cfg.get('monitoring', {})
    data = analyze(rows, settings.get('loss_spike_factor', 10.))
    status = read_json(run/'status.json', {'status':'not_started'})
    progress = read_json(run/'progress.json', {})
    step = int(rows[-1]['step']) if rows else 0
    age = time.time()-(run/'training.csv').stat().st_mtime if (run/'training.csv').exists() else None
    if status['status'] == 'running' and age is not None and age > settings.get('stale_seconds', 180):
        data['alerts'].append(dict(severity='notice', kind='no_recent_log_progress', seconds=age))
    if status['status'] == 'failed':
        data['alerts'].append(dict(severity='error', kind='controller_failed', error=status.get('error')))
    data.update(status=status, step=step, total_steps=cfg['training']['total_env_steps'],
        falls=sum(r['fell']=='True' for r in rows), bc_recoveries=progress.get('bc_recoveries', int(rows[-1].get('bc_recoveries') or 0) if rows else 0),
        episode_count=len(episodes), source_training_sha256=checksum, source_episodes_sha256=episode_checksum,
        source_config=payload, generated_at=time.time(), progress=progress)
    out = run/'monitor'; out.mkdir(exist_ok=True)
    write_text(out/'monitor.json', json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))
    plots(data, episodes, out)
    latest = data['latest_update'] or {}
    values = [('运行状态', status['status']), ('用途', cfg.get('artifact_purpose', 'research')), ('环境步数', f"{step} / {data['total_steps']}"),
        ('有效更新记录', data['update_count']), ('训练跌倒', data['falls']), ('BC恢复', data['bc_recoveries']),
        ('Actor loss', latest.get('actor_loss')), ('Critic loss', latest.get('critic_loss')),
        ('Q / Target Q', f"{latest.get('q_mean')} / {latest.get('target_q_mean')}"),
        ('阶段', latest.get('stage')), ('Actor / Critic LR', f"{latest.get('actor_lr')} / {latest.get('critic_lr')}")]
    lines = ['# DDPG 训练监测', '', f'运行目录：`{run}`', '', '| 项目 | 数值 |', '|---|---|']
    lines += [f'| {k} | {v if v is not None else "暂无数据"} |' for k,v in values]
    lines += ['', '![训练曲线](curves.png)', '',
        'Actor loss = -mean Q，可为负；Critic loss = Bellman target均方误差。仅统计真实更新行，warmup时Actor loss缺失是正常的。',
        '滚动均值在BC恢复、阶段或学习率变化时重新计算。尖峰只是排查提示，不等于发散。NaN/Inf或非法梯度会停止新的训练更新并写入失败状态。',
        '训练回合位移不是独立Evaluation；历史日志缺少的Q/梯度/回合位移不会补造。', '',
        f"异常/提示记录：{len(data['alerts'])}，详见 monitor.json。", '',
        '## 实际参数', '', '```json', json.dumps(cfg['training'], indent=2), '```']
    evaluations = []
    for path in sorted(run.parent.glob('straight_eval_*/effective_config.json')):
        item = read_json(path, {})
        cp = item.get('options', {}).get('checkpoint')
        if cp and Path(cp).resolve().parent == run/'checkpoints' and (path.parent/'result.json').exists():
            evaluations.append(evaluation_report(path.parent))
    lines += ['', '## 独立评估', '']
    lines += [f'- [{p.parent.name}]({p.as_posix()})' for p in evaluations] if evaluations else ['尚无与本次训练Checkpoint对应的独立Evaluation。']
    write_text(out/'TRAINING.md', '\n'.join(lines)+'\n')
    cells = ''.join(f'<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v)) if v is not None else "暂无数据"}</td></tr>' for k,v in values)
    notices = html.escape(json.dumps(data['alerts'][-10:], ensure_ascii=False, indent=2))
    refresh = '<meta http-equiv="refresh" content="15">' if status['status'] == 'running' else ''
    write_text(out/'index.html', f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">{refresh}
<title>DDPG训练监测</title><style>body{{font:16px system-ui;max-width:1080px;margin:32px auto;padding:0 20px;background:#f5f7fa;color:#182230}}table{{border-collapse:collapse;background:white;width:100%}}th,td{{padding:9px;text-align:left;border-bottom:1px solid #ddd}}img{{width:100%;margin-top:20px}}pre{{white-space:pre-wrap}}h1{{font-size:26px}}</style>
<h1>DDPG 训练监测</h1><p>{html.escape(run.name)}</p><table>{cells}</table><img src="curves.png?{time.time()}" alt="训练曲线">
<p>Actor loss 可以为负。loss 下降不代表走得更好；训练回合数据不替代独立评估。</p>
<p><a href="TRAINING.md">训练报告</a> · <a href="monitor.json">详细监测数据</a></p><h2>最近提示</h2><pre>{notices}</pre></html>''')
    return out/'index.html'
