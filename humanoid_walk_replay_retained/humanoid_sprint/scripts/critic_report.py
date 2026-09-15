"""Observe logged Critic update losses without modifying training or Evaluation."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def describe(values):
    if not values:return dict(count=0,mean=None,median=None,minimum=None,maximum=None)
    return dict(count=len(values),mean=statistics.mean(values),median=statistics.median(values),
                minimum=min(values),maximum=max(values))


def analyze(rows):
    updates=[];resets=[];groups={};history=[];previous_group=None
    missing=nonfinite=negative=malformed=0
    segment=0;stage_segment=0
    for row in rows:
        if row.get('event_name')=='reset' and row.get('bc_reset_applied')=='True':
            resets.append(int(row['env_steps']));segment+=1;history=[]
        if row.get('type')!='update':continue
        group=(segment,row.get('training_stage') or 'unknown',row.get('actor_lr') or 'unknown',row.get('critic_lr') or 'unknown')
        if group!=previous_group:history=[];stage_segment+=1
        previous_group=group
        text=row.get('critic_loss','')
        invalid=None;loss=None
        if text is None or not str(text).strip():missing+=1;invalid='missing'
        else:
            try:loss=float(text)
            except (ValueError,TypeError):malformed+=1;invalid='malformed'
            if invalid is None:
                if not math.isfinite(loss):nonfinite+=1;invalid='nonfinite'
                elif loss<0:negative+=1;invalid='negative_mse'
        if invalid:loss=None
        baseline=statistics.median(history[-20:]) if len(history)>=5 else None
        spike=loss is not None and baseline is not None and loss>10*max(baseline,1e-12)
        if loss is not None:history.append(loss)
        else:history=[]
        point=dict(env_steps=int(row['env_steps']),global_step=row.get('global_step'),
            critic_loss=loss,invalid=invalid,training_stage=group[1],actor_lr=group[2],critic_lr=group[3],
            bc_segment=segment,stage_segment=stage_segment,rolling_mean_20=statistics.mean(history[-20:]) if loss is not None else None,
            spike=spike)
        updates.append(point)
        key=group[1:]
        if loss is not None:groups.setdefault(key,[]).append(loss)
    valid=[r for r in updates if r['critic_loss'] is not None]
    values=[r['critic_loss'] for r in valid]
    # Compare early/late windows only inside the final uninterrupted stage/LR/BC segment.
    tail=[]
    for r in reversed(updates):
        if not updates:break
        last=updates[-1]
        if last['bc_segment']!=segment:break
        if any(r[k]!=last[k] for k in ['bc_segment','training_stage','actor_lr','critic_lr']):break
        if r['critic_loss'] is not None:tail.append(r['critic_loss'])
    tail.reverse();window=min(20,len(tail)//2)
    trend=dict(segment_updates=len(tail),window=window,
        first_mean=statistics.mean(tail[:window]) if window else None,
        last_mean=statistics.mean(tail[-window:]) if window else None)
    return dict(update_rows=len(updates),missing=missing,nonfinite=nonfinite,negative=negative,malformed=malformed,
        overall=describe(values),last_valid=valid[-1] if valid else None,
        peak=max(valid,key=lambda r:r['critic_loss']) if valid else None,
        stages=[dict(training_stage=k[0],actor_lr=k[1],critic_lr=k[2],**describe(v)) for k,v in groups.items()],
        final_segment_trend=trend,spikes=[r for r in updates if r['spike']],bc_reset_steps=resets,updates=updates)


def add_critic_section(log_path,out):
    log_path=Path(log_path);out=Path(out)
    lines=['','## Critic loss 观测（训练日志）','']
    if not log_path.exists():return lines+['训练日志不可用，Critic loss 待补。']
    # One immutable in-memory read: source hash and statistics refer to the same snapshot.
    raw=log_path.read_bytes()
    data=analyze(list(csv.DictReader(raw.decode('utf-8-sig').splitlines())))
    data.update(source=str(log_path.resolve()),source_sha256=hashlib.sha256(raw).hexdigest())
    (out/'critic_loss.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    if data['updates']:
        with (out/'critic_updates.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data['updates'][0]));writer.writeheader();writer.writerows(data['updates'])
    def n(value):return '—' if value is None else f'{value:.6g}'
    lines+=['只统计 `type=update` 的 critic_loss；不重复计算 progress / reset 行，也不把缺失值当成0。',
        '当前实现是训练批次上的 Q 与 Bellman target 的均方误差；一轮含多次 Critic 优化时，日志取这些优化 loss 的平均值。它与固定 BatchNorm 模式下另外计算的 td_mse 不保证相同。','',
        '| 指标 | 数值 |','|---|---:|',
        f"| 更新行 / 有效 loss | {data['update_rows']} / {data['overall']['count']} |",
        f"| 均值 / 中位数 | {n(data['overall']['mean'])} / {n(data['overall']['median'])} |",
        f"| 最小值 / 最大值 | {n(data['overall']['minimum'])} / {n(data['overall']['maximum'])} |",
        f"| 缺失 / 非数值 / NaN或Inf / 负MSE | {data['missing']} / {data['malformed']} / {data['nonfinite']} / {data['negative']} |"]
    if data['last_valid']:
        last=data['last_valid'];peak=data['peak']
        lines+=[f"| 最后有效值（env step） | {n(last['critic_loss'])}（{last['env_steps']}） |",
                f"| 最大值所在 env step | {peak['env_steps']} |"]
    lines+=['','### 分训练阶段和学习率','',
        '| 阶段 | Actor LR | Critic LR | 有效更新数 | loss均值 | loss中位数 |',
        '|---|---:|---:|---:|---:|---:|']
    for s in data['stages']:lines.append(f"| {s['training_stage']} | {s['actor_lr']} | {s['critic_lr']} | {s['count']} | {n(s['mean'])} | {n(s['median'])} |")
    trend=data['final_segment_trend']
    lines+=['','### 趋势与异常','',
        f"发现 {len(data['spikes'])} 个局部尖峰候选：比同一BC恢复区间、同阶段/学习率下此前最多20次有效更新的中位数高10倍；至少先有5个样本，分母下限1e-12。这是排查提示，不是发散判定。"]
    if trend['window']>=5:
        lines.append(f"最后一个连续BC恢复区间/阶段/学习率组合中，前{trend['window']}次均值 {n(trend['first_mean'])}，后{trend['window']}次均值 {n(trend['last_mean'])}。两窗口不重叠。")
    else:lines.append('最后连续阶段的有效更新不足10次，暂不判断该阶段loss趋势。')
    if data['nonfinite'] or data['negative'] or data['malformed']:
        lines.append('存在非有限值、负MSE或无法解析的loss，需要检查对应更新记录；这些值不参与均值统计。')
    if data['overall']['count']:
        plot(data,out/'critic_loss.png');lines+=['','![Critic loss](critic_loss.png)']
    lines+=['','曲线为逐更新原始loss和最多20次更新的滚动均值；在BC恢复、训练阶段或学习率变化处重新统计滚动窗口。虚线标记BC恢复。',
        'loss下降仅表示当前训练批次的目标拟合误差减小，不证明Q估值准确、策略稳定或走得更好；loss尖峰也可能与跌倒样本、Replay变化、BC恢复或奖励尺度有关，不能仅凭时间接近确定原因。',
        '增大方向Reward权重会改变target Q的尺度和分布，因此不同奖励配置的绝对loss大小不能直接当作优劣分数。Evaluation完整报告里的本节仍引用训练日志，不是在Evaluation中重新训练或计算Critic loss。']
    return lines


def plot(data,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    groups={}
    for r in data['updates']:
        key=r['stage_segment']
        groups.setdefault(key,[]).append(r)
    for i,rows in enumerate(groups.values()):
        x=[r['env_steps'] for r in rows]
        ax.plot(x,[r['critic_loss'] if r['critic_loss'] is not None else math.nan for r in rows],color='#84b3d2',alpha=.7,lw=.8,label='Update loss' if i==0 else None)
        ax.plot(x,[r['rolling_mean_20'] if r['rolling_mean_20'] is not None else math.nan for r in rows],color='#165d91',lw=1.5,label='Rolling mean (up to 20 updates)' if i==0 else None)
    for i,step in enumerate(data['bc_reset_steps']):ax.axvline(step,color='#b97028',lw=.7,ls='--',alpha=.6,label='BC recovery' if i==0 else None)
    low=data['overall']['minimum'];high=data['overall']['maximum']
    if low>0:
        ax.set_yscale('log');ax.set_ylim(low*.8,high*1.25);scale='log'
    else:
        ax.set_yscale('symlog',linthresh=1e-8);ax.set_ylim(0,max(high*1.25,1e-8));scale='symlog'
    ax.set(xlabel='Environment steps',ylabel=f'Critic loss (MSE, {scale})',title='Training Critic loss: lower loss alone does not imply better walking')
    ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(path,dpi=150);plt.close(fig)
