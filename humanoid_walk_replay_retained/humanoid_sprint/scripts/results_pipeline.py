"""Training -> quick report; independent Evaluation -> full comparison report."""
import argparse
import copy
import csv
from datetime import datetime, timezone
import html
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from configs.runtime import load_config, validate, file_hash, project_path
from launch import experiment_root, prepare_evaluation, run_evaluation_batch, validate_models
from webots_trials import prepare_driver, execute
from critic_report import add_critic_section

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def completed_training(config):
    """Do not overwrite or silently resume existing training."""
    root=experiment_root(config)
    runs=sorted((root/'runs').glob('train_*/effective_config.json'))
    if not runs:
        return None
    snapshot=runs[-1];payload=read(snapshot)
    status=snapshot.parent/'run_status.json'
    if not status.exists() or read(status).get('status')!='completed':
        raise ValueError('现有训练尚未正常完成；请先处理原训练，避免并行覆盖。')
    old=validate(copy.deepcopy(payload['config']));new=validate(copy.deepcopy(config))
    for item in (old,new):
        item['webots'].pop('world_template',None)
        # Evaluation selection can be changed without changing the completed training.
        item.pop('evaluation',None)
    if old!=new or payload['source_world_sha256']!=file_hash(project_path(config['webots']['world_template'])):
        raise ValueError('当前训练设置与已完成版本不同。新训练请换实验名；仅评估请用 eval-compare。')
    rows=list(csv.DictReader((root/'metrics/train/metrics.csv').open(encoding='utf-8-sig')))
    ends=[r for r in rows if r.get('event_name')=='session_end']
    if not ends or int(ends[-1]['env_steps'])!=config['training']['total_env_steps']:
        raise ValueError('未找到达到训练预算的完成记录，不能自动进入评估。')
    return snapshot

def train(config):
    existing=completed_training(config)
    if existing:
        print('训练已完成，保留现有模型，继续 Evaluation。',flush=True)
        return existing
    entry=ROOT/'controllers/sprinter/sprinter.py'
    driver=f'''import importlib.util
spec=importlib.util.spec_from_file_location('pipeline_train',{str(entry)!r})
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
instances=[]
original=module.Sprinter
class TrackedSprinter(original):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        instances.append(self)
module.Sprinter=TrackedSprinter
try:
    module.main()
except BaseException:
    import traceback
    status.write_text(json.dumps(dict(status='failed',error=traceback.format_exc())),encoding='utf-8')
    if instances: instances[0].simulationQuit(1)
    raise
else:
    status.write_text(json.dumps(dict(status='completed')),encoding='utf-8')
    instances[0].simulationQuit(0)
'''
    prepared=prepare_driver(config,'train',driver)
    validate_models(prepared[2]);execute(*prepared)
    return completed_training(config)

def latest_batch(config):
    paths=sorted((experiment_root(config)/'runs').glob('eval_batch_*/batch_status.json'))
    return paths[-1] if paths else None

def environment_signature(config):
    return dict(environment=config['environment'],
        webots={k:config['webots'][k] for k in ['initial_translation','initial_rotation','basic_time_step_ms','control_time_step_ms']},
        world_sha256=file_hash(project_path(config['webots']['world_template'])))

def load_batch(path,expected_experiment=None):
    manifest=read(path)
    if manifest.get('status')!='completed' or not manifest.get('runs'):
        raise ValueError('最新 Evaluation 未完整完成；不能用旧批次或部分结果冒充完成结果。')
    rows=[];signature=None;seen=set()
    for run in manifest['runs']:
        snapshot=read(run['snapshot']);result=read(Path(run['snapshot']).parent/'evaluation_result.json')
        cfg=snapshot['config'];name=cfg['experiment']['name']
        if expected_experiment is not None and name!=expected_experiment:
            raise ValueError('Evaluation 批次不属于所选实验')
        if (run['status']!='completed' or result['eval_run_id']!=snapshot['run_id']
                or result['checkpoint_sha256']!=run['sha256']
                or snapshot['sources'][0]['sha256']!=run['sha256']
                or file_hash(run['checkpoint'])!=run['sha256']):
            raise ValueError('Evaluation 结果与 Checkpoint / 快照不匹配')
        episodes=result['episodes']
        if len(episodes)!=1:
            raise ValueError('本入口按当前每模型单次 Evaluation 协议比较，拒绝混合多回合均值。')
        e=episodes[0];key=(run['global_step'],run['sha256'])
        if key in seen: raise ValueError('重复 Checkpoint 结果')
        seen.add(key)
        conditions=dict(**environment_signature(cfg),evaluation={k:v for k,v in cfg['evaluation'].items() if k not in ['selected_global_step','only_latest','start_global_step','metrics_filename']},
                        budget=cfg['training']['total_env_steps'],seed=cfg['experiment']['seed'])
        if signature is not None and conditions!=signature: raise ValueError('同一批次的评估条件不一致')
        signature=conditions
        success=bool(e['reached']) and not e['fell']
        time=e['reach_time'] if success else None
        for value in [e['forward'],e['lateral']]+([time] if success else []):
            if value is None or not math.isfinite(float(value)): raise ValueError('评估指标缺失或非有限数')
        rows.append(dict(checkpoint=Path(run['checkpoint']).name,global_step=run['global_step'],
            time_s=time,lateral_m=float(e['lateral']),abs_lateral_m=abs(float(e['lateral'])),
            forward_m=float(e['forward']),success=success,fell=bool(e['fell']),
            end_reason=e['end_reason'],checkpoint_sha256=run['sha256'],eval_run_id=result['eval_run_id']))
    return rows,signature

def select(rows):
    success=[r for r in rows if r['success'] and not r['fell']]
    return dict(fastest=min(success,key=lambda r:(r['time_s'],r['abs_lateral_m'],r['global_step'])) if success else None,
                straightest=min(success,key=lambda r:(r['abs_lateral_m'],r['time_s'],r['global_step'])) if success else None)

def references(config):
    # Reference artifacts are explicitly tied to the current initialization, not silently replaced by a newer result.
    design=project_path(config['initialization']['bc_checkpoint']).parent
    refs=[]
    for name,label in [('bc_baseline','BC baseline'),('verify_strong','Strong Expert')]:
        path=design/'trials'/name/'result.json'
        if not path.exists(): continue
        data=read(path);arrival=data.get('arrival_x5')
        if not arrival: continue
        if name=='bc_baseline' and data.get('checkpoint_sha256')!=file_hash(project_path(config['initialization']['bc_checkpoint'])):
            continue
        snapshots=sorted((design/'trial_runs'/name).glob('seed_*/runs/train_*/effective_config.json'))
        signature=environment_signature(read(snapshots[-1])['config']) if snapshots else None
        refs.append(dict(label=label,source=str(path),stop_x_m=5.,initial_yaw_rad=data['initial_yaw_rad'],environment_signature=signature,
                         **arrival,success=bool(data['reached']),fell=bool(data['fell'])))
    baseline=ROOT/'baselines/pre_s1_20260724_v1/pre_s1_20260724.csv'
    if baseline.exists():
        if file_hash(baseline).lower()!='463cf1723419b789e84209efe3d326dbf16d20e7583a45f11d03cb5c18d6f482':
            raise ValueError('固定 baseline 校验值发生变化')
        for r in csv.DictReader(baseline.open(encoding='utf-8-sig')):
            refs.append(dict(label='pre_s1_20260724_v1 / gs'+r['global_step'],source=str(baseline),stop_x_m=4.5,
                time_s=float(r['reach_time_mean']),lateral_m=float(r['lateral_mean']),abs_lateral_m=abs(float(r['lateral_mean'])),
                forward_m=float(r['forward_mean']),success=float(r['eval_success_rate'])==1.,fell=float(r['fell_rate'])>0))
    return refs

def report(config,batch=None):
    chosen=Path(batch) if batch else latest_batch(config)
    manifest=read(chosen) if chosen else None
    ready=bool(manifest and manifest.get('status')=='completed')
    rows,conditions=load_batch(chosen,config['experiment']['name']) if ready else ([],None)
    selected=select(rows);refs=references(config)
    previous_name=config['experiment']['previous_experiment']
    previous=dict(experiment=previous_name,status='待比较：上一版独立 Evaluation 不可用；保留版本身份，不用旧 BC 代替。')
    if previous_name:
        old_cfg=copy.deepcopy(config);old_cfg['experiment']['name']=previous_name
        old_batch=latest_batch(old_cfg)
        if old_batch and read(old_batch).get('status')=='completed':
            old_rows,old_conditions=load_batch(old_batch,previous_name)
            previous.update(status='条件相同，可比较' if conditions==old_conditions else '评估条件或训练预算不同，仅作参考',
                            batch=str(old_batch),selected=select(old_rows))
    identifier=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'_'+uuid.uuid4().hex[:6]
    out=experiment_root(config)/'reports'/identifier;out.mkdir(parents=True,exist_ok=False)
    data=dict(experiment=config['experiment']['name'],batch=str(chosen) if chosen else None,
              status='completed' if ready else 'pending_evaluation',batch_status=manifest.get('status') if manifest else None,conditions=conditions,selection=selected,
              previous=previous,references=refs,results=rows,
              selection_rule='Successful non-fallen models only; fastest by time then abs lateral, straightest by abs lateral then time; final tie by update.')
    write(out/'comparison.json',data)
    if rows:
        with (out/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    def number(value): return '—' if value is None else f'{float(value):.4f}'
    def row(label,r):
        return f"| {label} | {number(r.get('time_s'))} | {number(r['lateral_m'])} | {number(r['abs_lateral_m'])} | {number(r['forward_m'])} | {int(r['success'])}/1 | {int(r['fell'])}/1 |"
    lines=[f"# {config['experiment']['name']} 测评与比较",'',
        '## 当前结果','', (f"Evaluation 尚未完整完成（状态：{manifest.get('status') if manifest else '尚未运行'}）；以下 baseline 不代表当前 DDPG 成绩。" if not ready else f'批次：{chosen.parent.name}。共 {len(rows)} 个 Checkpoint，到达 {sum(r["success"] for r in rows)}/{len(rows)}，跌倒 {sum(r["fell"] for r in rows)}/{len(rows)}。'),
        '', '所有结果均绑定同一 Checkpoint 的同次 Evaluation。失败模型不参与最快/最小偏移排序。不同 Checkpoint 的到达比例不是同一模型的重复成功率。','',
        '| 模型 | 到达时间(s) | 带符号横移(m) | 绝对横移(m) | 净前进(m) | 成功 | 跌倒 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for label,key in [('最快','fastest'),('偏移最小','straightest')]:
        if selected[key]: lines.append(row(label+' '+selected[key]['checkpoint'],selected[key]))
    if conditions:
        lines+=['',f"评估条件：seed={conditions['seed']}，训练预算={conditions['budget']}步；终点x={conditions['evaluation']['stop_x_m']}m，物理/控制周期={conditions['webots']['basic_time_step_ms']}/{conditions['webots']['control_time_step_ms']}ms；初始位置={conditions['webots']['initial_translation']}，初始旋转={conditions['webots']['initial_rotation']}；每Checkpoint独立1次、Evaluation无探索噪声。"]
    lines+=['','## 比较对象','',f"直接上一版：{previous_name}。{previous['status']}",'']
    if previous.get('selected'):
        lines+=['','| 上一版模型 | 到达时间(s) | 带符号横移(m) | 绝对横移(m) | 净前进(m) | 成功 | 跌倒 |',
                '|---|---:|---:|---:|---:|---:|---:|']
        for label,r in previous['selected'].items():
            if r:lines.append(row(label+' '+r['checkpoint'],r))
        lines.append('')
        if previous['status']=='条件相同，可比较':
            for key in ('fastest','straightest'):
                a,b=selected[key],previous['selected'][key]
                if a and b: lines.append(f"{key} 当前减上一版：时间 {number(a['time_s']-b['time_s'])} s；绝对横移 {number(a['abs_lateral_m']-b['abs_lateral_m'])} m。")
    lines+=['','当前 BC / Expert baseline 为单次 x=5 记录；只在起始姿态、仿真条件及终点一致时作直接参考。',
        '固定 pre_s1_20260724_v1 为 x=4.5，历史训练预算未对齐。当前常规 x=5，故不计算跨终点的时间优劣，也不自动声称超过固定 baseline。','',
        '| 参照 | 到达时间(s) | 带符号横移(m) | 绝对横移(m) | 净前进(m) | 成功 | 跌倒 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in refs:lines.append(row(r['label'],r))
    if conditions:
        lines+=['','### 当前模型与 BC / Expert 的差值','']
        for reference in refs:
            if 'environment_signature' not in reference:continue
            comparable=(reference['environment_signature']=={k:conditions[k] for k in ['environment','webots','world_sha256']}
                        and conditions['evaluation']['stop_x_m']==reference['stop_x_m'])
            if not comparable:
                lines.append(reference['label']+'：环境或终点未确认一致，暂不计算直接差值。');continue
            for key in ('fastest','straightest'):
                r=selected[key]
                if r:lines.append(f"{key} {r['checkpoint']} 减 {reference['label']}：用时 {number(r['time_s']-reference['time_s'])} s，绝对横移 {number(r['abs_lateral_m']-reference['abs_lateral_m'])} m。")
        lines.append('负值表示该指标减小；这是单次参考比较，不代表重复稳定性。')
    lines+=['','## 全部 Checkpoint','',
        '| 模型 | 到达时间(s) | 带符号横移(m) | 绝对横移(m) | 净前进(m) | 成功 | 跌倒 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows: lines.append(row(r['checkpoint']+(' ('+r['end_reason']+')' if not r['success'] else ''),r))
    lines+=add_critic_section(experiment_root(config)/'metrics/train/metrics.csv',out)
    save_document(out,lines)
    write(experiment_root(config)/'latest_comparison.json',dict(report=str(out.resolve()),status=data['status'],batch=data['batch']))
    print('完整比较报告：'+str(out/'report.html'),flush=True)
    return out

def save_document(out,lines):
    text='\n'.join(lines)+'\n';(out/'REPORT.md').write_text(text,encoding='utf-8')
    # Self-contained, readable without installing a Markdown viewer.
    body=[];in_table=False
    for line in lines:
        if line.startswith('|---'):continue
        if line.startswith('|'):
            if not in_table:body.append('<div class="scroll"><table>');in_table=True
            body.append('<tr>'+''.join('<td>'+html.escape(c.strip())+'</td>' for c in line.strip('|').split('|'))+'</tr>');continue
        if in_table:body.append('</table></div>');in_table=False
        if line=='![Critic loss](critic_loss.png)':
            body.append('<figure><img src="critic_loss.png" alt="Critic loss" style="max-width:100%;height:auto"></figure>');continue
        if line.startswith('# '):body.append('<h1>'+html.escape(line[2:])+'</h1>')
        elif line.startswith('## '):body.append('<h2>'+html.escape(line[3:])+'</h2>')
        elif line.startswith('### '):body.append('<h3>'+html.escape(line[4:])+'</h3>')
        elif line:body.append('<p>'+html.escape(line)+'</p>')
    if in_table:body.append('</table></div>')
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>训练测评比较</title><style>body{font:16px/1.6 system-ui;max-width:1150px;margin:40px auto;padding:0 20px;color:#172434}h1{font-size:26px}h2{margin-top:32px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%}td{padding:9px 12px;border-bottom:1px solid #ddd;white-space:nowrap}tr:first-child{background:#eaf0f7;font-weight:bold}p{overflow-wrap:anywhere}</style>'+''.join(body)+'</html>'
    (out/'report.html').write_text(page,encoding='utf-8')

def training_summary(path):
    rows=list(csv.DictReader(Path(path).open(encoding='utf-8-sig')))
    events=[r for r in rows if r.get('env_steps')]
    if not events:raise ValueError('训练日志为空，无法生成快速报告。')
    progress=[r for r in events if r['type']=='progress']
    resets=[r for r in events if r['event_name']=='reset']
    end=int(events[-1]['env_steps'])
    def mean_angle(samples):
        values=[abs(math.degrees(float(r['heading_error_rad']))) for r in samples if r.get('heading_error_rad')]
        return sum(values)/len(values) if values else None
    bins=[]
    for lo in range(0,max(end,1),10000):
        samples=[r for r in progress if lo<int(r['env_steps'])<=lo+10000]
        bins.append(dict(start_step=lo,end_step=min(lo+10000,end),samples=len(samples),mean_abs_heading_deg=mean_angle(samples)))
    last=events[-1]
    return dict(source=str(Path(path).resolve()),source_sha256=file_hash(path),env_steps=end,
        session_end=last.get('event_name')=='session_end',falls=sum(r.get('fell')=='True' for r in resets),
        goals=sum(r.get('reached_goal')=='True' for r in resets),
        stalls=sum(r.get('reason')=='stall_timeout_steps' for r in resets),
        bc_resets=sum(r.get('bc_reset_applied')=='True' for r in resets),
        last_fall_step=max([int(r['env_steps']) for r in resets if r.get('fell')=='True'] or [0]),
        last_sample_abs_heading_deg=abs(math.degrees(float(last['heading_error_rad']))) if last.get('heading_error_rad') else None,
        recent_mean_abs_heading_deg=mean_angle([r for r in progress if int(r['env_steps'])>max(0,end-10000)]),
        final_segment_estimated_forward_m=float(last['estimated_forward_m']) if last.get('estimated_forward_m') else None,
        final_stalled_steps=int(last['stalled_steps']) if last.get('stalled_steps') else None,bins=bins)

def quick_report(config):
    root=experiment_root(config);summary=training_summary(root/'metrics/train/metrics.csv')
    identifier=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'_'+uuid.uuid4().hex[:6]
    out=root/'training_reports'/identifier;out.mkdir(parents=True,exist_ok=False)
    previous_name=config['experiment']['previous_experiment'];previous=None
    if previous_name:
        old_cfg=copy.deepcopy(config);old_cfg['experiment']['name']=previous_name
        source=experiment_root(old_cfg)/'metrics/train/metrics.csv'
        if not source.exists():
            # Only the archived direct previous identity is eligible for this fallback.
            old_config=root.parent/'design/previous_config.json'
            fallback=root.parent/'analysis/training_heading/reference/previous_metrics.csv'
            if old_config.exists() and read(old_config)['experiment']['name']==previous_name and fallback.exists():source=fallback
        if source.exists():previous=training_summary(source)
    done=summary['session_end'] and summary['env_steps']==config['training']['total_env_steps']
    write(out/'training_summary.json',dict(experiment=config['experiment']['name'],status='completed' if done else 'incomplete',current=summary,previous_experiment=previous_name,previous=previous))
    def n(value):return '—' if value is None else f'{value:.3f}'
    lines=[f"# {config['experiment']['name']} 训练快速报告",'',
        f"状态：{'训练完成' if done else '训练尚未完整结束'}；{summary['env_steps']} / {config['training']['total_env_steps']} 步。",'',
        '## 训练记录','', '| 指标 | 本轮 | 直接上一版 |','|---|---:|---:|']
    for label,key in [('跌倒次数','falls'),('到训练终点次数','goals'),('停滞重置次数','stalls'),('恢复BC次数','bc_resets'),('最近一万步平均绝对偏向角(°)','recent_mean_abs_heading_deg')]:
        lines.append(f"| {label} | {n(summary[key])} | {n(previous[key]) if previous else '待比较'} |")
    lines+=['',f"直接上一版身份：{previous_name}。"+('训练预算不同时，不把次数直接视作性能差值。' if previous else '原始训练日志不可用，保持待比较。'),'',
        f"最后一次跌倒发生于第 {summary['last_fall_step']} 步。" if summary['falls'] else '记录中没有跌倒。',
        f"最终行走段的前进估计为 {n(summary['final_segment_estimated_forward_m'])} m；停滞计数为 {summary['final_stalled_steps']} 步。这是里程计估计，不是真实净位移。",'',
        '## 朝向趋势','', '| 步数范围 | 采样数 | 平均绝对偏向角(°) |','|---|---:|---:|']
    for b in summary['bins']:lines.append(f"| {b['start_step']}–{b['end_step']} | {b['samples']} | {n(b['mean_abs_heading_deg'])} |")
    if previous and summary['recent_mean_abs_heading_deg'] is not None and previous['recent_mean_abs_heading_deg'] is not None:
        delta=summary['recent_mean_abs_heading_deg']-previous['recent_mean_abs_heading_deg']
        lines+=['',f"最近一万步的平均绝对偏向角较上一版{'降低' if delta<0 else '增加'} {abs(delta):.3f}°。"]
        if summary['falls']>previous['falls']:lines.append('同时本轮跌倒更多，不能只凭角度变小认定整体改善。')
    lines+=['','## 结论边界','',
        '这是训练过程快速诊断：Actor持续更新、有探索噪声、可能恢复BC；不是固定Checkpoint的Evaluation。角度是约35步一次的即时采样均值，停住也可能角度很小。',
        '训练日志缺少连续世界Y和每段精确起点，无法提供可信的横向净偏移及精确净前进。时间/横移/成功率/跌倒率的完整对比留到独立Evaluation结束。',
        '本入口不会自动启动Evaluation；需要时运行start_evaluation.cmd。']
    lines+=add_critic_section(root/'metrics/train/metrics.csv',out)
    save_document(out,lines);write(root/'latest_training_report.json',dict(report=str(out.resolve()),status='completed' if done else 'incomplete'))
    print('训练快速报告：'+str(out/'report.html'),flush=True);return out

def evaluation_process_alive(batch):
    process_file=Path(batch).parent/'webots_process.json'
    if not process_file.exists():return False
    pid=read(process_file)['webots_pid']
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=kernel.OpenProcess(0x1000,False,pid)
        if not handle:return False
        try:
            code=wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle,ctypes.byref(code))) and code.value==259
        finally:kernel.CloseHandle(handle)
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False

def wait_for_evaluation(batch):
    if not batch:raise ValueError('没有可等待的Evaluation批次。')
    batch=Path(batch);last_count=None
    while True:
        status=read(batch)
        if status.get('status')=='completed':return
        if status.get('status')!='running':raise ValueError('Evaluation未运行或失败，不能生成完成报告。')
        if not evaluation_process_alive(batch):
            # Re-read in case the final manifest was flushed as Webots exited.
            if read(batch).get('status')=='completed':return
            raise RuntimeError('Evaluation进程已退出，但批次未标记完成，请检查webots.log。')
        count=sum(r.get('status')=='completed' for r in status['runs'])
        if count!=last_count:
            print(f"等待Evaluation完成：{count}/{len(status['runs'])}",flush=True);last_count=count
        time.sleep(5)

def run(mode,config,batch=None):
    if mode=='full-report':return report(config,batch)
    if mode=='quick-report':return quick_report(config)
    identifier=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'_'+uuid.uuid4().hex[:6]
    folder=experiment_root(config)/'pipelines'/identifier;folder.mkdir(parents=True,exist_ok=False)
    status=dict(mode=mode,status='running',stage='start');write(folder/'config.json',config)
    def stage(name,**values):
        status.update(stage=name,**values);write(folder/'status.json',status)
    try:
        if mode=='train-report':
            stage('training');snapshot=train(config);stage('training_completed',training_snapshot=str(snapshot))
            stage('quick_report');out=quick_report(config)
            stage('completed',status='completed',report=str(out));return out
        stage('evaluation_preparing')
        existing=latest_batch(config)
        if existing and read(existing).get('status') in ('running','prepared'):
            raise ValueError('已有Evaluation正在运行或已准备，请勿重复启动。完成后可用show_evaluation_report.cmd生成完整报告。')
        batch=prepare_evaluation(config);stage('evaluating',batch=str(batch))
        run_evaluation_batch(batch)
        stage('reporting');out=report(config,batch)
        stage('completed',status='completed',report=str(out));return out
    except BaseException as error:
        stage(status['stage'],status='failed',error=str(error));raise

def main():
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['train-report','eval-report','quick-report','full-report'])
    parser.add_argument('--batch',type=Path,help='Specific batch_status.json for full-report')
    parser.add_argument('--open',action='store_true',help='Open generated HTML report')
    parser.add_argument('--wait',action='store_true',help='With full-report, wait for an already running Evaluation; never starts Evaluation')
    args=parser.parse_args()
    if args.batch and args.mode!='full-report':parser.error('--batch is only valid with full-report')
    if args.wait and args.mode!='full-report':parser.error('--wait is only valid with full-report')
    config=load_config()
    if args.wait:
        args.batch=args.batch or latest_batch(config)
        wait_for_evaluation(args.batch)
    out=run(args.mode,config,args.batch)
    if args.open:os.startfile(str(out/'report.html'))

if __name__=='__main__':main()
