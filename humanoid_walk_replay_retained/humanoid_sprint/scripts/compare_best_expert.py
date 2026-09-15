"""Fresh-world, matched evaluation of report-selected models, BC and rule Expert."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'controllers'), str(ROOT/'controllers/common'), str(ROOT/'scripts')]


def simulate(supervisor, cfg, out, checkpoint):
    from configs.runtime import runtime_settings, task_settings, create_actor
    from tasks import load_task
    from checkpointing import load_actor_weights
    from experts.strong_heading import StrongHeadingExpertPolicy
    from envs.nao_locomotion_env import NaoLocomotionEnv
    from webots_platform.webots_runtime import WebotsRuntime
    from eval_stall import EvalStallMonitor
    torch.manual_seed(cfg['experiment']['seed'])
    if checkpoint:
        policy = create_actor(cfg)
        load_actor_weights(policy, checkpoint, torch.device('cpu')); policy.eval()
    else:
        policy = StrongHeadingExpertPolicy(); policy.reset()
    task = load_task(cfg['environment']['task_module'], cfg['environment']['task_class'], task_settings(cfg, 'eval'))
    env = NaoLocomotionEnv(WebotsRuntime(supervisor, **runtime_settings(cfg, 'eval')), task)
    env.initialize(); obs = env.reset(cfg['environment']['settle_steps'])
    initial = np.array(env.position)
    positions, states, actions = [initial], [obs.values], []
    e = cfg['evaluation']; dt = cfg['webots']['control_time_step_ms']/1000
    stall = EvalStallMonitor(initial[0], window_steps=e['stall_window_steps'],
        cycle_limit=e['stall_cycle_limit'], max_abs_progress_m=e['stall_max_abs_x_progress_m'])
    limit = int(e['max_steps']*e['first_episode_step_fraction'])
    reason = 'timeout'; fell = False; reached = False
    for i in range(limit):
        with torch.no_grad():
            chosen = policy(obs.values)[0].numpy() if checkpoint else policy.act(obs)
        step = env.step(chosen, i); obs = step.next_observation
        actions.append(step.action.values); states.append(obs.values); positions.append(np.array(env.position))
        fell = bool(step.info['fell']); reached = bool(step.info['reached_goal'])
        if reached: reason='goal'; break
        if fell: reason='fall'; break
        if stall.observe(env.position[0]): reason='stall'; break
    delta = positions[-1]-initial
    result = dict(time_s=len(actions)*dt if reached else None, elapsed_s=len(actions)*dt,
        lateral_m=float(delta[1]), abs_lateral_m=float(abs(delta[1])), forward_m=float(delta[0]),
        success=reached, fell=fell, end_reason=reason, steps=len(actions),
        initial_position=initial.tolist(), final_position=positions[-1].tolist(),
        final_heading_error_deg=float(np.degrees(np.arctan2(obs.values[24],obs.values[25]))),
        checkpoint=checkpoint, checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() if checkpoint else None)
    out = Path(out)
    np.savez_compressed(out/'trajectory.npz', positions=positions, states=states, actions=actions)
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


def main():
    from datetime import datetime, timezone
    from webots_trials import prepare_driver, execute
    from configs.runtime import project_path
    report = max(ROOT.glob('experiments/*/seed_001/reports/*/comparison.json'),key=lambda p:p.stat().st_mtime)
    summary = json.loads(report.read_text(encoding='utf-8-sig'))
    assert summary['status']=='completed'
    batch = Path(summary['batch']).parent
    cfg = json.loads(next((batch/f"gs{summary['selection']['fastest']['global_step']:06d}").glob('effective_config.json')).read_text(encoding='utf-8-sig'))['config']
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=ROOT/'experiments'/summary['experiment']/'analysis'/('best_vs_expert_'+stamp)
    out.mkdir(parents=True)
    variants=[]
    for key in ('fastest','straightest'):
        row=summary['selection'][key]
        checkpoint=ROOT/'experiments'/summary['experiment']/'seed_001/checkpoints/eval_candidates'/row['checkpoint']
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==row['checkpoint_sha256']
        variants.append((key,str(checkpoint)))
    variants.extend([('bc',str(project_path(cfg['initialization']['bc_checkpoint']))),('expert',None)])
    manifest=dict(source_report=str(report),source_report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
        original_selection=summary['selection'],conditions=summary['conditions'],
        protocol='One fresh Webots process per policy, same initial condition, noise off, original first-episode cap; no training.',
        results={},status='running')
    (out/'comparison.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('OUTPUT',out,flush=True)
    for label,checkpoint in variants:
        folder=out/label;folder.mkdir()
        config=copy.deepcopy(cfg)
        config['experiment'].update(name=label,output_root=str(out/'runs'))
        config['initialization'].update(use_bc=False,reset_to_bc_on_failure=False,resume='never')
        driver=f'''from controller import Supervisor
sys.path[:0] = {[str(ROOT),str(ROOT/'controllers'),str(ROOT/'controllers/common'),str(ROOT/'scripts')]!r}
from compare_best_expert import simulate
supervisor=Supervisor()
try:
    simulate(supervisor,{config!r},{str(folder)!r},{checkpoint!r})
except BaseException:
    import traceback
    status.write_text(json.dumps(dict(status='failed',error=traceback.format_exc())),encoding='utf-8')
    supervisor.simulationQuit(1)
    raise
else:
    status.write_text(json.dumps(dict(status='completed')),encoding='utf-8')
    supervisor.simulationQuit(0)
'''
        # Reuse isolated-world scaffolding only; the driver above has no learner.
        snapshot,world,payload=prepare_driver(config,'train',driver)
        payload['purpose']='matched_policy_evaluation_only'
        snapshot.write_text(json.dumps(payload,indent=2),encoding='utf-8')
        execute(snapshot,world,payload)
        result=json.loads((folder/'result.json').read_text())
        result['snapshot']=str(snapshot)
        manifest['results'][label]=result
        (out/'comparison.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        print(label,json.dumps(result),flush=True)
    manifest['status']='completed'
    (out/'comparison.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4))
    for label,_ in variants:
        data=np.load(out/label/'trajectory.npz');p=data['positions'];p=p-p[0]
        ax.plot(p[:,0],p[:,1],label=label)
    ax.set(xlabel='Net forward (m)',ylabel='Signed lateral displacement (m)',title='Matched policy evaluation')
    ax.legend();ax.grid(alpha=.3);fig.tight_layout();fig.savefig(out/'trajectories.png',dpi=160)
    print('COMPLETED',out,flush=True)


if __name__=='__main__':
    main()
