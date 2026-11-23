"""Single-node multi-GPU Isaac collectors training one shared omni SAC policy."""
import argparse
import csv
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys

from multigpu_worker import worker

HERE = Path(__file__).resolve().parent
FIELDS = ('step rank local_step episode updates motion_phase vx_cmd vy_cmd wz_cmd actual_vx '
          'actual_vy actual_wz reward reward_fall reward_tracking_yaw reward_yaw_rate '
          'tilt_rad fell truncated actor_loss critic_loss entropy_coefficient').split()


def receive(pipe, rank, timeout):
    if not pipe.poll(timeout):
        raise TimeoutError(f'GPU {rank} did not respond within {timeout}s')
    try:
        message = pipe.recv()
    except EOFError as exc:
        raise RuntimeError(f'GPU {rank} worker exited; check its Kit log') from exc
    if isinstance(message, dict) and 'worker_error' in message:
        raise RuntimeError(f"GPU {rank}: {message['worker_error']}")
    return message


def local_curriculum(cfg, workers):
    if 'walk_commands' in cfg['training']:
        return  # The Walk sequence has no global curriculum boundaries.
    # Keep 200-step task blocks and 100-step speed holds on each individual robot.
    spec = cfg['training']['omni_curriculum']
    period = 2 * spec['block_steps']
    for stage in spec['walk_stages']:
        start = stage['start_step']
        stage['start_step'] = ((start + workers * period - 1) // (workers * period)) * period


def contract(cfg):
    keys = ('experiment_name', 'environment', 'training', 'sac', 'network_inputs', 'actor_observation',
            'critic_observation', 'reward', 'reward_source_sha256', 'seed', 'support_observation')
    physics = {k: v for k, v in cfg['isaac'].items() if k not in {
        'device', 'asset', 'urdf_source', 'nao_meshes_path', 'isaac_python', 'isaac_sim_path',
        'isaac_lab_path', 'standing_ground'}}
    return {**{k: cfg[k] for k in keys}, 'physics': physics, 'bc': cfg['initialization']['sha256'],
            'collectors': cfg['multi_gpu']['collectors']}


def train(cfg, folder, n, device, timeout, resume=None, worker_fn=worker, *, session=None, worker_folder=None):
    import numpy as np
    import torch
    from basewalk.omni_resume import normalize_legacy_walk_config, recovery_state, restore_agent
    from basewalk.phase_fall import PhaseFallBuffer
    from basewalk.runner import write_json
    from basewalk.sac import SAC

    agent = SAC(cfg, device=device)
    t = cfg['training']
    total = t['total_env_steps']
    step, counts, episodes = 0, [0]*n, [1]*n
    fresh = [True] * n
    buffers = [PhaseFallBuffer(cfg['reward']['fall_penalty'], cfg) for _ in range(n)]
    if resume:
        saved = torch.load(resume, map_location='cpu', weights_only=True)
        state = saved.get('recovery', {}).get('collectors')
        old = normalize_legacy_walk_config(saved['config'], cfg)
        previous_budget = old['training']['total_env_steps']
        comparable = dict(old, training=dict(old['training'], total_env_steps=total))
        if state is None or total < previous_budget or contract(comparable) != contract(cfg):
            raise ValueError('Choose a compatible multi-GPU recovery checkpoint with the same collector count')
        step = saved['env_steps']
        counts = state['counts']
        if sum(counts) != step or len(counts) != n or not 0 <= step < total:
            raise ValueError('Invalid or completed collector checkpoint')
        restore_agent(agent, saved)
        if session is not None:
            from run_storage import restore_csv
            restore_csv(Path(resume), folder, state['csv_bytes'], step, session)
        episodes = [episode if at_reset else episode + 1
                    for episode, at_reset in zip(state['episodes'], state['fresh'], strict=True)]
        for buf, partial in zip(buffers, state['pending'], strict=True):
            for obs, action, reward, nxt, fell, row in partial:
                buf.append(obs.numpy(), action.numpy(), reward, nxt.numpy(), fell, {}, row, None)
        write_json(folder/'resume.json', dict(checkpoint=str(Path(resume).resolve()), step=step,
                                              parent_csv_bytes=state['csv_bytes'], replay=agent.replay.count,
                                              physics='fresh episodes; unfinished segments closed at checkpoint'))
        del saved, state

    def save(csv_bytes):
        state = recovery_state(agent)
        pending = []
        for buf in buffers:
            partial = []
            for (obs, action, reward, nxt, fell, _), row, _ in buf.pending:
                partial.append((torch.as_tensor(obs), torch.as_tensor(action), float(reward),
                                torch.as_tensor(nxt), bool(fell), row))
            pending.append(partial)
        state['collectors'] = dict(counts=counts, episodes=episodes, fresh=fresh,
                                   pending=pending, csv_bytes=csv_bytes)
        agent.save(folder/'checkpoints/recovery_latest.pth', step, recovery=state)

    ctx = mp.get_context('spawn')
    pipes, processes = [], []
    try:
        for rank in range(n):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=worker_fn, args=(child, rank, cfg, worker_folder or folder, counts[rank]))
            proc.start()
            child.close()
            pipes.append(parent)
            processes.append(proc)
        observations = [receive(p, rank, timeout) for rank, p in enumerate(pipes)]
        write_json(folder/'status.json', dict(status='running', step=step, workers=n))
        append = resume is not None and session is not None
        with (folder/'training.csv').open('a' if append else 'x', newline='') as stream:
            writer = csv.DictWriter(stream, FIELDS)
            if not append:
                writer.writeheader()
            for buf in buffers:
                buf.flush(agent.replay, writer)
            while step < total:
                before = step
                active = min(n, total-step)
                with torch.no_grad():
                    obs = torch.as_tensor(np.stack(observations[:active]), dtype=torch.float32, device=device)
                    actions = agent.actor.sample(obs)[0].cpu().numpy()
                for rank in range(active):
                    pipes[rank].send((actions[rank], total - before <= n))
                for rank in range(active):
                    nxt, executed, row, wrapped, reset_obs = receive(pipes[rank], rank, timeout)
                    counts[rank] += 1
                    step += 1
                    agent.schedule(step)
                    row.update(step=step, rank=rank, local_step=counts[rank], episode=episodes[rank],
                               updates=agent.updates)
                    fell = row['fell']
                    buf = buffers[rank]
                    buf.append(observations[rank], executed, row['reward'], nxt, fell, {}, row, None)
                    if step >= agent.learning_starts and agent.replay.ready(t['batch_size']):
                        metrics = agent.update()
                        for key in ('updates', 'actor_loss', 'critic_loss', 'entropy_coefficient'):
                            row[key] = metrics[key]
                    if fell or row['truncated'] or wrapped or total-before <= n:
                        buf.flush(agent.replay, writer, fell=fell, complete_cycle=wrapped)
                    fresh[rank] = fell or row['truncated']
                    if fresh[rank]:
                        episodes[rank] += 1
                    observations[rank] = nxt if reset_obs is None else reset_obs
                if step == total:
                    for buf in buffers[active:]:
                        buf.flush(agent.replay, writer)
                if before//500 != step//500 or step == total:
                    stream.flush()
                    save(stream.buffer.tell())
                    write_json(folder/'progress.json', dict(step=step, updates=agent.updates, worker_steps=counts))
                if before//t['checkpoint_every'] != step//t['checkpoint_every'] or step == total:
                    agent.save(folder/'checkpoints'/f'step_{step:08d}.pth', step)
        return dict(status='completed', step=step, worker_steps=counts, updates=agent.updates)
    finally:
        for pipe in pipes:
            try:
                pipe.send(None)
            except (OSError, EOFError):
                pass
            pipe.close()
        for proc in processes:
            proc.join(5)
            if proc.is_alive():
                proc.terminate()
                proc.join(5)
                if proc.is_alive():
                    proc.kill()
                    proc.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=('omni_step', 'omni', 'walk_only'), default='omni_step')
    parser.add_argument('--gpus', type=int, help='number of allocated visible GPUs; default all')
    parser.add_argument('--learner-device', choices=('cpu', 'cuda:0'), default='cuda:0')
    parser.add_argument('--steps', type=int, help='total transitions across all GPUs')
    parser.add_argument('--output', type=Path, help='parent directory for configuration groups')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--timeout', type=int, default=600, help='seconds to wait for a worker reply')
    args = parser.parse_args()
    import torch
    from main import sha
    from configuration import HERE as project_root, checkpoint_path, load_configuration
    from asset_contract import inspect_asset
    from runtime_versions import require_versions
    if args.resume:
        args.resume = checkpoint_path(args.resume, project_root)
    visible = torch.cuda.device_count()
    n = visible if args.gpus is None else args.gpus
    if not 1 <= n <= visible or args.timeout <= 0 or (args.steps is not None and args.steps < 1):
        parser.error(f'check GPU count (visible={visible}), steps and timeout')
    template = load_configuration(experiment=args.experiment)
    settings = template['isaac']
    settings.update(headless=True, device='cuda:0')
    if sys.platform == 'linux':
        if not os.environ.get('ISAACLAB_PATH'):
            parser.error('set ISAACLAB_PATH to the Linux Isaac Lab checkout containing apps/')
        settings.update(isaac_lab_path=os.environ['ISAACLAB_PATH'], isaac_sim_path=os.getenv('ISAACSIM_PATH'),
                        isaac_python=sys.executable)
    require_versions(settings)
    asset = Path(settings['asset'])
    if not inspect_asset(asset)['ready']:
        raise ValueError(f'NAO asset unavailable: {asset}')
    root = args.output or Path(template['project_layout']['experiment_folder'])/'multi_gpu_runs'
    from run_storage import selected_checkpoint, training_output
    settings['asset_sha256'] = sha(asset)
    apply_collector_settings(template, args, n)
    with training_output(root.resolve(), template, args.resume) as (folder, session):
        args.resume = selected_checkpoint(folder, args.resume)
        run_collectors(folder, session, template, asset, args, n)


def apply_collector_settings(cfg, args, n):
    cfg['multi_gpu'] = dict(collectors=n, learner=args.learner_device)
    if args.steps is not None:
        cfg['training']['total_env_steps'] = args.steps
        cfg['budget_source'].update(effective_budget=args.steps, purpose='multi_gpu_budget_override')
    local_curriculum(cfg, n)


def run_collectors(folder, session, template, asset, args, n):
    from main import sha, snapshot
    attempt = session if (folder/'source').exists() else folder
    cfg = snapshot(attempt, template['isaac'], asset, 'train', config=template)
    from basewalk.runner import write_json
    for name in ('multigpu.py', 'multigpu_worker.py'):
        shutil.copyfile(HERE/name, attempt/'source'/name)
        cfg['source_hashes'][name] = sha(attempt/'source'/name)
    write_json(attempt/'effective_config.json', cfg)
    write_json(folder/'effective_config.json', cfg)
    write_json(folder/'status.json', dict(status='starting', workers=n))
    print(f'Run: {folder}', flush=True)
    try:
        result = train(cfg, folder, n, args.learner_device, args.timeout, args.resume,
                       session=session, worker_folder=attempt)
        write_json(folder/'result.json', result)
        from scripts.tracking_report import write_report
        print('SAC report:', write_report(folder), flush=True)
        write_json(folder/'status.json', result)
    except BaseException as exc:
        write_json(folder/'status.json', dict(status='failed', error=repr(exc)))
        raise
    finally:
        shutil.copyfile(folder/'status.json', session/'status.json')
        if (folder/'resume.json').exists():
            shutil.copyfile(folder/'resume.json', session/'resume.json')


if __name__ == '__main__':
    main()
