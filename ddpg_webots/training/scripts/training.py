"""Current direction training: check, train, monitor, report, config."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from contextlib import ExitStack
from filelock import FileLock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from basewalk import TRAINING_ROOT
from basewalk.monitor import report, read_json
from basewalk.runner import write_json
from basewalk.artifacts import prepare, render_world
from basewalk.events import emit


def resolve_path(path):
    return (TRAINING_ROOT/path).resolve() if not Path(path).is_absolute() else Path(path).resolve()


def launch(config, mode, checkpoint=None, smoke=False, *, jobs=None, resume_model=None):
    # Register the prepared run so plotting also runs after a failed/interrupted run.
    prepared = []
    try:
        with ExitStack() as locks:
            if resume_model:
                resume_model.mkdir(parents=True, exist_ok=True)
                locks.enter_context(FileLock(str(resume_model/'.writer.lock'), timeout=0))
            return _launch(config, mode, checkpoint, smoke, prepared, jobs, resume_model, locks)
    finally:
        if mode == 'train' and prepared:
            try:
                from basewalk.reward_plots import plot_rewards
                output = plot_rewards(prepared[0])
                if output:
                    (output/'normal_exit.done').write_text('completed', encoding='utf-8')
                    print(f'Reward plots: {output}', flush=True)
            except Exception as error:
                print(f'Reward plotting failed: {error}', flush=True)


def _launch(config, mode, checkpoint, smoke, prepared, jobs, resume_model, locks):
    options = dict(mode=mode, smoke=smoke, checkpoint=checkpoint, suite='commands', play_steps=0)
    if mode == 'expert_collection':
        if not jobs:
            raise ValueError('Expert collection requires an explicit job list')
        options['jobs'] = jobs
    folder, world = prepare(config, options, resume_model=resume_model)
    if resume_model:
        from basewalk.checkpoint_history import prepare_resume
        prepare_resume(resume_model, folder, config['training']['checkpoint_start'])
        folder = resume_model
    elif mode == 'train':
        locks.enter_context(FileLock(str(folder/'.writer.lock'), timeout=0))
    prepared.append(folder)
    live = config['monitoring'].get('live_report_enabled', True)
    if config.get('launch_reference'):
        write_json(Path(config['launch_reference']), dict(run=str(folder.resolve()),experiment=config['experiment']))
    if mode == 'train' and os.name == 'nt' and live:
        # Detached from the console so closing CMD/terminating this launcher
        # cannot bypass plots for data already flushed to disk.
        helper = ROOT/'basewalk/exit_plotter.py'
        subprocess.Popen([sys.executable, str(helper), str(os.getpid()), str(folder)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            close_fds=True, creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
    write_json(folder/'status.json', dict(status='launching'))
    print(f'Run: {folder}', flush=True)
    print(f'Report: {folder / ("monitor/index.html" if mode == "train" else "EVALUATION.md")}', flush=True)
    executable = Path(config['webots']['executable'])
    environment = os.environ.copy()
    environment['PATH'] = str(executable.parent)+os.pathsep+environment.get('PATH', '')
    started = time.monotonic(); last_report = 0.
    with (folder/'webots.log').open('a', encoding='utf-8') as log:
        try:
            command = [str(executable), '--batch', '--mode=fast', '--stdout', '--stderr']
            if config['webots'].get('port'):
                command.append('--port='+str(config['webots']['port']))
            process = subprocess.Popen(command+[str(world)],
                cwd=str(executable.parent), env=environment, stdout=log, stderr=subprocess.STDOUT)
        except OSError as exc:
            emit(folder,config,'launch_failed','训练启动失败',str(exc))
            write_json(folder/'status.json', dict(status='failed', error=str(exc), source='launcher'))
            raise
        write_json(folder/'process.json', dict(pid=process.pid, status='launched'))
        try:
            if not live:
                process.wait(timeout=43200)
            while live and process.poll() is None:
                now = time.monotonic()
                if mode == 'train' and now-last_report >= config['monitoring']['refresh_seconds']:
                    report(folder); last_report = now
                if now-started > 43200:
                    raise TimeoutError('Run exceeded 12h')
                time.sleep(1)
        except BaseException as exc:
            process.terminate(); process.wait(timeout=30)
            write_json(folder/'status.json', dict(status='failed', error=str(exc), source='launcher'))
            emit(folder,config,'launcher_failed','训练停止',str(exc))
            raise
    write_json(folder/'process.json', dict(pid=process.pid, status='exited', exit_code=process.returncode))
    status = read_json(folder/'status.json')
    if status['status'] not in ('completed', 'paused') or process.returncode != 0:
        if status['status'] not in ('failed',):
            emit(folder,config,'process_failed','训练异常停止',f"Webots exit={process.returncode}, status={status['status']}")
        if mode == 'train': report(folder)
        raise RuntimeError(f'Run failed: {folder}: {status}')
    if mode != 'expert_collection':
        print(f'Report: {report(folder)}', flush=True)
    if mode == 'train' and status['status'] == 'completed' and config['monitoring'].get('generate_cycle_report'):
        layout = config['project_layout']
        report_script = Path(layout['project_root'])/'report.py'
        generated = subprocess.run([sys.executable, '-X', 'utf8', str(report_script), config['experiment'],
                                    '--config', layout['config_path'], '--run', str(folder), '--no-compare'])
        if generated.returncode:
            print(f'Cycle report failed (exit {generated.returncode}). The training run remains saved.', flush=True)
    if status['status'] == 'paused':
        print('Paused: '+str(status.get('reason', 'unspecified'))+'; no automatic restart.',flush=True)
    return folder

