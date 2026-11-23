"""Launch a frozen SAC run in its own Webots instance."""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import socket
from tempfile import NamedTemporaryFile
from uuid import uuid4
from filelock import FileLock, Timeout
from basewalk.artifacts import ROOT, prepare
from basewalk.runner import write_json
from basewalk.world_reload import load_continuation
from basewalk.run_storage import belongs_to_run_root
from .turn_report import write_report


def launch(cfg):
    folder, world = prepare(cfg, dict(mode='train', smoke=False, checkpoint=None))
    run_webots(cfg, folder, world)


def resume(folder, target):
    folder = folder.resolve(strict=True)
    if ROOT.parent not in folder.parents or not folder.name.startswith('train_'):
        raise ValueError('Select a local SAC Webots training run folder')
    snapshot = json.loads((folder/'effective_config.json').read_text(encoding='utf-8'))
    cfg = snapshot['config']
    if (cfg.get('algorithm') != 'sac' or not belongs_to_run_root(folder, cfg['project_layout']['runs_root'])
            or cfg['environment']['reset_mode'] != 'world_reload'):
        raise ValueError('Run does not support same-run Webots continuation')
    world = ROOT/'.webots_runs'/('bw_' + folder.name[len('train_'):])/'worlds/basewalk.wbt'
    if hashlib.sha256(world.read_bytes()).hexdigest() != snapshot['world_sha256']:
        raise ValueError('Runtime world differs from the run snapshot')
    if target != cfg['training']['total_env_steps'] and not snapshot.get('supports_budget_extension'):
        raise ValueError('This frozen controller does not support a budget change')
    run_webots(cfg, folder, world, target=target)


def align_training_log(cfg, folder, target, completed):
    payload = load_continuation(folder, cfg)
    if payload is None:
        raise ValueError('Missing episode-boundary continuation')
    next_step = payload['state']['next_step']
    if completed is not None and next_step != completed + 1:
        raise ValueError('This completed run has no final continuation state')
    if target < next_step:
        raise ValueError(f'Increase the total budget above the saved step {next_step - 1}')
    log = folder/'training.csv'
    lines = log.read_bytes().splitlines(keepends=True)
    rows = list(csv.DictReader(line.decode('utf-8') for line in lines))
    if (not 2 <= next_step <= len(rows) + 1 or
            any(int(row['step']) != n for n, row in enumerate(rows, 1)) or
            int(rows[next_step-2]['updates']) != payload['agent'].updates):
        raise ValueError('Training log and saved SAC state do not agree')
    if next_step <= len(rows):
        archive = folder/f'interrupted_steps_{next_step}_{len(rows)}_{uuid4().hex[:8]}.csv'
        with archive.open('xb') as tail:
            tail.writelines(lines[:1] + lines[next_step:])
            tail.flush()
            os.fsync(tail.fileno())
        with NamedTemporaryFile(dir=folder, delete=False) as tmp:
            tmp.writelines(lines[:next_step])
        os.replace(tmp.name, log)
        print('Archived interrupted steps:', archive, flush=True)
    print('Resuming at step', next_step, flush=True)


def restore_layout(cfg, world):
    layout = Path(cfg['webots']['world_template']).with_name('.basewalk.wbproj')
    destination = world.with_name('.' + world.stem + '.wbproj')
    with NamedTemporaryFile(dir=world.parent, delete=False) as tmp:
        staging = Path(tmp.name)
    try:
        shutil.copyfile(layout, staging)
        os.replace(staging, destination)
    finally:
        staging.unlink(missing_ok=True)


def run_webots(cfg, folder, world, target=None):
    port = cfg['webots']['port']
    lock = FileLock(str(ROOT/f'.webots-port-{port}.lock'), timeout=0)
    try:
        with lock:
            # Also detect simulators started by an older or external launcher.
            with socket.socket() as probe:
                try:
                    probe.bind(('127.0.0.1', port))
                except OSError as exc:
                    raise ValueError(f'Webots port {port} is unavailable') from exc
            if target is not None:
                result = folder/'result.json'
                completed = json.loads(result.read_text())['env_steps'] if result.exists() else None
                align_training_log(cfg, folder, target, completed)
                restore_layout(cfg, world)
                write_json(folder/'continuation_budget.json', dict(total_env_steps=target))
                if result.exists():
                    result.replace(folder/f'result_{completed}_{uuid4().hex[:8]}.json')
            start_process(cfg, folder, world)
    except Timeout as exc:
        raise ValueError(f'A SAC launcher is already using port {port}') from exc


def start_process(cfg, folder, world):
    print('SAC run:', folder, flush=True)
    exe = Path(cfg['webots']['executable'])
    args = [str(exe), '--batch', '--mode=fast', '--stdout', '--stderr',
            '--port=' + str(cfg['webots']['port']), str(world)]
    env = os.environ.copy()
    env['PATH'] = str(exe.parent) + os.pathsep + env.get('PATH', '')
    env['PYTHONFAULTHANDLER'] = '1'
    env['PYTHONUNBUFFERED'] = '1'
    status = folder/'status.json'
    write_json(status, dict(status='launching'))
    with (folder/'webots.log').open('a', encoding='utf-8') as log:
        with subprocess.Popen(args, cwd=exe.parent, env=env, stdout=log, stderr=subprocess.STDOUT) as process:
            returncode = process.wait()
        log.write(f'\nWebots exit code: {returncode}\n')
    if returncode or json.loads(status.read_text())['status'] != 'completed':
        raise RuntimeError('SAC controller did not complete. See ' + str(folder))
    print('SAC report:', write_report(folder), flush=True)
