"""Isolated Webots trial helpers for the current Expert and BC."""
import json
import os
from pathlib import Path
import subprocess
import sys
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from configs.runtime import project_path
from launch import prepare_run

def execute(snapshot, world, payload):
    folder = snapshot.parent
    executable = project_path(payload['config']['webots']['executable'])
    environment = os.environ.copy()
    environment['PATH'] = str(executable.parent) + os.pathsep + environment.get('PATH', '')
    print(f'Running {folder}', flush=True)
    with (folder / 'webots.log').open('w', encoding='utf-8') as log:
        result = subprocess.run([str(executable), '--batch', '--minimize', '--mode=fast',
                                 '--stdout', '--stderr', str(world.resolve())],
                                cwd=executable.parent, env=environment, stdout=log,
                                stderr=subprocess.STDOUT)
    status = folder / 'run_status.json'
    if result.returncode or not status.exists() or json.loads(status.read_text())['status'] != 'completed':
        raise RuntimeError(f'Run failed: {folder / "webots.log"}')


def prepare_driver(config, mode, driver):
    snapshot, world, payload = prepare_run(config, mode)
    wrapper = world.parent.parent / 'controllers/configured_run/configured_run.py'
    text = wrapper.read_text(encoding='utf-8')
    start = text.index('try:\n')
    text = text[:start] + driver
    compile(text, str(wrapper), 'exec')
    wrapper.write_text(text, encoding='utf-8')
    return snapshot, world, payload

