"""Route the shared double-click entries before importing backend-specific code."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent


def selected_backend(config_path, experiment=None):
    path = Path(config_path).resolve()
    document = json.loads(path.read_text(encoding='utf-8-sig'))
    if document.get('backend') == 'isaac_lab_physx':
        backend = document['backend']
    else:
        backend = document.get('current_training_backend', 'webots')
    if backend == 'webots':
        return None
    if backend != 'isaac_lab_physx':
        raise ValueError('Unknown training backend: ' + str(backend))
    selected = experiment or document.get('current_experiment', document.get('experiment_id'))
    if selected != '001':
        raise ValueError('Isaac SAC requires the three-digit experiment ID 001')
    return selected


def dispatch(config_path, experiment=None, *, mode):
    if selected_backend(config_path, experiment) is None:
        return None
    if mode == 'pretrain':
        print('Isaac SAC initializes fresh twin Critics and learns online. Offline DDPG pretraining is not used.')
        print('No simulation or optimizer updates were started.')
        return 0
    launcher = ROOT.parent / 'sac_isaac_lab' / 'launch.ps1'
    return subprocess.call(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass',
                            '-File', str(launcher), '-Mode', mode])
