"""Layout shared by the Webots launchers; historical flat run paths remain valid."""
import hashlib
import json
from pathlib import Path
import shutil


def group_name(cfg):
    identity = {key: value for key, value in cfg.items()
                if key not in ('project_layout', 'reward_module_path', 'launch_reference')}
    serial = json.dumps(identity, sort_keys=True, allow_nan=False).encode()
    return 'config_' + hashlib.sha256(serial).hexdigest()[:16]


def belongs_to_run_root(folder, root):
    folder, root = Path(folder).resolve(), Path(root).resolve()
    return folder.name.startswith('train_') and (
        folder.parent == root or (folder.parent.parent == root and folder.parent.name.startswith('config_')))


def freeze_training_config(layout, source):
    if 'training_config_path' in layout:
        target = source/'training_config.json'
        shutil.copyfile(layout['training_config_path'], target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != layout['training_config_sha256']:
            raise ValueError('Shared training config changed before launch')
