"""Keep independent model histories together by effective training configuration."""
from contextlib import contextmanager
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import uuid
from datetime import datetime, timezone


def config_group(cfg):
    identity = {k: v for k, v in cfg.items() if k not in
                ('project_layout', 'source_hashes', 'budget_source', 'reward_module_path')}
    identity['initialization'] = {k: v for k, v in cfg['initialization'].items() if k != 'actor_path'}
    identity['isaac'] = {k: v for k, v in cfg['isaac'].items() if k not in (
        'device', 'learner_device', 'headless', 'isaac_python', 'isaac_sim_path',
        'isaac_lab_path', 'asset', 'urdf_source', 'nao_meshes_path', 'standing_ground')}
    identity['source_webots'] = {k: v for k, v in cfg['source_webots'].items()
                                 if k not in ('python', 'executable', 'world_template')}
    encoded = json.dumps(identity, sort_keys=True, allow_nan=False).encode()
    return 'config_' + hashlib.sha256(encoded).hexdigest()[:16]


def launch_id():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:8]


def continue_checkpoint(root, selector):
    """Resolve a run inside this experiment before creating any session files."""
    run_name = selector['run']
    name = selector.get('checkpoint', 'latest.pth')
    if not isinstance(name, str) or not re.fullmatch(r'latest\.pth|recovery_latest\.pth|step_[0-9]+\.pth', name):
        raise ValueError('checkpoint must be latest.pth, recovery_latest.pth or step_<number>.pth')
    if run_name == 'latest':
        pointer = root/'latest_run.txt'
        if not pointer.is_file():
            raise FileNotFoundError(f'Missing {pointer}. Set continue_training.run in sac_isaac_lab/config.json.')
        run_name = pointer.read_text(encoding='utf-8').strip()
    if not isinstance(run_name, str) or not re.fullmatch(r'(?:config_[\w-]+/)?train_[\w-]+', run_name):
        raise ValueError('run must be latest, train_<id> or config_<hash>/train_<id>')
    run = root/run_name
    if not run.is_dir():
        raise FileNotFoundError(f'Run directory not found: {run}')
    if not (run/'training.csv').is_file():
        raise FileNotFoundError(f'Continue needs original run history: {run / "training.csv"}')
    checkpoints = run/'checkpoints'
    native = checkpoints/name
    if native.is_file():
        return native
    if name == 'latest.pth':
        import torch
        numbered = [p for p in checkpoints.glob('step_*.pth') if p.stem[5:].isdigit()]
        candidates = [(int(p.stem[5:]), False, p) for p in numbered]
        recovery = checkpoints/'recovery_latest.pth'
        if recovery.is_file():
            step = torch.load(recovery, map_location='meta', weights_only=True)['env_steps']
            candidates.append((int(step), True, recovery))
        if candidates:
            return max(candidates)[2]
    raise FileNotFoundError(f'No {name} checkpoint for {run}')


@contextmanager
def exclusive_writer(folder):
    # OS-owned locks are released even if Kit or the host process crashes.
    with (folder/'.writer.lock').open('a+b') as lock:
        lock.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f'Another training process is writing to {folder}') from exc
        yield


@contextmanager
def training_output(root, cfg, checkpoint):
    group = root / config_group(cfg)
    stamp = launch_id()
    model = group / (model_name(checkpoint) if checkpoint else 'train_' + stamp)
    if checkpoint:
        origin = checkpoint.parent.parent
        if origin.parent.parent == root and origin.parent.name.startswith('config_'):
            model = origin
        elif origin.parent == root:
            imports = [record.parent for record in root.glob('config_*/train_*/imported_checkpoint.json')
                       if Path(json.loads(record.read_text())['source']) == checkpoint]
            if len(imports) > 1:
                raise ValueError('Select a checkpoint inside the intended imported run')
            if imports:
                model = imports[0]
    model.mkdir(parents=True, exist_ok=bool(checkpoint))
    with exclusive_writer(model):
        session = model/'sessions'/stamp
        session.mkdir(parents=True)
        log = model/'logs'/stamp
        log.mkdir(parents=True)
        hyperparameters = dict(seed=cfg['seed'], sac=cfg['sac'],
                               training={key: cfg['training'][key] for key in
                                         ('batch_size', 'replay_size', 'checkpoint_every')})
        (log/'training_config.json').write_text(json.dumps(hyperparameters, indent=2) + '\n', encoding='utf-8')
        yield model, session


def model_name(checkpoint):
    model = checkpoint.parent.parent
    if model.parent.name.startswith('config_'):
        return model.name
    # Old launchers created a sibling for every continuation of a single model.
    seen = {model.name}
    while (model/'resume.json').is_file():
        record = json.loads((model/'resume.json').read_text(encoding='utf-8-sig'))
        origin = Path(record['checkpoint'].replace('\\', '/')).parent.parent.name
        previous = model.parent/origin
        if not origin.startswith('train_') or origin in seen or not previous.is_dir():
            break
        seen.add(origin)
        model = previous
    return model.name


def selected_checkpoint(model, configured, *, follow_latest=False):
    """Resolve the live model's continuation after acquiring its writer lock."""
    if configured is None:
        return None
    record = model/'imported_checkpoint.json'
    current = model/'checkpoints/recovery_latest.pth'
    imported = record.is_file() and Path(json.loads(record.read_text())['source']) == configured
    if follow_latest and (model == configured.parent.parent or imported):
        candidates = {configured}
        candidates.update(p for p in current.parent.glob('step_*.pth') if p.stem[5:].isdigit())
        candidates.update(p for p in (current, current.with_name('latest.pth')) if p.is_file())
    elif imported and configured.name == current.name and current.is_file():
        candidates = {configured, current}
    else:
        return configured
    if len(candidates) == 1:
        return configured

    def priority(path):
        if path.stem.startswith('step_'):
            step = int(path.stem[5:])
        else:
            import torch
            step = int(torch.load(path, map_location='meta', weights_only=True)['env_steps'])
        return step, path.name == current.name, path.parent == current.parent, path == configured

    return max(candidates, key=priority)


def preserve_file(original, backup):
    if original.is_file():
        backup.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(original, backup)
        except OSError:
            shutil.copy2(original, backup)


def archive_after_checkpoint(folder, session, step, selected):
    checkpoints = folder/'checkpoints'
    # Numbered checkpoints beyond a rollback belong to the previous history.
    if not checkpoints.is_dir():
        return
    for path in checkpoints.glob('step_*.pth'):
        suffix = path.stem[5:]
        if suffix.isdigit() and int(suffix) > step:
            target = session/'previous_checkpoints'/path.name
            target.parent.mkdir(exist_ok=True)
            path.replace(target)
    recovery = checkpoints/'recovery_latest.pth'
    preserve_file(recovery, session/'previous_checkpoints'/recovery.name)
    if recovery.exists() and recovery != selected:
        import torch
        if torch.load(recovery, map_location='meta', weights_only=True)['env_steps'] > step:
            recovery.unlink()


def archive_csv_tail(path, offset, session):
    with path.open('r+b') as stream:
        if stream.seek(0, os.SEEK_END) < offset:
            raise ValueError('Training CSV is shorter than its checkpoint')
        stream.seek(offset)
        if stream.read(1):
            stream.seek(offset)
            with (session/'training_after_checkpoint.rows').open('xb') as backup:
                shutil.copyfileobj(stream, backup)
            stream.truncate(offset)


def restore_csv(checkpoint, folder, offset, step, session):
    source = checkpoint.parent.parent/'training.csv'
    live = folder/'training.csv'
    if offset < 0 or source.stat().st_size < offset:
        raise ValueError('Invalid checkpoint CSV offset')
    if source == live:
        archive_after_checkpoint(folder, session, step, checkpoint)
        archive_csv_tail(live, offset, session)
        return
    staging = (session or folder)/'training.import'
    with source.open('rb') as src, staging.open('xb') as dst:
        remaining = offset
        while remaining:
            data = src.read(min(remaining, 1024 * 1024))
            if not data:
                raise ValueError('Training CSV changed while reading its checkpoint')
            dst.write(data)
            remaining -= len(data)
    if live.exists():
        if session is None:
            raise FileExistsError(live)
        preserve_file(live, session/'previous_training.csv')
        archive_after_checkpoint(folder, session, step, checkpoint)
    staging.replace(live)
    if session is not None:
        (folder/'imported_checkpoint.json').write_text(json.dumps({'source': str(checkpoint)}))
