"""Render the current scene and save isolated experiment run artifacts."""
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
import shutil
import uuid
import copy
import json
from .core import CONTRACT
from .world_template import absolute_resources, render_template
from .runner import write_json
from .run_storage import freeze_training_config, group_name

from . import TRAINING_ROOT as ROOT

SOURCE = Path(__file__).resolve().parents[1]


def render_world(config):
    # Reuse the existing resource resolver, but a separate flat arena is required
    # for lateral/backward commands. Do not modify the historical sprint world.
    text = render_template(ROOT/config['webots']['world_template'], config, 'basewalk_run')
    # Resolve the reference world's backgrounds from the standalone SAC asset bundle.
    background_root = ROOT.parent/'model/resources/backgrounds'
    for name in ('TexturedBackground.proto', 'TexturedBackgroundLight.proto'):
        url = 'https://raw.githubusercontent.com/cyberbotics/webots/R2022b/projects/objects/backgrounds/protos/' + name
        text = text.replace(url, (background_root/name).as_posix())
    if config.get('physical_gait', {}).get('imu_full_axes'):
        model=ROOT.parent/'model/Nao.proto'
        if not model.is_file():raise FileNotFoundError('Run prepare_imu_model.py first')
        text=text.replace('https://raw.githubusercontent.com/cyberbotics/webots/R2022b/projects/robots/softbank/nao/protos/Nao.proto',model.as_posix())
    text = re.sub(r'(DEF TRACK Floor\s*\{\s*translation\s+)[^\n]+', r'\g<1>0 0 0.39', text)
    text = re.sub(r'(\bsize\s+)11 3', r'\g<1>30 30', text)
    text = re.sub(r'\nRaceStopwatch\s*\{[^}]*\}', '', text)
    text = text.replace('window "humanoid_sprint"', 'window "<none>"')
    text = text.replace('basicTimeStep '+str(config['webots']['basic_time_step_ms']),
                        'randomSeed '+str(config['seed'])+'\n  basicTimeStep '+str(config['webots']['basic_time_step_ms']))
    return text


def prepare(config, options):
    """Freeze root configuration, selected Reward and runtime code before launch."""
    from .reward_dispatch import implementation
    implementation(config)
    layout = config['project_layout']
    if hashlib.sha256(Path(layout['config_path']).read_bytes()).hexdigest() != layout['config_sha256']:
        raise ValueError('Project config changed after validation; reload before launch')
    descriptor = Path(layout['experiment_folder'])/'config.json'
    if hashlib.sha256(descriptor.read_bytes()).hexdigest() != layout['descriptor_sha256']:
        raise ValueError('Experiment descriptor changed after validation')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:6]
    label = 'smoke' if options['smoke'] else options['mode']
    if config.get('bc_rollout',{}).get('collection_mode') and not options['smoke']:
        label = 'bc_rollout'
    parent = Path(layout['runs_root'])
    if label == 'train':
        parent = parent/group_name(config)
    folder = parent/(label+'_'+stamp)
    folder.mkdir(parents=True, exist_ok=False)
    source = folder/'source'
    source.mkdir()
    freeze_training_config(layout, source)
    config_root = Path(layout['config_path']).parent
    runtime_root = ROOT if config_root == Path(layout['project_root']) else config_root
    runtime = runtime_root/'.webots_runs'/('bw_'+stamp)
    (runtime/'worlds').mkdir(parents=True)
    controller = runtime/'controllers/basewalk_run'
    controller.mkdir(parents=True)
    hashes = {}
    for group in ('basewalk', 'scripts'):
        files = list((SOURCE/group).glob('*.py'))
        if group == 'scripts':
            files.append(SOURCE/group/'turn_view.html')
        for original in files:
            relative = str(original.relative_to(SOURCE))
            target = source/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
            hashes[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
    from . import webots_devices
    shared = Path(webots_devices.__file__).parent
    for name in ('webots_devices.py', 'run_storage.py', 'world_template.py', 'foot_feedback.py', 'report_diagnostics.py'):
        target = source/'basewalk'/name
        shutil.copyfile(shared/name, target)
        hashes['basewalk/' + name] = hashlib.sha256(target.read_bytes()).hexdigest()
    frozen_reward = source/'experiment_reward.py'
    shutil.copyfile(config['reward_module_path'], frozen_reward)
    if hashlib.sha256(frozen_reward.read_bytes()).hexdigest() != config['reward_source_sha256']:
        raise ValueError('Reward changed while snapshotting')
    hashes['experiment_reward.py'] = config['reward_source_sha256']
    for name in ('train_sac.py', 'project.py'):
        entry_root = ROOT.parent if name == 'project.py' else SOURCE.parent
        shutil.copyfile(entry_root/name, source/name)
        hashes[name] = hashlib.sha256((source/name).read_bytes()).hexdigest()
    shutil.copyfile(layout['config_path'], folder/'project_config.json')
    shutil.copyfile(descriptor, folder/'experiment.json')
    cfg = copy.deepcopy(config)
    if cfg.get('initialization', {}).get('use_bc'):
        init = cfg['initialization']
        target = source/'expert_actor.pth'
        shutil.copyfile(init['actor_path'], target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != init['sha256']:
            raise ValueError('BC Actor changed while snapshotting')
        shutil.copyfile(Path(init['actor_path']).with_name('manifest.json'), source/'bc_manifest.json')
        hashes[target.name] = init['sha256']
        init['actor_path'] = str(target.resolve())
    cfg['reward_module_path'] = str(frozen_reward.resolve())
    text = render_world(cfg)
    # Preserve relative dependencies inside each run's local Webots asset tree.
    resource_hashes = {}
    resources = [ROOT.parent/'model/Nao.proto', Path(cfg['webots']['world_template']).parent/'textures/direction_circle_10m.png']
    for original in resources:
        if original.is_file():
            target = source/'resources'/original.name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(original, target)
            text = text.replace(original.resolve().as_posix(), target.resolve().as_posix())
            resource_hashes[original.name] = hashlib.sha256(target.read_bytes()).hexdigest()
            if original.name == 'Nao.proto':
                asset_root = original.parent/'resources'
                frozen_root = target.parent/'resources'
                shutil.copytree(asset_root, frozen_root)
                text = text.replace(asset_root.resolve().as_posix()+'/', frozen_root.resolve().as_posix()+'/')
                for asset in frozen_root.rglob('*'):
                    if asset.is_file():
                        key = asset.relative_to(target.parent).as_posix()
                        resource_hashes[key] = hashlib.sha256(asset.read_bytes()).hexdigest()
    world = runtime/'worlds/basewalk.wbt'
    world.write_text(text, encoding='utf-8')
    view = Path(cfg['webots']['world_template']).with_name('.basewalk.wbproj')
    shutil.copyfile(view, world.with_name('.basewalk.wbproj'))
    shutil.copyfile(view, folder/'view.wbproj')
    resource_hashes['view.wbproj'] = hashlib.sha256(view.read_bytes()).hexdigest()
    shutil.copyfile(world, folder/'world.wbt')
    snapshot = folder/'effective_config.json'
    payload = dict(config=cfg, contract=CONTRACT, options=options, source_hashes=hashes,
                   supports_budget_extension=True,
                   resource_hashes=resource_hashes, comparison_status='pending',
                   world_sha256=hashlib.sha256(world.read_bytes()).hexdigest())
    if options['checkpoint']:
        payload['checkpoint_sha256'] = hashlib.sha256(Path(options['checkpoint']).read_bytes()).hexdigest()
    write_json(snapshot, payload)
    (folder/'README.md').write_text(
        '# '+cfg['experiment']+'\n\nConfiguration and exact Reward implementation are frozen in this run.\n'
        'effective_config.json is authoritative; project_config.json records live input at launch.\n'
        'source/experiment_reward.py is the implementation used by the controller.\n'
        'training.csv contains finalized rewards, Turn rates, falls, and SAC metrics.\n'
        'Online logs are not independent Evaluation. No policy-improvement conclusion is implied.\n', encoding='utf-8')
    controller_module = 'expert_rollout' if options['mode'] == 'expert_collection' else 'runner'
    (controller/'basewalk_run.py').write_text(
        'import sys\nsys.path.insert(0, '+repr(str(source.resolve()))+')\n'
        f'from basewalk.{controller_module} import main\nmain('+repr(str(snapshot.resolve()))+')\n', encoding='utf-8')
    (controller/'runtime.ini').write_text('[python]\nCOMMAND = '+config['webots']['python']+'\n', encoding='utf-8')
    return folder, world

