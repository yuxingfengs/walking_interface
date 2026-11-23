"""Resolve binary/pip runtime versions without starting an Isaac simulation."""
import importlib.metadata
import json
import sys
from pathlib import Path

WARP_VERSION = '1.8.1'
GRAPH_WRAPPER_VERSION = '1.7.1-project.1'
D3D12_SHADER_EXTENSION = 'omni.gpu_foundation.shadercache.d3d12-1.0.0+8131b85d.wx64.r'


def runtime_path(here):
    root = Path(here)
    config = root/'config.json'
    if config.is_file():
        return (root/json.loads(config.read_text(encoding='utf-8-sig')).get('runtime_root', 'runtime')).resolve()
    return root/'runtime'


def warp_graph_path(here):
    return runtime_path(here) / 'kit_extensions' / ('omni.warp-' + GRAPH_WRAPPER_VERSION)


def warp_extension_path(here):
    return runtime_path(here) / 'kit_extensions' / ('omni.warp.core-' + WARP_VERSION + '+wx64.project')


def dependency_status():
    from packaging.requirements import Requirement
    missing, incompatible = [], []
    try:
        requirements = importlib.metadata.requires('isaaclab')
    except importlib.metadata.PackageNotFoundError:
        return dict(complete=False, missing=['isaaclab'], incompatible=[])
    manifest = Path(__file__).with_name('runtime_requirements.txt')
    if not manifest.is_file():
        manifest = Path(__file__).resolve().parent / 'setup/runtime_requirements.txt'
    extra = manifest.read_text(encoding='utf-8').splitlines()
    for text in list(requirements or []) + [line for line in extra if line.strip() and not line.startswith('#')]:
        item = Requirement(text)
        if item.marker and not item.marker.evaluate():
            continue
        try:
            installed = importlib.metadata.version(item.name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(str(item))
            continue
        if item.specifier and not item.specifier.contains(installed, prereleases=True):
            incompatible.append(dict(requirement=str(item), installed=installed))
    return dict(complete=not missing and not incompatible, missing=missing, incompatible=incompatible)


def versions(settings):
    result = {}
    for package, key in (('isaacsim', 'isaac_sim_path'), ('isaaclab', 'isaac_lab_path')):
        root = settings.get(key)
        version_file = Path(root) / 'VERSION' if root else None
        if version_file and version_file.is_file():
            result[package] = version_file.read_text(encoding='utf-8-sig').strip()
        else:
            try:
                result[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                result[package] = None
    return result


def require_versions(settings):
    result = versions(settings)
    if not str(result['isaacsim']).startswith('5.0.') or not str(result['isaaclab']).startswith('2.2.'):
        raise RuntimeError('Expected Isaac Sim 5.0.x and Isaac Lab 2.2.x: ' + str(result))
    return result


def offline_headless_experience(settings, folder, *, standing=False):
    apps = Path(settings['isaac_lab_path']).resolve() / 'apps'
    template = (apps / 'isaaclab.python.headless.kit').read_text(encoding='utf-8')
    dependency = '"omni.kit.telemetry" = {}'
    if template.splitlines().count(dependency) != 1:
        raise RuntimeError('Unexpected headless Kit dependencies; check the offline experience before starting.')
    content = template.replace(dependency, '').replace('${app}', apps.as_posix())
    content = content.replace('registryEnabled = true', 'registryEnabled = false')
    if standing:
        # The NAO standing check uses the core articulation API directly.
        for name in ('isaaclab_mimic', 'isaaclab_rl', 'isaaclab_tasks'):
            if not (apps.parent/'source'/name/'config/extension.toml').is_file():
                content = content.replace(f'"{name}" = {{order = 1000}}', '')
    path = folder / 'omni_offline_headless.kit'
    path.write_text(content, encoding='utf-8')
    return str(path)


def kit_arguments(settings, here, *, state_root=None, native_warp=False):
    # Portable mode does not relocate the Windows Documents tokens by itself.
    # Keep the Kit state and document-derived scripts inside this experiment.
    # User scripts and Kit preferences stay with the project when software moves.
    config = Path(here) / 'config.json'
    options = json.loads(config.read_text(encoding='utf-8-sig')) if config.is_file() else {}
    state_dir = options.get('kit_state_root', 'runtime')
    root = (Path(here) / state_dir / 'kit_state').resolve()
    if settings.get('experiment_id') == 'omni':
        root = root.with_name('kit_state_omni')
    if state_root is not None:
        root = Path(state_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    documents = root / 'user_files'
    tokens = dict(omni_documents=documents, shared_documents=documents/'shared',
                  app_documents=documents/'Isaac-Sim')
    for folder in (*tokens.values(), tokens['app_documents']/'scripts'/'new_stage'):
        folder.mkdir(parents=True, exist_ok=True)
    # Kit requires a separate value for --portable-root; the previous '=path'
    # form was ignored, as confirmed by 'Portable Root' in the startup log.
    args = ['--portable', '--portable-root', root.as_posix()]
    if settings.get('experiment_id') == 'omni':
        args.extend(['--/app/extensions/excluded/0=omni.kit.telemetry',
                     '--/telemetry/enableAnonymousData=false'])
    args.extend('--/app/tokens/' + name + '=' + path.as_posix()
                for name, path in tokens.items())
    if native_warp:
        return ' '.join(args)
    extension = warp_extension_path(here).resolve()
    if not (extension / 'config' / 'extension.toml').is_file():
        raise RuntimeError('Project Warp extension is missing. Run Install Isaac Dependencies.cmd.')
    args.extend(['--ext-path', extension.as_posix(), '--enable', 'omni.warp.core-' + WARP_VERSION])
    if settings.get('headless') is False:
        graph = warp_graph_path(here).resolve()
        if not (graph/'config'/'extension.toml').is_file():
            raise RuntimeError('Missing project GUI Warp wrapper. Run configure_warp_extension.py.')
        args.extend(['--ext-path', graph.as_posix(), '--enable', 'omni.warp-' + GRAPH_WRAPPER_VERSION])
        if sys.platform == 'win32':
            cache = graph.parent / D3D12_SHADER_EXTENSION
            if not (cache/'config'/'extension.toml').is_file():
                raise RuntimeError('Missing local D3D12 shader cache extension. Run configure_warp_extension.py.')
            args.extend(['--/app/vulkan=false', '--ext-path', cache.as_posix(), '--/renderer/multiGpu/enabled=false'])
    return ' '.join(args)
