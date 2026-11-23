"""Create a project-local Kit wrapper around the complete compatible Warp wheel.

The original Sim installation is read-only. No simulation or learner is started.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import tempfile
import tomllib


import sys

ENVIRONMENT = Path(__file__).resolve().parents[1]
sys.path.append(str(ENVIRONMENT))
HERE = ENVIRONMENT.parents[1]/'sac_isaac_lab'
from runtime_versions import D3D12_SHADER_EXTENSION, GRAPH_WRAPPER_VERSION, WARP_VERSION, warp_extension_path, warp_graph_path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_extension(source, package, destination):
    source, package, destination = map(Path, (source, package, destination))
    metadata = tomllib.loads((source/'config/extension.toml').read_text(encoding='utf-8'))
    if metadata['package']['version'] != '1.7.1':
        raise ValueError('Expected the Isaac Sim 5.0 Warp 1.7.1 extension wrapper')
    binaries = ('bin/warp.dll', 'bin/warp-clang.dll')
    hashes = {name: digest(package/name) for name in binaries}
    manifest = dict(wrapper_source=str(source), warp_source=str(package),
                    warp_version=WARP_VERSION, binaries=hashes)
    if destination.exists():
        record = json.loads((destination/'project_manifest.json').read_text(encoding='utf-8'))
        if record != manifest or any(digest(destination/'warp'/name) != value for name,value in hashes.items()):
            raise ValueError('Existing project Warp extension differs from the installed wheel')
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='warp-staging-', dir=destination.parent))
    # Copy the complete extension wrapper and the complete matching wheel.
    # Python files and native DLLs must always come from the same Warp release.
    for child in source.iterdir():
        if child.name in ('warp', '__pycache__'):
            continue
        if child.is_dir():
            shutil.copytree(child, staging/child.name, ignore=shutil.ignore_patterns('__pycache__'))
        else:
            shutil.copy2(child, staging/child.name)
    shutil.copytree(package, staging/'warp', ignore=shutil.ignore_patterns('__pycache__'))
    config = staging/'config/extension.toml'
    text, count = re.subn(r'(?m)^version\s*=\s*"1\.7\.1"', 'version = "'+WARP_VERSION+'"',
                         config.read_text(encoding='utf-8'), count=1)
    if count != 1:
        raise ValueError('Unexpected Warp wrapper metadata')
    config.write_text(text, encoding='utf-8')
    (staging/'project_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    staging.rename(destination)
    return destination


def build_graph_wrapper(source, destination):
    config = Path('config/extension.toml')
    original = (source/config).read_text(encoding='utf-8')
    pin = '"omni.warp.core" = { version = "1.7.1", exact = true }'
    if original.count(pin) != 1 or tomllib.loads(original)['package']['version'] != '1.7.1':
        raise ValueError('Unexpected Sim 5.0 Warp graph dependency')
    updated = original.replace(pin, pin.replace('1.7.1', WARP_VERSION))
    updated = updated.replace('version = "1.7.1"', f'version = "{GRAPH_WRAPPER_VERSION}"', 1)
    if destination.exists():
        if (destination/config).read_text(encoding='utf-8') != updated:
            raise ValueError('Project GUI Warp wrapper differs from expected manifest')
        return destination
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        tree = Path(temporary)/'wrapper'
        shutil.copytree(source, tree, ignore=shutil.ignore_patterns('__pycache__'))
        (tree/config).write_text(updated, encoding='utf-8')
        tree.rename(destination)
    return destination


def main():
    if importlib.metadata.version('warp-lang') != WARP_VERSION:
        raise RuntimeError('Install warp-lang=='+WARP_VERSION+' first')
    settings = json.loads((ENVIRONMENT/'setup/simulator_config.json').read_text(encoding='utf-8-sig'))
    sim = Path(settings['isaac_sim_path'])
    if not sim.is_absolute():
        sim = (HERE/sim).resolve()
    source = sim/'extscache/omni.warp.core-1.7.1+wx64'
    dist = importlib.metadata.distribution('warp-lang')
    package = Path(dist.locate_file('warp')).resolve()
    print(build_extension(source, package, warp_extension_path(HERE)), flush=True)
    print(build_graph_wrapper(sim/'extscache/omni.warp-1.7.1', warp_graph_path(HERE)), flush=True)
    # Kit creates driver-cache files within this extension's own directory.
    shader = warp_graph_path(HERE).parent / D3D12_SHADER_EXTENSION
    if not shader.exists():
        with tempfile.TemporaryDirectory(dir=shader.parent) as temporary:
            tree = Path(temporary)/'shader'
            shutil.copytree(sim/'extscache'/D3D12_SHADER_EXTENSION, tree)
            tree.rename(shader)


if __name__ == '__main__':
    main()
