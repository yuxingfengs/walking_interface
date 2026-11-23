"""Rebuild the installed Windows Isaac environment offline, then switch the launcher."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import venv


MODULE = Path(__file__).resolve().parents[3]/'sac_isaac_lab'
CONFIG = MODULE / 'config.json'


def build(source, target):
    target.mkdir(parents=True, exist_ok=False)
    skip = shutil.ignore_patterns('__pycache__', '*.pyc')
    for name in ('IsaacLab/apps', 'IsaacLab/source/isaaclab', 'IsaacLab/source/isaaclab_assets', 'kit_extensions'):
        shutil.copytree(source / name, target / name, ignore=skip)
    for name in ('VERSION', 'LICENSE', 'isaaclab.bat'):
        shutil.copy2(source / 'IsaacLab' / name, target / 'IsaacLab' / name)
    venv.EnvBuilder(system_site_packages=True, with_pip=False).create(target / 'python')
    python = target / 'python/Scripts/python.exe'
    packages = Path('python/Lib/site-packages')
    shutil.copytree(source / packages, target / packages, dirs_exist_ok=True, ignore=skip)
    # Re-register the two editable packages and regenerate console launchers for this venv.
    with tempfile.TemporaryDirectory(dir=target) as temp:
        env = dict(os.environ, TMP=temp, TEMP=temp, PIP_NO_CACHE_DIR='1')
        for package in ('isaaclab', 'isaaclab_assets'):
            subprocess.run([python, '-m', 'pip', 'install', '--no-index', '--no-deps',
                            '--no-build-isolation', '-e', target / 'IsaacLab/source' / package],
                           check=True, env=env)
    subprocess.run([python, '-c', """
from importlib import metadata
from pathlib import Path
import sys
from pip._vendor.distlib.scripts import ScriptMaker
scripts = Path(sys.executable).parent
maker = ScriptMaker(None, str(scripts))
maker.variants = {''}
for dist in metadata.distributions(path=[str(scripts.parent / 'Lib/site-packages')]):
    for ep in dist.entry_points:
        if ep.group == 'console_scripts':
            maker.make(f'{ep.name} = {ep.value}')
"""], check=True)
    subprocess.run([python, '-c', """
import importlib.util, pathlib, sys
import h5py, torch, warp
root = pathlib.Path(sys.argv[1]).resolve()
for name in ('isaaclab', 'isaaclab_assets', 'h5py', 'warp'):
    assert pathlib.Path(importlib.util.find_spec(name).origin).resolve().is_relative_to(root), name
assert h5py.__version__ == '3.14.0' and warp.__version__ == '1.8.1'
sys.path.insert(0, sys.argv[2])
from runtime_versions import dependency_status
status = dependency_status()
assert status['complete'], status
for ext in ('omni.warp.core-*', 'omni.warp-*', 'omni.gpu_foundation.shadercache.d3d12-*'):
    assert any((root/'kit_extensions').glob(ext+'/config/extension.toml')), ext
print('Validated external Python, Isaac Lab, h5py, Warp and Kit extensions.')
""", target, MODULE.parent/'environments/isaac_lab'], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=Path('F:/issac/nao-runtime'))
    parser.add_argument('--rehearse', action='store_true')
    args = parser.parse_args()
    original = CONFIG.read_bytes()
    cfg = json.loads(original.decode('utf-8-sig'))
    source = (MODULE / cfg['runtime_root']).resolve()
    target = args.destination.resolve()
    if args.rehearse:
        with tempfile.TemporaryDirectory(dir=MODULE.parent) as tmp:
            build(source, Path(tmp) / 'nao-runtime')
        return
    if target == source:
        print('External runtime already selected:', target)
        return
    if target.is_relative_to(MODULE.parent):
        raise ValueError('Choose a destination outside the project')
    build(source, target)
    if CONFIG.read_bytes() != original:
        raise RuntimeError('Launcher config changed during installation. External install kept.')
    cfg['runtime_root'] = target.as_posix()
    with tempfile.NamedTemporaryFile(dir=MODULE, delete=False, mode='w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
        f.write('\n')
    os.replace(f.name, CONFIG)
    print('Active Isaac runtime:', target)


if __name__ == '__main__':
    main()
