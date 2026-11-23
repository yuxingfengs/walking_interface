"""Materialize the H25 V5 URDF using an installed ROS nao_meshes package."""
import argparse
import hashlib
import json
import os
import re
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import sys

ENVIRONMENT = Path(__file__).resolve().parents[1]
sys.path.append(str(ENVIRONMENT))
from asset_contract import validate_urdf

SOURCE = ('https://raw.githubusercontent.com/ros-naoqi/nao_robot/master/'
          'nao_description/urdf/naoV50_generated_urdf/nao.urdf')


def import_mesh_path(source, folder):
    """Alias STL basenames that Sim 5.0 otherwise uses as invalid USD prim names."""
    source, folder = Path(source), Path(folder)
    if source.suffix.lower() != '.stl' or re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', source.stem):
        return source
    stem = re.sub(r'[^A-Za-z0-9_]', '_', source.stem)
    if not stem or stem[0].isdigit():
        stem = 'mesh_' + stem
    data = source.read_bytes()
    checksum = hashlib.sha256(data).hexdigest()
    target = folder/'import_meshes'/(stem+'_'+checksum[:12]+'.stl')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != data:
            raise ValueError('Import mesh alias differs from its source: ' + str(target))
    else:
        target.write_bytes(data)
    return target


def prepare(source, meshes):
    source = Path(source) if source else None
    meshes = Path(meshes).resolve()
    original = source.read_bytes() if source else urllib.request.urlopen(SOURCE, timeout=30).read()
    robot = ET.fromstring(original)
    if robot.get('name') != 'NaoH25V50':
        raise ValueError('Expected ROS NaoH25V50 URDF')
    used, aliases = {}, {}
    folder = ENVIRONMENT/'assets'
    for mesh in robot.iter('mesh'):
        name = mesh.attrib['filename']
        if not name.startswith('package://nao_meshes/'):
            raise ValueError('Unexpected mesh URI: ' + name)
        path = (meshes / name[len('package://nao_meshes/'):]).resolve()
        if not path.is_relative_to(meshes.resolve()) or not path.is_file():
            raise FileNotFoundError(path)
        if name not in used:
            used[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        prepared = import_mesh_path(path, folder)
        if prepared != path:
            aliases[name] = dict(source=os.path.relpath(path,folder).replace('\\','/'),
                                 prepared=os.path.relpath(prepared,folder).replace('\\','/'),
                                 sha256=used[name])
        mesh.set('filename', os.path.relpath(prepared,folder).replace('\\','/'))
    folder.mkdir(exist_ok=True)
    dest = folder/'nao.urdf'
    ET.ElementTree(robot).write(dest, encoding='utf-8', xml_declaration=True)
    contract = validate_urdf(dest)
    original_path = folder/'source_nao.urdf'
    original_path.write_bytes(original)
    record = dict(source=str(source or SOURCE), source_sha256=hashlib.sha256(original).hexdigest(),
                  asset_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(), meshes=used,
                  resolved_meshes={os.path.relpath(k,folder).replace('\\','/'):v for k,v in contract['meshes'].items()},
                  hip_coupling=contract['hip_coupling'], mesh_aliases=aliases)
    (folder/'asset_manifest.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(dest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--urdf', type=Path, default=os.environ.get('NAO_URDF'))
    parser.add_argument('--meshes', type=Path, default=os.environ.get('NAO_MESHES'))
    args = parser.parse_args()
    if not args.meshes:
        parser.error('Set NAO_MESHES to the package directory containing meshes/V40.')
    prepare(args.urdf, args.meshes)
