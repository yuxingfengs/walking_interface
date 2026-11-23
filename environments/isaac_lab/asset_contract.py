"""Validate local NAO assets without importing Isaac or altering the source model."""
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

LEG_JOINTS = tuple(side + joint for side in ('L', 'R') for joint in
                  ('HipYawPitch', 'HipRoll', 'HipPitch', 'KneePitch', 'AnklePitch', 'AnkleRoll'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_urdf(path):
    path = Path(path).resolve()
    robot = ET.fromstring(path.read_bytes())
    if robot.get('name') != 'NaoH25V50':
        raise ValueError('Expected the ROS-NAOqi NaoH25V50 model')
    joints = robot.findall('joint')
    names = [joint.get('name') for joint in joints]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate URDF joint names')
    by_name = {joint.get('name'): joint for joint in joints}
    for name in LEG_JOINTS:
        joint = by_name.get(name)
        if joint is None or joint.get('type') != 'revolute':
            raise ValueError('Missing revolute leg joint: ' + name)
        limit = joint.find('limit')
        if limit is None:
            raise ValueError('Missing joint limit: ' + name)
        values = [float(limit.get(key)) for key in ('lower', 'upper', 'effort', 'velocity')]
        if not all(map(math.isfinite, values)) or not values[0] < values[1] or min(values[2:]) <= 0:
            raise ValueError('Invalid limits: ' + name)
    mimic = by_name['RHipYawPitch'].find('mimic')
    if (mimic is None or mimic.get('joint') != 'LHipYawPitch' or
            float(mimic.get('multiplier', '1')) != 1. or float(mimic.get('offset', '0')) != 0.):
        raise ValueError('RHipYawPitch must physically mimic LHipYawPitch at 1:1')
    links = {link.get('name'): link for link in robot.findall('link')}
    for name in ('l_ankle', 'r_ankle'):
        if name not in links or links[name].find('collision/geometry') is None:
            raise ValueError('Missing foot collision geometry: ' + name)
    meshes = {}
    for mesh in robot.iter('mesh'):
        name = mesh.get('filename', '')
        if '://' in name:
            raise ValueError('Resolve URDF mesh URI before simulation: ' + name)
        location = Path(name)
        if not location.is_absolute():
            location = path.parent / location
        location = location.resolve()
        if not location.is_file():
            raise FileNotFoundError('Missing NAO geometry: ' + str(location))
        meshes[str(location)] = digest(location)
    if not meshes:
        raise ValueError('NAO asset contains no mesh geometry')
    return dict(robot=robot.get('name'), asset_sha256=digest(path),
                leg_joints=list(LEG_JOINTS), hip_coupling='physical_mimic_1_to_1', meshes=meshes)


def inspect_asset(path):
    path = Path(path)
    if not path.is_file():
        return dict(ready=False, reason='NAO asset not prepared', path=str(path))
    if path.suffix.lower() == '.urdf':
        contract = validate_urdf(path)
        manifest = path.with_name('asset_manifest.json')
        if manifest.is_file():
            record = json.loads(manifest.read_text(encoding='utf-8-sig'))
            if record['asset_sha256'] != contract['asset_sha256']:
                raise ValueError('Prepared NAO URDF changed after asset preparation')
            recorded = record.get('resolved_meshes', contract['meshes'])
            recorded = {str((path.parent/Path(k)).resolve()) if not Path(k).is_absolute() else str(Path(k).resolve()):v
                        for k,v in recorded.items()}
            if recorded != contract['meshes']:
                raise ValueError('NAO mesh bytes or resolved paths changed after preparation')
        return dict(ready=True, kind='urdf', **contract)
    if path.suffix.lower() in ('.usd', '.usda', '.usdc'):
        return dict(ready=True, kind='usd', asset_sha256=digest(path),
                    validation='Articulation, foot and physical mimic checks required in Isaac')
    raise ValueError('Use a floating-base NAO URDF or USD')
