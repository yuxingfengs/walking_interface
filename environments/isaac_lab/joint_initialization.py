"""Validate initial poses and express implicit URDF finger bounds in USD."""
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def _joints(urdf):
    robot = ET.fromstring(Path(urdf).read_bytes())
    if robot.get('name') != 'NaoH25V50':
        raise ValueError('Expected the original NAO H25 V5 joint model')
    return {joint.get('name'): joint for joint in robot.findall('joint')}


def derive_mimic_limits(urdf):
    """Return finite follower bounds implied by q_f = multiplier*q_m + offset."""
    joints = _joints(urdf)

    def bounds(name, visiting):
        if name in visiting or name not in joints:
            raise ValueError('Missing or cyclic mimic reference: ' + name)
        joint = joints[name]
        limit = joint.find('limit')
        if joint.get('type') != 'continuous' and limit is not None:
            low, high = float(limit.get('lower')), float(limit.get('upper'))
        else:
            mimic = joint.find('mimic')
            if mimic is None:
                raise ValueError('Cannot derive finite limits for ' + name)
            master_low, master_high = bounds(mimic.get('joint'), visiting | {name})
            multiplier = float(mimic.get('multiplier', '1'))
            offset = float(mimic.get('offset', '0'))
            low, high = sorted((multiplier*master_low+offset, multiplier*master_high+offset))
        if not all(map(math.isfinite, (low, high))) or low >= high:
            raise ValueError('Invalid derived mimic range: ' + name)
        return low, high

    result = {}
    for name, joint in joints.items():
        mimic = joint.find('mimic')
        if joint.get('type') == 'continuous' and mimic is not None:
            low, high = bounds(name, set())
            result[name] = dict(master=mimic.get('joint'),
                                multiplier=float(mimic.get('multiplier', '1')),
                                offset=float(mimic.get('offset', '0')),
                                lower_rad=low, upper_rad=high)
    return result


def validate_initial_pose(urdf, positions):
    joints = _joints(urdf)
    unknown = set(positions) - set(joints)
    if unknown:
        raise ValueError('Unknown initial joint names: ' + str(sorted(unknown)))
    derived = derive_mimic_limits(urdf)
    for name, joint in joints.items():
        limit = joint.find('limit')
        if joint.get('type') == 'revolute' and limit is not None:
            low, high = float(limit.get('lower')), float(limit.get('upper'))
        elif name in derived:
            low, high = derived[name]['lower_rad'], derived[name]['upper_rad']
        else:
            continue
        value = float(positions.get(name, 0.))
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'Initial joint {name}={value} outside [{low}, {high}] rad')


def apply_mimic_limits(stage, root_path, rules):
    """Author bounds/references in the runtime layer before PhysX initialization.

    The original URDF and imported USD files remain untouched. Existing finite
    bounds must contain the entire source-derived interval; mimic references
    must agree with the source. All joints are checked before authoring.
    """
    from pxr import Usd, UsdPhysics
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise ValueError('Missing NAO prim: ' + str(root_path))
    joints = {}
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdPhysics.RevoluteJoint):
            if prim.GetName() in joints:
                raise ValueError('Duplicate imported joint: ' + prim.GetName())
            joints[prim.GetName()] = prim
    plans = []
    for name, rule in rules.items():
        if name not in joints or rule['master'] not in joints:
            raise ValueError('Missing imported mimic joint or master: ' + name)
        prim, master = joints[name], joints[rule['master']]
        tokens = prim.GetMetadata('apiSchemas')
        schemas = set(prim.GetAppliedSchemas())
        if tokens:
            schemas.update(tokens.explicitItems + tokens.prependedItems + tokens.appendedItems)
        mimic = [item for item in schemas if item.startswith('PhysxMimicJointAPI:')]
        if len(mimic) != 1:
            raise ValueError('Expected one physical mimic API for ' + name)
        prefix = 'physxMimicJoint:' + mimic[0].split(':', 1)[1] + ':'
        for attr, expected in (('gearing', -rule['multiplier']), ('offset', -rule['offset'])):
            actual = prim.GetAttribute(prefix+attr).Get()
            if actual is None or not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError('Imported mimic coefficient differs from URDF: ' + name)
        relation = prim.GetRelationship(prefix+'referenceJoint')
        targets = relation.GetTargets() if relation else []
        if targets and targets != [master.GetPath()]:
            raise ValueError('Imported mimic reference differs from URDF: ' + name)
        joint = UsdPhysics.RevoluteJoint(prim)
        desired = tuple(math.degrees(rule[key]) for key in ('lower_rad', 'upper_rad'))
        previous = (joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get())
        for index, (actual, expected) in enumerate(zip(previous, desired)):
            # The importer already pads the four thumb ranges. Align those
            # with the interval implied by the original bounded hand master.
            if actual is not None and math.isfinite(actual):
                excludes_source = actual > expected+1e-5 if index==0 else actual < expected-1e-5
                if excludes_source:
                    raise ValueError('Imported finger limit excludes a source-valid angle: ' + name)
        plans.append((name, rule, joint, prefix, master, targets, desired, previous))
    records = []
    with Usd.EditContext(stage, stage.GetRootLayer()):
        for name, rule, joint, prefix, master, targets, desired, previous in plans:
            joint.CreateLowerLimitAttr(desired[0])
            joint.CreateUpperLimitAttr(desired[1])
            if not targets:
                joint.GetPrim().CreateRelationship(prefix+'referenceJoint', custom=False).SetTargets([master.GetPath()])
            records.append(dict(joint=name, path=str(joint.GetPath()), **rule,
                                reference_joint=str(master.GetPath()), reference_restored=not bool(targets),
                                previous_limits_deg=[str(value) for value in previous],
                                authored_limits_deg=[joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()]))
    return records
