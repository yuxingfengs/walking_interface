"""Read-only privileged diagnostics. Never feed these fields into Actor or Replay."""
import math
import numpy as np
from .core import JOINTS


class JointTruthTracker:
    """Independent physical joint-coordinate samples, reset separately from encoders."""
    def __init__(self):
        self.previous = None

    def update(self, positions, time_s):
        if len(positions) != len(JOINTS):
            raise ValueError('Expected 12 physical joint coordinates')
        current = [float(q) if q is not None and math.isfinite(float(q)) else None for q in positions]
        velocities = [None]*len(JOINTS)
        if self.previous is not None:
            before, previous_time = self.previous
            dt = time_s-previous_time
            if dt <= 0:
                raise ValueError('Non-increasing physical joint sample time')
            velocities = [(q-old)/dt if q is not None and old is not None else None for q,old in zip(current,before)]
        self.previous = current.copy(), float(time_s)
        return current, velocities


def bind_joint_truth(robot_node):
    """Read actual JointParameters.position, never Motor target or PositionSensor."""
    refs = []
    for name in JOINTS:
        joint = robot_node.getFromProtoDef(name)
        field = joint.getField('jointParameters') if joint is not None else None
        parameters = field.getSFNode() if field is not None else None
        refs.append(parameters.getField('position') if parameters is not None else None)
    return refs


def capture_joint_truth(env, time_s):
    if not hasattr(env, 'joint_truth_fields'):
        env.joint_truth_fields = bind_joint_truth(env.node)
    if not hasattr(env, 'joint_truth_tracker'):
        env.joint_truth_tracker = JointTruthTracker()
    values = [field.getSFFloat() if field is not None else None for field in env.joint_truth_fields]
    env.truth_joint_positions, env.truth_joint_velocities = env.joint_truth_tracker.update(values,time_s)


def rotation_vector(rotation):
    """SO(3) log, including small and near-pi rotations; result in the matrix frame."""
    angle = math.acos(float(np.clip((np.trace(rotation)-1)*.5, -1., 1.)))
    skew = np.array([rotation[2,1]-rotation[1,2], rotation[0,2]-rotation[2,0],
                     rotation[1,0]-rotation[0,1]], dtype=float)
    if angle < 1e-7:
        return .5*skew
    if math.pi-angle < 1e-5:
        values, vectors = np.linalg.eigh((rotation+rotation.T)*.5)
        axis = vectors[:, np.argmax(values)]
        if np.dot(axis, skew) < 0:
            axis = -axis
        return axis*angle
    return skew*(angle/(2*math.sin(angle)))


class TruthTracker:
    def __init__(self):
        self.previous = None

    def update(self, position, rotation, time_s):
        position, rotation = np.asarray(position, dtype=float), np.asarray(rotation, dtype=float)
        if position.shape != (3,) or rotation.shape != (3,3) or not np.isfinite(np.r_[position,rotation.ravel(),time_s]).all():
            raise ValueError('Invalid Supervisor diagnostic pose')
        if not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation)<0:
            raise ValueError('Invalid Supervisor diagnostic rotation')
        linear = angular = None
        yaw_rate = None
        if self.previous is not None:
            old_position, old_rotation, old_time = self.previous
            dt = time_s-old_time
            if dt <= 0:
                raise ValueError('Non-increasing diagnostic pose time')
            linear = rotation.T@((position-old_position)/dt)
            world_angular = rotation_vector(rotation@old_rotation.T)/dt
            angular = rotation.T@world_angular
            yaw = math.atan2(rotation[1,0], rotation[0,0])
            old_yaw = math.atan2(old_rotation[1,0], old_rotation[0,0])
            yaw_rate = math.atan2(math.sin(yaw-old_yaw),math.cos(yaw-old_yaw))/dt
        self.previous = position.copy(), rotation.copy(), float(time_s)
        gravity = rotation.T@np.array([0.,0.,-1.])
        return dict(global_vx=float(linear[0]) if linear is not None else None,
                    global_vy=float(linear[1]) if linear is not None else None,
                    global_wz=yaw_rate,
                    global_gyro_x=float(angular[0]) if angular is not None else None,
                    global_gyro_y=float(angular[1]) if angular is not None else None,
                    global_roll=math.atan2(rotation[2,1],rotation[2,2]),
                    global_pitch=math.atan2(-rotation[2,0],math.hypot(rotation[2,1],rotation[2,2])),
                    global_gx=float(gravity[0]),global_gy=float(gravity[1]),global_gz=float(gravity[2]),
                    simulation_time_s=float(time_s))


def diagnostic_row(env):
    """Capture post-action values with the source command handled by the writer."""
    fields = dict(env.truth_diagnostics)
    gravity = np.asarray(env.gravity)
    fields.update(sensor_roll=math.atan2(-float(gravity[1]),-float(gravity[2])),
                  sensor_pitch=math.asin(float(np.clip(gravity[0],-1,1))),
                  sensor_gx=float(gravity[0]),sensor_gy=float(gravity[1]),sensor_gz=float(gravity[2]),
                  sensor_gyro_x=float(env.gyro_value[0]),sensor_gyro_y=float(env.gyro_value[1]),
                  sensor_sources=dict(roll='imu',pitch='imu',gx='imu',gy='imu',gz='imu',
                                      gyro_x='gyro',gyro_y='gyro',vx='global',vy='global',wz='global'),
                  global_height=float(env.height))
    if getattr(env, 'actor_body_gyro', False):
        fields['sensor_gyro_z'] = env.actor_gyro_z
        fields['sensor_sources']['gyro_z'] = 'gyro'
    displacement = env.position-env.start_position
    heading = env.heading_reference
    fields['global_forward'] = float(math.cos(heading)*displacement[0]+math.sin(heading)*displacement[1])
    fields['global_lateral'] = float(-math.sin(heading)*displacement[0]+math.cos(heading)*displacement[1])
    fields['global_fell'] = bool(env.height < env.cfg['fall_height'] or
                                -fields['global_gz'] < math.cos(env.cfg['fall_tilt_rad']))
    fields.update(reward_input_gx=float(gravity[0]),reward_input_gy=float(gravity[1]),reward_input_gz=float(gravity[2]),
                  reward_input_gyro_x=float(env.gyro_value[0]),reward_input_gyro_y=float(env.gyro_value[1]))
    for index,name in enumerate(JOINTS):
        fields['global_position_'+name] = env.truth_joint_positions[index]
        fields['global_velocity_'+name] = env.truth_joint_velocities[index]
        fields['sensor_position_'+name] = float(env.q[index])
        fields['sensor_velocity_'+name] = float(env.qvel[index])
        fields['sensor_sources']['position_'+name] = 'joint_encoder'
        fields['sensor_sources']['velocity_'+name] = 'encoder_difference'
    # Contact geometry is an independent observation, never a replacement for force.
    if not hasattr(env,'truth_foot_nodes'):
        env.truth_foot_nodes = [env.node.getFromProtoDef(name) for name in ('LFsr','RFsr')]
    contacts = {}
    for side,node in zip(('left','right'),env.truth_foot_nodes):
        points = node.getContactPoints(False) if node is not None else None
        fields['global_'+side+'_contact_count'] = len(points) if points is not None else None
        fields['global_'+side+'_contact'] = int(bool(points)) if points is not None else None
        fields['global_'+side+'_foot_height'] = float(node.getPosition()[2]-env.cfg['floor_z']) if node is not None else None
        contacts[side] = [list(point.point) for point in points] if points is not None else None
        fields['global_'+side+'_normal_load'] = None
        fields['sensor_'+side+'_normal_load'] = float(env.support_feedback.loads[0 if side=='left' else 1])
    fields['foot_contact_points'] = contacts
    fields['sensor_support_load_difference'] = float(env.support_feedback.value)
    fields['global_support_load_difference'] = None
    fields['sensor_sources']['support_load_difference'] = 'filtered_foot_force_difference'
    fields['truth_unavailable'] = dict(support_load_difference='Independent force unavailable; contact geometry is not force')
    return fields
