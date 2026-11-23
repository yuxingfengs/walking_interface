"""Fixed-target NAO standing comparisons by spawn height or physics step."""
import csv
import json
import math


# Matched pair for the approved H25 pose. See JOINT_INITIALIZATION.md.
HEIGHTS_M = {'baseline': 0.306172, 'foot_geometry': 0.3275423}
TIMESTEPS = {'step_20ms': (0.020, 2), 'step_5ms': (0.005, 8)}


def configure_standing_pose(settings, case, assets):
    pose = json.loads((assets/'ddpg_standing_pose.json').read_text(encoding='utf-8'))
    configure_case(settings, case, pose['settled_height_above_floor_m'])
    settings.update(standing_joint_positions=pose['joint_positions_rad'],
                    standing_pose_source=pose['ddpg_run'],
                    standing_ground=str(assets/'standing_ground.usda'))


def configure_case(settings, case, ddpg_height):
    for key in ('self_collisions', 'solver_position_iterations', 'solver_velocity_iterations',
                'leg_stiffness', 'leg_damping'):
        settings[key] = settings['standing'][key]
    if case in TIMESTEPS:
        settings['spawn_height'] = HEIGHTS_M['baseline']
        settings['physics_dt'], settings['decimation'] = TIMESTEPS[case]
    elif case == 'ddpg_settled':
        settings['spawn_height'] = ddpg_height
    else:
        settings['spawn_height'] = HEIGHTS_M[case]


def check_standing(env, cfg, case, folder):
    import numpy as np
    from basewalk.core import JOINTS
    from environment import UprightResetError

    thresholds = cfg['environment']
    floor = thresholds['floor_z']
    minimum_z = thresholds['fall_height']
    maximum_tilt = thresholds['fall_tilt_rad']
    fields = ['step', 'time_s', 'root_z_m', 'tilt_deg', 'hip_difference_rad', 'left_net_fz_n',
              'right_net_fz_n', 'below_height', 'over_tilt', *JOINTS]
    summary = dict(case=case, spawn_height_m=env.settings['spawn_height'],
                   minimum_height_above_floor_m=minimum_z, floor_z_m=floor,
                   maximum_tilt_deg=math.degrees(maximum_tilt),
                   control_period_s=env.dt, first_threshold_crossing=None,
                   first_hip_violation=None, training_started=False)
    hip_tolerance = env.settings['hip_tolerance_rad']
    summary.update(physics_dt=env.settings['physics_dt'], decimation=env.settings['decimation'],
                   hip_tolerance_rad=hip_tolerance, peak_tilt_deg=0., peak_hip_difference_rad=0.)
    body_poses = []

    with (folder/'standing.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()

        def record(step, forces):
            if case in TIMESTEPS:
                data = env.robot.data
                pos = data.body_link_pos_w[0].detach().cpu().numpy()
                quat = data.body_link_quat_w[0].detach().cpu().numpy()
                body_poses.append(np.concatenate((pos, quat), axis=1))
            vertical = float(-env.gravity[2])
            height = float(env.pos[2])
            hip_difference = abs(float(env.q[0]) - float(env.q[6]))
            row = dict(step=step, time_s=step*env.dt, root_z_m=height,
                       hip_difference_rad=hip_difference,
                       tilt_deg=math.degrees(math.acos(max(-1., min(1., vertical)))),
                       left_net_fz_n=None, right_net_fz_n=None,
                       below_height=height-floor < minimum_z,
                       over_tilt=vertical < math.cos(maximum_tilt))
            if forces is not None:
                row.update(left_net_fz_n=float(forces[0, 2]), right_net_fz_n=float(forces[1, 2]))
            summary['peak_tilt_deg'] = max(summary['peak_tilt_deg'], row['tilt_deg'])
            summary['peak_hip_difference_rad'] = max(summary['peak_hip_difference_rad'], hip_difference)
            if summary['first_threshold_crossing'] is None and (row['below_height'] or row['over_tilt']):
                summary['first_threshold_crossing'] = row.copy()
            if summary['first_hip_violation'] is None and hip_difference > hip_tolerance:
                summary['first_hip_violation'] = row.copy()
            summary['final'] = row.copy()
            writer.writerow({**row, **dict(zip(JOINTS, map(float, env.q)))})
            stream.flush()

        try:
            env.reset((0., 0., 0.11), observe_settle=record)
        except UprightResetError as exc:
            summary.update(final_reset_passed=False, reason=str(exc))
        else:
            summary['final_reset_passed'] = True
    summary['upright'] = summary['final_reset_passed'] and summary['first_threshold_crossing'] is None
    summary['final_both_feet_loaded'] = all((summary['final'][key] or 0.) > 0.
                                            for key in ('left_net_fz_n', 'right_net_fz_n'))
    summary['passed'] = (summary['upright'] and summary['first_hip_violation'] is None
                         and summary['final_both_feet_loaded'])
    if body_poses:
        np.savez_compressed(folder/'body_poses.npz', names=env.robot.body_names,
                            poses=body_poses, control_period_s=env.dt)
    return summary
