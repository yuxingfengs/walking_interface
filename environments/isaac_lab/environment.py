"""Single free-base PhysX articulation, implementing the SAC Turn environment contract."""
import math
import json
from pathlib import Path
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.converters import UrdfConverterCfg
from isaaclab.utils.math import quat_apply_inverse
from pxr import Usd, UsdPhysics, UsdGeom
from basewalk.core import JOINTS, DEFAULT_POSE, MINIMUM, MAXIMUM, joint_targets, observation
from basewalk.reward_dispatch import motion_reward
from basewalk.reward_velocity import RewardVelocity
from basewalk.support_observation import SupportObservation
from joint_initialization import derive_mimic_limits, validate_initial_pose, apply_mimic_limits


def host(tensor):
    return tensor.detach().cpu().numpy().copy()


class UprightResetError(RuntimeError):
    """The free-standing robot exceeded a configured fall threshold after settling."""


class IsaacTurn:
    def __init__(self, sim, settings, cfg, asset, cache):
        self.sim, self.settings, self.cfg = sim, settings, cfg
        self.env = cfg['environment']
        self.dt = settings['decimation'] * settings['physics_dt']
        self.phase = 0.
        self.persistent_fixed_cycle_stages = None
        self.render = not settings['headless']
        material = sim_utils.RigidBodyMaterialCfg(static_friction=settings['friction'],
                                                  dynamic_friction=settings['friction'])
        if 'standing_ground' in settings:
            plane = sim_utils.UsdFileCfg(usd_path=settings['standing_ground'])
            plane.func('/World/Ground', plane)
            material.func('/World/Ground/physicsMaterial', material)
            sim_utils.bind_physics_material('/World/Ground/Floor', '/World/Ground/physicsMaterial')
        else:
            plane = sim_utils.GroundPlaneCfg(physics_material=material)
            plane.func('/World/Ground', plane)
        light = sim_utils.DomeLightCfg(intensity=1600.)
        light.func('/World/Light', light)
        spawn = dict(activate_contact_sensors=True, articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=settings.get('self_collisions', True),
            solver_position_iteration_count=settings.get('solver_position_iterations', 8),
            solver_velocity_iteration_count=settings.get('solver_velocity_iterations', 2)))
        if asset.suffix.lower()=='.urdf':
            drive = UrdfConverterCfg.JointDriveCfg(gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0., damping=0.))
            # In the installed Lab 2.2.1 converter this misleadingly named field
            # is forwarded directly to Sim 5.0 ImportConfig.set_parse_mimic().
            # True therefore enables physical URDF mimic import in this SDK.
            spawn = sim_utils.UrdfFileCfg(asset_path=str(asset), usd_dir=str(cache), fix_base=False,
                merge_fixed_joints=True, convert_mimic_joints_to_normal_joints=True, joint_drive=drive, **spawn)
        else:
            spawn = sim_utils.UsdFileCfg(usd_path=str(asset), **spawn)
        if 'standing_joint_positions' in settings:
            pose = settings['standing_joint_positions']
        else:
            pose = dict(zip(JOINTS, map(float, DEFAULT_POSE)))
            pose.update(settings['upper_joint_positions'])
        joint_source = asset if asset.suffix.lower()=='.urdf' else settings['urdf_source']
        validate_initial_pose(joint_source, pose)
        finger_limits = derive_mimic_limits(joint_source)
        legs = '|'.join(JOINTS)
        servo = settings.get('webots_leg_servo')
        driven = [j for j in JOINTS if j != 'RHipYawPitch']
        leg_drive = dict(stiffness=settings['leg_stiffness'], damping=settings['leg_damping'])
        if servo:
            # Webots motor limits are [nominal rad/s, available N m] per joint.
            limits = servo['limits_by_joint']
            speed = {j: limits[j[1:]][0] * self.env['motor_velocity_scale'] for j in JOINTS}
            leg_drive = dict(stiffness=0., damping=servo['physx_velocity_damping'],
                             effort_limit_sim={j: limits[j[1:]][1] for j in driven},
                             velocity_limit_sim={j: speed[j] for j in driven})
        self.count = settings['num_envs']
        robot_path = '/World/NAO'
        roots = [robot_path]
        if self.count > 1:
            from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
            self.scene = InteractiveScene(InteractiveSceneCfg(num_envs=self.count, env_spacing=4.,
                                                               replicate_physics=False))
            roots = [path + '/NAO' for path in self.scene.env_prim_paths]
            robot_path = self.scene.env_regex_ns + '/NAO'
        if self.render:
            scale_asset = (Path(__file__).parent / 'assets' / 'walk_ground_scale.usda').resolve()
            origins = self.scene.env_origins.tolist() if self.count > 1 else [(0., 0., 0.)]
            for index, origin in enumerate(origins):
                scale = UsdGeom.Xform.Define(sim.stage, f'/World/GroundScales/Lane_{index}')
                scale.GetPrim().GetReferences().AddReference(scale_asset.as_posix())
                scale.AddTranslateOp().Set(tuple(origin))
        robot_cfg = ArticulationCfg(prim_path=robot_path, spawn=spawn,
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.,0.,settings['spawn_height']),joint_pos=pose),
            actuators={'legs':ImplicitActuatorCfg(joint_names_expr=driven, **leg_drive),
                # The source URDF couples the right hip to the driven left hip.
                'follower':ImplicitActuatorCfg(joint_names_expr=['RHipYawPitch'],stiffness=0.,damping=0.),
                'upper':ImplicitActuatorCfg(joint_names_expr=['(?!(?:'+legs+')$).*'],
                stiffness=settings['other_stiffness'],damping=settings['other_damping'])})
        self.robot = Articulation(robot_cfg)
        finger_record = [entry for path in roots for entry in apply_mimic_limits(sim.stage, path, finger_limits)]
        if self.count > 1:
            self.scene.filter_collisions(['/World/Ground'])
        initialization_record = dict(initial_positions_rad=pose, source=str(joint_source),
                                     finger_overrides=finger_record, physics_initialized=False)
        initialization_file = cache.parent/'joint_initialization.json'
        initialization_file.write_text(json.dumps(initialization_record,indent=2),encoding='utf-8')
        prims = list(Usd.PrimRange(sim.stage.GetPrimAtPath(roots[0])))
        hips = [p for p in prims if p.GetName() == 'RHipYawPitch' and p.IsA(UsdPhysics.RevoluteJoint)]
        hip_record = dict(expected='physical_mimic_1_to_1',
                          joints=[dict(path=str(p.GetPath()), schemas=p.GetAppliedSchemas(),
                                       authored_schemas=str(p.GetMetadata('apiSchemas'))) for p in hips])
        (cache.parent/'hip_import.json').write_text(json.dumps(hip_record,indent=2),encoding='utf-8')
        if len(hips) != 1 or not any('MimicJointAPI' in api for api in hips[0].GetAppliedSchemas()):
            raise ValueError('The right hip requires a physical PhysX mimic constraint. '
                             'See hip_import.json for imported joint/schema details.')
        self.feet = []
        for name in settings['feet']:
            matches = [p for p in prims if p.GetName()==name and p.HasAPI(UsdPhysics.RigidBodyAPI)]
            if len(matches)!=1:
                raise ValueError('Expected one rigid foot named '+name)
            self.feet.append(ContactSensor(ContactSensorCfg(prim_path=str(matches[0].GetPath()).replace(roots[0], robot_path, 1))))
        sim.reset()
        if not self.robot.is_initialized:
            raise RuntimeError('NAO PhysX articulation initialization failed; see the Kit log')
        initialization_record['physics_initialized'] = True
        initialization_file.write_text(json.dumps(initialization_record,indent=2),encoding='utf-8')
        if self.robot.num_instances != self.count:
            raise ValueError('PhysX articulation count differs from configured environment count')
        if self.robot.is_fixed_base:
            raise ValueError('NAO requires a floating base')
        self.ids, names = self.robot.find_joints(list(JOINTS), preserve_order=True)
        if tuple(names)!=JOINTS:
            raise ValueError('NAO leg joint contract mismatch: '+str(names))
        limits=host(self.robot.data.joint_pos_limits[0,self.ids])
        margin=self.env['joint_margin']
        self.lower=np.maximum(limits[:,0],MINIMUM)+margin
        self.upper=np.minimum(limits[:,1],MAXIMUM)-margin
        if np.any(self.lower >= self.upper):
            raise ValueError('NAO leg joint limits do not intersect the SAC Action contract')
        if servo:
            speeds = self.robot.data.joint_vel_limits.new_tensor([[speed[j] for j in JOINTS]])
            self.servo_ids = [self.ids[i] for i, j in enumerate(JOINTS) if j in driven]
            self.servo_speed = speeds[:, [i for i, j in enumerate(JOINTS) if j in driven]]
        else:
            speeds=self.robot.data.joint_vel_limits[:,self.ids]*self.env['motor_velocity_scale']
        self.robot.write_joint_velocity_limit_to_sim(speeds.expand(self.count, -1).contiguous(),joint_ids=self.ids)
        self.targets=self.robot.data.default_joint_pos.clone()
        self.yaw_filter=RewardVelocity('command_ema',
            -math.expm1(-self.dt/cfg['reward']['biped_tracking']['turn_yaw_lowpass_tau_s']),
            continuous_turn='yaw_ramp' in cfg['training'] or 'omni_curriculum' in cfg['training'])
        if self.render:
            x, y, _ = origins[0]
            if 'standing_joint_positions' in settings:
                sim.set_camera_view(eye=(x+.88, y+.73, .54), target=(x, y, .27))
            else:
                sim.set_camera_view(eye=(x+1.2, y+1.2, .85), target=(x, y, .23))

    def _advance_physics(self, trace=None):
        self.robot.set_joint_position_target(self.targets)
        if trace is not None:
            trace(0, None)
        for substep in range(1, self.settings['decimation'] + 1):
            drive_velocity = None
            if 'webots_leg_servo' in self.settings:
                error = self.targets[:, self.servo_ids] - self.robot.data.joint_pos[:, self.servo_ids]
                velocity = self.settings['webots_leg_servo']['position_gain_per_s'] * error
                # A failed lane can still be present in the shared PhysX view during cooldown.
                # Never feed its invalid measured joint angles back into a drive target.
                velocity.nan_to_num_(nan=0., posinf=0., neginf=0.)
                drive_velocity = velocity.clamp(-self.servo_speed, self.servo_speed)
                self.robot.set_joint_velocity_target(drive_velocity, joint_ids=self.servo_ids)
            self.robot.write_data_to_sim()
            self.sim.step(render=self.render)
            self.robot.update(self.settings['physics_dt'])
            for foot in self.feet:
                foot.update(self.settings['physics_dt'])
            if trace is not None:
                trace(substep, drive_velocity)
        return torch.stack([foot.data.net_forces_w[:, 0] for foot in self.feet], dim=1)

    def _advance(self):
        return host(self._advance_physics()[0])

    def _read(self):
        data=self.robot.data
        self.pos=host(data.root_link_pos_w[0])
        quat=data.root_link_quat_w[:1]
        w,x,y,z=host(quat[0])
        self.yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
        self.gravity=host(data.projected_gravity_b[0])
        self.gyro=host(data.root_ang_vel_b[0])
        self.gyro_value = self.gyro[:2]
        self.actor_gyro_z=float(self.gyro[2])
        self.q=host(data.joint_pos[0,self.ids])
        if not np.isfinite(np.r_[self.pos, self.gravity, self.gyro, self.q]).all():
            raise FloatingPointError('Nonfinite PhysX feedback')
        return quat

    def _obs(self):
        base=observation(self.gravity,self.gyro[:2],self.command,self.phase,self.q,self.qvel,
            self.last,self.env,yaw_rate=self.velocity[2],body_velocity=self.velocity[:2],
            support_load_difference=self.support.value)
        scale=self.env['yaw_rate_scale']
        key = self.yaw_filter.command_key(self.command, 0)
        valid=self.yaw_filter.key == key and np.linalg.norm(self.command[:2]) <= 1e-8
        ema=self.yaw_filter.value[2] if valid else 0.
        return np.r_[base,self.actor_gyro_z*scale,ema*scale,float(valid),0.,0.].astype(np.float32)

    def _fell(self):
        return bool(self.pos[2]-self.env['floor_z']<self.env['fall_height'] or
                    -self.gravity[2]<math.cos(self.env['fall_tilt_rad']))

    def reset(self,command, *, observe_settle=None):
        retry_log = getattr(self, 'reset_retry_log', None)
        attempts = 3 if retry_log and observe_settle is None else 1
        for attempt in range(1, attempts + 1):
            try:
                return self._reset_once(command, observe_settle=observe_settle)
            except UprightResetError as exc:
                if retry_log:
                    import json
                    with retry_log.open('a', encoding='utf-8') as stream:
                        stages = self.persistent_fixed_cycle_stages.state()
                        stream.write(json.dumps(dict(attempt=attempt, stages=stages, error=str(exc))) + '\n')
                if attempt == attempts:
                    raise

    def _reset_once(self, command, *, observe_settle=None):
        self.command=np.asarray(command,np.float32)
        self.phase=float(self.env.get('initial_gait_phase',0.))
        self.elapsed=0.
        self.last=np.zeros(12,np.float32)
        self.velocity=np.zeros(3,np.float32)
        self.qvel=np.zeros(12,np.float32)
        self.yaw_filter.reset()
        robot=self.robot
        state=robot.data.default_root_state.clone()
        robot.write_root_pose_to_sim(state[:,:7])
        robot.write_root_velocity_to_sim(torch.zeros_like(state[:,7:]))
        self.targets=robot.data.default_joint_pos.clone()
        robot.write_joint_state_to_sim(self.targets,torch.zeros_like(self.targets))
        robot.reset()
        for sensor in self.feet:
            sensor.reset()
        history=[]
        if observe_settle is not None:
            self._read()
            observe_settle(0, None)
        for step in range(1, self.env['settle_steps'] + 1):
            forces = self._advance()
            history.append(forces)
            if observe_settle is not None:
                self._read()
                observe_settle(step, forces)
        self._read()
        if self._fell():
            tilt = math.degrees(math.acos(np.clip(-self.gravity[2], -1., 1.)))
            raise UprightResetError(f'NAO failed unsupported upright reset '
                                   f'(root z={self.pos[2]:.5f} m, tilt={tilt:.2f} deg)')
        self.support=SupportObservation(self.cfg['support_observation'],history)
        self.qvel = np.zeros(12, np.float32)
        return self._obs()

    def step(self,action):
        qtarget,executed=joint_targets(action,self.env['action_scale'],self.lower,self.upper)
        self.targets[:,self.ids]=torch.as_tensor(qtarget,device=self.sim.device)
        old_pos,old_yaw,old_q=self.pos.copy(),self.yaw,self.q.copy()
        self.support.advance(self._advance())
        quat=self._read()
        world_vel=torch.as_tensor((self.pos-old_pos)/self.dt,device=quat.device,dtype=quat.dtype)[None]
        linear=host(quat_apply_inverse(quat,world_vel)[0])
        wz=math.atan2(math.sin(self.yaw-old_yaw),math.cos(self.yaw-old_yaw))/self.dt
        return self.finish_step(executed, old_pos, old_q, linear, wz)

    def finish_step(self, executed, old_pos, old_q, linear, wz):
        self.velocity=np.asarray([*linear[:2],wz],np.float32)
        self.qvel=(self.q-old_q)/self.dt
        gyro_velocity=np.array([*linear[:2],self.actor_gyro_z],np.float32)
        command = self.command
        mode = 2 if np.linalg.norm(command[:2]) > 1e-8 else 0
        reward_velocity = gyro_velocity.copy()
        if mode == 0:
            filtered = self.yaw_filter.advance(gyro_velocity, command, mode)
            if not self.cfg['reward']['biped_tracking'].get('turn_yaw_use_raw_gyro'):
                reward_velocity[2] = filtered[2]
        else:
            self.yaw_filter.reset()
        reward,terms=motion_reward(command,reward_velocity,self.gyro[:2],self.gravity,mode,self.cfg)
        fell=self._fell()
        terms['fall']=-self.cfg['reward']['fall_penalty'] if fell else 0.
        before=self.phase
        self.phase=(before+self.dt*self.env['gait_frequency_hz'])%1.
        if self.persistent_fixed_cycle_stages is not None:
            self.persistent_fixed_cycle_stages.advance_phase(before,self.phase,fell=fell)
            self.command = self.persistent_fixed_cycle_stages.command(0, 0)
        self.elapsed+=self.dt
        self.last=executed
        truncated=not self.env.get('disable_time_limit',False) and self.elapsed+1e-8>=self.env['episode_seconds']
        info=dict(executed_action=executed,command=command, motion_phase='walk' if mode == 2 else 'turn', actual_velocity=self.velocity,
                  position=self.pos,yaw=self.yaw,reward_terms=terms,reward_velocity=reward_velocity,
                  support_loads=self.support.loads.copy(),
                  cycle_actual=[*(self.pos-old_pos)[:2], wz*self.dt], cycle_dt=self.dt)
        return self._obs(), reward+terms['fall'], fell, truncated, info
