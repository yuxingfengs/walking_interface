"""One isolated headless Isaac Sim instance per allocated GPU."""
import math
from argparse import Namespace
import sys
import traceback
from pathlib import Path


def simulate(pipe, env, stages):
    env.persistent_fixed_cycle_stages = stages
    pipe.send(env.reset(command=stages.command(0, 0)))
    while True:
        request = pipe.recv()
        if request is None:
            return
        action, last = request
        before = env.phase
        nxt, reward, fell, truncated, info = env.step(action)
        vx, vy, wz = map(float, info['command'])
        ax, ay, az = map(float, info['actual_velocity'])
        terms = info['reward_terms']
        row = dict(motion_phase=info['motion_phase'], vx_cmd=vx, vy_cmd=vy, wz_cmd=wz,
                   actual_vx=ax, actual_vy=ay, actual_wz=az, reward=float(reward),
                   reward_fall=float(terms['fall']), reward_tracking_yaw=float(terms['tracking_yaw']),
                   reward_yaw_rate=float(info['reward_velocity'][2]), fell=bool(fell),
                   truncated=bool(truncated), tilt_rad=math.atan2(math.hypot(*env.gravity[:2]), -env.gravity[2]))
        wrapped = env.phase < before
        following = env.reset(command=stages.command(0, 0)) if (fell or truncated) and not last else None
        pipe.send((nxt, info['executed_action'], row, wrapped, following))


def worker(pipe, rank, cfg, folder, local_step):
    app = None
    try:
        here = Path(cfg['project_layout']['project_root'])
        sys.path.insert(0, str(folder/'source'))
        from runtime_versions import kit_arguments, project_experience
        from isaaclab.app import AppLauncher
        work = folder/f'gpu_{rank}'
        work.mkdir()
        settings = dict(cfg['isaac'], device=f'cuda:{rank}', headless=True)
        args = kit_arguments(settings, here, state_root=work/'kit', native_warp=sys.platform == 'linux')
        launch = Namespace(headless=True, device=settings['device'], multi_gpu=False, experience='',
                           kit_args=args + ' --/renderer/multiGpu/enabled=false')
        launch.experience = project_experience(launch, settings, work)
        launcher = AppLauncher(launch)
        app = launcher.app
        import isaaclab.sim as sim_utils
        import torch
        from environment import IsaacTurn
        from basewalk.fixed_cycle import online_stages
        torch.cuda.set_device(rank)
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
            dt=settings['physics_dt'], device=settings['device'], render_interval=settings['decimation']))
        env = IsaacTurn(sim, settings, cfg, Path(settings['asset']), work/'usd')
        if env.robot.data.joint_pos.device != torch.device(settings['device']):
            raise RuntimeError(f'GPU {rank} received articulation on {env.robot.data.joint_pos.device}')
        env.reset_retry_log = work/'reset_retries.jsonl'
        from main import preflight
        from basewalk.runner import write_json
        check = preflight(env, cfg, settings['check_steps'], work)
        check.update(rank=rank, physics_device=str(env.robot.data.joint_pos.device),
                     gpu_name=torch.cuda.get_device_name(rank))
        write_json(work/'check.json', check)
        simulate(pipe, env, online_stages(cfg, {'step': local_step}))
    except (BrokenPipeError, EOFError):
        return
    except BaseException:
        pipe.send({'worker_error': traceback.format_exc()})
        raise
    finally:
        pipe.close()
        if app is not None:
            app.close(wait_for_replicator=False)
