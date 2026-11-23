"""Check the single-robot SAC contract without simulation or optimizer updates."""
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from configuration import CODE, HERE, SAC, SIM, launch_configuration, load_configuration, read_json, sha
from asset_contract import inspect_asset


def runtime_status(settings):
    python = settings.get('isaac_python')
    if not python:
        checkout = os.environ.get('ISAACLAB_PATH')
        if checkout:
            # The wrapper may select its bundled Python; the real simulation check
            # is still required to validate that installation.
            launcher = Path(checkout) / 'isaaclab.bat'
            return dict(ready=launcher.is_file(), launcher=str(launcher), verified=False)
        return dict(ready=False, reason='isaac_python and ISAACLAB_PATH are unset')
    if not Path(python).is_file():
        return dict(ready=False, reason='Configured Isaac Python is missing', python=python)
    script = ("import json,sys,importlib.util; sys.path.insert(0,sys.argv[1]); "
              "from runtime_versions import versions,dependency_status; s=json.loads(sys.argv[2]); "
              "v=versions(s); found={n:bool(importlib.util.find_spec(n)) for n in ('isaacsim','isaaclab','torch')}; "
              "print(json.dumps(dict(python=list(sys.version_info[:2]),packages=v,modules=found,dependencies=dependency_status())))")
    try:
        result = subprocess.run([python, '-c', script, str(SIM), json.dumps(settings)],
                                capture_output=True, text=True, timeout=20)
        if result.returncode:
            raise ValueError(result.stderr.strip())
        versions = json.loads(result.stdout.strip())
        valid = (versions['python'] == [3, 11] and
                 str(versions['packages']['isaacsim']).startswith('5.0.') and
                 str(versions['packages']['isaaclab']).startswith('2.2.') and all(versions['modules'].values())
                 and versions['dependencies']['complete'])
        reason = None if valid else 'Expected Python 3.11, Isaac Sim 5.0.x, Isaac Lab 2.2.x'
        if not versions['dependencies']['complete']:
            reason = 'Isaac Lab dependencies are incomplete; use Install Isaac Dependencies.cmd'
        elif not all(versions['modules'].values()):
            reason = 'Required Isaac modules are not importable in the configured Python'
        return dict(ready=valid, python=python, versions=versions, reason=reason)
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return dict(ready=False, reason=str(exc), python=python)


def check(output=None, experiment=None, *, cfg=None):
    if cfg is None:
        cfg = load_configuration(experiment=experiment)
    sys.path.insert(0, str(CODE / 'training'))
    import numpy as np
    from basewalk.sac import SAC as Agent
    from basewalk.core import DEFAULT_POSE, joint_targets, observation
    from basewalk.network_inputs import actor_input_dim, observation_dim
    from basewalk.reward_dispatch import motion_reward
    from basewalk.reward_velocity import RewardVelocity
    from basewalk.fixed_cycle import online_stages

    agent = Agent(cfg, device='cpu')
    env = cfg['environment']
    walk = cfg['training'].get('walk_commands')
    initial_command = online_stages(cfg).command(0, 0)
    obs = np.r_[observation([0., 0., -1.], [0., 0.], initial_command, 0.,
        DEFAULT_POSE, np.zeros(12), np.zeros(12), env, yaw_rate=0.,
        body_velocity=[0., 0.], support_load_difference=0.), np.zeros(5)].astype(np.float32)
    action = agent.act(obs, explore=False)
    targets, executed = joint_targets(action, env['action_scale'])
    if (obs.shape != (55,) or action.shape != (12,) or not np.isfinite(action).all() or
            action[0] != action[6] or targets[0] != targets[6] or executed[0] != executed[6]):
        raise ValueError('BC Observation or coupled Action contract failed')
    params = cfg['reward']['biped_tracking']
    if params['turn_yaw_underspeed_width_rad_s'] != params['turn_yaw_gaussian_width_rad_s']:
        raise ValueError('Expected the approved symmetric yaw Reward')
    width = params['turn_yaw_gaussian_width_rad_s']
    peak = params['turn_yaw_gaussian_weight']
    vectors = []
    turn_probes = [] if walk else [(0., 0., .11), (0., 0., .03), (0., 0., .19),
                                   (.03, .05, .31), (0., 0., -.11)]
    for vx, vy, yaw in turn_probes:
        value, terms = motion_reward([0., 0., .11], [vx, vy, yaw], [0., 0.], [0., 0., -1.], 0, cfg)
        expected = peak * math.exp(-((yaw - .11) / width) ** 2)
        expected_x = .5 * params['turn_xy_penalty_scale'] * (math.exp(-(vx / .03)**2) - 1.)
        expected_y = .5 * params['turn_xy_penalty_scale'] * (math.exp(-(vy / .05)**2) - 1.)
        if not np.allclose([terms['tracking_yaw'], terms['tracking_x'], terms['tracking_y']],
                           [expected, expected_x, expected_y], rtol=1e-5, atol=1e-7):
            raise ValueError('Source SAC Reward test vector mismatch')
        vectors.append(dict(velocity=[vx, vy, yaw], reward=float(value)))
    dt = cfg['isaac']['physics_dt'] * cfg['isaac']['decimation']
    alpha = -math.expm1(-dt / params['turn_yaw_lowpass_tau_s'])
    ramp = 'yaw_ramp' in cfg['training']
    omni = cfg['training'].get('omni_curriculum')
    filt = RewardVelocity('command_ema', alpha, continuous_turn=bool(ramp or omni))
    command = np.array([0., 0., .11], np.float32)
    filt.advance([0., 0., .1], command, 0)
    if ramp or omni:
        command[2] = 0.
    after = filt.advance([0., 0., .3], command, 0)[2]
    if not math.isclose(float(after), .1 + alpha * .2, abs_tol=1e-7):
        raise ValueError('Yaw EMA update mismatch')
    filt.reset()
    if filt.value is not None or not np.isclose(filt.advance([0., 0., .2], command, 0)[2], .2):
        raise ValueError('Yaw EMA reset must seed from the first new sample')
    stages = online_stages(cfg)
    if walk:
        cycle_vx = initial_command[0]
        random_forward = cfg['experiment_name'] == 'walk_lateral_random_vx'
        if random_forward:
            low, high = cfg['environment']['command_ranges'][0]
            if not low <= cycle_vx <= high:
                raise ValueError('Sampled Walk vx outside configured range')
        for index, xy in enumerate(walk['commands_m_s']):
            for offset in (0, walk['hold_steps'] - 1):
                clock = index * walk['hold_steps'] + offset
                cmd = online_stages(cfg, {'step': clock}).command(0, 0)
                expected_vx = cycle_vx if random_forward else xy[0]
                np.testing.assert_allclose(cmd, [expected_vx, xy[1], 0.])
            obs[5:8] = cmd
            if not np.isfinite(agent.act(obs, explore=False)).all():
                raise ValueError('Invalid Walk policy output')
            value, _ = motion_reward(cmd, cmd, [0., 0.], [0., 0., -1.], 2, cfg)
            np.testing.assert_allclose(value, 2 * params['walk_linear_peak'])
            wrong = [cmd[0], -cmd[1], 0.] if cmd[1] else [-cmd[0], 0., 0.]
            missed, _ = motion_reward(cmd, wrong, [0., 0.], [0., 0., -1.], 2, cfg)
            if missed >= value:
                raise ValueError('Wrong Walk direction scored as highly as the requested velocity')
            vectors.append(dict(command=cmd.tolist(), matched=value, reversed_axis=missed))
    elif ramp:
        commands = []
        for step in range(stages.period):
            commands.append(stages.command(0, 0)[2])
            stages.advance_phase(.2, .3, fell=(step == stages.hold))
        levels = (1, 0, -1, 0, 1)
        for i in range(4):
            expected = np.r_[np.full(stages.hold, levels[i]),
                             np.linspace(levels[i], levels[i+1], stages.ramp + 1)[1:]] * stages.rate
            np.testing.assert_allclose(np.asarray(commands).reshape(4, -1)[i], expected, atol=1e-7)
        if stages.state()['ramp_step'] != 0:
            raise ValueError('Yaw ramp did not repeat')
    elif omni:
        for stage in omni['walk_stages']:
            stage_start = stage['start_step']
            for index, xy in enumerate(stage['commands_m_s']):
                walk_block, within_block = divmod(index * stages.walk_speed_steps, stages.block)
                step = stage_start + 2 * walk_block * stages.block + within_block
                probe = online_stages(cfg, {'step': step})
                cmd = probe.command(0, 0)
                np.testing.assert_allclose(cmd, [*xy, 0.], atol=1e-7)
                obs[5:8] = cmd
                if not np.isfinite(agent.act(obs, explore=False)).all():
                    raise ValueError('Invalid policy output for planar command')
                _, terms = motion_reward(cmd, cmd, [0., 0.], [0., 0., -1.], 2, cfg)
                np.testing.assert_allclose(terms['tracking_x'], params['walk_linear_peak'])
        if 'turn_ramp_steps' not in omni:
            levels = list(range(12)) + list(range(10, -12, -1)) + list(range(-10, 1))
            targets = []
            for index, level in enumerate(levels):
                for edge in (0, 99):
                    block, offset = divmod(index * 100 + edge, stages.block)
                    targets.append(((2 * block + 1) * stages.block + offset, level * 0.01))
        else:
            targets = ((stages.block, 0.), (stages.block + stages.ramp, stages.peak),
                       (3*stages.block + stages.ramp, -stages.peak))
        for step, target in targets:
            probe = online_stages(cfg, {'step': step - 1})
            probe.advance_phase(.4, .5, fell=True)
            cmd = probe.command(0, 0)
            np.testing.assert_allclose(cmd, [0., 0., target], atol=1e-7)
            _, terms = motion_reward(cmd, cmd, [0., 0.], [0., 0., -1.], 0, cfg)
            np.testing.assert_allclose(terms['tracking_yaw'], peak)
    else:
        for _ in range(7):
            stages.advance_phase(.99, .01)
        stages.advance_phase(.99, .01, fell=True)
        if stages.cycles != 8:
            raise ValueError('Cycle-end fall must count exactly one attempt')
        restored = online_stages(cfg, stages.state())
        for _ in range(restored.limit - restored.cycles):
            restored.advance_phase(.99, .01)
        if restored.cycles != 0 or restored.stage != restored.order[1 % len(restored.order)]:
            raise ValueError('Online scheduler continuation failed')
    if agent.updates or len(agent.actor_optimizer.state) or agent.replay.count:
        raise ValueError('Offline check unexpectedly trained or populated Replay')
    asset = inspect_asset(cfg['isaac']['asset'])
    runtime = runtime_status(cfg['isaac'])
    manifest = read_json(Path(cfg['initialization']['actor_path']).with_name('manifest.json'))
    dataset_verified = False
    if 'dataset' in manifest:
        dataset = Path(cfg['initialization']['actor_path']).parent / manifest['dataset']
        if dataset.is_file():
            if sha(dataset) != manifest['dataset_sha256']:
                raise ValueError('BC source expert dataset hash changed')
            dataset_verified = True
    elif not manifest.get('demonstration_sha256'):
        raise ValueError('BC manifest lacks demonstration provenance')
    report = dict(status='offline_contract_passed', simulation_validated=False,
        training_started=False, runtime_ready=runtime['ready'], asset_ready=asset['ready'],
        runtime=runtime, asset=asset, num_envs=cfg["isaac"]["num_envs"],
        sim_device=cfg['isaac']['device'], learner_device=cfg['isaac'].get('learner_device', cfg['isaac']['device']),
        observation_dim=observation_dim(cfg),
        actor_input_dim=actor_input_dim(cfg), action_dim=12, independent_action_dim=11,
        optimizer_updates=agent.updates, replay_count=agent.replay.count,
        budget=cfg['budget_source'], command_rad_s=.11, yaw_width_rad_s=width,
        yaw_ema_alpha=alpha, reward_sha256=cfg['reward_source_sha256'],
        bc_actor_sha256=cfg['initialization']['sha256'],
        bc_manifest=manifest, expert_dataset_hash_verified=dataset_verified, reward_vectors=vectors)
    if walk:
        report.update(command_rad_s=0., walk_hold_steps=walk['hold_steps'],
                      tracking_scale_m_s=params['tracking_scale_m_s'])
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment')
    parser.add_argument('--sim-device', choices=('cpu', 'cuda:0'))
    parser.add_argument('--learner-device', choices=('cpu', 'cuda:0'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    cfg = launch_configuration(args.experiment)
    if args.sim_device:
        cfg['isaac']['device'] = args.sim_device
    if args.learner_device:
        cfg['isaac']['learner_device'] = args.learner_device
    reports = Path(cfg['project_layout']['reports_root'])
    if reports == HERE/'reports':
        reports = SAC/'experiments'/cfg['isaac']['experiment_id']/'reports'
    result = check(args.output or reports/'offline_check.json', cfg=cfg)
    print(json.dumps({key: result[key] for key in ('status', 'runtime_ready', 'asset_ready',
          'num_envs', 'sim_device', 'learner_device', 'observation_dim', 'actor_input_dim',
          'optimizer_updates', 'training_started')}, indent=2))
    print('Online budget:', result['budget']['effective_budget'], 'physical transitions')
    print('Physics and NAO simulation checks remain separate from dependency and asset readiness.')
