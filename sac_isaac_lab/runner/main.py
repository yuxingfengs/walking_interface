"""Independent Isaac Lab launcher for NAO SAC Turn. Check runs no optimizer steps."""
import argparse
import csv
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import uuid

from configuration import CODE, HERE, SAC, SIM, checkpoint_path, experiment_directory, launch_configuration, load_configuration
from asset_contract import inspect_asset
from runtime_versions import require_versions, kit_arguments, dependency_status, project_experience


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(folder, settings, asset, mode='check', *, config=None):
    cfg = (deepcopy(config) if config is not None else
           load_configuration(experiment=settings.get('experiment_variant', settings['experiment_id'])))
    if mode == 'standing':
        control_dt = settings['physics_dt'] * settings['decimation']
        hold = settings['standing']['hold_seconds']
        if not math.isfinite(hold) or hold <= 0:
            raise ValueError('Standing hold_seconds must be finite and positive')
        cfg['environment']['settle_steps'] = math.ceil(hold / control_dt)
    if mode == 'smoke' and settings.get('experiment_variant') in ('walk_lateral', 'walk_lateral_random_vx'):
        total = cfg['sac']['learning_starts'] + cfg['training']['batch_size']
        cfg['training']['total_env_steps'] = total
        cfg['budget_source'].update(effective_budget=total, purpose='batched_optimizer_check')
    elif mode == 'smoke':
        ramp = cfg['training']['yaw_ramp']
        period = 4 * (ramp['quarter_ramp_steps'] + ramp['hold_steps'])
        total = max(period + 1, cfg['sac']['learning_starts'] + 1)
        cfg['training']['total_env_steps'] = total
        cfg['budget_source'].update(effective_budget=total, purpose='one_reversal_cycle_and_optimizer_check')
    cfg['isaac'].update(settings, asset=str(asset), asset_sha256=sha(asset))
    source=folder/'source'
    source.mkdir()
    experiment_folder = Path(cfg['project_layout']['experiment_folder'])
    recipe = (CODE/'training/legacy_fixed.py' if experiment_folder == HERE
              else experiment_folder/'training.py')
    shutil.copyfile(recipe, source/'training.py')
    if not settings['headless']:
        (source/'assets').mkdir()
        for name in ('walk_ground_scale.usda', 'walk_ground_scale.png'):
            shutil.copyfile(SIM/'assets'/name, source/'assets'/name)
    shutil.copytree(CODE/'training/basewalk',source/'basewalk',ignore=shutil.ignore_patterns('__pycache__'))
    sources = {
        SIM: ('environment.py', 'batched_environment.py', 'asset_contract.py', 'joint_initialization.py',
              'runtime_versions.py', 'standing_check.py'),
        CODE/'runner': ('main.py', 'configuration.py', 'walk_batch.py', 'run_storage.py'),
        SIM/'setup': ('runtime_requirements.txt',),

    }
    for origin, names in sources.items():
        for name in names:
            shutil.copyfile(origin/name, source/name)
    shutil.copyfile(SIM/'setup/simulator_config.json', source/'config.json')
    shutil.copyfile(SAC/'config.json', source/'project_config.json')
    shutil.copyfile(CODE/'training_config.json', source/'isaac_training_config.json')
    if settings['experiment_id'] in ('004', 'omni'):
        source_id = settings['experiment_id']
        if source_id == 'omni':
            shutil.copyfile(experiment_directory('omni')/'config.json', source/'omni_config.json')
            if settings.get('experiment_variant') in ('walk_only', 'walk_variable', 'walk_lateral', 'walk_lateral_random_vx'):
                shutil.copyfile(experiment_directory('walk_only')/'config.json', source/'walk_config.json')
                variant = settings['experiment_variant']
                if variant not in ('walk_only', 'walk_lateral_random_vx'):
                    shutil.copyfile(experiment_directory(variant)/'config.json', source/f'{variant}_config.json')
            elif 'experiment_variant' in settings:
                shutil.copyfile(experiment_directory(settings['experiment_variant'])/'turn.json', source/'turn.json')
            source_id = '002'
        descriptor = experiment_directory('004') if source_id == '004' else SAC/'experiments'/source_id
        descriptor_file = descriptor/'config.json'
        shutil.copyfile(descriptor_file, source/'experiment_config.json')
        task_file = json.loads(descriptor_file.read_text(encoding='utf-8-sig')).get('training_config', 'training_config.json')
        shutil.copyfile(descriptor/task_file, source/'experiment_training_config.json')
    if 'standing_ground' in settings:
        shutil.copyfile(SIM/'assets/ddpg_standing_pose.json', source/'ddpg_standing_pose.json')
        shutil.copyfile(settings['standing_ground'], source/'standing_ground.usda')
    shutil.copytree(CODE/'training/scripts',source/'scripts', ignore=shutil.ignore_patterns('__pycache__'))
    reward=source/'reward.py'
    shutil.copyfile(cfg['reward_module_path'],reward)
    if sha(reward)!=cfg['reward_source_sha256']:
        raise ValueError('Reward changed during snapshot')
    cfg['reward_module_path']=str(reward)
    if cfg.get('initialization',{}).get('use_bc'):
        init=cfg['initialization']
        actor=source/'actor.pth'
        shutil.copyfile(init['actor_path'],actor)
        shutil.copyfile(Path(init['actor_path']).with_name('manifest.json'),source/'bc_manifest.json')
        if sha(actor)!=init['sha256']:
            raise ValueError('BC checkpoint changed during snapshot')
        init['actor_path']=str(actor)
    record=asset.with_name('asset_manifest.json')
    if record.exists():
        shutil.copyfile(record,source/'asset_manifest.json')
    cfg['source_hashes']={p.relative_to(source).as_posix():sha(p) for p in source.rglob('*') if p.is_file()}
    (folder/'effective_config.json').write_text(json.dumps(cfg,indent=2), encoding='utf-8')
    sys.path[:0]=[str(source)]
    return cfg


def preflight(env, cfg, steps, folder):
    import numpy as np
    import torch
    from basewalk.sac import SAC
    from basewalk.fixed_cycle import online_stages
    stages = online_stages(cfg)
    ramp = 'yaw_ramp' in cfg['training']
    omni = cfg['training'].get('omni_curriculum')
    walk = cfg['training'].get('walk_commands')
    env.persistent_fixed_cycle_stages=stages
    agent=SAC(cfg)
    rates=[]
    falls = 0
    max_hip=0.
    fields = ('trial step reward body_gyro_z reward_yaw_rate actual_wz x y height '
              'left_load right_load hip_difference attempts ramp_step fell truncated task vx_cmd vy_cmd wz_cmd').split()
    with (folder / 'fixed_policy.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for trial in range(cfg['isaac']['check_resets']):
            if walk:
                stages = online_stages(cfg, {'step': (1 + trial) * walk['hold_steps']})
                env.persistent_fixed_cycle_stages = stages
            elif omni:
                # Exercise both lateral Walk and a signed Turn before starting a fresh training clock.
                offset = omni['walk_stages'][-1]['start_step'] + (2 + trial) * omni['block_steps']
                stages = online_stages(cfg, {'step': offset})
                env.persistent_fixed_cycle_stages = stages
            before = stages.state()
            obs = env.reset(stages.command(0,0))
            if stages.state() != before or obs[51] != 0 or obs[52] != 0 or np.any(obs[34:46]):
                raise ValueError('Reset must preserve attempts and clear Action/EMA history')
            for step in range(steps):
                if obs.shape!=(55,) or not np.isfinite(obs).all():
                    raise ValueError('Invalid NAO Observation')
                act=agent.act(obs,explore=False)
                if act.shape!=(12,) or act[0]!=act[6]:
                    raise ValueError('Invalid coupled Action')
                state = stages.state()
                active = stages.command(0, 0)
                obs,reward,fell,truncated,info=env.step(act)
                if walk and (info['motion_phase'] != 'walk' or not np.array_equal(active, info['command'])
                             or not np.array_equal(obs[5:8], stages.command(0, 0))):
                    raise ValueError('Body-frame Walk command mismatch')
                if ramp and (not np.array_equal(active, info['command']) or
                             not np.array_equal(obs[5:8], stages.command(0, 0)) or obs[52] != 1 or
                             state['ramp_step'] == stages.state()['ramp_step']):
                    raise ValueError('Isaac command schedule or continuous yaw EMA mismatch')
                if not np.isfinite(reward) or not np.isfinite(obs).all():
                    raise ValueError('Invalid post-action feedback')
                rate=env.actor_gyro_z
                target=float(info['command'][2])
                p=cfg['reward']['biped_tracking']
                scored_rate = float(info['reward_velocity'][2])
                if info['motion_phase'] == 'turn':
                    underspeed = np.sign(target) * scored_rate < abs(target)
                    width = p['turn_yaw_underspeed_width_rad_s'] if underspeed else p['turn_yaw_gaussian_width_rad_s']
                    expected = p['turn_yaw_gaussian_weight'] * np.exp(-((scored_rate-target)/width)**2)
                else:
                    expected = p['velocity_weight'] * p['yaw_weight'] * np.expm1(
                        -((scored_rate-target)/p['tracking_widths'][2])**2)
                if not np.isclose(info['reward_terms']['tracking_yaw'],expected):
                    raise ValueError('Isaac yaw tracking reward mismatch')
                rates.append(rate)
                hip_difference = float(abs(env.q[0]-env.q[6]))
                max_hip=max(max_hip,hip_difference)
                writer.writerow(dict(trial=trial+1, step=step+1, reward=float(reward),
                    body_gyro_z=rate, reward_yaw_rate=scored_rate, actual_wz=float(info['actual_velocity'][2]),
                    x=float(env.pos[0]), y=float(env.pos[1]), height=float(env.pos[2]),
                    left_load=float(env.support.loads[0]), right_load=float(env.support.loads[1]),
                    task=info['motion_phase'], vx_cmd=info['command'][0], vy_cmd=info['command'][1], wz_cmd=target,
                    hip_difference=hip_difference, attempts=stages.state().get('cycles'),
                    ramp_step=state.get('ramp_step'), fell=fell, truncated=truncated))
                stream.flush()
                if hip_difference > cfg['isaac']['hip_tolerance_rad']:
                    raise ValueError(f'HipYawPitch tracking difference {hip_difference:.4f} rad')
                if fell or truncated:
                    if ramp or omni or walk:
                        falls += int(fell)
                        break
                    raise RuntimeError(f'Fixed policy ended at trial {trial+1}, step {step+1}; inspect asset and drives')
    if agent.updates or agent.replay.count:
        raise ValueError('Preflight unexpectedly performed learning')
    return dict(status='simulation_check_passed', steps=len(rates), resets=cfg['isaac']['check_resets'], falls=falls,
                mean_body_gyro_z=float(np.mean(rates)),max_measured_hip_difference=max_hip,
                optimizer_updates=agent.updates, training_started=False,
                limitation='Fixed-policy integration test; independent Evaluation pending')


def main():
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError('Isaac Sim 5.0 requires a separate Python 3.11 environment')
    if not dependency_status()['complete']:
        raise RuntimeError('Incomplete Isaac Lab dependencies; run Install Isaac Dependencies.cmd first')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('check','train','standing','smoke'),default='check')
    parser.add_argument('--experiment')
    parser.add_argument('--learner-device', choices=('cpu', 'cuda:0'))
    resume_options = parser.add_mutually_exclusive_group()
    resume_options.add_argument('--resume', type=Path)
    resume_options.add_argument('--continue-configured-run', action='store_true')
    from standing_check import HEIGHTS_M, TIMESTEPS, configure_standing_pose
    parser.add_argument('--standing-case', choices=(*HEIGHTS_M, *TIMESTEPS, 'ddpg_settled'))
    parser.add_argument('--standing-group', type=uuid.UUID)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(device=None)
    args=parser.parse_args()
    config = (load_configuration(experiment=args.experiment) if args.mode == 'standing'
              else launch_configuration(args.experiment))
    settings = config['isaac']
    args.device = args.device or settings['device']
    if (args.resume or args.continue_configured_run) and args.mode != 'train':
        parser.error('Continue requires --mode train')
    follow_latest = False
    if args.continue_configured_run:
        from run_storage import continue_checkpoint
        selector = json.loads((CODE/'config.json').read_text(encoding='utf-8-sig'))['continue_training']
        args.resume = continue_checkpoint(Path(config['project_layout']['runs_root']), selector)
        follow_latest = selector['run'] == 'latest' and selector.get('checkpoint', 'latest.pth') == 'latest.pth'
    if args.resume:
        args.resume = checkpoint_path(args.resume, HERE)
    installed_versions = require_versions(settings)
    if args.mode == 'smoke' and settings['experiment_id'] != '004' and settings.get('experiment_variant') not in ('walk_lateral', 'walk_lateral_random_vx'):
        parser.error('smoke requires experiment 004 or walk_lateral')
    if args.mode == 'standing':
        settings['num_envs'] = 1
        if args.standing_case is None or args.standing_group is None:
            parser.error('standing requires a case and comparison UUID')
        configure_standing_pose(settings, args.standing_case, SIM/'assets')
    if settings['headless']:
        args.headless = True
    settings['headless'] = args.headless
    if 'standing_ground' in settings and not args.headless and args.rendering_mode is None:
        args.rendering_mode = 'performance'
    args.kit_args = (args.kit_args + ' ' + kit_arguments(settings, HERE)).strip()
    settings['device']=args.device
    if args.learner_device:
        settings['learner_device'] = args.learner_device
    asset=Path(settings['asset'])
    asset_check = inspect_asset(asset)
    if not asset_check['ready']:
        raise FileNotFoundError(f'Missing NAO asset: {asset}. Use Prepare NAO Asset.cmd or set NAO_ISAAC_ASSET.')
    if args.mode == 'train':
        from run_storage import selected_checkpoint, training_output
        identity = dict(config, isaac=dict(settings, asset_sha256=sha(asset)))
        with training_output(Path(config['project_layout']['runs_root']), identity, args.resume) as (model, session):
            args.resume = selected_checkpoint(model, args.resume, follow_latest=follow_latest)
            if args.resume:
                print(f'Continue checkpoint: {args.resume}', flush=True)
            execute(args, settings, asset, asset_check, installed_versions, model, session, config=config)
        return
    if args.mode == 'standing':
        folder = SIM/'checks'/'standing'/args.standing_group.hex/args.standing_case
    else:
        root = Path(config['project_layout']['experiment_folder'])
        prefix = args.mode + '_' if settings['experiment_id'] in ('004', 'omni') else ''
        folder=root/'checks'/(prefix + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_')+uuid.uuid4().hex[:6])
    folder.mkdir(parents=True,exist_ok=False)
    execute(args, settings, asset, asset_check, installed_versions, folder, config=config)


def execute(args, settings, asset, asset_check, installed_versions, folder, session=None, *, config=None):
    from isaaclab.app import AppLauncher
    from standing_check import check_standing
    attempt = session if session and (folder/'source').exists() else folder
    if session:
        for name in ('status.json', 'resume.json', 'result.json'):
            if (folder/name).is_file():
                shutil.copyfile(folder/name, session/('previous_' + name))
    app = None
    try:
        (folder/'status.json').write_text(json.dumps(dict(status='starting_kit', mode=args.mode)))
        args.experience = project_experience(args, settings, attempt)
        launcher=AppLauncher(args)
        app=launcher.app
        cfg=snapshot(attempt,settings,asset,args.mode,config=config)
        import isaaclab.sim as sim_utils
        from environment import IsaacTurn
        from basewalk.runner import run,write_json
        print('Isaac SAC:',folder,flush=True)
        write_json(folder/'status.json',dict(status='checking',mode=args.mode))
        sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=settings['physics_dt'],device=args.device,
            render_interval=settings['decimation']))
        env=IsaacTurn(sim,settings,cfg,asset,attempt/'usd')
        if args.mode == 'train' and settings['experiment_id'] == 'omni':
            env.reset_retry_log = folder/'reset_retries.jsonl'
        if args.mode == 'standing':
            result = check_standing(env, cfg, args.standing_case, folder)
            write_json(folder/'standing.json', result)
            write_json(folder/'status.json', dict(status='completed', mode=args.mode,
                                                  upright=result['upright'], passed=result['passed']))
            print('Standing result:', result, flush=True)
            if env.render:
                from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
                import asyncio
                import time
                viewport = get_active_viewport()
                if viewport is not None:
                    image = folder/'standing.png'
                    capture = capture_viewport_to_file(viewport, str(image))
                    pending = asyncio.ensure_future(capture.wait_for_result())
                    deadline = time.monotonic() + 15.
                    while app.is_running() and not pending.done() and time.monotonic() < deadline:
                        sim.render()
                    if pending.done():
                        pending.result()
                    else:
                        pending.cancel()
                    if image.is_file():
                        print('Standing viewport:', image, flush=True)
            return
        check_fn = preflight
        if settings.get('experiment_variant') in ('walk_lateral', 'walk_lateral_random_vx'):
            from walk_batch import check_batch, run_batch
            check_fn, run = check_batch, run_batch
        check=check_fn(env,cfg,settings['check_steps'],attempt)
        check.update(installed_versions)
        check['asset']=asset_check
        check['converted_hashes']={str(p.relative_to(attempt)):sha(p) for p in (attempt/'usd').rglob('*') if p.is_file()}
        write_json(attempt/'check.json',check)
        if args.mode in ('train','smoke'):
            write_json(folder/'status.json',dict(status='running'))
            continuation = None
            if args.resume:
                from basewalk.omni_resume import restore
                continuation = restore(args.resume, cfg, folder, session)
            if attempt != folder:
                write_json(folder/'effective_config.json', cfg)
            result=run(env,cfg,folder, resume=continuation)
            if args.mode == 'smoke' and result['updates'] < 1:
                raise RuntimeError('Smoke run did not update the SAC optimizer')
            write_json(folder/'result.json',result)
            from scripts.tracking_report import write_report
            print('SAC report:', write_report(folder), flush=True)
        write_json(folder/'status.json',dict(status='completed',mode=args.mode))
    except BaseException as exc:
        if folder.is_dir():
            (folder/'status.json').write_text(json.dumps(dict(status='failed',error=str(exc))))
        raise
    finally:
        if session and (folder/'status.json').exists():
            shutil.copyfile(folder/'status.json', session/'status.json')
        if app is not None:
            if args.mode == 'standing' or settings['experiment_id'] in ('004', 'omni'):
                # The isolated check has already closed its files. A wedged Kit
                # teardown must not prevent the next fresh process from starting.
                import threading
                exit_code = 0 if sys.exc_info()[0] is None else 1
                timer = threading.Timer(15., os._exit, args=(exit_code,))
                timer.daemon = True
                timer.start()
                try:
                    app.close(wait_for_replicator=False)
                finally:
                    timer.cancel()
            else:
                app.close()


if __name__=='__main__':
    main()
