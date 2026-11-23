"""Config-driven online DDPG entry. --check never launches Webots or updates weights."""
import argparse
import sys
from pathlib import Path
from project import CODE, ROOT, resolve_entry_config, training_configuration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', nargs='?')
    parser.add_argument('--config', type=Path, default=ROOT/'config.json')
    parser.add_argument('--resume-checkpoint', type=Path)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--run-reference', type=Path)
    args = parser.parse_args()
    args.config, args.experiment = resolve_entry_config(args.config, args.experiment)
    try:
        from backend_dispatch import dispatch
        result = dispatch(args.config, args.experiment, mode='offline' if args.check else 'train')
        if result is not None:
            return result
        cfg = training_configuration(args.config, args.experiment)
        sys.path.insert(0, str(CODE/'training'))
        from basewalk.configuration import training_config
        from basewalk.learning import create_algorithm
        from basewalk.artifacts import render_world
        from basewalk.reward_dispatch import implementation
        implementation(cfg)
        cfg = training_config(cfg)
        for key in ('executable', 'python', 'world_template'):
            if not Path(cfg['webots'][key]).is_file():
                raise FileNotFoundError('Missing webots.'+key+': '+cfg['webots'][key])
        render_world(cfg)
        agent = create_algorithm(cfg)
        if agent.updates != 0:
            raise ValueError('Initialization unexpectedly performed optimizer updates')
        print('Experiment:', cfg['experiment'])
        print('Critic pretraining updates:', getattr(agent, 'pretrained_critic_updates', 0))
        print('Runtime updates:', agent.updates)
        from basewalk.network_inputs import actor_input_dim, critic_input_dim, observation_dim
        print(f'State{observation_dim(cfg)} / Action12')
        attempts = cfg['training'].get('joint_update_every_turn_attempts')
        gait_attempts = cfg['training'].get('joint_update_every_gait_attempts')
        if gait_attempts is not None:
            print(f'Critic and Actor update together every {gait_attempts} Turn/Walk gait attempts')
        elif attempts is not None:
            actor_attempts = cfg['training'].get('actor_update_every_turn_attempts', attempts)
            label = ('Walk' if cfg['training'].get('walk_commands') else
                     'Turn/Walk' if cfg['training'].get('turn_and_walk', {}).get('enabled') else 'Turn')
            print(f'Critic every {attempts} {label} attempts; Actor every {actor_attempts} {label} attempts')
        else:
            print('Critic cadence:', cfg['training']['update_every'],
                  '; Actor cadence:', cfg['training']['actor_update_every'])
        print('Actor neural input:',actor_input_dim(cfg))
        from basewalk.core import critic_extra_dim
        print('Critic input:', critic_input_dim(cfg,critic_extra_dim(cfg)), '; context:', cfg['critic_observation'])
        print('Runs:', cfg['project_layout']['runs_root'])
        if args.resume_checkpoint:
            from basewalk.run_storage import belongs_to_run_root, group_name
            import torch
            from project import digest
            from basewalk.core import CONTRACT
            path = args.resume_checkpoint.resolve(strict=True)
            runs = Path(cfg['project_layout']['runs_root']).resolve()
            if (path.parent.name != 'checkpoints' or not path.stem.startswith('step_')
                    or not path.stem[5:].isdigit() or not belongs_to_run_root(path.parent.parent, runs)):
                raise ValueError('Choose a step checkpoint in the selected experiment runs directory')
            saved = torch.load(path, map_location='cpu', weights_only=False)
            step = saved['env_steps']
            if saved['contract'] != CONTRACT or not 0 < step < cfg['training']['total_env_steps']:
                raise ValueError('Incompatible DDPG checkpoint or checkpoint already at the configured budget')
            for key in ('actor', 'critic', 'actor_target', 'critic_target', 'actor_optimizer', 'critic_optimizer'):
                getattr(agent, key).load_state_dict(saved[key])
            checksum = digest(path)
            origin = path.parent.parent
            model = runs/group_name(cfg)/origin.name
            if origin.parent != runs:
                from project import read_json
                old = read_json(origin/'effective_config.json')['config']
                training = dict(old['training'], total_env_steps=cfg['training']['total_env_steps'])
                if (origin/'resume.json').is_file():
                    training.pop('checkpoint_start', None)
                    if 'checkpoint_start' in cfg['training']:
                        training['checkpoint_start'] = cfg['training']['checkpoint_start']
                if group_name(dict(old, training=training)) == group_name(cfg):
                    model = origin
            branch = cfg['training'].get('checkpoint_start') or {}
            if branch.get('sha256') != checksum:
                branch = {}
            cfg['training']['checkpoint_start'] = dict(branch, path=str(path), sha256=checksum, step=step)
            print('DDPG checkpoint validated; next step:', step + 1)
        if args.check:
            print('CHECK PASSED. No simulation, training or optimizer update.')
            return
        del agent
        if args.run_reference:
            cfg['launch_reference'] = str(args.run_reference.resolve())
        from scripts.training import launch
        launch(cfg, 'train', None, False, resume_model=model if args.resume_checkpoint else None)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.exit(2, 'Training configuration failed: '+str(exc)+'\n')


if __name__ == '__main__':
    raise SystemExit(main())
