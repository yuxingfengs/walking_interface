"""Independent SAC Turn entry point. --check never launches Webots."""
import argparse
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
from project import ROOT, read_json, training_configuration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--resume', type=Path, metavar='RUN_FOLDER',
                      help='Continue the selected run up to the configured total budget')
    parser.add_argument('--experiment')
    args = parser.parse_args()

    sys.path.insert(0, str(CODE / 'training'))
    try:
        if args.resume is not None:
            from basewalk.run_storage import belongs_to_run_root
            from scripts.training import resume
            experiment = args.experiment or read_json(CODE/'config.json')['current_experiment']
            selected = training_configuration(experiment=experiment, load_bc=False)
            if not belongs_to_run_root(args.resume, selected['project_layout']['runs_root']):
                raise ValueError('Resume run belongs to another experiment')
            resume(args.resume, selected['training']['total_env_steps'])
            return 0

        from basewalk.artifacts import render_world
        from basewalk.sac import SAC
        from basewalk.reward_dispatch import implementation

        webots = read_json(ROOT/'config.json').get('backend_overrides', {}).get('webots', {})
        cfg = training_configuration(experiment=args.experiment or webots.get('current_experiment'))
        if cfg.get('algorithm') != 'sac':
            raise ValueError('The standalone SAC project accepts only algorithm="sac"')
        if cfg.get('expert_pipeline') or cfg.get('three_class_online'):
            raise ValueError('Standalone SAC must use online-only experience')
        implementation(cfg)
        for key in ('executable', 'python', 'world_template'):
            if not Path(cfg['webots'][key]).is_file():
                raise FileNotFoundError('Missing webots.' + key + ': ' + cfg['webots'][key])
        render_world(cfg)
        agent = SAC(cfg)
        if agent.__class__.__name__ != 'SAC' or agent.updates != 0:
            raise ValueError('SAC initialization contract failed')
        print('Project: sac_webots')
        print('Experiment:', cfg['experiment'])
        print('Algorithm:', cfg['algorithm'].upper())
        print('Task:', cfg['experiment_name'])
        print('Turn rate magnitude:', cfg['training']['online_turn_rate_rad_s'], 'rad/s')
        print('Online budget:', cfg['training']['total_env_steps'], 'environment steps')
        print('Runs:', cfg['project_layout']['runs_root'])
        print('Runtime updates:', agent.updates)
        if args.check:
            print('CHECK PASSED. No simulation, training or optimizer update.')
            return 0

        from scripts.training import launch
        del agent
        launch(cfg)
        return 0
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.exit(2, 'SAC configuration failed: ' + str(exc) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
