"""Bounded empty-scene test of the installed runtime; never creates an RL agent."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback

sys.path.append(str(Path(__file__).resolve().parents[3]/'sac_isaac_lab/runner'))
from configuration import HERE, SIM, simulator_settings
from runtime_versions import (require_versions, kit_arguments, dependency_status, project_experience,
                              WARP_VERSION, warp_extension_path)


def main():
    settings = simulator_settings()
    report = dict(training_started=False, optimizer_updates=0, physics_steps=0,
                  status='running', phase='dependencies',
                  started_at=datetime.now(timezone.utc).isoformat())
    folder = SIM / 'reports'
    folder.mkdir(exist_ok=True)

    def save_report():
        (folder / 'runtime_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    save_report()
    app = None
    try:
        report['versions'] = require_versions(settings)
        report['dependencies'] = dependency_status()
        if not report['dependencies']['complete']:
            raise RuntimeError('Isaac Lab dependencies are incomplete. Use Install Isaac Dependencies.cmd. '
                               + str(report['dependencies']))
        report['phase'] = 'import_app_launcher'
        save_report()
        from isaaclab.app import AppLauncher
        parser = argparse.ArgumentParser(description=__doc__)
        AppLauncher.add_app_launcher_args(parser)
        parser.set_defaults(device=settings['device'], headless=settings['headless'])
        args = parser.parse_args()
        settings.update(headless=args.headless, device=args.device)
        args.experience = project_experience(args, settings, folder)
        args.kit_args = (args.kit_args + ' ' + kit_arguments(settings, HERE)).strip()
        report['phase'] = 'starting_app'
        save_report()
        launcher = AppLauncher(args)
        app = launcher.app
        report['phase'] = 'validate_runtime_imports'
        save_report()
        import h5py
        import warp as wp
        import isaaclab_assets
        if wp.__version__ != WARP_VERSION or not Path(wp.__file__).resolve().is_relative_to(warp_extension_path(HERE).resolve()):
            raise RuntimeError('Kit did not select the project Warp extension: ' + str(wp.__file__))
        wp.init()
        # The dependency check exercises CUDA even when PhysX runs on the CPU.
        warp_device = wp.get_device('cuda:0' if args.device == 'cpu' else args.device)
        if not warp_device.is_cuda:
            raise RuntimeError('Warp CUDA device is unavailable')
        # Exercise native device allocation/copy without compiling an RL kernel.
        wp.zeros(1, dtype=wp.float32, device=warp_device).numpy()
        report.update(warp_version=wp.__version__, warp_module=wp.__file__,
                      warp_cuda_validated=True, physics_device=settings['device'], h5py_version=h5py.__version__,
                      asset_extension_module=isaaclab_assets.__file__)
        report['phase'] = 'import_simulation'
        save_report()
        import torch
        import isaaclab.sim as sim_utils
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
            dt=settings['physics_dt'], device=settings['device']))
        report['phase'] = 'reset_simulation'
        save_report()
        sim.reset()
        for _ in range(10):
            sim.step(render=not settings['headless'])
            report['physics_steps'] += 1
        report.update(status='passed', phase='completed', cuda_available=torch.cuda.is_available(),
                      gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__)
        return 0
    except Exception as exc:
        report.update(status='failed', error=str(exc), traceback=traceback.format_exc())
        print(report['traceback'], flush=True)
        return 1
    finally:
        # Save before closing Kit, which can itself terminate the process.
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        save_report()
        print(json.dumps({k:v for k,v in report.items() if k != 'traceback'}, indent=2), flush=True)
        if app is not None:
            app.close()


if __name__ == '__main__':
    raise SystemExit(main())
