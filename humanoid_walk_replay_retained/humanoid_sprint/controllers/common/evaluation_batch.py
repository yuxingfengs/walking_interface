"""Continue a frozen evaluation batch without closing the Webots application."""
import json
from pathlib import Path

PROTOCOL = 'shared_webots_world_reload_v3'


def finish_evaluation(controller, payload):
    batch_path = payload.get('evaluation_batch')
    if not batch_path:
        controller.simulationQuit(0)
        return
    path = Path(batch_path)
    manifest = json.loads(path.read_text(encoding='utf-8'))
    index = payload['evaluation_batch_index']
    run = manifest['runs'][index]
    folder = Path(run['snapshot']).parent
    result = json.loads((folder / 'evaluation_result.json').read_text(encoding='utf-8'))
    if result['eval_run_id'] != payload['run_id'] or result['checkpoint_sha256'] != run['sha256']:
        raise RuntimeError('Evaluation result does not match the frozen batch')
    run.update(status='completed', result=result)
    (folder / 'run_status.json').write_text(json.dumps({'status': 'completed'}), encoding='utf-8')
    more = index + 1 < len(manifest['runs'])
    manifest['status'] = 'running' if more else 'completed'
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if more:
        # A full world load resets devices/physics and starts a fresh Controller,
        # while retaining the existing Webots application process and window.
        controller.worldLoad(manifest['runs'][index + 1]['world'])
        while controller.step(controller.timeStep) != -1:
            pass
    else:
        controller.simulationQuit(0)
