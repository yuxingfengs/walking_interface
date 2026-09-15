"""Portable, isolated evaluation of the published gs330 checkpoint."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from configs.runtime import load_config
from webots_trials import prepare_driver, execute


def main():
    config = load_config()
    checkpoint = ROOT/'experiments/replay_retained_half_noise_actor_decay/seed_001/checkpoints/eval_candidates/ckpt_gs000330.pth'
    expected = 'e467a02b77a2554761e2aa6222f67140b4cc8b1ca41a9a292da3400447f6e4d4'
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != expected:
        raise ValueError('Published checkpoint hash mismatch')
    out = ROOT/'.local_evaluation'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True)
    config['experiment'].update(name='gs330', output_root=str(out/'runs'))
    config['initialization'].update(use_bc=False, reset_to_bc_on_failure=False, resume='never')
    paths = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'controllers'), str(ROOT/'controllers/common')]
    driver = f'''from controller import Supervisor
sys.path[:0] = {paths!r}
from compare_best_expert import simulate
supervisor=Supervisor()
try:
    simulate(supervisor,{config!r},{str(out)!r},{str(checkpoint)!r})
except BaseException:
    import traceback
    status.write_text(json.dumps(dict(status='failed',error=traceback.format_exc())),encoding='utf-8')
    supervisor.simulationQuit(1)
    raise
else:
    status.write_text(json.dumps(dict(status='completed')),encoding='utf-8')
    supervisor.simulationQuit(0)
'''
    snapshot, world, payload = prepare_driver(config, 'train', driver)
    payload['purpose'] = 'published_gs330_evaluation_only'
    snapshot.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    execute(snapshot,world,payload)
    print((out/'result.json').read_text(encoding='utf-8'))
    print('Results:', out)


if __name__ == '__main__':
    main()
