"""Read-only readiness diagnostics; never starts collection or optimization."""
from pathlib import Path
import torch
from .expert_pipeline import signature


def missing_assets(cfg, stage):
    pipeline=cfg['expert_pipeline']
    dataset=Path(pipeline['dataset'])
    if not dataset.is_file():
        partial=dataset.with_suffix('.partial.pth')
        if not partial.is_file():
            return 'Expert dataset is missing. Run Collect gate3 experts.cmd first.'
        saved=torch.load(partial,map_location='cpu',weights_only=False)
        if saved.get('signature')!=signature(cfg):
            return 'Partial expert data does not match the current task/reward configuration.'
        episodes=saved.get('episodes',[])
        quota=pipeline['successes_per_angle']
        missing=[a for a in cfg['direction_training']['angles_deg'] if sum(
            e.get('task')=='navigation' and e.get('success') and e['angle']==a for e in episodes)<quota]
        stand=sum(e.get('task')=='stand' and e.get('success') for e in episodes)
        counts=[sum(row[0]==i for e in episodes for row in e['transitions']) for i in range(3)]
        return ('Expert collection is incomplete; partial data retained. Missing navigation angles: '+
                ', '.join(f'{a:+g}' for a in missing)+
                f'; Stand success {stand}/{quota}; Turn/Stand/Walk transitions={counts}. '
                'Full angle coverage is optional. Pretraining requires nonempty Turn and Walk categories; Stand is unused. '
                'Inspect collection results before retrying; no automatic collection will start.')
    if stage=='train' and not Path(pipeline['critic_checkpoint']).is_file():
        return 'Matching pretrained Critic is missing. Run Pretrain Critic.cmd after complete expert collection.'
    return None
