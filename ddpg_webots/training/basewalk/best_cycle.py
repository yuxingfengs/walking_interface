"""Preserve the policy that generated a record complete, nonfallen cycle."""
import hashlib
import json
import math
from pathlib import Path


class BestRewardCycle:
    def __init__(self, folder, agent, archive):
        self.folder = Path(folder) / 'best_reward_cycle'
        self.agent = agent
        self.archive = archive
        self.best = -math.inf
        metadata = self.folder / 'metadata.json'
        if metadata.exists():
            self.best = json.loads(metadata.read_text(encoding='utf-8'))['mean_reward']

    def __call__(self, cycle):
        if self.archive is not None:
            self.archive(cycle)
        if not cycle:
            return
        rows = [item[5] for item in cycle]
        stage = 'walk' if self.agent.config['training'].get('walk_commands') else 'turn'
        if any(row['motion_phase'] != stage for row in rows):
            return
        score = math.fsum(float(item[2]) for item in cycle) / len(cycle)
        if not math.isfinite(score) or score <= self.best:
            return
        start, end = int(rows[0]['step']), int(rows[-1]['step'])
        self.folder.mkdir(parents=True, exist_ok=True)
        # Versioned files make the metadata switch safe and retain former records.
        checkpoint = self.folder / f'cycle_{end:08d}.pth'
        self.agent.save(checkpoint, end)
        training = self.agent.config['training']
        per_step = bool(training.get('walk_commands')) and not (
            training.get('joint_update_every_turn_attempts') or training.get('joint_update_every_gait_attempts'))
        metadata = dict(mean_reward=score, start_step=start, end_step=end,
                        transitions=len(cycle), checkpoint=str(checkpoint.resolve()),
                        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        critic_updates=self.agent.updates,
                        saved_before_cycle_end_update=not per_step,
                        updates_during_cycle=per_step,
                        exploration_enabled=True,
                        purpose='frozen_actor_retest_not_exact_training_resume')
        temporary = self.folder / 'metadata.tmp'
        temporary.write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
        temporary.replace(self.folder / 'metadata.json')
        self.best = score
