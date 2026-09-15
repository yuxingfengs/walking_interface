"""Stable paths shared by all Webots controllers.

Webots starts each controller with its own working directory.  Never use the
working directory for experiment artifacts: resolve them from this file instead.
"""

from dataclasses import dataclass
from pathlib import Path
import sys


CONTROLLERS_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = CONTROLLERS_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
from configs.runtime import load_config, project_path

EXPERIMENTS_DIR = project_path(load_config()["experiment"]["output_root"])


@dataclass(frozen=True)
class ExperimentPaths:
    """Filesystem layout for one reproducible experiment seed."""

    experiment_name: str
    seed: int

    @property
    def root(self) -> Path:
        return EXPERIMENTS_DIR / self.experiment_name / f"seed_{self.seed:03d}"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def resume_checkpoint(self) -> Path:
        return self.checkpoints_dir / "resume.pth"

    @property
    def eval_candidates_dir(self) -> Path:
        return self.checkpoints_dir / "eval_candidates"

    def eval_checkpoint(self, global_step: int) -> Path:
        return self.eval_candidates_dir / f"ckpt_gs{int(global_step):06d}.pth"

    @property
    def metrics_dir(self) -> Path:
        return self.root / "metrics"

    @property
    def train_log_dir(self) -> Path:
        return self.metrics_dir / "train"

    @property
    def evaluation_log_dir(self) -> Path:
        return self.metrics_dir / "evaluation"

    @property
    def metadata_path(self) -> Path:
        return self.root / "metadata.json"

    def ensure_directories(self) -> None:
        for directory in (
            self.checkpoints_dir,
            self.eval_candidates_dir,
            self.train_log_dir,
            self.evaluation_log_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
