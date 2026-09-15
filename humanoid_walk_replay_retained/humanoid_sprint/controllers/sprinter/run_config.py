"""Compatibility adapter. Edit configs/project_config.py; start via launch.py."""
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from configs.runtime import load_config, task_settings, algorithm_settings, project_path

RUN_CONFIG = load_config('train')
EXPERIMENT_NAME = RUN_CONFIG['experiment']['name']
SEED = RUN_CONFIG['experiment']['seed']
TASK_MODULE = RUN_CONFIG['environment']['task_module']
TASK_CLASS = RUN_CONFIG['environment']['task_class']
ALGORITHM_MODULE = RUN_CONFIG['training']['algorithm_module']
ALGORITHM_CLASS = RUN_CONFIG['training']['algorithm_class']
USE_BC_INITIALIZATION = RUN_CONFIG['initialization']['use_bc']
TRAIN_EPISODES = RUN_CONFIG['training']['rollouts']
SEQUENCE_LENGTH = RUN_CONFIG['training']['rollout_steps']
PHASE_MODE = RUN_CONFIG['environment']['phase_mode']
PHASE_FREQUENCY_HZ = RUN_CONFIG['environment']['phase_coefficient']
PHASE_RESET_VALUE = RUN_CONFIG['environment']['phase_reset_value']
ACTOR_LR = RUN_CONFIG['training']['actor_lr']
CRITIC_LR = RUN_CONFIG['training']['critic_lr']
GAMMA = RUN_CONFIG['training']['gamma']
TAU = RUN_CONFIG['training']['tau']
BATCH_SIZE = RUN_CONFIG['training']['batch_size']
BUFFER_SIZE = RUN_CONFIG['training']['buffer_size']
MIN_BUFFER_FOR_TRAIN = RUN_CONFIG['training']['min_buffer_for_train']
TRAIN_EVERY_EPISODES = RUN_CONFIG['training']['train_every_rollouts']
STOP_X = RUN_CONFIG['training']['stop_x_m']
EVAL_CHECKPOINT_EVERY_UPDATES = RUN_CONFIG['training']['checkpoint_every_updates']
CONSECUTIVE_FALL_CYCLE_LIMIT = RUN_CONFIG['initialization']['consecutive_fall_rollouts']
STALL_CYCLE_LIMIT = RUN_CONFIG['initialization']['stall_rollouts']
STALL_MAX_ABS_X_PROGRESS_M = RUN_CONFIG['initialization']['stall_max_abs_x_progress_m']

def task_config():
    return task_settings(RUN_CONFIG, 'train')

BC_CHECKPOINT_FILENAME = str(project_path(RUN_CONFIG["initialization"]["bc_checkpoint"]))

def algorithm_config(*, device, task_id):
    return algorithm_settings(RUN_CONFIG, device, task_id)
