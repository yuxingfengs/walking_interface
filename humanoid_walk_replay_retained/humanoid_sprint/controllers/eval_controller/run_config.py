"""Compatibility adapter. Edit configs/project_config.py; start via launch.py."""
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from configs.runtime import load_config, task_settings

RUN_CONFIG = load_config('eval')
EXPERIMENT_NAME = RUN_CONFIG['experiment']['name']
SEED = RUN_CONFIG['experiment']['seed']
TASK_MODULE = RUN_CONFIG['environment']['task_module']
TASK_CLASS = RUN_CONFIG['environment']['task_class']
PHASE_MODE = RUN_CONFIG['environment']['phase_mode']
PHASE_FREQUENCY_HZ = RUN_CONFIG['environment']['phase_coefficient']
PHASE_RESET_VALUE = RUN_CONFIG['environment']['phase_reset_value']
EVAL_EPISODES = RUN_CONFIG['evaluation']['episodes']
STOP_X = RUN_CONFIG['evaluation']['stop_x_m']
MAX_STEPS = RUN_CONFIG['evaluation']['max_steps']
START_GLOBAL_STEP = RUN_CONFIG['evaluation']['start_global_step']
SELECTED_GLOBAL_STEP = RUN_CONFIG['evaluation']['selected_global_step']
EVAL_METRICS_FILENAME = RUN_CONFIG['evaluation']['metrics_filename']
ONLY_LATEST = RUN_CONFIG['evaluation']['only_latest']

def task_config():
    return task_settings(RUN_CONFIG, 'eval')
