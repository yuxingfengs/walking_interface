from pathlib import Path

__path__ = [str((Path(__file__).parent / '../../../../training/basewalk').resolve())]
DATA_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = DATA_ROOT / 'training'
__path__.append(str(Path(__path__[0]).parents[2] / 'environments/webots'))
