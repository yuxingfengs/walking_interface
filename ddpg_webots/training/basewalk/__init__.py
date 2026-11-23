"""Command-conditioned NAO locomotion, independent of legacy checkpoint contracts."""

from pathlib import Path

# In a frozen run these modules are materialized alongside the controller.
TRAINING_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = TRAINING_ROOT.parent
if not Path(__file__).with_name('webots_devices.py').is_file():
    DATA_ROOT = DATA_ROOT / 'data/runtime'
    TRAINING_ROOT = DATA_ROOT / 'training'
    __path__.append(str(Path(__file__).resolve().parents[3] / 'environments/webots'))
