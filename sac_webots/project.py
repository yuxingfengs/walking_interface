"""Read the SAC source descriptors used by Webots and Isaac."""
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parent

from data.project import ROOT, digest, read_json, training_configuration as source_configuration


def training_configuration(load_bc=True, experiment=None, *, descriptor_folder=None):
    cfg = source_configuration(load_bc=load_bc, experiment=experiment, descriptor_folder=descriptor_folder)
    path = CODE/'training_config.json'
    shared = read_json(path)
    cfg['seed'] = shared['seed']
    cfg['sac'] = shared['sac']
    cfg['training'].update(shared['training'])
    cfg['project_layout'].update(training_config_path=str(path), training_config_sha256=digest(path))
    return cfg
