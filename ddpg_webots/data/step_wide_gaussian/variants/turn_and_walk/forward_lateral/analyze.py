"""Offline signed lateral-with-forward command result analysis."""
import importlib.util
from pathlib import Path

OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('mixed_command_analysis',OUT.parent/'analyze.py')
analysis=importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)

if __name__=='__main__':
    analysis.main(out=OUT,run=OUT/'runs/train_20261121T052059_f38fce',lateral_vx=.04,
        lateral_levels=[0,.005,.01,.015,.02,.015,.01,.005,0,-.005,-.01,-.015,-.02,-.015,-.01,-.005,0])
