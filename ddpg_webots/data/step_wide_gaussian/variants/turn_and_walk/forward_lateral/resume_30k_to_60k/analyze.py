"""Offline branch result; retain cumulative attempt and step coordinates."""
import importlib.util
from pathlib import Path

OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('mixed_analysis',OUT.parents[1]/'analyze.py')
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)
if __name__=='__main__':
    analysis.main(out=OUT,run=OUT/'runs/train_20261121T060052_2cbf68',lateral_vx=.04,
        lateral_levels=[0,.005,.01,.015,.02,.015,.01,.005,0,-.005,-.01,-.015,-.02,-.015,-.01,-.005,0],
        start_attempt=782,windows=[(30001,35000),(35001,40000),(40001,45000),(45001,50000),(50001,55000),(55001,60000),(30001,60000)])
