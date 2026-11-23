"""Compatibility entry for the relocated report module."""
from pathlib import Path
import sys

__file__ = str((Path(__file__).resolve().parent / '../../report.py').resolve())
sys.path.insert(0, str(Path(__file__).parent))
exec(compile(Path(__file__).read_bytes(), __file__, 'exec'))
