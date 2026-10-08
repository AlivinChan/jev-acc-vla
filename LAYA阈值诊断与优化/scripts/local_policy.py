"""Check the experiment's persistent stop marker before a local stage."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def allowed():
    if (ROOT/'STOP_REQUESTED.json').exists():
        raise RuntimeError('This experiment is stopped')
    return {}
