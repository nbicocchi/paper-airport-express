"""Resolve manuscript assets, with an optional external manuscript override."""
import os
from pathlib import Path
CODE = Path(__file__).resolve().parent
CODE_REPO = CODE.parent
TEXT_ROOT = Path(os.environ.get(
    'AIRPORT_EXPRESS_TEXT_REPO',
    CODE_REPO,
)).expanduser().resolve()
PAPER = TEXT_ROOT / 'paper'
RESULTS = CODE / 'results'
