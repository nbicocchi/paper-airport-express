"""Resolve the private manuscript and its sibling reproducibility repository."""
import os
from pathlib import Path
CODE = Path(__file__).resolve().parent
CODE_REPO = CODE.parent
TEXT_ROOT = Path(os.environ.get(
    'AIRPORT_EXPRESS_TEXT_REPO',
    CODE_REPO.parent / 'paper-airport-express-text',
)).expanduser().resolve()
PAPER = TEXT_ROOT / 'paper'
RESULTS = CODE / 'results'
