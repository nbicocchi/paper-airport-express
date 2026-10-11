"""Read saved revision tables, including the compressed row-level artifacts."""
from pathlib import Path
import gzip
import pandas as pd

RESULTS = Path(__file__).resolve().parent / 'results' / 'revision'
ROW_FILES = ['predictions.csv', 'quantile_predictions.csv', 'recent_audit.csv']


def read_result(name, **kwargs):
    path = RESULTS / name
    return pd.read_csv(path if path.exists() else RESULTS / (name + '.gz'), **kwargs)


def compress_outputs():
    for name in ROW_FILES:
        path = RESULTS / name
        (RESULTS / (name + '.gz')).write_bytes(gzip.compress(path.read_bytes(), mtime=0))
