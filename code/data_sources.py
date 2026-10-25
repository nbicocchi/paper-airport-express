"""Resolve and verify the repository's immutable Excel inputs."""
from pathlib import Path
import hashlib
import json

CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CODE_ROOT.parent
DATA_ROOT = REPO_ROOT / "dataset"
MANIFEST = json.loads((CODE_ROOT / "source_manifest.json").read_text())


def source_items():
    return [item for item in MANIFEST["tree"] if item["type"] == "blob"]


def source_path(item):
    relative = Path(item["path"])
    if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "dataset":
        raise ValueError(f"Invalid dataset path: {relative}")
    return REPO_ROOT / relative


def verify_source(item):
    path = source_path(item)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run python code/download.py from repository root.")
    content = path.read_bytes()
    digest = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
    if digest != item["sha"]:
        raise ValueError(f"Source hash mismatch: {path}; the pinned input has changed.")
    return path


def verify_sources():
    return [verify_source(item) for item in source_items()]
