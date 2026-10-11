"""Verify dataset/ offline, or restore missing inputs from the pinned Git commit."""
import argparse
import concurrent.futures
import hashlib
import urllib.parse
import urllib.request
from data_sources import MANIFEST, source_items, source_path, verify_source, verify_sources


def fetch(item):
    path = source_path(item)
    if path.exists():
        return verify_source(item)
    url = ("https://raw.githubusercontent.com/nbicocchi/paper-airport-express/"
           + MANIFEST["sha"] + "/" + urllib.parse.quote(item["path"], safe="/"))
    with urllib.request.urlopen(url, timeout=90) as response:
        content = response.read()
    digest = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
    if digest != item["sha"]:
        raise ValueError(f"Downloaded source hash mismatch: {item['path']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(content)
    return verify_source(item)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify local files only; never download or write.")
    args = parser.parse_args()
    if args.check:
        paths = verify_sources()
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            paths = list(pool.map(fetch, source_items()))
    print(f"Verified {len(paths)} pinned Excel inputs in dataset/.")
