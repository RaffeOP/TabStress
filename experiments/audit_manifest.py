"""Check that every file listed in the ```files block of PIPELINE_MANIFEST.md exists and print its sha256.
Exit status 1 if any file is missing."""

import hashlib
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MANIFEST = os.path.join(ROOT, "PIPELINE_MANIFEST.md")
CHUNK = 1 << 20


def listed_files():
    text = open(MANIFEST, encoding="utf-8").read()
    block = re.search(r"```files\n(.*?)```", text, re.S)
    if block is None:
        raise SystemExit("no ```files block in PIPELINE_MANIFEST.md")
    return [line.strip() for line in block.group(1).splitlines() if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    missing = []
    for rel in listed_files():
        path = os.path.join(ROOT, rel)
        if os.path.isfile(path):
            print(f"{sha256(path)}  {rel}")
        else:
            print(f"MISSING  {rel}")
            missing.append(rel)
    print(f"{len(missing)} missing")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
