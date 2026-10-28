"""Download the pinned public model assets, or verify local copies.

Only Python's standard library is required. Downloads use the public GitHub
Release URLs without authentication. Existing files are never repaired or
replaced automatically: a mismatch must be resolved by the user first.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parent
RELEASE_BASE = "https://github.com/Anone92-yuntech/attention/releases/download/v1.0.0"
CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class ModelAsset:
    relative_path: str
    size: int
    sha256: str

    @property
    def url(self) -> str:
        # Release asset names are flat, even when local files live in models/.
        return f"{RELEASE_BASE}/{Path(self.relative_path).name}"


ASSETS = (
    ModelAsset(
        "best.pt", 40516645,
        "f5fce7480e8fb4aebd800eeaae24299979cd7aa6df49c8085b0a67ee6ecf906f",
    ),
    ModelAsset(
        "models/yolox_nano.onnx", 3659407,
        "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d",
    ),
    ModelAsset(
        "models/pose_landmarker_lite.task", 5777746,
        "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",
    ),
)


class ModelIntegrityError(RuntimeError):
    """A local file or downloaded response differs from the pinned manifest."""


def _check_integrity(asset: ModelAsset, size: int, digest: str, path: Path):
    if size != asset.size:
        raise ModelIntegrityError(
            f"{path}: size mismatch (expected {asset.size} bytes, got {size}); "
            "existing files are preserved"
        )
    if digest.lower() != asset.sha256.lower():
        raise ModelIntegrityError(
            f"{path}: SHA256 mismatch (expected {asset.sha256}, got {digest}); "
            "existing files are preserved"
        )


def _verify_existing(path: Path, asset: ModelAsset):
    if not path.is_file():
        raise ModelIntegrityError(f"{path}: existing path is not a regular file; left unchanged")
    size = path.stat().st_size
    if size != asset.size:
        _check_integrity(asset, size, "", path)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    _check_integrity(asset, size, digest.hexdigest(), path)


def _exists(path: Path) -> bool:
    # A broken symlink is still an existing path that must be preserved.
    return path.exists() or path.is_symlink()


def ensure_model(asset: ModelAsset, *, root: Path = ROOT, verify_only: bool = False) -> str:
    """Return ``verified`` or ``downloaded``; raise without altering bad files.

    A new download is held in a unique temporary file alongside its destination
    and checked for exact byte count and SHA256 before the atomic replacement.
    Only this invocation's temporary file is removed if anything fails.
    """
    path = Path(root) / asset.relative_path
    if _exists(path):
        _verify_existing(path, asset)
        return "verified"
    if verify_only:
        raise FileNotFoundError(f"{path}: model is missing (--verify-only does not download)")

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".part", dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        digest = hashlib.sha256()
        size = 0
        request = urllib.request.Request(
            asset.url, headers={"User-Agent": "attention-model-fetcher/1.0"},
        )
        with os.fdopen(descriptor, "wb") as destination:
            with urllib.request.urlopen(request, timeout=60) as response:
                while chunk := response.read(CHUNK_SIZE):
                    size += len(chunk)
                    if size > asset.size:
                        raise ModelIntegrityError(
                            f"{path}: download exceeds expected size ({asset.size} bytes)"
                        )
                    destination.write(chunk)
                    digest.update(chunk)
            destination.flush()
            os.fsync(destination.fileno())
        _check_integrity(asset, size, digest.hexdigest(), path)

        # Another process may have supplied a file during the download. Check
        # again and preserve it rather than silently overwriting its contents.
        if _exists(path):
            _verify_existing(path, asset)
            return "verified"
        os.replace(temporary_path, path)
        return "downloaded"
    finally:
        temporary_path.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only", action="store_true",
        help="check all model files without downloading or changing them",
    )
    arguments = parser.parse_args(argv)
    failed = False
    for asset in ASSETS:
        try:
            status = ensure_model(asset, verify_only=arguments.verify_only)
        except (OSError, ModelIntegrityError) as error:
            print(f"ERROR: {error}", file=sys.stderr)
            failed = True
        else:
            print(f"{status}: {asset.relative_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
