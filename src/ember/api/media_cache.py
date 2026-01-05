"""Media cache implementations for Ember's data streaming API."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.parse import unquote, urlparse

from ember.api.record import MediaAsset


class DiskMediaCache:
    """Disk-backed cache that attaches lazy openers to :class:`MediaAsset` objects.

    The cache understands local paths as well as remote URIs. Remote assets are
    downloaded into :attr:`root` and reused across calls by keying on the asset
    checksum when available.
    """

    def __init__(self, root: Optional[Path] = None, *, create: bool = True) -> None:
        default_root = Path.home() / ".cache" / "ember" / "media"
        self._root = Path(root) if root is not None else default_root
        if create:
            self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        """Return the directory used for cached assets."""

        return self._root

    def bind(self, asset: MediaAsset) -> MediaAsset:
        """Attach a lazy opener for cached assets when possible."""

        local_path = self._resolve_local_path(asset)
        if local_path is None:
            return asset

        def _open(path: Path | None = local_path) -> Any:
            if path is None:
                raise ValueError("Local path not resolved")
            return path.open("rb")

        return asset.with_opener(_open)

    def _resolve_local_path(self, asset: MediaAsset) -> Optional[Path]:
        uri = asset.uri
        parsed = urlparse(uri)

        if parsed.scheme == "file":
            return Path(unquote(parsed.path))

        if parsed.scheme in {"http", "https"}:
            return self._download_http(asset)

        if parsed.scheme == "hf":
            return self._download_hf(asset)

        if parsed.scheme:
            return None

        raw_path = Path(uri)
        if raw_path.is_absolute() and raw_path.exists():
            return raw_path

        candidate = self._root / raw_path
        if candidate.exists():
            return candidate

        return raw_path if raw_path.exists() else None

    def _download_http(self, asset: MediaAsset) -> Optional[Path]:
        filename = asset.sha256 or Path(urlparse(asset.uri).path).name
        if not filename:
            filename = hashlib.sha256(asset.uri.encode("utf-8")).hexdigest()

        target = self._root / filename
        if target.exists():
            return target

        try:
            with urllib.request.urlopen(asset.uri) as response:  # noqa: S310
                data = response.read()
        except Exception:
            return None

        if asset.sha256:
            digest = hashlib.sha256(data).hexdigest()
            if digest != asset.sha256:
                return None

        target.write_bytes(data)
        return target

    def _download_hf(self, asset: MediaAsset) -> Optional[Path]:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:  # pragma: no cover - optional dependency
            return None

        remainder = asset.uri[len("hf://") :]
        if remainder.startswith("datasets/"):
            remainder = remainder[len("datasets/") :]

        if "/" not in remainder:
            return None

        repo_id, _, filename = remainder.partition("/")
        if not filename:
            return None

        metadata: Mapping[str, object] = asset.metadata
        revision = metadata.get("revision") if isinstance(metadata, Mapping) else None
        subfolder = metadata.get("subfolder") if isinstance(metadata, Mapping) else None

        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision if isinstance(revision, str) else None,
                subfolder=subfolder if isinstance(subfolder, str) else None,
                cache_dir=str(self._root),
            )
        except Exception:  # pragma: no cover - network errors
            return None

        return Path(downloaded)


__all__ = ["DiskMediaCache"]
