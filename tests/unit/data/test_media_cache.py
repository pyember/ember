"""Tests for media cache helpers."""

import hashlib
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from ember.api.media_cache import DiskMediaCache
from ember.api.record import MediaAsset


def test_disk_media_cache_binds_local_file(tmp_path):
    payload = tmp_path / "asset.bin"
    payload.write_bytes(b"data")
    asset = MediaAsset(uri=str(payload), kind="image")

    cache = DiskMediaCache(root=tmp_path)
    bound = cache.bind(asset)

    with bound.open() as handle:
        assert handle.read() == b"data"


def test_disk_media_cache_noop_for_remote(tmp_path):
    cache = DiskMediaCache(root=tmp_path)
    asset = MediaAsset(uri="hf://datasets/demo/file.png", kind="image")

    rebound = cache.bind(asset)
    assert rebound is asset


def test_disk_media_cache_downloads_http(tmp_path):
    payload = tmp_path / "remote.bin"
    payload.write_bytes(b"remote")

    class QuietHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(tmp_path), **kwargs)

        def log_message(self, format, *args):  # noqa: D401,N802 - silence server logs
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), QuietHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        uri = f"http://127.0.0.1:{port}/remote.bin"
        sha = hashlib.sha256(b"remote").hexdigest()
        cache = DiskMediaCache(root=tmp_path / "cache")
        asset = MediaAsset(uri=uri, kind="image", sha256=sha)

        rebound = cache.bind(asset)

        assert rebound is not asset
        cached_path = cache.root / sha
        assert cached_path.exists()
        with rebound.open() as handle:
            assert handle.read() == b"remote"
    finally:
        server.shutdown()
        thread.join()
