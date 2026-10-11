from pathlib import Path
from types import SimpleNamespace
import hashlib
import io
import json
import sys

import pytest

from recog import dependency_download as downloads, environment
from recog.service import Service
from recog.store import atomic_json


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {"Content-Length": str(len(data))}


@pytest.mark.parametrize("supports_range", [True, False])
def test_resumes_bytes_or_restarts_if_server_ignores_range(tmp_path, monkeypatch, supports_range):
    data = b"a" * (1024 * 1024) + b"b" * 1024
    partial = tmp_path / "torch.whl.part"
    partial.write_bytes(data[:1024 * 1024])
    events = []

    def open_url(request, **kwargs):
        assert request.get_header("Range") == "bytes=1048576-"
        return Response(data[1048576:], 206, {"Content-Range": f"bytes 1048576-{len(data)-1}/{len(data)}"}) if supports_range else Response(data)

    monkeypatch.setattr(downloads, "urlopen", open_url)
    final = downloads.download("https://example.test/torch.whl", hashlib.sha256(data).hexdigest(), tmp_path, lambda **event: events.append(event))
    assert final.read_bytes() == data
    assert not partial.exists()
    assert any(event.get("downloaded") == len(data) for event in events)


def test_valid_cache_is_reused_and_corrupt_partial_is_retried(tmp_path, monkeypatch):
    data = b"valid wheel"
    (tmp_path / "torch.whl.part").write_bytes(b"corrupt")
    calls = []

    def open_url(request, **kwargs):
        calls.append(request)
        if len(calls) == 1:
            return Response(b"bad", 206, {"Content-Range": "bytes 7-9/10"})
        assert request.get_header("Range") is None
        return Response(data)

    monkeypatch.setattr(downloads, "urlopen", open_url)
    sha = hashlib.sha256(data).hexdigest()
    events = []
    downloads.download("https://example.test/torch.whl", sha, tmp_path, lambda **event: events.append(event))
    assert len(calls) == 2
    downloads.download("https://example.test/torch.whl", sha, tmp_path, lambda **event: events.append(event))
    assert len(calls) == 2 and events[-1]["cached"]


def test_interrupted_download_preserves_partial_bytes_and_progress(tmp_path, monkeypatch):
    data = b"a" * (1024 * 1024) + b"tail"

    class Interrupted(Response):
        def read(self, size=-1):
            if self.tell():
                raise OSError("connection lost")
            return super().read(size)

    monkeypatch.setattr(downloads, "urlopen", lambda *args, **kwargs: Interrupted(data))
    events = []
    with pytest.raises(OSError, match="connection lost"):
        downloads.download("https://example.test/torch.whl", hashlib.sha256(data).hexdigest(), tmp_path,
                           lambda **event: events.append(event))
    assert (tmp_path / "torch.whl.part").stat().st_size == 1024 * 1024
    assert events[-1]["downloaded"] == 1024 * 1024


def test_recovery_reattaches_live_job_or_relaunches_dead_job(tmp_path, monkeypatch):
    service = Service(tmp_path, sys.executable)
    job = {"state": "running", "pid": 123, "python": sys.executable, "kind": "gpu",
           "target": str(service.overlay_path), "downloaded": 100, "total": 200, "percent": 50, "phase": "download"}
    atomic_json(tmp_path / "environment-status.json", job)
    launched = []
    monkeypatch.setattr(environment, "launch", lambda *args: launched.append(args))
    monkeypatch.setattr(environment, "process_alive", lambda pid: True)
    try:
        environment.recover(service)
        assert not launched
        assert environment.describe(service, {})["progress_percent"] == 50
        monkeypatch.setattr(environment, "process_alive", lambda pid: False)
        environment.recover(service)
        assert len(launched) == 1 and launched[0][2]["downloaded"] == 100
        service.store.set_setting("worker_mode", "custom")
        service.configure_python()
        environment.recover(service)
        assert len(launched) == 1
        assert json.loads((tmp_path / "environment-status.json").read_text(encoding="utf-8"))["state"] == "interrupted"
    finally:
        service.close()
