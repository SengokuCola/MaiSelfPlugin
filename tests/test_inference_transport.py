from pathlib import Path
import subprocess
import sys

from recog.service import Service


def worker(service: Service, code: str):
    with (service.store.root / "inference.log").open("a", encoding="utf-8") as log:
        return subprocess.Popen(
            [sys.executable, "-u", "-c", code], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=log, text=True, encoding="utf-8",
        )


def test_worker_exit_reports_current_error_and_reaps_process(tmp_path: Path, monkeypatch):
    service = Service(tmp_path, sys.executable)
    (tmp_path / "inference.log").write_text("OLD ERROR\n", encoding="utf-8")
    processes = []

    def spawn(*args, **kwargs):
        process = worker(service, "import sys;print(\"ModuleNotFoundError: No module named 'torch'\",file=sys.stderr);sys.exit(1)")
        processes.append(process)
        return process

    monkeypatch.setattr(service, "_spawn", spawn)
    try:
        result = service.request({"operation": "load"}, timeout=2)
        assert result["status"] == "model_error"
        assert result["error"] == "未安装 PyTorch"
        assert not service.inference_enabled
        assert "OLD ERROR" not in result["error"]
        assert processes[0].poll() == 1
        assert processes[0].stdin.closed and processes[0].stdout.closed
        assert service.inference is None
    finally:
        service.close()


def test_timeout_is_distinguished_from_worker_exit(tmp_path: Path, monkeypatch):
    service = Service(tmp_path, sys.executable)
    processes = []

    def spawn(*args, **kwargs):
        process = worker(service, "import sys,time;sys.stdin.readline();time.sleep(10)")
        processes.append(process)
        return process

    monkeypatch.setattr(service, "_spawn", spawn)
    try:
        result = service.request({"operation": "load"}, timeout=0.1)
        assert "推理响应超时" in result["error"]
        assert processes[0].poll() is not None
        assert service.inference is None
    finally:
        service.close()


def test_stop_blocks_requests_until_manual_start_or_plugin_reload(tmp_path: Path, monkeypatch):
    service = Service(tmp_path, sys.executable)
    processes = []

    def spawn(*args, **kwargs):
        process = worker(service, 'import sys;\nfor line in sys.stdin: print(\'{"status":"ready"}\', flush=True)')
        processes.append(process)
        return process

    monkeypatch.setattr(service, "_spawn", spawn)
    try:
        assert service.request({"operation": "load"})["status"] == "ready"
        service.stop_inference()
        assert processes[0].poll() is not None
        for _ in range(2):
            assert service.request({"image": "unused"})["status"] == "inference_stopped"
        assert len(processes) == 1
        assert service.request({"operation": "load"}, resume=True)["status"] == "ready"
        assert len(processes) == 2
        service.stop_inference()
    finally:
        service.close()
    reopened = Service(tmp_path, sys.executable)
    try:
        assert reopened.inference_enabled
    finally:
        reopened.close()
