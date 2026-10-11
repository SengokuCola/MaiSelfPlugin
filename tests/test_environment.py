from types import SimpleNamespace
import sys
import pytest
from recog import environment
from recog.service import Service
from recog.store import atomic_json


def test_shared_environment_manages_only_overlay(tmp_path):
    service = Service(tmp_path, sys.executable)
    try:
        service.environment_cache = {"installed": True, "cuda": False, "torch_path": str(tmp_path / "base/torch/__init__.py")}
        service.environment_checked = __import__('time').monotonic()
        result = environment.status(service)
        assert result["managed"] and result["can_manage"] and result["overlay"]
        assert result["cpu_action"] == "cpu"  # Never offer uninstall for inherited packages.
        assert result["package_source"] == "底层环境（只读）"
        assert service.overlay_path == tmp_path / "dependencies/packages"
    finally:
        service.close()


def test_custom_environment_is_read_only_and_survives_reload(tmp_path):
    service = Service(tmp_path, sys.executable)
    service.store.set_setting("worker_override", sys.executable)
    service.store.set_setting("worker_mode", "custom")
    service.configure_python()
    try:
        assert service.overlay_path is None
        assert service.runtime_command("-c", "pass") == [sys.executable, "-X", "utf8", "-u", "-c", "pass"]
        info = environment.describe(service, {"installed": True, "torch": "test"})
        assert not info["managed"] and not info["can_manage"]
        assert info["mode_text"] == "自定义环境"
        for kind in ("cpu", "gpu", "remove"):
            with pytest.raises(ValueError, match="自定义环境"):
                environment.submit(service, kind)
    finally:
        service.close()
    reopened = Service(tmp_path)
    try:
        assert reopened.overlay_path is None and reopened.python == sys.executable
    finally:
        reopened.close()


def test_install_blocks_new_workers_and_persists_environment(tmp_path):
    service = Service(tmp_path, sys.executable)
    try:
        import os

        atomic_json(tmp_path / "environment-status.json", {"state": "running", "pid": os.getpid()})
        with pytest.raises(ValueError):
            service._spawn("serve", pipe=True)
        with pytest.raises(ValueError):
            service.submit("blocked")
    finally:
        service.close()
    (tmp_path / "environment-status.json").unlink()
    service.store.set_setting("worker_override", sys.executable)
    reopened = Service(tmp_path)
    try:
        assert reopened.python == sys.executable
    finally:
        reopened.close()


def test_overlay_precedence_and_base_fallback(tmp_path):
    import json
    import os
    import subprocess
    from recog.service import PACKAGE_ROOT

    base = tmp_path / "base"
    base.mkdir()
    (base / "overlay_sample.py").write_text("value='base'", encoding="utf-8")
    (base / "base_only_sample.py").write_text("value='inherited'", encoding="utf-8")
    service = Service(tmp_path / "data", sys.executable)
    try:
        service.overlay_path.mkdir(parents=True)
        (service.overlay_path / "overlay_sample.py").write_text("value='overlay'", encoding="utf-8")
        extra = service.overlay_path / "extra"
        extra.mkdir()
        (extra / "pth_sample.py").write_text("value='pth'", encoding="utf-8")
        (service.overlay_path / "extra.pth").write_text("extra\nimport overlay_sample\n", encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(base)}
        code = "import json,overlay_sample,base_only_sample,pth_sample; print(json.dumps([overlay_sample.value,base_only_sample.value,pth_sample.value]))"
        result = subprocess.run(service.runtime_command("-c", code), cwd=PACKAGE_ROOT, env=env, capture_output=True, text=True, check=True)
        assert json.loads(result.stdout) == ["overlay", "inherited", "pth"]
        ordinary = subprocess.run([sys.executable, "-c", "import overlay_sample; print(overlay_sample.value)"], env=env, capture_output=True, text=True, check=True)
        assert ordinary.stdout.strip() == "base"
    finally:
        service.close()


@pytest.mark.parametrize("kind", ["cpu", "gpu"])
def test_target_install_and_uninstall_never_modify_base(tmp_path, monkeypatch, kind):
    import json
    from pathlib import Path

    overlay = tmp_path / "dependencies/packages"
    leftover = overlay.parent / "staging-interrupted"
    leftover.mkdir(parents=True)
    (leftover / "partial-install").write_text("incomplete", encoding="utf-8")
    calls = []
    monkeypatch.setattr(environment.shutil, "which", lambda _: "uv")
    monkeypatch.setattr(environment, "prepare_wheels", lambda *args: [Path("torch.whl"), Path("torchvision.whl")])

    def run(command, **kwargs):
        calls.append(command)
        if "--target" in command:
            stage = Path(command[command.index("--target") + 1])
            for name in ("torch", "torchvision"):
                (stage / name).mkdir(exist_ok=True)
                (stage / name / "__init__.py").write_text("# installed", encoding="utf-8")
            return SimpleNamespace(returncode=0)
        stage = Path(command[command.index("recog.bootstrap") + 1])
        return SimpleNamespace(returncode=0, stdout=json.dumps({"installed": True, "cuda": kind == "gpu", "torch": "2.11.0", "torchvision": "0.26.0",
            "torch_path": str(stage / "torch/__init__.py"), "torchvision_path": str(stage / "torchvision/__init__.py")}), stderr="")

    monkeypatch.setattr(environment.subprocess, "run", run)
    monkeypatch.setattr(environment, "run_installer", lambda command, update: run(command))
    environment.install(tmp_path, sys.executable, kind, overlay)
    assert (overlay / "overlay.json").exists()
    assert not leftover.exists()
    build = "cpu" if kind == "cpu" else "cu128"
    install = next(call for call in calls if "torch.whl" in call)
    assert "--no-config" in install and "torchvision.whl" in install
    assert install[install.index("--index-url") + 1].endswith(build)
    assert all("--target" in call for call in calls if call[0] == "uv")
    calls.clear()
    environment.install(tmp_path, sys.executable, "remove", overlay)
    assert not overlay.exists() and not calls
    with pytest.raises(ValueError, match="禁止修改底层"):
        environment.install(tmp_path, sys.executable, "cpu")


def test_failed_overlay_verification_retains_previous_layer(tmp_path, monkeypatch):
    import json

    overlay = tmp_path / "dependencies/packages"
    overlay.mkdir(parents=True)
    (overlay / "keep.txt").write_text("old", encoding="utf-8")
    monkeypatch.setattr(environment.shutil, "which", lambda _: "uv")
    monkeypatch.setattr(environment, "prepare_wheels", lambda *args: [])
    monkeypatch.setattr(environment, "run_installer", lambda *args: None)
    monkeypatch.setattr(environment.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=json.dumps({"installed": False, "cuda": False})))
    with pytest.raises(ValueError, match="依赖加载失败"):
        environment.install(tmp_path, sys.executable, "cpu", overlay)
    assert (overlay / "keep.txt").read_text(encoding="utf-8") == "old"
    assert list(overlay.parent.glob('staging-*')) == []
