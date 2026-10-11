"""Explicit dependency management confined to the plugin-owned environment."""

from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import time
import threading
import uuid

from .store import atomic_json
from .processes import process_alive
from .dependency_download import prepare_wheels

PROBE = """import json,sys,importlib.util
r={'python':sys.executable,'installed':False,'cuda':False}
r['missing']=[name for name in ('torch','torchvision') if importlib.util.find_spec(name) is None]
try:
 import torch,torchvision
 r.update(installed=True,torch=torch.__version__,torchvision=torchvision.__version__,build_cuda=torch.version.cuda,cuda=torch.cuda.is_available(),torch_path=torch.__file__,torchvision_path=torchvision.__file__)
 if r['cuda']:
  x=torch.ones(2,device='cuda'); r['cuda']=bool((x+x).sum().item()==4); r['gpu']=torch.cuda.get_device_name(0)
except Exception as e:r['error']=str(e)
print(json.dumps(r))
"""


def status(service):
    try:
        service.check_environment_idle()
    except ValueError:
        return describe(service, {"installed": False, "cuda": False, "detecting": True})
    now = time.monotonic()
    if service.environment_cache is None or now - service.environment_checked > 60:
        service.environment_checked = now
        service.environment_cache = {"installed": False, "cuda": False, "detecting": True}
        threading.Thread(target=probe_environment,
                         args=(service, service.python, service.environment_cache, service.runtime_command("-c", PROBE)),
                         daemon=True).start()
    r = dict(service.environment_cache)
    return describe(service, r)


def probe_environment(service, python, token, command):
    """Dependency imports can be slow; queries must stay inside the API budget."""
    try:
        result = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        value = (
            json.loads(result.stdout)
            if result.returncode == 0
            else {"installed": False, "cuda": False, "error": result.stderr[-1000:]}
        )
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        value = {"installed": False, "cuda": False, "error": str(exc)}
    if service.python == python and service.environment_cache is token:
        service.environment_cache = value


def describe(service, r):
    """Combine cached runtime information with current operation logs."""
    private = service.overlay_path is None
    job = {}
    path = service.store.root / "environment-status.json"
    if path.exists():
        job = json.loads(path.read_text("utf-8"))
        if job.get("state") == "running":
            if not process_alive(job["pid"]):
                job.update(state="interrupted", error="安装控制器已退出，请手动重试")
                atomic_json(path, job)
    overlay = service.overlay_path
    owned = private if overlay is None else bool(r.get("torch_path") and Path(r["torch_path"]).resolve().is_relative_to(overlay.resolve()))
    cpu = owned and r.get("installed") and not r.get("build_cuda")
    gpu = owned and r.get("installed") and bool(r.get("build_cuda"))
    running = job.get("state") == "running"
    training = any(process.poll() is None for process in service.processes.values())
    inference = service.inference is not None and service.inference.poll() is None
    detecting = bool(r.get("detecting"))
    states = {"running": "处理中", "complete": "已完成", "failed": "失败", "interrupted": "已中断"}
    operation = states.get(job.get("state"), "尚无任务")
    phases = {"prepare": "准备安装", "recover": "恢复上次任务", "resolve": "查询官方安装包",
              "download": "下载安装包", "verify_download": "校验安装包", "install": "安装依赖",
              "verify": "核验环境", "complete": "已完成", "failed": "失败", "waiting_previous": "等待上次安装进程退出"}
    progress_text = phases.get(job.get("phase"), operation)
    if job.get("file"):
        progress_text += f" · {job['file']}"
    if job.get("total"):
        progress_text += f" · {job.get('downloaded', 0) / 1e9:.2f} / {job['total'] / 1e9:.2f} GB"
        if job.get("percent") is not None:
            progress_text += f" · {job['percent']:.1f}%"
    failure = job.get("error", "")
    if failure.startswith("安装后核验未通过："):
        failure = "GPU安装核验失败：CUDA不可用，请检查驱动或选择CPU版。" if job.get("kind") == "gpu" else "安装后依赖核验失败，详情见日志。"
    r.update(
        managed=not private,
        overlay=overlay is not None,
        overlay_path=str(overlay) if overlay is not None else "",
        package_source="插件覆盖层" if owned and overlay is not None else "自定义环境（只读）" if private else "底层环境（只读）",
        can_manage=not private and not running and not detecting and not training and not inference,
        can_switch=not running and not training,
        can_use_host=private and not running and not training,
        can_use_private=not running and not training,
        inference_running=inference,
        inference_enabled=service.inference_enabled,
        is_private=private,
        mode_text="自定义环境" if private else "共享覆盖层",
        readiness_text="检测中" if detecting else "已就绪" if r.get("installed") else "未就绪",
        dependency_text="检测中" if detecting else "GPU版已安装" if r.get("installed") and r.get("build_cuda") else "CPU版已安装" if r.get("installed") else "未安装",
        cuda_text="检测中" if detecting else "可用" if r.get("cuda") else "不可用",
        operation_status=operation,
        progress_text=progress_text,
        progress_percent=job.get("percent"),
        progress_running=running,
        operation_detail=f"{operation} · {'安装GPU版' if job.get('kind') == 'gpu' else '安装CPU版' if job.get('kind') == 'cpu' else '移除依赖' if job.get('kind') == 'remove' else '未提交安装任务'}" + (f"\n{failure}" if failure else ""),
        python_path=service.python,
        dependency_path=str(overlay) if overlay is not None else str(Path(service.python).parents[1]),
        version_text="检测中…" if detecting else f"PyTorch：{r.get('torch', '未安装')}\ntorchvision：{r.get('torchvision', '未安装')}",
        source_text="检测中…" if detecting else "插件覆盖层" if owned and overlay is not None else "自定义环境（只读）" if private else "MaiBot/指定Python的已有依赖（只读）",
        usage_text="训练正在运行，结束后才能修改依赖。" if training else "推理服务运行中，修改依赖前请先停止推理。" if inference else "依赖任务正在运行，请等待完成。" if running else "正在检测依赖，请稍候。" if detecting else "可以安装或移除依赖。CPU与GPU版会相互替换。",
        host_label="已使用：共享覆盖层" if not private else "切换为共享覆盖层",
        private_label="更换自定义环境" if private else "切换为自定义环境",
        host_mode="host",
        private_mode="private",
        cpu_action="remove" if cpu else "cpu",
        gpu_action="remove" if gpu else "gpu",
        cpu_label="卸载 PyTorch（CPU）" if cpu else "安装 PyTorch（CPU）",
        gpu_label="卸载 PyTorch（GPU）" if gpu else "安装 PyTorch（GPU）",
        summary=f"Python：{service.python}\nPyTorch：{r.get('torch', '检测中' if r.get('detecting') else '未安装')} · torchvision：{r.get('torchvision', '检测中' if r.get('detecting') else '未安装')}\nCUDA：{'检测中' if r.get('detecting') else '可用' if r.get('cuda') else '不可用'} · 环境：{'插件独立' if overlay is None else '继承底层Python + 插件依赖覆盖层'}\n依赖来源：{'插件覆盖层' if owned and overlay is not None else '插件独立环境' if private else '底层环境（只读）'}\n{r.get('error', '')}",
        operation_text=f"依赖任务：{job.get('state', '无')} {job.get('error', '')}",
        guide=f'覆盖层目录：{overlay}。安装/卸载仅修改此目录，底层Python及其依赖不修改。卸载覆盖层后，底层已有的包仍可能可用。' if overlay is not None else '',
        log=((service.store.root / "environment.log").read_text("utf-8", errors="replace")[-5000:]
             if (service.store.root / "environment.log").exists() else "尚无安装日志。")
            + (f"\n\n失败详情：\n{job['error']}" if job.get("error") else ""),
    )
    return r


def submit(service, kind):
    if service.overlay_path is None:
        raise ValueError("自定义环境仅检测和使用，依赖请在该环境中手动管理")
    if kind not in {"cpu", "gpu", "remove"}:
        raise ValueError("无效依赖操作")
    with service.train_lock, service.infer_lock:
        r = status(service)
        if service.environment_process and service.environment_process.poll() is None:
            raise ValueError("依赖任务正在运行")
        if not r["can_manage"]:
            raise ValueError("环境检测或依赖任务尚未结束，请稍后重试")
        if any(p.poll() is None for p in service.processes.values()):
            raise ValueError("训练正在使用环境")
        if service.inference is not None and service.inference.poll() is None:
            raise ValueError("推理进程正在使用环境，请先停止推理服务")
        if kind == "gpu":
            if not shutil.which("nvidia-smi"):
                raise ValueError("未检测到NVIDIA驱动；请先安装驱动")
            try:
                test = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                    capture_output=True,
                    timeout=3,
                )
            except subprocess.TimeoutExpired as exc:
                raise ValueError("NVIDIA驱动检查超时，请检查驱动") from exc
            if test.returncode:
                raise ValueError("NVIDIA驱动检查失败")
        root = service.store.root
        return launch(service, kind)


def launch(service, kind, previous=None):
    root = service.store.root
    path = root / "environment-status.json"
    # Publish the pending task before spawn, so the parent never overwrites a
    # newer byte-progress record produced by a fast child.
    atomic_json(path, {**(previous or {}), "state": "running", "pid": os.getpid(),
                       "python": service.python, "kind": kind, "target": str(service.overlay_path),
                       "phase": "recover" if previous else "prepare"})
    try:
        with (root / "environment.log").open("a" if previous else "w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, "-X", "utf8", "-m", "recog.environment", str(root), service.python, kind,
                 str(service.overlay_path) if service.overlay_path is not None else ""],
                cwd=Path(__file__).resolve().parents[1], stdout=log, stderr=log,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
    except OSError as exc:
        atomic_json(path, {"state": "failed", "kind": kind, "error": str(exc)})
        raise
    service.environment_process = process
    service.environment_cache = None
    return {"state": "running", "pid": process.pid}


def recover(service):
    """Reattach live installers; retry only an interrupted installation at plugin load."""
    path = service.store.root / "environment-status.json"
    if not path.exists():
        return
    job = json.loads(path.read_text("utf-8"))
    if job.get("state") not in {"running", "interrupted"}:
        return
    if job.get("pid") and process_alive(job["pid"]):
        return  # The independent controller still owns progress and logs.
    if (service.overlay_path is None or job.get("kind") not in {"cpu", "gpu"}
            or job.get("python", service.python) != service.python or job.get("target") != str(service.overlay_path)):
        atomic_json(path, {**job, "state": "interrupted", "error": "上次任务环境已改变，请手动重新安装"})
        return
    with service.train_lock, service.infer_lock:
        launch(service, job["kind"], job)


def remove_owned(path: Path, root: Path) -> None:
    """Only remove a verified child of the plugin dependency directory."""
    resolved = path.resolve()
    parent = (root / "dependencies").resolve()
    if (not parent.is_relative_to(root.resolve()) or (root / "dependencies").is_symlink()
            or resolved == parent or not resolved.is_relative_to(parent) or path.is_symlink()):
        raise ValueError("依赖目录必须位于插件自己的dependencies内")
    if path.exists():
        shutil.rmtree(path)


def run_installer(command, update):
    process = subprocess.Popen(command, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    update(child_pid=process.pid)
    code = process.wait()
    update(child_pid=None)
    if code:
        raise subprocess.CalledProcessError(code, command)


def install(root: Path, python: str, kind: str, overlay: Path | None = None) -> None:
    # This OS lock is released even after a forced exit. Two recovering hosts
    # cannot simultaneously download, clean staging, or replace the overlay.
    root.mkdir(parents=True, exist_ok=True)
    with (root / "environment.lock").open("a+b") as lock:
        try:
            if os.name == "nt":
                import msvcrt
                lock.write(b"0"); lock.flush(); lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return
        try:
            _install(root, python, kind, overlay)
        finally:
            if os.name == "nt":
                lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)


def _install(root: Path, python: str, kind: str, overlay: Path | None = None) -> None:
    path = root / "environment-status.json"
    previous = json.loads(path.read_text("utf-8")) if path.exists() else {}
    record = {"state": "running", "pid": os.getpid(), "kind": kind, "python": python,
              "target": str(overlay) if overlay is not None else python, "phase": "prepare",
              "started_at": previous.get("started_at", time.time()) if previous.get("state") in {"running", "interrupted"} and previous.get("kind") == kind else time.time(), "percent": None}

    def update(**fields):
        record.update(fields, updated_at=time.time())
        atomic_json(path, record)

    update()
    stage = None
    try:
        if kind not in {"cpu", "gpu", "remove"}:
            raise ValueError("无效依赖操作")
        if overlay is not None:
            if (overlay.resolve() != (root / "dependencies" / "packages").resolve()
                    or not overlay.resolve().is_relative_to(root.resolve())
                    or (root / "dependencies").is_symlink() or overlay.is_symlink()):
                raise ValueError("覆盖层路径不属于本插件")
            overlay.parent.mkdir(parents=True, exist_ok=True)
            # An orphaned uv process may still be writing its old staging
            # directory. Wait for it; never kill unrelated processes.
            child = previous.get("child_pid")
            while child and process_alive(child):
                update(phase="waiting_previous", child_pid=child)
                time.sleep(1)
            update(child_pid=None)
            for leftover in overlay.parent.glob("staging-*"):
                remove_owned(leftover, root)
            backups = list(overlay.parent.glob("previous-*"))
            if not overlay.exists() and len(backups) == 1:
                backups[0].rename(overlay)
            if len(backups) > 1 and not overlay.exists():
                raise ValueError("发现多个旧覆盖层备份，请手动核对，未删除")
            if kind == "remove":
                remove_owned(overlay, root)
                update(state="complete", phase="complete", percent=100)
                return
            stage = overlay.parent / ("staging-" + uuid.uuid4().hex)
            stage.mkdir()
        else:
            raise ValueError("禁止修改底层Python环境，请使用插件覆盖层")
        uv = shutil.which("uv")
        if not uv:
            raise ValueError("未找到uv，请先安装uv")
        if not Path(python).exists():
            raise ValueError("底层Python不存在")
        if kind == "remove":
            subprocess.run([uv, "pip", "uninstall", "--python", python, "torch", "torchvision"], check=True)
        else:
            # A target installs packages without creating an interpreter or
            # changing the base site-packages. Include dependencies for ABI consistency.
            destination = ["--target", str(stage)] if stage is not None else []
            reinstall = ["--reinstall-package", "torch", "--reinstall-package", "torchvision"] if stage is None else []
            index = "https://download.pytorch.org/whl/" + ("cpu" if kind == "cpu" else "cu128")
            build = "cpu" if kind == "cpu" else "cu128"
            wheels = prepare_wheels(root, python, build, update)
            update(phase="install", percent=None, file="", downloaded=0, total=0)
            # Parent-project indexes otherwise outrank --index-url and may select CPU wheels for GPU installs.
            run_installer([uv, "pip", "install", "--no-config", "--python", python, *destination,
                           *map(str, wheels), "--index-url", index, *reinstall], update)
            run_installer([uv, "pip", "install", "--no-config", "--python", python, *destination,
                           "Pillow>=11,<13", "numpy>=2,<3", "psutil>=6,<8"], update)
            update(phase="verify", percent=None)
            command = [python, "-X", "utf8"]
            if stage is not None:
                command.extend(["-m", "recog.bootstrap", str(stage)])
            probe = subprocess.run([*command, "-c", PROBE], cwd=Path(__file__).resolve().parents[1],
                                   capture_output=True, text=True, encoding="utf-8", check=True)
            verified = json.loads(probe.stdout)
            print(json.dumps(verified, ensure_ascii=False), flush=True)
            if not verified["installed"]:
                raise ValueError(f"依赖加载失败：{verified.get('error', '未能加载PyTorch/torchvision')}")
            if kind == "gpu" and not verified["cuda"]:
                raise ValueError("GPU安装核验失败：CUDA不可用，请检查驱动或选择CPU版。")
            if stage is not None:
                for field in ("torch_path", "torchvision_path"):
                    if not Path(verified[field]).resolve().is_relative_to(stage.resolve()):
                        raise ValueError("核验加载了底层包，未实际加载覆盖层")
                atomic_json(stage / "overlay.json", {"python": python, "kind": kind,
                                                       "torch": verified["torch"], "torchvision": verified["torchvision"]})
                backup = overlay.parent / ("previous-" + uuid.uuid4().hex)
                if overlay.exists():
                    overlay.rename(backup)
                try:
                    stage.rename(overlay)
                    stage = None
                except Exception:
                    if backup.exists():
                        backup.rename(overlay)
                    raise
                if backup.exists():
                    remove_owned(backup, root)
        update(state="complete", phase="complete", percent=100)
    except Exception as exc:
        update(state="failed", phase="failed", error=str(exc))
        raise
    finally:
        if stage is not None:
            remove_owned(stage, root)


if __name__ == "__main__":
    install(Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else None)
