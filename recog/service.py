"""Fast task submission and isolated, cached inference worker management."""

from pathlib import Path
import asyncio
import base64
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid

from .store import Store, atomic_json, digest
from .processes import process_alive

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def inference_error_message(detail: str) -> str:
    """Keep diagnostic tracebacks in logs; translate missing packages for the UI."""
    missing = re.findall(r"No module named ['\"]([^'\"]+)['\"]", detail)
    if missing:
        names = {"torch": "PyTorch", "torchvision": "torchvision", "PIL": "Pillow", "numpy": "NumPy", "psutil": "psutil"}
        packages = dict.fromkeys(names.get(name.split('.')[0], name.split('.')[0]) for name in missing)
        return "；".join(f"未安装 {name}" for name in packages)
    return "推理服务启动或运行失败，请查看 inference.log"


class Service:
    def __init__(self, root: Path, python: str = ""):
        self.store = Store(root)
        self.configure_python(python)
        self.processes = {}
        self.inference = None
        self.inference_enabled = True
        self.train_lock = threading.Lock()
        self.infer_lock = threading.Lock()
        self.closed = threading.Event()
        self.environment_process = None
        self.environment_cache = None
        self.environment_checked = 0.0
        with self.store.db() as db:
            for row in db.execute("SELECT id FROM jobs WHERE state IN ('queued','running','pausing','cancelling')"):
                (root / "jobs" / row[0] / "cancel").touch()
            db.execute(
                "UPDATE jobs SET state='interrupted',error='控制器重启；可手动重试' WHERE state IN ('queued','running','pausing','cancelling')"
            )
        self.monitor = threading.Thread(target=self._monitor, daemon=True)
        self.monitor.start()

    def update_config(self, character: str, python: str) -> None:
        # Existing training and inference processes keep their environment; new
        # processes use the updated executable without cancelling active work.
        with self.train_lock:
            self.store.set_setting("character", character)
            self.configure_python(python)
            self.environment_cache = None

    def configure_python(self, python: str = "") -> None:
        override = self.store.setting("worker_override")
        custom = override == "private" or self.store.setting("worker_mode") == "custom"
        self.overlay_path = None if custom else self.store.root / "dependencies" / "packages"
        self.python = (
            str(PACKAGE_ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
            if override == "private" else override or python or sys.executable
        )

    def runtime_command(self, *args: str, overlay: Path | None = None) -> list[str]:
        command = [self.python, "-X", "utf8", "-u"]
        target = self.overlay_path if overlay is None else overlay
        if target is not None:
            command.extend(["-m", "recog.bootstrap", str(target)])
        return [*command, *args]

    def _spawn(self, operation: str, job: str = "", pipe: bool = False):
        self.check_environment_idle()
        if not Path(self.python).is_file():
            raise ValueError(
                "插件训练环境尚未安装，请在插件目录执行 uv venv .venv 与 uv pip install --python .venv -r requirements-worker.txt"
            )
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        command = self.runtime_command("-m", "recog.worker", operation, str(self.store.root))
        if job:
            command.append(job)
        if pipe:
            stderr = open(self.store.root / "inference.log", "a", encoding="utf-8")
            try:
                return subprocess.Popen(
                    command,
                    cwd=PACKAGE_ROOT,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=stderr,
                    text=True,
                    encoding="utf-8",
                    creationflags=flags,
                )
            finally:
                stderr.close()
        log = open(self.store.root / "jobs" / job / "worker.log", "a", encoding="utf-8")
        try:
            return subprocess.Popen(command, cwd=PACKAGE_ROOT, stdout=log, stderr=log, creationflags=flags)
        finally:
            log.close()

    def submit(
        self, request_key: str, mode: str = "incremental", device: str = "cpu", character: str = "Bot自己"
    ) -> dict:
        if mode not in {"fresh", "incremental"} or device not in {"cpu", "cuda"}:
            raise ValueError("无效训练模式或设备")
        if not request_key or len(request_key) > 128:
            raise ValueError("需要长度1到128的幂等请求编号")
        with self.train_lock:
            self.check_environment_idle()
            with self.store.db() as db:
                db.execute("BEGIN IMMEDIATE")
                existing = db.execute("SELECT * FROM jobs WHERE request_key=?", (request_key,)).fetchone()
                if existing:
                    return dict(existing)
                if any(process.poll() is None for process in self.processes.values()):
                    raise ValueError("上一个训练进程尚未退出")
                active = db.execute(
                    "SELECT id FROM jobs WHERE state IN ('queued','running','pausing','cancelling')"
                ).fetchone()
                if active:
                    raise ValueError("已有训练任务")
                # Previous workers acknowledge cancel; never overlap an orphan after restart.
                for row in db.execute("SELECT id,pid FROM jobs WHERE state='interrupted' AND pid IS NOT NULL"):
                    if not process_alive(row[1]):
                        continue
                    raise ValueError("旧训练进程尚未退出，请稍后手动重试")
                job = uuid.uuid4().hex
                db.execute("INSERT OR REPLACE INTO settings VALUES ('character',?)", (character,))
                snapshot = self.store.freeze(db, job, mode, device)
                directory = self.store.root / "jobs" / job
                directory.mkdir()
                atomic_json(directory / "snapshot.json", snapshot)
                (directory / "lease").touch()
                db.execute(
                    "INSERT INTO jobs VALUES (?,?,'queued',NULL,'',?,?)",
                    (job, request_key, time.time(), digest(snapshot)),
                )
            try:
                process = self._spawn("train", job)
                self.processes[job] = process
                with self.store.db() as db:
                    db.execute("UPDATE jobs SET pid=? WHERE id=?", (process.pid, job))
            except Exception as exc:
                with self.store.db() as db:
                    db.execute("UPDATE jobs SET state='failed',error=? WHERE id=?", (str(exc), job))
        return self.store.job(job)

    def _monitor(self):
        while not self.closed.wait(1):
            with self.train_lock:
                for job, process in list(self.processes.items()):
                    directory = self.store.root / "jobs" / job
                    if process.poll() is None:
                        (directory / "lease").touch()
                        if (directory / "cancel").exists() and time.time() - (
                            directory / "cancel"
                        ).stat().st_mtime > 10:
                            process.terminate()
                    else:
                        with self.store.db() as db:
                            state = "cancelled" if (directory / "cancel").exists() else "failed"
                            db.execute(
                                "UPDATE jobs SET state=?,error=? WHERE id=? AND state IN ('queued','running','pausing','cancelling')",
                                (state, f"训练进程退出，code={process.returncode}", job),
                            )
                        self.processes.pop(job)

    def cancel(self, job: str) -> dict:
        self.store.job(job)
        with self.store.db() as db:
            db.execute("UPDATE jobs SET state='cancelled' WHERE id=? AND state='paused'", (job,))
            db.execute(
                "UPDATE jobs SET state='cancelling' WHERE id=? AND state IN ('queued','running','pausing')", (job,)
            )
        (self.store.root / "jobs" / job / "cancel").touch()
        return self.store.job(job)

    def pause(self, job: str) -> dict:
        with self.train_lock, self.store.db() as db:
            row = db.execute("SELECT state FROM jobs WHERE id=?", (job,)).fetchone()
            if row is None or row[0] not in {"queued", "running"}:
                raise ValueError("仅运行中的任务可暂停")
            path = self.store.root / "jobs" / job / "progress.json"
            if path.exists() and json.loads(path.read_text("utf-8")).get("phase") == "test":
                raise ValueError("已进入最终测试核验，请等待完成或取消")
            (self.store.root / "jobs" / job / "pause").touch()
            db.execute("UPDATE jobs SET state='pausing' WHERE id=?", (job,))
        return self.store.job(job)

    def resume(self, job: str) -> dict:
        with self.train_lock:
            self.check_environment_idle()
            if any(p.poll() is None for p in self.processes.values()):
                raise ValueError("训练进程尚未退出，请稍后继续")
            with self.store.db() as db:
                row = db.execute("SELECT state FROM jobs WHERE id=?", (job,)).fetchone()
                if row is None or row[0] != "paused":
                    raise ValueError("仅暂停的任务可继续")
                if db.execute(
                    "SELECT 1 FROM jobs WHERE state IN ('queued','running','pausing','cancelling')"
                ).fetchone():
                    raise ValueError("已有其他训练任务")
                directory = self.store.root / "jobs" / job
                if not (directory / "resume.pt").exists():
                    raise ValueError("恢复断点不存在")
                (directory / "pause").unlink(missing_ok=True)
                (directory / "lease").touch()
                db.execute("UPDATE jobs SET state='queued',error='' WHERE id=?", (job,))
            try:
                process = self._spawn("train", job)
                self.processes[job] = process
                with self.store.db() as db:
                    db.execute("UPDATE jobs SET pid=? WHERE id=?", (process.pid, job))
            except Exception as exc:
                with self.store.db() as db:
                    db.execute("UPDATE jobs SET state='paused',error=? WHERE id=?", (str(exc), job))
                raise
        return self.store.job(job)

    def jobs(self) -> dict:
        with self.store.db() as db:
            rows = [dict(r) for r in db.execute("SELECT * FROM jobs ORDER BY created DESC LIMIT 50")]
        for row in rows:
            path = self.store.root / "jobs" / row["id"] / "progress.json"
            try:
                row["progress"] = json.loads(path.read_text("utf-8"))
            except (OSError, ValueError):
                # Windows readers can race an atomic replace; the next poll recovers.
                row["progress"] = {"phase": row["state"]}
            p = row["progress"]
            labels = {
                "queued": "排队中",
                "preparing": "准备模型和数据",
                "running": "运行中",
                "training": "训练中",
                "validation": "验证中",
                "test": "测试和核验",
                "complete": "完成",
                "failed": "失败",
                "paused": "已暂停",
                "pausing": "正在保存断点",
                "cancelled": "已取消",
                "cancelling": "取消中",
                "interrupted": "已中断",
            }
            row["name"] = time.strftime("训练 %m-%d %H:%M", time.localtime(row["created"]))
            row["state_text"] = labels.get(row["state"], row["state"])
            fraction = ((p.get("epoch", 1) - 1) + p.get("step", 0) / max(1, p.get("steps", 1))) / max(
                1, p.get("epochs", 1)
            )
            row["percent"] = (
                100 if row["state"] == "complete" else min(99, round(fraction * 100, 1)) if p.get("epochs") else 0
            )
            seconds = p.get("seconds", 0)
            row["time_text"] = f"已用 {seconds:.0f} 秒"
            if p.get("phase") == "training" and fraction > 0.05 and seconds > 5:
                row["time_text"] += f" · 预计还需约 {seconds * (1 / fraction - 1):.0f} 秒（不含最终核验）"
            row["progress_text"] = labels.get(p.get("phase"), "准备中") + (
                f" · 第{p['epoch']}/{p['epochs']}轮 · {p.get('seen', 0)}/{p.get('total_images', 0)}张 · {p.get('step', 0)}/{p.get('steps', 0)}批"
                if p.get("epochs")
                else ""
            )
            row["can_pause"] = row["state"] in {"queued", "running"} and p.get("phase") != "test"
            row["can_resume"] = row["state"] == "paused"
            row["can_cancel"] = row["state"] in {"queued", "running", "pausing", "paused"}
            row["has_report"] = row["state"] == "complete"
        return {"rows": rows}

    def stop_inference(self) -> None:
        with self.infer_lock:
            self.inference_enabled = False
            process = self.inference
            if process and process.poll() is None:
                process.terminate()
                process.wait(5)
            self.inference = None

    def request(self, payload: dict, timeout: float = 60, *, resume: bool = False) -> dict:
        deadline = time.monotonic() + timeout
        if not self.infer_lock.acquire(timeout=timeout):
            return {"status": "busy", "error": "推理繁忙"}
        try:
            if not self.inference_enabled and not resume:
                return {"status": "inference_stopped", "error": "推理服务已停止，请手动开启或重新加载插件"}
            self.check_environment_idle()
            if resume:
                self.inference_enabled = True
            if self.inference is None or self.inference.poll() is not None:
                log = self.store.root / "inference.log"
                self.inference_log_start = log.stat().st_size if log.exists() else 0
                self.inference = self._spawn("serve", pipe=True)
            process = self.inference
            # A worker failure cannot leave the host coroutine waiting indefinitely.
            output = []

            def read_response():
                try:
                    # Include pipe writes in the budget: a cold or wedged worker
                    # can otherwise block a large image before response reading.
                    process.stdin.write(json.dumps(payload) + "\n")
                    process.stdin.flush()
                    output.append(process.stdout.readline())
                except (OSError, UnicodeError):
                    output.append("")

            thread = threading.Thread(target=read_response, daemon=True)
            thread.start()
            thread.join(max(0, deadline - time.monotonic()))
            if thread.is_alive() or not output or not output[0]:
                timed_out = thread.is_alive() and process.poll() is None
                if timed_out:
                    process.terminate()
                try:
                    process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=0.5)
                thread.join(0.5)
                if not thread.is_alive():
                    process.stdin.close()
                    process.stdout.close()
                self.inference = None
                self.inference_enabled = False
                if timed_out:
                    error = f"推理响应超时（{timeout:g} 秒），请查看运行环境和 inference.log"
                else:
                    with (self.store.root / "inference.log").open("rb") as stream:
                        stream.seek(0, 2)
                        stream.seek(max(self.inference_log_start, stream.tell() - 2000))
                        detail = stream.read().decode("utf-8", errors="replace").strip()
                    error = inference_error_message(detail)
                return {"status": "model_error", "error": error}
            result = json.loads(output[0])
            if resume and result.get("status") != "ready":
                self.inference_enabled = False
                process.terminate()
                process.wait(5)
                process.stdin.close()
                process.stdout.close()
                self.inference = None
            return result
        finally:
            self.infer_lock.release()

    def check_environment_idle(self) -> None:
        """Never start a worker while dependencies are being replaced."""
        if self.environment_process and self.environment_process.poll() is None:
            raise ValueError("依赖安装/卸载正在运行，请等待完成")
        path = self.store.root / "environment-status.json"
        if path.exists():
            job = json.loads(path.read_text("utf-8"))
            if job.get("state") == "running":
                if not process_alive(job["pid"]):
                    return
                raise ValueError("依赖安装/卸载正在运行，请等待完成")

    def infer(self, raw: bytes, timeout: float = 8) -> dict:
        if not self.store.setting("active"):
            return {"status": "no_model"}
        try:
            return self.request({"image": base64.b64encode(raw).decode()}, timeout)
        except Exception as exc:
            return {"status": "model_error", "error": str(exc)}

    def activate(self, version: str) -> dict:
        if not any(r["id"] == version for r in self.store.models()):
            raise ValueError("版本不存在")
        result = self.request({"operation": "load", "version": version}, timeout=8)
        if result["status"] != "ready":
            raise ValueError(result.get("error", "模型加载失败"))
        with self.store.db() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT value FROM settings WHERE key='active'").fetchone()
            if old and old[0] != version:
                db.execute("INSERT OR REPLACE INTO settings VALUES ('previous',?)", (old[0],))
            db.execute("INSERT OR REPLACE INTO settings VALUES ('active',?)", (version,))
        return result

    def rollback(self) -> dict:
        previous = self.store.setting("previous")
        if not previous:
            raise ValueError("无可回退版本")
        return self.activate(previous)

    def close(self) -> None:
        self.closed.set()
        self.monitor.join(3)
        with self.train_lock:
            for job, process in self.processes.items():
                self.cancel(job)
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                with self.store.db() as db:
                    db.execute(
                        "UPDATE jobs SET state='cancelled',error='插件卸载' WHERE id=? AND state!='complete'", (job,)
                    )
        with self.infer_lock:
            if self.inference and self.inference.poll() is None:
                self.inference.terminate()
                self.inference.wait(5)


async def offload(function, *args, **kwargs):
    return await asyncio.to_thread(function, *args, **kwargs)
