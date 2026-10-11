"""Admin-trained self-character identity recognition for MaiBot."""

from pathlib import Path
from typing import Any, Literal, Optional
import base64
import json
import uuid
import hashlib
import time

from maibot_sdk import API, Tool, HookHandler, MaiBotPlugin, PluginConfigBase, Field
from maibot_sdk.types import ToolParameterInfo, ToolParamType

from .recog.service import Service, offload


class PluginSectionConfig(PluginConfigBase):
    enabled: bool = Field(default=True, description="是否启用自身角色识别插件")
    config_version: str = Field(default="0.1.1", description="配置版本")


class Config(PluginConfigBase):
    plugin: PluginSectionConfig = Field(default_factory=PluginSectionConfig)
    character_name: str = Field(default="Bot自己", description="Bot自身角色名")
    worker_python: str = Field(default="", description="留空继承MaiBot Python并使用插件依赖覆盖层；可指定其他底层Python")
    input_size: Literal[224, 320, 448] = Field(default=320, description="新训练任务的图片尺寸；已训练模型使用自己的尺寸")
    auto_recognize: bool = Field(default=True, description="自动识别收到的图片，阳性结果作为上下文提示")


class SelfRecognition(MaiBotPlugin):
    config_model = Config
    service: Optional[Service] = None

    async def on_load(self) -> None:
        self.service = await offload(Service, self.ctx.paths.data_dir, self.config.worker_python)
        from .recog.environment import recover
        await offload(recover, self.service)
        await offload(self.service.store.set_setting, "character", self.config.character_name)
        self._image_cache = {}

    @HookHandler("chat.receive.before_process", timeout_ms=8000)
    async def annotate_received_images(self, message: dict, **kwargs: Any) -> dict:
        """Annotate individual received media without collecting training examples."""
        if not self.config.auto_recognize or not self.service.inference_enabled:
            return {"action": "continue"}
        version = await offload(self.service.store.setting, "active")
        if not version:
            return {"action": "continue"}
        deadline = time.monotonic() + 6

        async def visit(parts):
            result = []
            for part in parts:
                if part.get("type") == "forward":
                    nodes = []
                    for node in part.get("data", []):
                        nodes.append({**node, "content": await visit(node.get("content", []))})
                    result.append({**part, "data": nodes})
                    continue
                result.append(part)
                if part.get("type") not in {"image", "emoji"} or time.monotonic() >= deadline:
                    continue
                binary = part.get("binary_data_base64")
                if not isinstance(binary, str) or len(binary) > 28 * 1024 * 1024:
                    continue
                try:
                    raw = base64.b64decode(binary, validate=True)
                    key = (hashlib.sha256(raw).hexdigest(), version, self.config.character_name)
                    prediction = self._image_cache.get(key)
                    if prediction is None:
                        prediction = await offload(self.service.infer, raw, max(0.1, deadline - time.monotonic()))
                        if prediction.get("status") == "ok" and prediction.get("version", version) == version:
                            if len(self._image_cache) >= 512:
                                self._image_cache.pop(next(iter(self._image_cache)))
                            self._image_cache[key] = prediction
                    if prediction.get("status") == "ok" and prediction["score"] >= 0.6:
                        result.append(
                            {
                                "type": "text",
                                "data": f"\n[提示：此图中的角色看起来很可能是 “{self.config.character_name}”]\n",
                            }
                        )
                except (ValueError, OSError):
                    continue
            return result

        updated = dict(message)
        updated["raw_message"] = await visit(message.get("raw_message", []))
        return {"action": "continue", "modified_kwargs": {**kwargs, "message": updated}}

    async def on_unload(self) -> None:
        # Registration may fail before on_load creates the service.
        if self.service is not None:
            await offload(self.service.close)

    async def on_config_update(self, scope: str, config_data: dict[str, Any], version: str) -> None:
        """The runner applies typed self configuration before invoking this hook."""
        if scope != "self":
            return
        await offload(self.service.update_config, self.config.character_name, self.config.worker_python)

    async def _action(self, function, *args, **kwargs) -> dict:
        try:
            return await offload(function, *args, **kwargs)
        except (ValueError, OSError) as exc:
            return {"success": False, "error": str(exc)}

    @API("self_upload", version="1")
    async def upload(self, upload_id: str, upload_label: str = "self") -> dict:
        if upload_label not in {"self", "other"}:
            return {"success": False, "error": "上传身份必须为是Bot或不是Bot"}
        claimed = await self.ctx.webui.claim_upload(upload_id)
        result = await self._action(self.service.store.ingest, Path(claimed["path"]), upload_label)
        if result.get("already_exists"):
            result["message"] = "图片已存在，复用原标注"
        return result

    @API("self_images", version="1")
    async def images(self, page: int = 1, view: str = "all", job_id: str = "") -> dict:
        errors = job_id if view == "errors" else ""
        result = await offload(self.service.store.images, page, view, job_id)
        result.update(
            errors_active=bool(errors),
            source_text=f"错判图片 · 任务 {errors[:8]}（人工标签与模型判断不一致）" if errors else "图片库",
        )
        return result

    @API("self_label", version="1")
    async def label(self, image_id: str, label: str, remove: bool | None = None) -> dict:
        return await self._action(self.service.store.label, image_id, label, remove)

    @API("self_label_positive", version="1")
    async def label_positive(self, image_id: str) -> dict:
        return await self.label(image_id, "self")

    @API("self_label_negative", version="1")
    async def label_negative(self, image_id: str) -> dict:
        return await self.label(image_id, "other")

    @API("self_label_clear", version="1")
    async def label_clear(self, image_id: str) -> dict:
        return await self.label(image_id, "unlabelled")

    @API("self_image_remove", version="1")
    async def remove_image(self, image_id: str) -> dict:
        return await self._action(self.service.store.set_removed, image_id, True)

    @API("self_image_restore", version="1")
    async def restore_image(self, image_id: str) -> dict:
        return await self._action(self.service.store.set_removed, image_id, False)

    @API("self_images_batch", version="1")
    async def batch_images(self, image_ids: str, operation: str) -> dict:
        if operation not in {"self", "other", "remove"}:
            return {"success": False, "error": "无效批量操作"}
        return await self._action(self.service.store.batch_update, image_ids, operation)

    @API("self_images_remove_all", version="1")
    async def remove_all_images(self) -> dict:
        return await self._action(self.service.store.batch_update, "", "remove_all")

    @API("self_images_batch_positive", version="1")
    async def batch_positive(self, image_ids: str) -> dict:
        return await self.batch_images(image_ids, "self")

    @API("self_images_batch_negative", version="1")
    async def batch_negative(self, image_ids: str) -> dict:
        return await self.batch_images(image_ids, "other")

    @API("self_images_batch_remove", version="1")
    async def batch_remove(self, image_ids: str) -> dict:
        return await self.batch_images(image_ids, "remove")

    @API("self_training_submit", version="1")
    async def submit(self, request_key: str, mode: str = "incremental", device: str = "cpu", input_size: int = 0) -> dict:
        from .recog.environment import status

        environment = await offload(status, self.service)
        if environment.get("detecting"):
            return {"success": False, "error": "正在检测运行环境，请稍后再开始训练"}
        if not environment.get("installed") or device == "cuda" and not environment.get("cuda"):
            return {"success": False, "error": "所选设备不可用，请先检查运行环境或安装匹配的PyTorch"}
        result = await self._action(self.service.submit, request_key, mode, device, self.config.character_name, input_size or self.config.input_size)
        if result.get("id") and await offload(self.service.store.setting, "form_request_key") == request_key:
            await offload(self.service.store.set_setting, "form_request_key", uuid.uuid4().hex)
        if result.get("state") in {"failed", "cancelled", "interrupted"}:
            return {
                **result,
                "success": False,
                "error": f"请求已对应任务 {result['id']}（{result['state']}），未启动新训练。"
                f"请检查设备和环境后再次点击“开始训练”。原失败原因：{result.get('error') or '无'}",
            }
        return result

    @API("self_training_pause", version="1")
    async def pause_training(self, job_id: str) -> dict:
        return await self._action(self.service.pause, job_id)

    @API("self_training_resume", version="1")
    async def resume_training(self, job_id: str) -> dict:
        return await self._action(self.service.resume, job_id)

    @API("self_select_report", version="1")
    async def select_report(self, job_id: str) -> dict:
        await offload(self.service.store.job, job_id)
        return {"navigate_page": "training", "navigate_params": {"job_id": job_id}}

    @API("self_select_errors", version="1")
    async def select_errors(self, job_id: str) -> dict:
        if (await offload(self.service.store.job, job_id))["state"] != "complete":
            return {"success": False, "error": "任务未完成，错判报告尚未生成"}
        return {"navigate_page": "images", "navigate_params": {"job_id": job_id, "view": "errors"}}

    @API("self_clear_errors", version="1")
    async def clear_errors(self) -> dict:
        return {"navigate_page": "images", "navigate_params": {"view": "all"}}

    @API("self_environment", version="1")
    async def environment(self) -> dict:
        from .recog.environment import status

        return await offload(status, self.service)

    @API("self_environment_manage", version="1")
    async def manage_environment(self, kind: str) -> dict:
        from .recog.environment import submit

        return await self._action(submit, self.service, kind)

    @API("self_environment_mode", version="1")
    async def environment_mode(self, mode: str, python_path: str = "") -> dict:
        return await self._action(self.change_environment, mode, python_path)

    def change_environment(self, mode: str, python_path: str = "") -> dict:
        import sys

        if mode not in {"host", "custom"}:
            raise ValueError("无效环境")
        python = sys.executable
        if mode == "custom":
            candidate = Path(python_path.strip()).expanduser()
            if not candidate.is_absolute():
                raise ValueError("请填写已有 Python 解释器或环境目录的绝对路径")
            if candidate.is_dir():
                names = ("Scripts/python.exe", "python.exe") if sys.platform == "win32" else ("bin/python", "bin/python3")
                candidate = next((candidate / name for name in names if (candidate / name).is_file()), candidate / names[0])
            if not candidate.is_file():
                raise ValueError("Python 解释器不存在；插件不会创建环境")
            python = str(candidate.resolve())
        with self.service.train_lock:
            self.service.check_environment_idle()
            if any(p.poll() is None for p in self.service.processes.values()):
                raise ValueError("训练中不能切换环境")
            self.stop_inference_sync()
            self.service.store.set_setting("worker_override", python)
            self.service.store.set_setting("worker_mode", mode)
            self.service.configure_python()
            self.service.environment_cache = None
        return {"success": True}

    def stop_inference_sync(self):
        self.service.stop_inference()

    @API("self_stop_inference", version="1")
    async def stop_inference(self) -> dict:
        await offload(self.stop_inference_sync)
        return {"success": True}

    @API("self_start_inference", version="1")
    async def start_inference(self) -> dict:
        cache = self.service.environment_cache or {}
        if cache.get("missing"):
            await offload(self.service.stop_inference)
            names = {"torch": "PyTorch", "torchvision": "torchvision"}
            return {"success": False, "error": "；".join(f"未安装 {names.get(name, name)}" for name in cache["missing"])}
        result = await offload(self.service.request, {"operation": "load"}, 8, resume=True)
        if result.get("status") != "ready":
            return {"success": False, "error": result.get("error", "尚未启用模型"), "result": result}
        return {"success": True, "result": result}

    @API("self_training_form", version="1")
    async def training_form(self) -> dict:
        key = await offload(self.form_key)
        if not key:
            key = uuid.uuid4().hex
            await offload(self.service.store.set_setting, "form_request_key", key)
        return {"request_key": key, "input_size": self.config.input_size, **await offload(self.service.store.training_readiness)}

    def form_key(self):
        with self.service.store.db() as db:
            db.execute("BEGIN IMMEDIATE")
            r = db.execute("SELECT value FROM settings WHERE key='form_request_key'").fetchone()
            key = r[0] if r else uuid.uuid4().hex
            existing = db.execute("SELECT state FROM jobs WHERE request_key=?", (key,)).fetchone()
            if existing and existing[0] in {"failed", "complete", "cancelled", "interrupted"}:
                key = uuid.uuid4().hex
            db.execute("INSERT OR REPLACE INTO settings VALUES ('form_request_key',?)", (key,))
        return key

    @API("self_training_new_request", version="1")
    async def new_request(self) -> dict:
        key = uuid.uuid4().hex
        await offload(self.service.store.set_setting, "form_request_key", key)
        return {"request_key": key}

    @API("self_training_jobs", version="1")
    async def jobs(self) -> dict:
        return await offload(self.service.jobs)

    @API("self_training_cancel", version="1")
    async def cancel(self, job_id: str) -> dict:
        return await self._action(self.service.cancel, job_id)

    @API("self_training_report", version="1")
    async def report(self, job_id: str = "") -> dict:
        if not job_id:
            return {"summary": "选择任务以查看报告", "metrics": [], "gallery": [], "pages": 1, "technical": ""}
        try:
            report = await offload(self.service.store.report, job_id)
        except ValueError as exc:
            return {"summary": str(exc), "metrics": [], "technical": ""}
        if not report:
            return {"summary": "报告尚未完成", "metrics": [], "gallery": [], "pages": 1, "technical": ""}
        summary = {
            k: v for k, v in report.items() if k not in {"history", "test_predictions", "validation_predictions"}
        }
        summary["评测性质"] = "历史留出评测" if report["historical_test"] else "首次封存测试"
        summary["比较提示"] = (
            "留出真值或成员改变，与旧报告不可直接比较"
            if not report["comparable_to_previous"]
            else "留出真值及成员未改变"
        )
        rows = []
        for name, m in (("验证", report["selection"]["validation"]), ("测试", report["test"])):
            rows.append(
                {
                    "split": name,
                    "precision": f"{m['precision_numerator']}/{m['precision_denominator']} = {m['precision']:.1%}",
                    "recall": m["recall"],
                    "f1": m["f1"],
                    "accuracy": m["accuracy"],
                    "count": m["count"],
                    "balanced_accuracy": m["balanced_accuracy"],
                    "other_recall": m["other_recall"],
                    "matrix": f"真非Bot → 非Bot/是Bot：{m['confusion_matrix'][0][0]}/{m['confusion_matrix'][0][1]}；真Bot → 非Bot/是Bot：{m['confusion_matrix'][1][0]}/{m['confusion_matrix'][1][1]}",
                }
            )
        readable = f"所选第{report['selection']['epoch']}轮 · 总用时{report['total_seconds']:.1f}秒 · 训练（含验证）{report['training_seconds_including_validation']:.1f}秒\n{summary['评测性质']} · {summary['比较提示']}\n核验：{'通过' if report['audit']['passed'] else '未通过'}"
        for row in rows:
            for field in ("recall", "f1", "accuracy", "other_recall", "balanced_accuracy"):
                row[field] = f"{row[field]:.1%}"
        return {"summary": readable, "metrics": rows, "technical": json.dumps(summary, ensure_ascii=False)}

    @API("self_models", version="1")
    async def versions(self) -> dict:
        rows = await offload(self.service.store.models)
        active = await offload(self.service.store.setting, "active")
        previous = await offload(self.service.store.setting, "previous")
        for r in rows:
            m = json.loads(r["metrics"])
            r.update(
                name=time.strftime("模型 %m-%d %H:%M", time.localtime(r["created"])),
                status="使用中" if r["id"] == active else "未启用",
                can_activate=r["id"] != active,
                input_size=m.get("input_size", 320),
            )
            for split in ("validation", "test"):
                metric = m[split]
                r[split + "_text"] = (
                    f"精确率 {metric['precision']:.1%}（{metric['precision_numerator']}/{metric['precision_denominator']}） · 召回 {metric['recall']:.1%} · F1 {metric['f1']:.1%} · 准确率 {metric['accuracy']:.1%} · {metric['count']}张"
                )
        return {
            "rows": rows,
            "active": next((r["name"] for r in rows if r["id"] == active), "未启用"),
            "can_rollback": bool(previous),
            "rollback_text": next(("回退目标：" + r["name"] for r in rows if r["id"] == previous), "无可回退版本"),
        }

    @API("self_model_activate", version="1")
    async def activate(self, version: str) -> dict:
        return await self._action(self.service.activate, version)

    @API("self_model_rollback", version="1")
    async def rollback(self) -> dict:
        return await self._action(self.service.rollback)

    @API("self_test_upload", version="1")
    async def test_upload(self, upload_id: str = "", result_id: str = "") -> dict:
        if result_id:
            raw = await offload(self.read_test_image, result_id)
            result = await offload(self.run_test, raw, result_id)
        else:
            claimed = await self.ctx.webui.claim_upload(upload_id)
            path = Path(claimed["path"])
            try:
                result = await offload(self.run_test, await offload(path.read_bytes), uuid.uuid4().hex)
            finally:
                await offload(path.unlink, missing_ok=True)
        return result

    def read_test_image(self, result_id: str) -> bytes:
        if len(result_id) != 32 or any(c not in "0123456789abcdef" for c in result_id):
            raise ValueError("测试图片不存在")
        history = json.loads(self.service.store.setting("test_history", "[]"))
        if not any(r.get("id") == result_id for r in history):
            raise ValueError("测试图片已过期")
        path = self.service.store.root / "test-images" / (result_id + ".image")
        if not path.is_file():
            raise ValueError("测试图片已过期，请重新上传")
        return path.read_bytes()

    def run_test(self, raw: bytes, result_id: str) -> dict:
        from .recog.images import thumbnail
        started = time.monotonic()
        result = self.service.infer(raw)
        directory = self.service.store.root / "test-images"
        directory.mkdir(exist_ok=True)
        path = directory / (result_id + ".image")
        path.write_bytes(raw)
        result.update(id=result_id, thumbnail=thumbnail(path), elapsed=f"耗时 {time.monotonic() - started:.2f} 秒")
        result["summary"] = (
            f"{result.get('character', '')} · 得分 {result.get('score', 0):.3f}"
            if result.get("status") == "ok" else f"识别未完成：{result.get('error') or result.get('status')}"
        )
        result["version_text"] = "模型版本：" + result.get("version", "无")[:8]
        self.save_test_history(result)
        self.service.store.set_setting("last_test", json.dumps(result, ensure_ascii=False))
        return result

    @API("self_test_result", version="1")
    async def test_result(self) -> dict:
        return {"rows": json.loads(await offload(self.service.store.setting, "test_history", "[]"))}

    def save_test_history(self, result):
        # Serialize history changes and retain the files needed for re-recognition.
        with self.service.store.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT value FROM settings WHERE key='test_history'").fetchone()
            history = json.loads(row[0]) if row else []
            history = [r for r in history if not result.get("id") or r.get("id") != result["id"]]
            combined = [result, *history]
            db.execute("INSERT OR REPLACE INTO settings VALUES ('test_history',?)", (json.dumps(combined[:24]),))
            for expired in combined[24:]:
                image_id = expired.get("id", "")
                if len(image_id) == 32 and all(c in "0123456789abcdef" for c in image_id):
                    (self.service.store.root / "test-images" / (image_id + ".image")).unlink(missing_ok=True)

    @staticmethod
    def _tool_result(result: dict) -> dict:
        return {**result, "name": "recognize_self_image", "content": json.dumps(result, ensure_ascii=False)}

    @Tool(
        "recognize_self_image",
        description="识别消息中的图片是否包含自身形象。",
        parameters=[
            ToolParameterInfo(
                name="message_id", param_type=ToolParamType.STRING, description="当前聊天中的真实消息ID", required=True
            ),
            ToolParameterInfo(
                name="image_index",
                param_type=ToolParamType.INTEGER,
                description="消息图片/表情零基序号",
                required=False,
            ),
        ],
    )
    async def recognize(self, message_id: str, image_index: int = 0, stream_id: str = "", **kwargs: Any) -> dict:
        del kwargs
        if not stream_id:
            return self._tool_result({"status": "missing_chat_context"})
        if type(image_index) is not int or image_index < 0:
            return self._tool_result({"status": "invalid_image_index"})
        message = await self.ctx.message.get_by_id(message_id, stream_id=stream_id, include_binary_data=True)
        if not message or message.get("session_id") != stream_id:
            return self._tool_result({"status": "message_not_found"})
        media = [part for part in message.get("raw_message", []) if part.get("type") in {"image", "emoji"}]
        if image_index >= len(media):
            return self._tool_result({"status": "image_not_found"})
        binary = media[image_index].get("binary_data_base64")
        if not binary:
            return self._tool_result({"status": "image_expired"})
        if len(binary) > 28 * 1024 * 1024:
            return self._tool_result({"status": "image_too_large"})
        try:
            raw = base64.b64decode(binary, validate=True)
        except ValueError:
            return self._tool_result({"status": "invalid_image"})
        return self._tool_result(await offload(self.service.infer, raw))


def create_plugin() -> SelfRecognition:
    return SelfRecognition()
