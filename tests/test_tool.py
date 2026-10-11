import asyncio
import base64
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

spec = importlib.util.spec_from_file_location(
    "self_plugin_test",
    Path(__file__).parents[1] / "plugin.py",
    submodule_search_locations=[str(Path(__file__).parents[1])],
)
module = importlib.util.module_from_spec(spec)
import sys

sys.modules[spec.name] = module
spec.loader.exec_module(module)


def instance(message, model_status="ok"):
    plugin = module.create_plugin()
    plugin._ctx = SimpleNamespace(message=SimpleNamespace(get_by_id=AsyncMock(return_value=message)))
    seen = []
    plugin.service = SimpleNamespace(
        infer=lambda raw: seen.append(raw) or {"status": model_status, "judgment": "不确定"}
    )
    return plugin, seen


def test_unload_before_load_and_after_service_creation():
    plugin = module.create_plugin()
    asyncio.run(plugin.on_unload())
    closed = []
    plugin.service = SimpleNamespace(close=lambda: closed.append(True))
    asyncio.run(plugin.on_unload())
    assert closed == [True]


def test_report_navigation_does_not_write_shared_settings():
    plugin = module.create_plugin()
    calls = []
    plugin.service = SimpleNamespace(store=SimpleNamespace(
        job=lambda job_id: {"state": "complete"},
        images=lambda page, view, job_id: calls.append((page, view, job_id)) or {},
    ))
    assert asyncio.run(plugin.select_report("first")) == {
        "navigate_page": "training", "navigate_params": {"job_id": "first"}
    }
    assert asyncio.run(plugin.select_errors("second")) == {
        "navigate_page": "images", "navigate_params": {"job_id": "second", "view": "errors"}
    }
    assert asyncio.run(plugin.images(1, "errors", "second"))["errors_active"]
    assert not asyncio.run(plugin.images(1, "all"))["errors_active"]
    assert calls == [(1, "errors", "second"), (1, "all", "")]
    assert asyncio.run(plugin.clear_errors()) == {"navigate_page": "images", "navigate_params": {"view": "all"}}


def test_required_config_update_and_scope():
    from maibot_sdk import MaiBotPlugin

    plugin = module.create_plugin()
    assert type(plugin).on_config_update is not MaiBotPlugin.on_config_update
    plugin._plugin_config_instance = module.Config(character_name="新角色", worker_python="new-python.exe")
    calls = []
    plugin.service = SimpleNamespace(update_config=lambda *args: calls.append(args))
    asyncio.run(plugin.on_config_update("bot", {}, "1"))
    asyncio.run(plugin.on_config_update("model", {}, "1"))
    assert not calls
    asyncio.run(plugin.on_config_update("self", {}, "2"))
    assert calls == [("新角色", "new-python.exe")]


def test_default_config_has_required_version():
    from maibot_sdk.config import extract_plugin_config_version
    from src.plugin_runtime.runner.plugin_loader import PluginLoader
    from src.plugin_runtime.runner.runner_main import PluginRunner

    plugin = module.create_plugin()
    PluginLoader._validate_sdk_plugin_contract("maibot.mai-recog-self", plugin)
    defaults = plugin.get_default_config()
    assert extract_plugin_config_version(defaults) == "0.1.0"
    runner = object.__new__(PluginRunner)
    result = runner._normalize_plugin_config(plugin, {}, default_config=defaults, suppress_errors=False)
    assert result.normalized_config["plugin"]["config_version"] == "0.1.0"
    assert result.normalized_config["plugin"]["enabled"] is True
    plugin.set_plugin_config(result.normalized_config)
    assert plugin.config.character_name == "Bot自己"


def test_multimedia_index_and_readonly():
    plugin, seen = instance(
        dict(
            session_id="chat",
            raw_message=[
                dict(type="text", data="hi"),
                dict(type="image", binary_data_base64=base64.b64encode(b"first").decode()),
                dict(type="emoji", binary_data_base64=base64.b64encode(b"second").decode()),
            ],
        )
    )
    result = asyncio.run(plugin.recognize("message", 1, stream_id="chat"))
    assert result["judgment"] == "不确定" and seen == [b"second"]
    plugin.ctx.message.get_by_id.assert_awaited_once_with("message", stream_id="chat", include_binary_data=True)


@pytest.mark.parametrize(
    "message,index,stream,status",
    [
        (None, 0, "chat", "message_not_found"),
        (dict(session_id="other", raw_message=[]), 0, "chat", "message_not_found"),
        (dict(session_id="chat", raw_message=[]), 0, "chat", "image_not_found"),
        (dict(session_id="chat", raw_message=[dict(type="image")]), 0, "chat", "image_expired"),
        (None, 0, "", "missing_chat_context"),
        (None, -1, "chat", "invalid_image_index"),
    ],
)
def test_concrete_errors_and_stream_isolation(message, index, stream, status):
    plugin, seen = instance(message)
    assert asyncio.run(plugin.recognize("message", index, stream_id=stream))["status"] == status
    assert not seen


def test_failed_training_request_shows_manual_retry_hint(monkeypatch):
    plugin = module.create_plugin()
    plugin._plugin_config_instance = module.Config()
    calls = []

    def submit(*args):
        calls.append(args)
        return {"id": "old-job", "state": "failed", "error": "CUDA unavailable"}

    monkeypatch.setattr(
        module.__package__ + ".recog.environment.status", lambda service: {"installed": True, "cuda": False}
    )
    plugin.service = SimpleNamespace(submit=submit, store=SimpleNamespace(setting=lambda key: "other-key"))
    result = asyncio.run(plugin.submit("same-key", device="cpu"))
    assert result["success"] is False
    assert result["id"] == "old-job"
    assert "CUDA unavailable" in result["error"]
    assert "old-job" in result["error"]
    assert len(calls) == 1


def test_automatic_media_annotations_cache_and_model_switch():
    plugin = module.create_plugin()
    plugin._plugin_config_instance = module.Config(character_name="Bot")
    plugin._image_cache = {}
    version = ["first"]
    seen = []

    def infer(raw, timeout):
        seen.append(raw)
        return {"status": "ok", "score": 0.8 if raw == b"positive" else 0.3, "version": version[0]}

    plugin.service = SimpleNamespace(store=SimpleNamespace(setting=lambda key: version[0]), infer=infer, inference_enabled=True)

    def media(raw, kind="image"):
        return {"type": kind, "binary_data_base64": base64.b64encode(raw).decode()}

    message = {
        "raw_message": [
            media(b"positive"),
            media(b"negative", "emoji"),
            {"type": "forward", "data": [{"content": [media(b"positive")]}]},
        ]
    }
    result = asyncio.run(plugin.annotate_received_images(message))["modified_kwargs"]["message"]["raw_message"]
    assert len(result) == 4 and result[1]["type"] == "text"
    assert len(result[3]["data"][0]["content"]) == 2
    assert seen == [b"positive", b"negative"]
    assert len(message["raw_message"]) == 3
    asyncio.run(plugin.annotate_received_images(message))
    assert len(seen) == 2
    plugin.service.inference_enabled = False
    assert asyncio.run(plugin.annotate_received_images(message)) == {"action": "continue"}
    assert len(seen) == 2
    plugin.service.inference_enabled = True
    version[0] = "second"
    asyncio.run(plugin.annotate_received_images(message))
    assert len(seen) == 4


def test_automatic_media_disabled_or_untrained():
    plugin = module.create_plugin()
    plugin._plugin_config_instance = module.Config()
    plugin.service = SimpleNamespace(store=SimpleNamespace(setting=lambda key: ""), inference_enabled=True)
    assert asyncio.run(plugin.annotate_received_images({"raw_message": []})) == {"action": "continue"}
