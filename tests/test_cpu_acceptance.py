"""Real default 15+10 epoch CPU workers and atomic activation/rollback."""

from pathlib import Path
import json
import sys
import time

from recog.service import Service
from recog.store import atomic_json
from test_catalog import corpus, upload


def wait(service, job, timeout=240):
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        row = service.store.job(job)
        if row["state"] in {"complete", "failed", "cancelled"}:
            assert row["state"] == "complete", row
            return service.store.report(job)
        time.sleep(0.5)
    raise AssertionError("CPU training timeout")


def test_cpu_acceptance(tmp_path):
    store = corpus(tmp_path)
    service = Service(tmp_path, sys.executable)
    try:
        first = service.submit("initial", "fresh", input_size=224)
        assert service.submit("initial")["id"] == first["id"]
        service.pause(first["id"])
        deadline = time.monotonic() + 60
        while service.store.job(first["id"])["state"] == "pausing" and time.monotonic() < deadline:
            time.sleep(.25)
        assert service.store.job(first["id"])["state"] == "paused"
        assert (tmp_path / "jobs" / first["id"] / "resume.pt").exists()
        while any(p.poll() is None for p in service.processes.values()) and time.monotonic() < deadline:
            time.sleep(.25)
        assert service.resume(first["id"])["id"] == first["id"]
        report = wait(service, first["id"])
        assert len(report["history"]) == 15 and report["audit"]["passed"]
        assert report["counts"] == dict(train=28, val=6, test=6)
        assert report["input_size"] == 224
        assert report["peak_rss_bytes"] > 0
        assert service.activate(first["id"])["status"] == "ready"
        path = store.root / store.images()["rows"][0]["path"]
        result = service.infer(path.read_bytes())
        assert result["version"] == first["id"] and 0 <= result["score"] <= 1
        assert result["score_is_calibrated_probability"] is False
        assert result["input_size"] == 224
        for i in range(40, 46):
            upload(store, i, "self" if i < 43 else "other")
        second = service.submit("continuation", input_size=448)
        snapshot = json.loads((tmp_path / "jobs" / second["id"] / "snapshot.json").read_text("utf-8"))
        assert snapshot["parent"] == first["id"] and snapshot["mode"] == "incremental"
        assert {r["id"] for r in snapshot["images"] if r["split"] == "train"}.isdisjoint({
            r["id"]
            for r in json.loads((tmp_path / "jobs" / first["id"] / "snapshot.json").read_text("utf-8"))["images"]
            if r["split"] == "train"
        })
        assert len(snapshot["trained_sources"]) == 34
        report2 = wait(service, second["id"])
        assert len(report2["history"]) == 10 and report2["historical_test"]
        assert report2["counts"]["train"] == 6
        assert service.activate(second["id"])["status"] == "ready"
        prediction = service.infer(path.read_bytes())
        assert prediction["version"] == second["id"] and prediction["input_size"] == 448
        assert service.rollback()["version"] == first["id"]
        assert service.infer(path.read_bytes())["input_size"] == 224
        assert service.infer(path.read_bytes())["version"] == first["id"]
        # Tampering rejects activation and leaves old serving model intact.
        weights = tmp_path / "models" / second["id"] / "weights.pt"
        with weights.open("ab") as stream:
            stream.write(b"tamper")
        try:
            service.activate(second["id"])
        except ValueError:
            pass
        else:
            raise AssertionError("Corrupt model activated")
        assert store.setting("active") == first["id"]
        results = {
            "fixture": "40 synthetic images; validates mechanics, not identity accuracy",
            "fresh": report,
            "incremental": report2,
        }
        atomic_json(Path(__file__).parents[1] / "test-results" / "cpu-acceptance.json", results)
    finally:
        service.close()
