"""Dedicated train / persistent CPU inference processes, stdio JSON protocol."""

from pathlib import Path
import argparse
import base64
import hashlib
import json
import os
import sys
import time
import traceback

from .store import Store, atomic_json, digest
from .images import judgment


def train_job(root: Path, job_id: str) -> None:
    store = Store(root)
    store.job(job_id)
    directory = root / "jobs" / job_id
    snapshot = json.loads((directory / "snapshot.json").read_text("utf-8"))
    if digest(snapshot) != store.job(job_id)["snapshot_sha"]:
        raise ValueError("快照SHA不符")
    from .model import train, TrainingPaused

    def progress(value):
        atomic_json(directory / "progress.json", value)

    def cancelled():
        lease = directory / "lease"
        return (directory / "cancel").exists() or not lease.exists() or time.time() - lease.stat().st_mtime > 120

    try:
        with store.db() as db:
            db.execute(
                "UPDATE jobs SET state=CASE WHEN state='queued' THEN 'running' ELSE state END,pid=? WHERE id=?",
                (os.getpid(), job_id),
            )
        if cancelled():
            raise InterruptedError("训练已取消")
        progress({"phase": "preparing"})
        report = train(root, snapshot, progress, cancelled, paused=lambda: (directory / "pause").exists())
        report["total_job_seconds"] = time.time() - snapshot["created"]
        if cancelled():
            raise InterruptedError("训练已取消")
        destination = root / "models" / job_id
        destination.mkdir(exist_ok=True)
        (directory / "selected.pt").replace(destination / "weights.pt")
        sha = hashlib.sha256((destination / "weights.pt").read_bytes()).hexdigest()
        atomic_json(directory / "report.json", report)
        atomic_json(destination / "audit.json", {**report["audit"], "sha256": sha, "snapshot_sha": digest(snapshot)})
        with store.db() as db:
            db.execute(
                "INSERT INTO models VALUES (?,?,?,?,?)",
                (
                    job_id,
                    job_id,
                    sha,
                    json.dumps({"validation": report["selection"]["validation"], "test": report["test"]}),
                    time.time(),
                ),
            )
            db.execute("UPDATE jobs SET state='complete' WHERE id=?", (job_id,))
            db.execute("INSERT OR REPLACE INTO settings VALUES ('last_held_truth',?)", (snapshot["held_truth"],))
        progress(
            {"phase": "complete", "selected_epoch": report["selection"]["epoch"], "seconds": report["total_seconds"]}
        )
        (directory / "resume.pt").unlink(missing_ok=True)
        (directory / "resume-audit.json").unlink(missing_ok=True)
    except TrainingPaused:
        with store.db() as db:
            db.execute("UPDATE jobs SET state='paused',pid=NULL WHERE id=?", (job_id,))
        value = json.loads((directory / "progress.json").read_text("utf-8"))
        progress({**value, "phase": "paused"})
    except BaseException as exc:
        state = "cancelled" if isinstance(exc, InterruptedError) else "failed"
        with store.db() as db:
            db.execute("UPDATE jobs SET state=?,error=? WHERE id=?", (state, str(exc), job_id))
        progress({"phase": state, "error": str(exc)})
        traceback.print_exc(file=sys.stderr)
        raise


def serve(root: Path) -> None:
    import torch
    from .model import loaded_model, infer

    torch.set_num_threads(min(4, torch.get_num_threads()))
    store = Store(root)
    current = ""
    model = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            version = request.get("version") or store.setting("active")
            if not version:
                result = {"status": "no_model"}
            else:
                if version != current:
                    record = next((r for r in store.models() if r["id"] == version), None)
                    if record is None:
                        raise ValueError("模型版本不存在")
                    path = root / "models" / version / "weights.pt"
                    audit = json.loads((path.parent / "audit.json").read_text("utf-8"))
                    if not audit["passed"] or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha"]:
                        raise ValueError("模型核验失败")
                    candidate = loaded_model(path)  # assign only after successful load
                    model = candidate
                    current = version
                if request.get("operation") == "load":
                    result = {"status": "ready", "version": current}
                else:
                    score = infer(model, base64.b64decode(request["image"], validate=True))
                    result = {
                        "status": "ok",
                        "character": store.setting("character", "Bot自己"),
                        "version": current,
                        "score": score,
                        "judgment": judgment(score),
                        "score_is_calibrated_probability": False,
                    }
        except Exception as exc:
            result = {"status": "model_error", "error": str(exc)}
        print(json.dumps(result, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["train", "serve"])
    parser.add_argument("root", type=Path)
    parser.add_argument("job", nargs="?")
    args = parser.parse_args()
    if args.operation == "train":
        train_job(args.root, args.job)
    else:
        serve(args.root)


if __name__ == "__main__":
    main()
