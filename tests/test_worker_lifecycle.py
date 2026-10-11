import sys
import time

from recog.service import Service
from test_catalog import corpus


def test_cancel_and_unload_are_terminal(tmp_path):
    corpus(tmp_path)
    service = Service(tmp_path, sys.executable)
    first = service.submit("cancel")
    service.cancel(first["id"])
    started = time.monotonic()
    while service.store.job(first["id"])["state"] == "cancelling" and time.monotonic() - started < 30:
        time.sleep(0.25)
    assert service.store.job(first["id"])["state"] == "cancelled"
    service.close()
    reopened = Service(tmp_path, sys.executable)
    try:
        # Restart never repeats the previous request or starts any worker automatically.
        assert not reopened.processes
        assert reopened.submit("cancel")["id"] == first["id"]
        assert reopened.submit("cancel")["state"] == "cancelled"
    finally:
        reopened.close()


def test_restart_marks_pending_task_interrupted(tmp_path):
    store = corpus(tmp_path)
    directory = tmp_path / "jobs" / "pending"
    directory.mkdir(parents=True)
    with store.db() as db:
        db.execute("INSERT INTO jobs VALUES ('pending','old','queued',NULL,'',?, '')", (time.time(),))
    service = Service(tmp_path, sys.executable)
    try:
        assert service.store.job("pending")["state"] == "interrupted"
        assert (directory / "cancel").exists()
        assert not service.processes
    finally:
        service.close()
