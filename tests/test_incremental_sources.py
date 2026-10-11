import json
import pytest
from test_catalog import corpus, upload
from recog.store import atomic_json


def publish(store, snapshot):
    job = snapshot["job_id"]
    atomic_json(store.root / "jobs" / job / "snapshot.json", snapshot)
    with store.db() as db:
        db.execute("INSERT INTO models VALUES (?,?,?,?,?)", (job,job,"sha","{}",1))
    store.set_setting("active", job)


def test_lineage_new_only_and_relabel(tmp_path):
    store = corpus(tmp_path)
    with store.db() as db:
        first = store.freeze(db, "first", "fresh", "cpu")
    publish(store, first)
    assert store.training_readiness()["pending_count"] == 0
    with store.db() as db, pytest.raises(ValueError, match="待训练"):
        store.freeze(db, "empty", "incremental", "cpu")
    for seed in range(40,46):
        upload(store, seed, "self" if seed < 43 else "other")
    with store.db() as db:
        second = store.freeze(db, "second", "incremental", "cpu")
    old = {r["id"] for r in first["images"] if r["split"] == "train"}
    new = {r["id"] for r in second["images"] if r["split"] == "train"}
    assert len(new) == 6 and not old & new
    assert {r["id"] for r in second["trained_sources"]} == old | new
    publish(store, second)
    assert store.training_readiness()["pending_count"] == 0
    # A changed truth is eligible once again, even if pixels are unchanged.
    target = next(r for r in first["images"] if r["split"] == "train" and r["label"] == "self")
    store.label(target["id"], "other")
    with store.db() as db:
        third = store.freeze(db, "third", "incremental", "cpu")
    assert [r["id"] for r in third["images"] if r["split"] == "train"] == [target["id"]]
    # Failed or unactivated jobs do not change the active model's source record.
    assert store.training_readiness()["pending_count"] == 1
    rows = store.images()["rows"] + store.images(page=2)["rows"]
    assert next(r for r in rows if r["id"] == target["id"])["training_text"] == "待训练"
    publish(store, third)
    store.label(target["id"], "self")
    with store.db() as db:
        fourth = store.freeze(db, "fourth", "incremental", "cpu")
    assert [r["id"] for r in fourth["images"] if r["split"] == "train"] == [target["id"]]


def test_legacy_snapshot_sources_and_rollback(tmp_path):
    store = corpus(tmp_path)
    with store.db() as db:
        first = store.freeze(db,"legacy","fresh","cpu")
    del first["trained_sources"]
    publish(store,first)
    assert store.training_readiness()["pending_count"] == 0
    upload(store,99,"self")
    with store.db() as db:
        fresh = store.freeze(db,"fresh","fresh","cpu")
    assert len([r for r in fresh["images"] if r["split"] == "train"]) >= 28
    assert len(first["images"]) == 40
