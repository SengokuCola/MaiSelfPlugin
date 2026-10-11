import io
import time

import numpy as np
from PIL import Image
import pytest

from recog.images import decode, whole_image, judgment, thumbnail
from recog.store import Store, digest
from recog.service import Service


def upload(store, seed, label=None, variant=False):
    image = Image.fromarray(np.random.default_rng(seed).integers(0, 256, (96, 120, 3), dtype=np.uint8))
    if variant:
        image = image.resize((240, 192))
    path = store.root / "uploads" / f"{seed}-{time.time_ns()}.png"
    image.save(path)
    result = store.ingest(path)
    if label:
        store.label(result["id"], label)
    return result["id"]


def corpus(root):
    store = Store(root)
    for i in range(40):
        upload(store, i, "self" if i < 20 else "other")
    return store


def test_thumbnail_keeps_detail_within_page_budget(tmp_path):
    import base64

    path = tmp_path / "large.jpg"
    Image.new("RGB", (1280, 720), "red").save(path)
    raw = base64.b64decode(thumbnail(path).split(",", 1)[1])
    assert len(raw) <= 12_000
    with Image.open(io.BytesIO(raw)) as image:
        assert image.size == (512, 288)


def test_upload_label_and_cached_annotation_refresh(tmp_path, monkeypatch):
    import recog.images as images

    store = Store(tmp_path)
    path = store.root / "uploads" / "negative.png"
    image = Image.fromarray(np.random.default_rng(321).integers(0, 256, (96, 120, 3), dtype=np.uint8))
    image.save(path)
    result = store.ingest(path, "other")
    rows = store.images()["rows"]
    assert rows[0]["label"] == "other"
    cached = rows[0]["thumbnail"]
    monkeypatch.setattr(images, "decode", lambda _: (_ for _ in ()).throw(AssertionError("Cache miss")))
    store.label(result["id"], "self")
    assert store.images()["rows"][0]["thumbnail"] == cached
    assert store.images()["rows"][0]["label"] == "self"
    store.set_removed(result["id"], True)
    assert store.images()["rows"] == []
    assert store.images(view="removed")["rows"][0]["id"] == result["id"]


def test_reupload_reuses_existing_image_and_keeps_label(tmp_path):
    store = Store(tmp_path)
    ident = upload(store, 95, "self")
    with store.db() as db:
        original = dict(db.execute("SELECT * FROM images WHERE id=?", (ident,)).fetchone())
    for recompressed in (False, True):
        incoming = store.root / "uploads" / "renamed.png"
        if recompressed:
            with Image.open(store.root / original["path"]) as image:
                image.save(incoming, compress_level=0)
        else:
            incoming.write_bytes((store.root / original["path"]).read_bytes())
        result = store.ingest(incoming)
        assert result["already_exists"] and result["id"] == ident
        assert not incoming.exists()
    with store.db() as db:
        assert db.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 1
        assert db.execute("SELECT label FROM images WHERE id=?", (ident,)).fetchone()[0] == "self"
    assert len(list((store.root / "images").iterdir())) == 1


@pytest.mark.parametrize("format", ["JPEG", "PNG", "WEBP"])
def test_supported_static_formats(format):
    buffer = io.BytesIO()
    Image.new("RGB", (16, 24), "red").save(buffer, format=format)
    assert decode(buffer.getvalue()).size == (16, 24)


def test_decode_and_letterbox():
    image = Image.new("RGB", (20, 100), "red")
    out = whole_image(image)
    assert out.size == (320, 320) and out.getpixel((0, 0)) == (124, 116, 104)
    buffer = io.BytesIO()
    image.save(buffer, format="GIF")
    with pytest.raises(ValueError):
        decode(buffer.getvalue())


@pytest.mark.parametrize(
    "score,expected",
    [
        (0, "不是自己"),
        (0.4, "不是自己"),
        (0.4001, "不确定"),
        (0.5, "不确定"),
        (0.5999, "不确定"),
        (0.6, "是自己"),
        (1, "是自己"),
    ],
)
def test_three_state_thresholds(score, expected):
    assert judgment(score) == expected


def test_solid_duplicate_conflict_and_restore(tmp_path):
    store = Store(tmp_path)
    solid = tmp_path / "uploads" / "solid.png"
    Image.new("RGB", (64, 64), "white").save(solid)
    assert store.ingest(solid)["reason"] == "solid"
    a = upload(store, 80, "self")
    b = upload(store, 80, "other", variant=True)
    assert {r["reason"] for r in store.images()["rows"] if r["id"] in {a, b}} == {"label_conflict"}
    store.label(b, "self")
    assert {r["reason"] for r in store.images()["rows"] if r["id"] in {a, b}} == {"", "duplicate"}
    store.label(a, "self", True)
    assert next(r for r in store.images()["rows"] if r["id"] == b)["reason"] == ""


def test_snapshot_preserves_split_and_frozen_truth(tmp_path):
    store = corpus(tmp_path)
    with store.db() as db:
        db.execute("BEGIN IMMEDIATE")
        first = store.freeze(db, "first", "fresh", "cpu")
    splits = {r["id"]: r["split"] for r in first["images"]}
    assert {s: sum(r["split"] == s for r in first["images"]) for s in ["train", "val", "test"]} == {
        "train": 28,
        "val": 6,
        "test": 6,
    }
    for i in range(40, 46):
        upload(store, i, "self" if i < 43 else "other")
    with store.db() as db:
        db.execute("BEGIN IMMEDIATE")
        second = store.freeze(db, "second", "fresh", "cpu")
    assert all(splits[r["id"]] == r["split"] for r in second["images"] if r["id"] in splits)
    target = next(r for r in second["images"] if r["split"] == "test")
    frozen = digest(second)
    store.set_setting("last_held_truth", second["held_truth"])
    store.label(target["id"], "other" if target["label"] == "self" else "self")
    with store.db() as db:
        third = store.freeze(db, "third", "fresh", "cpu")
    assert not third["comparable_to_previous"]
    assert digest(second) == frozen
    assert store.images(page=999)["pagination"]["page"] == 2


def test_bridging_train_and_holdout_is_excluded(tmp_path, monkeypatch):
    store = Store(tmp_path)
    a = upload(store, 1, "self")
    b = upload(store, 2, "other")
    with store.db() as db:
        db.execute("UPDATE groups SET split='train' WHERE id=(SELECT group_id FROM images WHERE id=?)", (a,))
        db.execute("UPDATE groups SET split='test' WHERE id=(SELECT group_id FROM images WHERE id=?)", (b,))
    monkeypatch.setattr("recog.store.similar", lambda a, b: True)
    c = upload(store, 3, "self")
    assert next(r for r in store.images()["rows"] if r["id"] == c)["reason"] == "split_conflict"
    assert next(r for r in store.images()["rows"] if r["id"] == a)["split"] == "train"


def test_submission_idempotent_and_missing_runtime(tmp_path):
    corpus(tmp_path)
    service = Service(tmp_path, python=str(tmp_path / "not-installed"))
    try:
        result = service.submit("same")
        assert result["state"] == "failed" and "环境" in result["error"]
        assert service.submit("same")["id"] == result["id"]
        assert service.infer(b"anything")["status"] == "no_model"
    finally:
        service.close()


def test_batch_across_pages_atomic_and_remove_all(tmp_path):
    store = corpus(tmp_path)
    first = store.images(1)["rows"][0]["id"]
    second = store.images(2)["rows"][0]["id"]
    result = store.batch_update(f"{first},{second}", "other")
    assert result["updated"] == 2
    with store.db() as db:
        assert all(db.execute("SELECT label FROM images WHERE id=?", (i,)).fetchone()[0] == "other" for i in (first, second))
    with pytest.raises(ValueError, match="不存在"):
        store.batch_update(f"{first},{'f' * 32}", "self")
    with store.db() as db:
        assert db.execute("SELECT label FROM images WHERE id=?", (first,)).fetchone()[0] == "other"
    store.batch_update(f"{first},{second}", "remove")
    assert store.images()["total"] == 38
    assert store.batch_update("", "remove_all")["updated"] == 38
    assert store.images()["total"] == 0
    assert store.images(view="removed")["total"] == 40
    store.set_removed(first, False)
    assert store.images()["total"] == 1
