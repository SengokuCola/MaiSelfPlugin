"""Input size belongs to each snapshot and saved model."""
from pathlib import Path
import torch
import pytest
from PIL import Image
from recog.model import Dataset, fresh_model, infer, loaded_model, train
from test_catalog import corpus


def test_resize_precedes_augmentation(tmp_path):
    Image.new("RGB", (1600, 900), "orange").save(tmp_path / "image.png")
    dataset = Dataset(tmp_path, [{"path": "image.png", "label": "self"}], True, 448)
    sizes = []
    def jitter(image):
        sizes.append(image.size)
        return image
    dataset.jitter = jitter
    tensor, label = dataset[0]
    assert sizes == [(448, 448)]
    assert tensor.shape == (3, 448, 448) and label == 1
    legacy = Dataset(tmp_path, dataset.rows, True, 320, False)
    legacy.jitter = jitter
    legacy[0]
    assert sizes[-1] == (1600, 900)


@pytest.mark.parametrize("size", [224, 448])
def test_size_survives_training_reload_and_inference(tmp_path, size):
    store = corpus(tmp_path)
    with store.db() as db:
        snapshot = store.freeze(db, "size", "fresh", "cpu", size)
    assert snapshot["input_size"] == size
    assert snapshot["preprocessing"] == "resize_before_augment"
    directory = tmp_path / "jobs" / "size"
    directory.mkdir()
    report = train(tmp_path, snapshot, lambda p: None, lambda: False, max_epochs=1)
    assert report["audit"]["passed"] and report["input_size"] == size
    model = loaded_model(directory / "selected.pt")
    assert model.input_size == size
    seen = []
    hook = model.register_forward_pre_hook(lambda model, args: seen.append(tuple(args[0].shape)))
    raw = (tmp_path / snapshot["images"][0]["path"]).read_bytes()
    assert 0 <= infer(model, raw) <= 1
    hook.remove()
    assert seen == [(1, 3, size, size)]


def test_legacy_weights_use_320(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save(fresh_model().state_dict(), path)
    assert loaded_model(path).input_size == 320


def test_invalid_size_rejected(tmp_path):
    store = corpus(tmp_path)
    with store.db() as db, pytest.raises(ValueError, match="224"):
        store.freeze(db, "invalid", "fresh", "cpu", 999)


def test_training_batch_may_contain_only_one_class(tmp_path):
    store = corpus(tmp_path)
    with store.db() as db:
        snapshot = store.freeze(db, "oneclass", "fresh", "cpu", 224)
    snapshot["images"] = [r for r in snapshot["images"] if r["split"] != "train" or r["label"] == "self"]
    (tmp_path / "jobs" / "oneclass").mkdir()
    report = train(tmp_path, snapshot, lambda p: None, lambda: False, max_epochs=1)
    assert report["audit"]["passed"]
    assert report["counts"]["train"] == 14
