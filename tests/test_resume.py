"""The pause boundary preserves optimizer, augmentation RNG and data ordering."""

import torch
import pytest
from recog.model import TrainingPaused, train
from test_catalog import corpus


@pytest.mark.parametrize("boundary", ["training", "validation"])
def test_resume_matches_uninterrupted_weights(tmp_path, boundary):
    store = corpus(tmp_path)
    with store.db() as db:
        snapshot = store.freeze(db, "exact", "fresh", "cpu")
    directory = tmp_path / "jobs" / "exact"
    directory.mkdir()
    full = train(tmp_path, snapshot, lambda p: None, lambda: False, max_epochs=2)
    expected = torch.load(directory / "selected.pt", weights_only=True)["state_dict"]
    stop = False

    def progress(p):
        nonlocal stop
        if p.get("phase") == boundary and (boundary == "validation" or p.get("step") == 1):
            stop = True

    with pytest.raises(TrainingPaused):
        train(tmp_path, snapshot, progress, lambda: False, max_epochs=2, paused=lambda: stop)
    resumed = train(tmp_path, snapshot, lambda p: None, lambda: False, max_epochs=2)
    actual = torch.load(directory / "selected.pt", weights_only=True)["state_dict"]
    assert all(torch.equal(expected[k], actual[k]) for k in expected)
    assert full["history"] == resumed["history"]
    assert resumed["audit"]["passed"]
