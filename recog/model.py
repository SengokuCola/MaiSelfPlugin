"""Small whole-image classifier. Heavy imports are worker-only."""

from pathlib import Path
import hashlib
import random
import time
import psutil

import numpy as np
import torch
from torch import nn
from torchvision import models, transforms

from .images import decode, whole_image


class TrainingPaused(Exception):
    """A verified checkpoint has been saved; release the worker process."""


BASE_SHA = "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f"
BASE_FILE = Path(__file__).resolve().parents[1] / "assets" / "mobilenet_v3_small-047dcff4.pth"
NORMALIZE = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def metrics(labels: list, scores: list) -> dict:
    predicted = [int(s >= 0.5) for s in scores]
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predicted))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predicted))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predicted))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": precision,
        "precision_numerator": tp,
        "precision_denominator": tp + fp,
        "recall": recall,
        "other_recall": tn / (tn + fp) if tn + fp else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "accuracy": (tp + tn) / len(labels) if labels else 0.0,
        "balanced_accuracy": (recall + (tn / (tn + fp) if tn + fp else 0.0)) / 2,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "count": len(labels),
    }


def seed_all() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def fresh_model() -> nn.Module:
    if not BASE_FILE.exists() or hashlib.sha256(BASE_FILE.read_bytes()).hexdigest() != BASE_SHA:
        raise ValueError("ImageNet底座缺失或SHA256校验失败")
    model = models.mobilenet_v3_small(weights=None)
    model.load_state_dict(torch.load(BASE_FILE, map_location="cpu", weights_only=True))
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model.input_size = 320
    return model


def loaded_model(path: Path) -> nn.Module:
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in artifact:
        state = artifact["state_dict"]
        input_size = artifact["input_size"]
        if type(input_size) is not int or input_size not in {224, 320, 448}:
            raise ValueError("模型图片尺寸无效")
    else:
        state, input_size = artifact, 320
    model.input_size = input_size
    if not state or any(not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("模型包含非有限权重")
    model.load_state_dict(state, strict=True)
    return model.eval()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root: Path, rows: list, augment: bool = False, input_size: int = 320, resize_first: bool = True):
        self.root = root
        self.rows = rows
        self.augment = augment
        self.input_size = input_size
        self.resize_first = resize_first
        self.jitter = transforms.ColorJitter(0.1, 0.1, 0.08, 0.02)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = decode((self.root / row["path"]).read_bytes())
        # Legacy paused tasks retain their original augmentation order.
        if self.resize_first:
            image = whole_image(image, self.input_size)
        if self.augment:
            image = self.jitter(image)
        if not self.resize_first:
            image = whole_image(image, self.input_size)
        return NORMALIZE(transforms.ToTensor()(image)), int(row["label"] == "self")


def evaluate(model: nn.Module, root: Path, rows: list, device: str, batch: int) -> tuple:
    model.eval()
    labels = []
    scores = []
    loader = torch.utils.data.DataLoader(Dataset(root, rows, input_size=model.input_size), batch_size=batch, shuffle=False, num_workers=0)
    with torch.inference_mode():
        for inputs, truth in loader:
            out = model(inputs.to(device)).float()
            if not torch.isfinite(out).all():
                raise ValueError("推理输出非有限")
            scores.extend(out.softmax(-1)[:, 1].cpu().tolist())
            labels.extend(truth.tolist())
    return metrics(labels, scores), [
        {"id": r["id"], "label": r["label"], "score": s, "predicted": "self" if s >= 0.5 else "other"}
        for r, s in zip(rows, scores)
    ]


def train(root: Path, snapshot: dict, progress, cancelled, max_epochs: int | None = None, paused=None) -> dict:
    started = time.monotonic()
    device = snapshot["device"]
    input_size = snapshot.get("input_size", 320)
    if type(input_size) is not int or input_size not in {224, 320, 448}:
        raise ValueError("图片尺寸无效")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA不可用，请选择CPU")
    seed_all()
    batch = 32 if device == "cuda" else 8
    parent = snapshot["parent"]
    if (
        parent
        and hashlib.sha256((root / "models" / parent / "weights.pt").read_bytes()).hexdigest() != snapshot["parent_sha"]
    ):
        raise ValueError("启用父模型SHA变化，拒绝增量训练")
    model = loaded_model(root / "models" / parent / "weights.pt") if parent else fresh_model()
    model.input_size = input_size
    model = model.to(device)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    rows = snapshot["images"]
    if any(hashlib.sha256((root / r["path"]).read_bytes()).hexdigest() != r["sha"] for r in rows):
        raise ValueError("冻结原图哈希变化")
    partitions = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    targets = [sum(r["label"] == label for r in partitions["train"]) for label in ("other", "self")]
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([sum(targets) / (2 * n) if n else 0.0 for n in targets], device=device))
    incremental = snapshot["mode"] == "incremental"
    epochs = max_epochs or (10 if incremental else 15)
    job_dir = root / "jobs" / snapshot["job_id"]
    generator = torch.Generator().manual_seed(42)
    history, coverage = [], []
    best, optimizer = None, None
    peak_rss = 0
    from .store import atomic_json, digest

    checkpoint = job_dir / "resume.pt"
    resume = None
    elapsed = 0.0
    if checkpoint.exists():
        metadata = __import__("json").loads((job_dir / "resume-audit.json").read_text("utf-8"))
        if (
            metadata["snapshot_sha"] != digest(snapshot)
            or metadata["sha"] != hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        ):
            raise ValueError("checkpoint integrity mismatch")
        # Only this worker writes checkpoints; user uploads never become pickle inputs.
        resume = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(resume["model"])
        initial, history, coverage, best = resume["initial"], resume["history"], resume["coverage"], resume["best"]
        generator.set_state(resume["generator"])
        elapsed, peak_rss = resume["seconds"], resume["peak_rss"]
    first_epoch = resume["epoch"] + int(resume.get("finished", False)) if resume else 1
    for epoch in range(first_epoch, epochs + 1):
        if cancelled():
            raise InterruptedError("training cancelled")
        tune = incremental or epoch > 3
        if optimizer is None or (epoch == 4 and not incremental):
            visual = []
            if tune:
                for module in list(model.features)[-4:]:
                    for parameter in module.parameters():
                        parameter.requires_grad = True
                        visual.append(parameter)
            groups = [
                {"params": model.classifier.parameters(), "lr": 1e-4 if incremental else (3e-4 if tune else 1e-3)}
            ]
            if visual:
                groups.append({"params": visual, "lr": 3e-5 if incremental else 1e-4})
            optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
        if resume:
            if not (resume.get("finished") and epoch == 4 and not incremental):
                optimizer.load_state_dict(resume["optimizer"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device)
            random.setstate(resume["random"])
            np.random.set_state(resume["numpy"])
            torch.set_rng_state(resume["torch"])
            if device == "cuda":
                torch.cuda.set_rng_state_all(resume["cuda"])
        model.train()
        for module in list(model.features)[:-4] if tune else list(model.features):
            module.eval()
        mid_epoch = resume and not resume.get("finished", False)
        order = resume["order"] if mid_epoch else torch.randperm(len(partitions["train"]), generator=generator).tolist()
        offset = resume["step"] if mid_epoch else 0
        total_loss, seen = (resume["loss"], resume["seen"]) if mid_epoch else (0.0, 0)
        steps = (len(order) + batch - 1) // batch

        def save_pause(next_step, finished=False):
            state = {
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer": optimizer.state_dict(),
                "initial": initial,
                "history": history,
                "coverage": coverage,
                "best": best,
                "generator": generator.get_state(),
                "epoch": epoch,
                "finished": finished,
                "step": next_step,
                "order": order,
                "loss": total_loss,
                "seen": seen,
                "random": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if device == "cuda" else [],
                "seconds": elapsed + time.monotonic() - started,
                "peak_rss": peak_rss,
            }
            temporary = checkpoint.with_suffix(".tmp")
            torch.save(state, temporary)
            temporary.replace(checkpoint)
            atomic_json(
                job_dir / "resume-audit.json",
                {"snapshot_sha": digest(snapshot), "sha": hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
            )
            raise TrainingPaused()

        # Explicit batch indices preserve ordering and augmentation RNG across restarts.
        dataset = Dataset(root, partitions["train"], True, input_size, snapshot.get("preprocessing") == "resize_before_augment")
        resume = None
        for step in range(offset, steps):
            if cancelled():
                raise InterruptedError("training cancelled")
            if paused and paused():
                save_pause(step)
            examples = [dataset[i] for i in order[step * batch : (step + 1) * batch]]
            inputs = torch.stack([example[0] for example in examples])
            truth = torch.tensor([example[1] for example in examples])
            optimizer.zero_grad(set_to_none=True)
            output = model(inputs.to(device))
            loss = loss_fn(output, truth.to(device))
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            seen += len(truth)
            total_loss += loss.item() * len(truth)
            memory = psutil.Process().memory_info()._asdict()
            peak_rss = max(peak_rss, memory["rss"], memory.get("peak_wset", 0))
            progress(
                {
                    "phase": "training",
                    "epoch": epoch,
                    "epochs": epochs,
                    "step": step + 1,
                    "steps": steps,
                    "seen": seen,
                    "total_images": len(order),
                    "seconds": elapsed + time.monotonic() - started,
                }
            )
            if paused and paused():
                save_pause(step + 1)
        coverage.append(seen)
        progress(
            {
                "phase": "validation",
                "epoch": epoch,
                "epochs": epochs,
                "seen": seen,
                "total_images": len(order),
                "step": steps,
                "steps": steps,
                "seconds": elapsed + time.monotonic() - started,
            }
        )
        val, predictions = evaluate(model, root, partitions["val"], device, batch)
        record = {"epoch": epoch, "loss": total_loss / seen, "validation": val}
        history.append(record)
        rank = (val["f1"], val["balanced_accuracy"], val["precision"], -epoch)
        if best is None or rank > best[0]:
            best = (rank, record, predictions)
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "input_size": input_size}, job_dir / "selected.pt")
        if paused and paused():
            save_pause(steps, finished=True)
    selection = {
        "epoch": best[1]["epoch"],
        "validation": best[1]["validation"],
        "rule": "F1,balanced_accuracy,precision,earlier_epoch",
        "threshold": 0.5,
    }
    atomic_json(job_dir / "selection.json", selection)
    selection_sha = hashlib.sha256((job_dir / "selection.json").read_bytes()).hexdigest()
    training_seconds = elapsed + time.monotonic() - started
    if cancelled():
        raise InterruptedError("训练已取消")
    progress(
        {
            "phase": "test",
            "selected_epoch": selection["epoch"],
            "epoch": epochs,
            "epochs": epochs,
            "seconds": training_seconds,
        }
    )
    selected = loaded_model(job_dir / "selected.pt").to(device)
    test, test_predictions = evaluate(selected, root, partitions["test"], device, batch)
    state = selected.state_dict()
    head_changed = any(
        not torch.equal(v.cpu(), initial[k])
        for k, v in state.items()
        if k.startswith("classifier.") and v.is_floating_point()
    )
    # Selection can legitimately choose a warm-up epoch with frozen vision.
    changed = any(not torch.equal(v.cpu(), initial[k]) for k, v in state.items() if v.is_floating_point())
    finite = all(torch.isfinite(v).all().item() for v in state.values())
    group_sets = [{r["group_id"] for r in partitions[s]} for s in partitions]
    isolation = all(not group_sets[i] & group_sets[j] for i in range(3) for j in range(i))
    for key in ("id", "sha", "decoded"):
        sets = [{r[key] for r in partitions[s]} for s in partitions]
        isolation = isolation and all(not sets[i] & sets[j] for i in range(3) for j in range(i))
    hashes = all(hashlib.sha256((root / r["path"]).read_bytes()).hexdigest() == r["sha"] for r in rows)
    recomputed = (
        metrics([int(r["label"] == "self") for r in test_predictions], [r["score"] for r in test_predictions]) == test
    )
    recomputed = (
        recomputed
        and metrics([int(r["label"] == "self") for r in best[2]], [r["score"] for r in best[2]])
        == best[1]["validation"]
    )
    # Independently reload the saved artifact on CPU and compare probabilities.
    reloaded = loaded_model(job_dir / "selected.pt")
    reload_val, reload_pred = evaluate(reloaded, root, partitions["val"], "cpu", 8)
    reload_close = all(abs(a["score"] - b["score"]) < 1e-3 for a, b in zip(best[2], reload_pred))
    selection_unchanged = hashlib.sha256((job_dir / "selection.json").read_bytes()).hexdigest() == selection_sha
    audit = {
        "finite": finite,
        "changed": changed,
        "head_changed": head_changed,
        "isolation": isolation,
        "image_hashes": hashes,
        "metrics_recomputed": recomputed,
        "reload_close": reload_close,
        "selection_unchanged": selection_unchanged,
        "training_coverage": all(n == len(partitions["train"]) for n in coverage),
    }
    audit["passed"] = all(audit.values())
    if not audit["passed"]:
        raise ValueError(f"模型核验失败: {audit}")
    return {
        "selection": selection,
        "selection_sha256_before_test": selection_sha,
        "history": history,
        "validation_predictions": best[2],
        "test": test,
        "test_predictions": test_predictions,
        "audit": audit,
        "training_seconds_including_validation": training_seconds,
        "total_seconds": elapsed + time.monotonic() - started,
        "peak_rss_bytes": peak_rss,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
        "historical_test": snapshot["historical_test"],
        "comparable_to_previous": snapshot["comparable_to_previous"],
        "counts": {s: len(r) for s, r in partitions.items()},
        "snapshot_sha": digest(snapshot),
        "device": device,
        "input_size": input_size,
        "trained_image_ids": [r["id"] for r in partitions["train"]],
    }


def infer(model: nn.Module, raw: bytes) -> float:
    image = decode(raw)
    tensor = NORMALIZE(transforms.ToTensor()(whole_image(image, model.input_size))).unsqueeze(0)
    with torch.inference_mode():
        score = model(tensor).float().softmax(-1)[0, 1].item()
    if not np.isfinite(score):
        raise ValueError("非有限模型得分")
    return score
