"""SQLite catalog, persistent variant groups and immutable task snapshots."""

from pathlib import Path
from typing import Any
from collections.abc import Iterator
from contextlib import contextmanager
import hashlib
import json
import random
import shutil
import sqlite3
import time
import uuid

from .images import decode, fingerprint, similar, thumbnail


def dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(dump(value).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(dump(value), encoding="utf-8")
    for attempt in range(6):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.02)


class Store:
    def __init__(self, root: Path):
        self.root = root.resolve()
        for name in ("images", "jobs", "models", "uploads"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        with self.db() as db:
            db.executescript("""
              PRAGMA journal_mode=WAL;
              CREATE TABLE IF NOT EXISTS images(id TEXT PRIMARY KEY, path TEXT, sha TEXT, decoded TEXT, phash TEXT, aspect REAL, width INTEGER, height INTEGER, group_id TEXT, label TEXT, removed INTEGER DEFAULT 0, reason TEXT DEFAULT '', created REAL);
              CREATE TABLE IF NOT EXISTS groups(id TEXT PRIMARY KEY, split TEXT);
              CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, request_key TEXT UNIQUE, state TEXT, pid INTEGER, error TEXT, created REAL, snapshot_sha TEXT);
              CREATE UNIQUE INDEX IF NOT EXISTS one_training ON jobs((1)) WHERE state IN ('queued','running','cancelling');
              CREATE UNIQUE INDEX IF NOT EXISTS one_training_v2 ON jobs((1)) WHERE state IN ('queued','running','pausing','cancelling');
              CREATE TABLE IF NOT EXISTS models(id TEXT PRIMARY KEY, job_id TEXT, sha TEXT, metrics TEXT, created REAL);
              CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT);
              CREATE INDEX IF NOT EXISTS images_group ON images(group_id);
            """)

    @contextmanager
    def db(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.root / "catalog.sqlite3", timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def setting(self, key: str, default: str = "") -> str:
        with self.db() as db:
            row = db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def set_setting(self, key: str, value: str) -> None:
        with self.db() as db:
            db.execute("INSERT OR REPLACE INTO settings VALUES (?,?)", (key, value))

    def _reconcile(self, db: sqlite3.Connection, group: str) -> None:
        rows = db.execute(
            "SELECT * FROM images WHERE group_id=? AND removed=0 AND reason NOT IN ('solid','split_conflict') ORDER BY created,id",
            (group,),
        ).fetchall()
        labels = {r["label"] for r in rows if r["label"] in {"self", "other"}}
        rows = sorted(rows, key=lambda r: (r["label"] not in {"self", "other"}, r["created"], r["id"]))
        for index, row in enumerate(rows):
            reason = "label_conflict" if len(labels) > 1 else ("duplicate" if index else "")
            db.execute("UPDATE images SET reason=? WHERE id=?", (reason, row["id"]))

    def ingest(self, path: Path, label: str | None = None) -> dict:
        if label not in {None, "self", "other"}:
            raise ValueError("上传身份必须为是Bot或不是Bot")
        path = path.resolve()
        if not path.is_relative_to(self.root / "uploads") or path.is_symlink():
            raise ValueError("只能领取本插件上传文件")
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        image = decode(raw)
        fp = fingerprint(image)
        ident = uuid.uuid4().hex
        destination = self.root / "images" / (ident + path.suffix)
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT id,reason FROM images WHERE sha=? OR decoded=? ORDER BY created,id LIMIT 1",
                (sha, fp["decoded"]),
            ).fetchone()
            if existing is not None:
                path.unlink()
                return {"id": existing["id"], "reason": existing["reason"], "already_exists": True}
            matches = [
                dict(r)
                for r in db.execute("SELECT * FROM images WHERE reason NOT IN ('solid','split_conflict')")
                if similar(fp, dict(r))
            ]
            groups = {r["group_id"] for r in matches}
            splits = {r[0] for g in groups for r in db.execute("SELECT split FROM groups WHERE id=?", (g,)) if r[0]}
            reason = "solid" if fp["solid"] else ("split_conflict" if len(splits) > 1 else "")
            group = min(groups) if groups and reason != "split_conflict" else ident
            db.execute(
                "INSERT OR IGNORE INTO groups VALUES (?,?)",
                (group, next(iter(splits), None) if reason != "split_conflict" else None),
            )
            if not reason:
                for old in groups - {group}:
                    db.execute("UPDATE images SET group_id=? WHERE group_id=?", (group, old))
                    db.execute("DELETE FROM groups WHERE id=?", (old,))
                if splits:
                    db.execute("UPDATE groups SET split=? WHERE id=?", (next(iter(splits)), group))
            shutil.move(path, destination)
            db.execute(
                "INSERT INTO images VALUES (?,?,?,?,?,?,?,?,?,?,0,?,?)",
                (
                    ident,
                    str(destination.relative_to(self.root)),
                    sha,
                    fp["decoded"],
                    fp["phash"],
                    fp["aspect"],
                    fp["width"],
                    fp["height"],
                    group,
                    label,
                    reason,
                    time.time(),
                ),
            )
            self._reconcile(db, group)
        return {"id": ident, "reason": reason}

    def label(self, image_id: str, label: str, remove: bool | None = None) -> dict:
        if label not in {"self", "other", "skip", "unlabelled"}:
            raise ValueError("无效标注")
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT group_id FROM images WHERE id=?", (image_id,)).fetchone()
            if row is None:
                raise ValueError("图片不存在")
            db.execute(
                "UPDATE images SET label=?,removed=COALESCE(?,removed) WHERE id=?",
                (None if label == "unlabelled" else label, int(remove) if remove is not None else None, image_id),
            )
            self._reconcile(db, row[0])
        return {"id": image_id, "label": label, "removed": remove}

    def set_removed(self, image_id: str, removed: bool) -> dict:
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT group_id FROM images WHERE id=?", (image_id,)).fetchone()
            if row is None:
                raise ValueError("图片不存在")
            db.execute("UPDATE images SET removed=? WHERE id=?", (int(removed), image_id))
            self._reconcile(db, row[0])
        return {"id": image_id, "removed": removed}

    def batch_update(self, image_ids: str, operation: str) -> dict:
        """Apply the entire selection in one transaction, without partial updates."""
        if operation not in {"self", "other", "remove", "remove_all"}:
            raise ValueError("无效批量操作")
        ids = list(dict.fromkeys(image_ids.split(","))) if image_ids else []
        if operation != "remove_all" and (not ids or len(ids) > 1000):
            raise ValueError("请先选择图片，每批最多1000张")
        if any(len(i) != 32 or any(c not in "0123456789abcdef" for c in i) for i in ids):
            raise ValueError("无效图片ID")
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            if operation == "remove_all":
                rows = db.execute("SELECT id,group_id FROM images WHERE removed=0").fetchall()
                db.execute("UPDATE images SET removed=1 WHERE removed=0")
            else:
                placeholders = ",".join("?" for _ in ids)
                rows = db.execute(f"SELECT id,group_id FROM images WHERE id IN ({placeholders})", ids).fetchall()
                if len(rows) != len(ids):
                    raise ValueError("选择中有不存在的图片，请清空选择后重新选择")
                if operation == "remove":
                    db.execute(f"UPDATE images SET removed=1 WHERE id IN ({placeholders})", ids)
                else:
                    if db.execute(f"SELECT 1 FROM images WHERE removed=1 AND id IN ({placeholders}) LIMIT 1", ids).fetchone():
                        raise ValueError("选择中有已删除图片，请先恢复或取消选择")
                    db.execute(f"UPDATE images SET label=? WHERE id IN ({placeholders})", [operation, *ids])
            for group in {row["group_id"] for row in rows}:
                self._reconcile(db, group)
        return {"updated": len(rows), "operation": operation}

    def images(self, page: int = 1, view: str = "all", job_id: str = "") -> dict:
        if page < 1 or view not in {"all", "eligible", "excluded", "unlabelled", "errors", "removed"}:
            raise ValueError("无效分页")
        where = {
            "all": "i.removed=0",
            "eligible": "i.removed=0 AND i.reason='' AND i.label IN ('self','other')",
            "excluded": "(i.removed=1 OR i.reason!='' OR i.label='skip')",
            "unlabelled": "i.label IS NULL",
            "errors": "1",
            "removed": "i.removed=1",
        }[view]
        error_ids = set()
        predictions = {}
        if view == "errors":
            if not job_id:
                return {
                    "rows": [],
                    "gallery": [],
                    "pages": 1,
                    "total": 0,
                    "pagination": {"page": 1, "pages": 1, "total": 0},
                }
            report = self.report(job_id)
            predictions = {r["id"]: r for r in report.get("test_predictions", []) + report.get("validation_predictions", [])}
            error_ids = {
                r["id"]
                for r in report.get("test_predictions", []) + report.get("validation_predictions", [])
                if r["predicted"] != r["label"]
            }
        parameters = []
        if view == "errors":
            if error_ids:
                where += " AND i.id IN (" + ",".join("?" for _ in error_ids) + ")"
                parameters = sorted(error_ids)
            else:
                where = "0"
        with self.db() as db:
            total = db.execute(f"SELECT COUNT(*) FROM images i WHERE {where}", parameters).fetchone()[0]
            page = min(page, max(1, (total + 23) // 24))
            rows = [dict(r) for r in db.execute(
                f"SELECT i.*,g.split FROM images i JOIN groups g ON g.id=i.group_id WHERE {where} ORDER BY i.created DESC LIMIT 24 OFFSET ?",
                [*parameters, (page - 1) * 24],
            )]
        gallery = []
        for r in rows:
            reasons = {"solid": "纯色/近纯色", "duplicate": "同图变体，仅保留一个代表", "label_conflict": "同图身份标签冲突", "split_conflict": "连接了不同历史划分"}
            r["status"] = "已移出训练库" if r["removed"] else reasons.get(r["reason"], "跳过" if r["label"] == "skip" else "可用")
            r["label_text"] = {"self": "是自己", "other": "不是自己", "skip": "跳过"}.get(r["label"], "未标注")
            prediction = predictions.get(r["id"])
            r["model_text"] = (f"模型：{'是自己' if prediction['predicted']=='self' else '不是自己'} · 得分 {prediction['score']:.3f} · 冻结真值：{'是自己' if prediction['label']=='self' else '不是自己'}" if prediction else "")
            gallery.append(
                {
                    "thumbnail": thumbnail(self.root / r["path"], self.root / "thumbnails" / (r["sha"] + "-512.jpg")),
                    "title": r["id"],
                    "description": f"{r['label_text']} / {r['status']} / {r['split'] or '未划分'}",
                    "badge": r["label_text"],
                    "subtitle": f"{r['width']}×{r['height']}",
                }
            )
        for row, preview in zip(rows, gallery):
            row["thumbnail"] = preview["thumbnail"]
        return {
            "rows": rows,
            "pages": max(1, (total + 23) // 24),
            "total": total,
            "pagination": {"page": page, "pages": max(1, (total + 23) // 24), "total": total},
        }

    def training_readiness(self) -> dict:
        counts = {"self": 0, "other": 0}
        with self.db() as db:
            for row in db.execute(
                "SELECT label,COUNT(DISTINCT group_id) AS count FROM images WHERE removed=0 AND reason='' AND label IN ('self','other') GROUP BY label"
            ):
                counts[row["label"]] = row["count"]
        return {
            "can_train": min(counts.values()) >= 20,
            "summary": f"可训练独立图组：是Bot {counts['self']} / 不是Bot {counts['other']}。每类至少20组，重复变体只算一组。",
            "groups": counts,
        }

    def freeze(self, db: sqlite3.Connection, job_id: str, mode: str, device: str) -> dict:
        rows = [
            dict(r)
            for r in db.execute(
                "SELECT i.*,g.split FROM images i JOIN groups g ON g.id=i.group_id WHERE i.removed=0 AND i.reason='' AND i.label IN ('self','other') ORDER BY i.id"
            )
        ]
        # A labelled variant can supply the group truth to its unlabelled representative.
        representatives = {r["group_id"]: r for r in rows}
        for r in db.execute("SELECT * FROM images WHERE removed=0 AND reason='' AND label IS NULL"):
            truth = db.execute(
                "SELECT label FROM images WHERE group_id=? AND label IN ('self','other') AND removed=0 LIMIT 1",
                (r["group_id"],),
            ).fetchone()
            if truth:
                value = dict(r)
                value["label"] = truth[0]
                value["split"] = db.execute("SELECT split FROM groups WHERE id=?", (r["group_id"],)).fetchone()[0]
                representatives[r["group_id"]] = value
        rows = list(representatives.values())
        rng = random.Random(42)
        for label in ("self", "other"):
            cohort = [r for r in rows if r["label"] == label]
            if len(cohort) < 20:
                name = "是Bot" if label == "self" else "不是Bot"
                raise ValueError(f"“{name}”至少需要20个可训练独立图片组，当前{len(cohort)}；请上传或补标该类图片")
            new = sorted((r for r in cohort if not r["split"]), key=lambda r: r["group_id"])
            rng.shuffle(new)
            # Fill cumulative targets while preserving every existing assignment.
            targets = {"test": max(2, round(len(cohort) * 0.15)), "val": max(2, round(len(cohort) * 0.15))}
            targets["train"] = len(cohort) - targets["test"] - targets["val"]
            counts = {s: sum(r["split"] == s for r in cohort) for s in targets}
            for r in new:
                split = max(targets, key=lambda s: targets[s] - counts[s])
                r["split"] = split
                counts[split] += 1
                db.execute("UPDATE groups SET split=? WHERE id=?", (split, r["group_id"]))
            if any(counts[s] < 2 for s in targets):
                raise ValueError(f"{label} 每个划分至少需要2张")
        parent = self.setting("active") if mode == "incremental" else ""
        parent_record = db.execute("SELECT sha FROM models WHERE id=?", (parent,)).fetchone() if parent else None
        if parent and parent_record is None:
            raise ValueError("当前启用模型的记录不存在")
        if mode == "incremental" and not parent:
            mode = "fresh"
        held_truth = digest(sorted((r["id"], r["label"]) for r in rows if r["split"] != "train"))
        previous = db.execute(
            "SELECT snapshot_sha FROM jobs WHERE state='complete' ORDER BY created DESC LIMIT 1"
        ).fetchone()
        old_truth = self.setting("last_held_truth")
        character = db.execute("SELECT value FROM settings WHERE key='character'").fetchone()
        return {
            "job_id": job_id,
            "mode": mode,
            "device": device,
            "seed": 42,
            "parent": parent,
            "parent_sha": parent_record[0] if parent_record else "",
            "character": character[0] if character else "Bot自己",
            "images": rows,
            "held_truth": held_truth,
            "comparable_to_previous": not old_truth or held_truth == old_truth,
            "historical_test": bool(previous),
            "threshold": 0.5,
            "created": time.time(),
        }

    def report(self, job_id: str) -> dict:
        self.job(job_id)  # reject arbitrary paths
        path = self.root / "jobs" / job_id / "report.json"
        return json.loads(path.read_text("utf-8")) if path.exists() else {}

    def job(self, job_id: str) -> dict:
        with self.db() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise ValueError("任务不存在")
        return dict(row)

    def models(self) -> list:
        with self.db() as db:
            return [dict(r) for r in db.execute("SELECT * FROM models ORDER BY created DESC")]
