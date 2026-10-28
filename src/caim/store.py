"""长期存储：SQLite 单表 + 巩固期的去重合并。

为什么用 SQLite 而不是内存 dict：
- 记忆需要跨进程/跨会话存活（"防止失忆"的前提）；
- 单文件、零依赖、可直接 inspect；
- 数据量在个人助手场景下完全够用（万级条目毫秒级查询）。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from .models import KIND_FACT, KIND_RAW, MemoryItem

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    kind         TEXT    NOT NULL,
    text         TEXT    NOT NULL,
    tokens       INTEGER NOT NULL,
    importance   REAL    NOT NULL,
    created_at   REAL    NOT NULL,
    last_access  REAL    NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    parent_id    INTEGER,
    meta         TEXT
);
CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);
CREATE INDEX IF NOT EXISTS idx_memories_access ON memories(last_access);
"""

_COLUMNS = (
    "id, kind, text, tokens, importance, created_at, last_access, "
    "access_count, parent_id, meta"
)


class MemoryStore:
    """记忆表。线程安全（内部加锁），可多线程共享同一实例。"""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ---------- 写入 ----------

    def add(
        self,
        kind: str,
        text: str,
        tokens: int,
        importance: float = 0.5,
        *,
        parent_id: int | None = None,
        meta: dict | None = None,
        now: float | None = None,
    ) -> MemoryItem:
        ts = time.time() if now is None else now
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO memories "
                "(kind, text, tokens, importance, created_at, last_access, "
                " access_count, parent_id, meta) VALUES (?,?,?,?,?,?,0,?,?)",
                (
                    kind,
                    text,
                    tokens,
                    float(importance),
                    ts,
                    ts,
                    parent_id,
                    json.dumps(meta or {}, ensure_ascii=False),
                ),
            )
            self._conn.commit()
            item_id = int(cur.lastrowid)
        item = self.get(item_id)
        assert item is not None
        return item

    def touch(self, item_id: int, now: float | None = None) -> None:
        """被召回一次 —— 人脑的"复述加固"：访问次数 +1，最后访问时间刷新。"""
        ts = time.time() if now is None else now
        with self._lock:
            self._conn.execute(
                "UPDATE memories SET access_count = access_count + 1, last_access = ? "
                "WHERE id = ?",
                (ts, item_id),
            )
            self._conn.commit()

    def set_importance(self, item_id: int, importance: float) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE memories SET importance = ? WHERE id = ?",
                (max(0.0, min(1.0, float(importance))), item_id),
            )
            self._conn.commit()

    def delete(self, item_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM memories WHERE id = ?", (item_id,))
            self._conn.commit()

    # ---------- 查询 ----------

    def get(self, item_id: int) -> MemoryItem | None:
        with self._lock:
            row = self._conn.execute(
                f"SELECT {_COLUMNS} FROM memories WHERE id = ?", (item_id,)
            ).fetchone()
        return _row_to_item(row) if row else None

    def all(self, kinds: tuple[str, ...] | None = None) -> list[MemoryItem]:
        sql = f"SELECT {_COLUMNS} FROM memories"
        params: tuple = ()
        if kinds:
            sql += " WHERE kind IN (" + ",".join("?" * len(kinds)) + ")"
            params = kinds
        sql += " ORDER BY id"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_item(r) for r in rows]

    def count(self, kinds: tuple[str, ...] | None = None) -> int:
        sql = "SELECT COUNT(*) FROM memories"
        params: tuple = ()
        if kinds:
            sql += " WHERE kind IN (" + ",".join("?" * len(kinds)) + ")"
            params = kinds
        with self._lock:
            return int(self._conn.execute(sql, params).fetchone()[0])

    def stats(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT kind, COUNT(*), COALESCE(SUM(tokens),0) FROM memories GROUP BY kind"
            ).fetchall()
        out: dict[str, int] = {}
        for kind, n, tokens in rows:
            out[f"{kind}_count"] = int(n)
            out[f"{kind}_tokens"] = int(tokens)
        out["total_count"] = sum(v for k, v in out.items() if k.endswith("_count"))
        return out

    # ---------- 巩固：去重合并 ----------

    def merge_similar_facts(self, similarity, threshold: float = 0.92) -> int:
        """把高度相似的事实合并（同一件事被反复说），返回合并条数。

        ``similarity`` 由 :mod:`caim.retrieval` 提供，避免循环导入。
        """
        facts = self.all((KIND_FACT,))
        merged = 0
        survivors: list[MemoryItem] = []
        for item in facts:
            target = None
            for kept in survivors:
                if similarity(item.text, kept.text) >= threshold:
                    target = kept
                    break
            if target is None:
                survivors.append(item)
                continue
            # 保留信息更全的那条，重要性取大
            if len(item.text) > len(target.text):
                self._replace_text(target.id, item.text, item.tokens)
            self.set_importance(
                target.id, max(target.importance, item.importance)
            )
            self.delete(item.id)
            merged += 1
        return merged

    def _replace_text(self, item_id: int, text: str, tokens: int) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE memories SET text = ?, tokens = ? WHERE id = ?",
                (text, tokens, item_id),
            )
            self._conn.commit()

    def prune_low_value(
        self,
        keep_top: int = 500,
        *,
        now: float | None = None,
        min_importance: float = 0.15,
    ) -> int:
        """截断：低重要性且长期未被访问的条目优先淘汰（"遗忘"）。

        只淘汰 ``raw``，事实与情景摘要默认保留 —— 这是"不失忆"的底线。
        """
        ts = time.time() if now is None else now
        raws = [m for m in self.all((KIND_RAW,)) if m.importance < min_importance]
        raws.sort(key=lambda m: (m.access_count, m.last_access))
        drop_n = max(0, len(raws) - keep_top)
        for item in raws[:drop_n]:
            self.delete(item.id)
        _ = ts
        return drop_n

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _row_to_item(row: sqlite3.Row) -> MemoryItem:
    try:
        meta = json.loads(row["meta"] or "{}")
    except json.JSONDecodeError:
        meta = {}
    return MemoryItem(
        id=int(row["id"]),
        kind=str(row["kind"]),
        text=str(row["text"]),
        tokens=int(row["tokens"]),
        importance=float(row["importance"]),
        created_at=float(row["created_at"]),
        last_access=float(row["last_access"]),
        access_count=int(row["access_count"]),
        parent_id=row["parent_id"],
        meta=meta,
    )
