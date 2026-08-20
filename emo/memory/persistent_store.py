"""M6: 持久存储层 — SQLite

职责:
    - 存储 MemoryFile（完整记忆档案）
    - 永不删除，即使热索引层已"遗忘"
    - 支持按 mem_id 精确查找（Skill 拉取时使用）
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import List, Optional

from .schema import MemoryFile

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS memory_files (
    mem_id             TEXT PRIMARY KEY,
    raw_content        TEXT NOT NULL,
    summary            TEXT DEFAULT '',
    speaker            TEXT DEFAULT '',
    category           TEXT DEFAULT '',
    sub_category       TEXT DEFAULT '',
    tags               TEXT DEFAULT '',
    source             TEXT DEFAULT 'interaction',
    fact_entries       TEXT DEFAULT '',
    rel_entries        TEXT DEFAULT '',
    keywords           TEXT DEFAULT '',
    context_description TEXT DEFAULT '',
    box_id             TEXT DEFAULT '',
    session_summary    TEXT DEFAULT '',
    event_description  TEXT DEFAULT '',
    original_date      TEXT DEFAULT '',
    timestamp          TEXT DEFAULT '',
    session_id         TEXT DEFAULT '',
    turn_index         INTEGER DEFAULT 0,
    temporal_context   TEXT DEFAULT '',
    event_sequence     TEXT DEFAULT '',
    event_timestamp    TEXT DEFAULT '',
    importance         REAL DEFAULT 0.5,
    status             TEXT DEFAULT 'raw',
    label1             TEXT DEFAULT '',
    label2             TEXT DEFAULT '',
    related_ids        TEXT DEFAULT '',
    superseded_by      TEXT DEFAULT ''
)
"""

# event_timestamp 为 "YYYY-MM-DD HH:MM:SS" 字符串，字典序即时间序
_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_memory_event_ts ON memory_files(event_timestamp)
"""


class PersistentStore:
    """持久存储层 — SQLite"""

    def __init__(self, db_path: str = ":memory:"):
        """
        Args:
            db_path: SQLite 数据库路径。":memory:" 为内存模式（测试用）
        """
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_CREATE_TABLE_SQL)
        self._conn.execute(_CREATE_INDEX_SQL)
        # 旧库迁移：superseded_by 列不存在则补
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(memory_files)")}
        if "superseded_by" not in cols:
            self._conn.execute("ALTER TABLE memory_files ADD COLUMN superseded_by TEXT DEFAULT ''")
        self._conn.commit()
        logger.info(f"PersistentStore initialized: db='{db_path}'")

    # ── 写入 ──

    def save(self, files: List[MemoryFile]) -> int:
        """批量保存 MemoryFile

        Args:
            files: MemoryFile 列表

        Returns:
            成功保存的数量
        """
        if not files:
            return 0

        rows = [
            (
                f.mem_id,
                f.raw_content,
                f.summary,
                f.speaker,
                f.category,
                f.sub_category,
                f.tags,
                f.source,
                f.fact_entries,
                f.rel_entries,
                f.keywords,
                f.context_description,
                f.box_id,
                f.session_summary,
                f.event_description,
                f.original_date,
                f.timestamp,
                f.session_id,
                f.turn_index,
                f.temporal_context,
                f.event_sequence,
                f.event_timestamp,
                f.importance,
                f.status,
                f.label1,
                f.label2,
                f.related_ids,
                f.superseded_by,
            )
            for f in files
        ]

        self._conn.executemany(
            """INSERT OR REPLACE INTO memory_files
               (mem_id, raw_content, summary, speaker, category, sub_category,
                tags, source, fact_entries, rel_entries, keywords, context_description, box_id,
                session_summary,
                event_description, original_date, timestamp, session_id,
                turn_index, temporal_context, event_sequence, event_timestamp,
                importance, status, label1, label2, related_ids, superseded_by)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self._conn.commit()
        logger.info(f"Saved {len(files)} memory files (total: {self.count()})")
        return len(files)

    # ── 读取 ──

    def get(self, mem_id: str) -> Optional[MemoryFile]:
        """按 mem_id 精确查找"""
        row = self._conn.execute(
            "SELECT * FROM memory_files WHERE mem_id = ?", (mem_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_file(row)

    def get_batch(self, mem_ids: List[str]) -> List[MemoryFile]:
        """批量按 mem_id 查找"""
        if not mem_ids:
            return []
        placeholders = ",".join("?" * len(mem_ids))
        rows = self._conn.execute(
            f"SELECT * FROM memory_files WHERE mem_id IN ({placeholders})",
            mem_ids,
        ).fetchall()
        return [self._row_to_file(r) for r in rows]

    def get_by_category(self, category: str, limit: int = 100) -> List[MemoryFile]:
        """按 L1 分类查找"""
        rows = self._conn.execute(
            "SELECT * FROM memory_files WHERE category = ? ORDER BY timestamp DESC LIMIT ?",
            (category, limit),
        ).fetchall()
        return [self._row_to_file(r) for r in rows]

    def get_related(self, mem_id: str, related_ids: List[str]) -> List[MemoryFile]:
        """获取关联记忆"""
        return self.get_batch(related_ids)

    # ── 工具方法 ──

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM memory_files").fetchone()
        return row[0]

    def search_text(self, keyword: str, limit: int = 20) -> List[MemoryFile]:
        """简单文本搜索（LIKE，仅用于调试）"""
        rows = self._conn.execute(
            "SELECT * FROM memory_files WHERE raw_content LIKE ? LIMIT ?",
            (f"%{keyword}%", limit),
        ).fetchall()
        return [self._row_to_file(r) for r in rows]

    def get_by_status(self, status: str, limit: int = None) -> List[MemoryFile]:
        """按 status 查找记忆（默认全量；做梦/训分类器依赖全量，截断会静默漏数据）"""
        sql = "SELECT * FROM memory_files WHERE status = ? ORDER BY timestamp"
        params: tuple = (status,)
        if limit is not None:
            sql += " LIMIT ?"
            params = (status, limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_file(r) for r in rows]

    def get_by_event_time_range(
        self, start: str, end: str, limit: int = 50
    ) -> List[MemoryFile]:
        """按事件发生时间范围查找（时间轴查询）

        Args:
            start: "YYYY-MM-DD HH:MM:SS"（含）
            end:   "YYYY-MM-DD HH:MM:SS"（不含，半开区间）
            limit: 最多返回条数（按事件时间升序）
        """
        rows = self._conn.execute(
            """SELECT * FROM memory_files
               WHERE event_timestamp != '' AND event_timestamp >= ? AND event_timestamp < ?
               ORDER BY event_timestamp LIMIT ?""",
            (start, end, limit),
        ).fetchall()
        return [self._row_to_file(r) for r in rows]

    def mark_settled(self) -> int:
        """将所有 dreamed 记忆标记为 settled"""
        cursor = self._conn.execute(
            "UPDATE memory_files SET status = 'settled' WHERE status = 'dreamed'"
        )
        self._conn.commit()
        count = cursor.rowcount
        if count:
            logger.info(f"Marked {count} memories as settled")
        return count

    def update_status(self, mem_id: str, status: str) -> None:
        """更新单条记忆的 status"""
        self._conn.execute(
            "UPDATE memory_files SET status = ? WHERE mem_id = ?",
            (status, mem_id),
        )
        self._conn.commit()

    def reset_dreamed_to_raw(self) -> int:
        """将所有 dreamed 记忆重置为 raw（用于重新结构化）"""
        cursor = self._conn.execute(
            "UPDATE memory_files SET status = 'raw' WHERE status = 'dreamed'"
        )
        self._conn.commit()
        return cursor.rowcount

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row_to_file(row: sqlite3.Row) -> MemoryFile:
        return MemoryFile(
            mem_id=row["mem_id"],
            raw_content=row["raw_content"],
            summary=row["summary"] or "",
            speaker=row["speaker"] or "",
            category=row["category"] or "",
            sub_category=row["sub_category"] or "",
            tags=row["tags"] or "",
            source=row["source"] or "interaction",
            fact_entries=row["fact_entries"] or "",
            rel_entries=row["rel_entries"] or "",
            keywords=row["keywords"] or "",
            context_description=row["context_description"] or "",
            box_id=row["box_id"] or "",
            session_summary=row["session_summary"] or "",
            event_description=row["event_description"] or "",
            original_date=row["original_date"] or "",
            timestamp=row["timestamp"] or "",
            session_id=row["session_id"] or "",
            turn_index=row["turn_index"] or 0,
            temporal_context=row["temporal_context"] or "",
            event_sequence=row["event_sequence"] or "",
            event_timestamp=row["event_timestamp"] or "",
            importance=row["importance"] or 0.5,
            status=row["status"] or "raw",
            label1=row["label1"] or "",
            label2=row["label2"] or "",
            related_ids=row["related_ids"] or "",
            superseded_by=row["superseded_by"] if "superseded_by" in row.keys() else "",
        )

    def mark_superseded(self, mem_id: str, by_id: str) -> None:
        """标记 mem_id 被 by_id 取代（失效但保留，供时间轴/审计追溯）"""
        self._conn.execute(
            "UPDATE memory_files SET superseded_by = ? WHERE mem_id = ?",
            (by_id, mem_id),
        )
        self._conn.commit()
