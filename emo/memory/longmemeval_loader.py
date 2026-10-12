"""LongMemEval data loader — 将单个实例的 haystack sessions 导入 EMO 记忆系统

LongMemEval 每个 question 自带一段编译好的聊天历史（haystack），500 实例互不共享，
因此按 question_id 建独立记忆库（chroma_dir/<qid> + sqlite_dir/<qid>.db）。

数据形态（longmemeval_s_cleaned.json / longmemeval_oracle.json）：
  question_id / question_type / question / answer / question_date
  haystack_dates:      ["2023/04/10 (Mon) 17:50", ...]   # 每 session 一个
  haystack_session_ids: ["answer_4be1b6b4_2", ...]
  haystack_sessions:   [[{"role": "user", "content": ..., "has_answer": true}, ...], ...]
  answer_session_ids:  ["answer_4be1b6b4_2", ...]        # session 级证据（官方召回口径）

存储策略：
  - 每个 turn → 一条记忆；event_timestamp = 所在 session 的日期
  - tags 带 session_id（供 session 级召回统计）与 has_answer（turn 级证据标记）
  - status="raw"：本数据集无 LoCoMo 那样的预结构化标注（observation），
    标 raw 让 --dream 的 Step1 真正执行结构化+相对时间解析——
    即 LoCoMo A0 报告遗留的"raw-turn 导入变体"实验，做梦完整价值在此可测。
    默认不做梦路径不受影响（index_store.search 不按 status 过滤）。

Usage:
    from memory.longmemeval_loader import LongMemEvalLoader
    loader = LongMemEvalLoader(index_store, persistent_store, session_id_prefix=qid)
    loader.load_instance(instance)
"""

import logging
import re
from datetime import datetime
from typing import List

from memory.schema import IndexEntry, MemoryFile

logger = logging.getLogger(__name__)


def parse_lme_datetime(date_str: str) -> str:
    """"2023/04/10 (Mon) 17:50" → "2023-04-10 17:50:00"（失败返回 ""）"""
    m = re.match(
        r"^(\d{4})/(\d{2})/(\d{2})\s*\(\w+\)\s*(\d{2}):(\d{2})$",
        (date_str or "").strip(),
    )
    if not m:
        return ""
    y, mo, d, h, mi = (int(x) for x in m.groups())
    try:
        return datetime(y, mo, d, h, mi, 0).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return ""


class LongMemEvalLoader:
    """将一个 LongMemEval 实例的 haystack 导入 EMO 记忆系统"""

    def __init__(self, index_store, persistent_store, session_id_prefix: str = "lme"):
        self._index = index_store
        self._persist = persistent_store
        self._prefix = session_id_prefix

    def load_instance(self, inst: dict) -> int:
        """导入一个实例的全部 haystack sessions，返回记忆条数"""
        sessions = inst["haystack_sessions"]
        dates = inst["haystack_dates"]
        session_ids = inst["haystack_session_ids"]

        index_entries: List[IndexEntry] = []
        memory_files: List[MemoryFile] = []
        embedding_docs: List[str] = []

        for sess_idx, (dialogs, date_str, sid) in enumerate(
            zip(sessions, dates, session_ids)
        ):
            event_ts = parse_lme_datetime(date_str)
            session_ts = event_ts or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            for turn_idx, turn in enumerate(dialogs):
                role = turn["role"]          # "user" / "assistant"
                text = turn["content"]
                has_answer = bool(turn.get("has_answer"))

                # mem_id 用 session 序号保证唯一：13/500 实例的
                # haystack_session_ids 有重复（同一 session 被编进多个 haystack 位置），
                # 用 sid 会撞 DuplicateIDError。sid 仍进 tags，召回匹配不受影响
                mem_id = f"{self._prefix}_s{sess_idx}_t{turn_idx}"
                summary = f"({date_str}) {role}: {text[:200]}"
                embedding_text = f"{date_str}. {role} said: \"{text}\""

                tags = [role, sid, f"{sid}_t{turn_idx}"]
                if has_answer:
                    tags.append("has_answer")

                entry = IndexEntry(
                    mem_id=mem_id,
                    summary=summary,
                    embedding_text=embedding_text,
                    category="对话",
                    sub_category="chat_history",
                    tags=tags,
                    source="longmemeval",
                    speaker=role,
                    original_date=date_str,
                    event_timestamp=event_ts,
                    base_weight=0.5,
                    created_at=session_ts,
                    last_access=session_ts,
                )
                index_entries.append(entry)
                embedding_docs.append(embedding_text)

                mf = MemoryFile(
                    mem_id=mem_id,
                    raw_content=(
                        f"DATE: {date_str}\n"
                        f"SESSION: {sid}\n"
                        f"ROLE: {role}\n"
                        f"TEXT: {text}"
                    ),
                    summary=summary,
                    speaker=role,
                    category="对话",
                    sub_category="chat_history",
                    tags=",".join(tags),
                    source="longmemeval",
                    fact_entries="[]",
                    original_date=date_str,
                    timestamp=session_ts,
                    event_timestamp=event_ts,
                    session_id=sid,
                    turn_index=turn_idx,
                    importance=0.6 if has_answer else 0.5,
                    status="raw",  # 见模块 docstring：让 --dream Step1 真正结构化
                )
                memory_files.append(mf)

        if index_entries:
            self._index.add(index_entries, documents=embedding_docs)
            self._persist.save(memory_files)

        logger.info(
            f"Imported {len(index_entries)} memories "
            f"({len(sessions)} sessions) for {self._prefix}"
        )
        return len(index_entries)
