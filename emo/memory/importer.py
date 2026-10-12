"""M8: 数据集导入器

职责:
    - 从 JSONL 文件导入初始数据（用户提供的历史对话数据集）
    - 每条记录 → IndexEntry（热索引层）+ MemoryFile（持久存储层）
    - 生成语义摘要、embedding 文本、结构化事实/关系条目

当前支持:
    - Aditi SFT 格式 (scenario_id, user_message, context, assistant_reply, tags)
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from .schema import IndexEntry, MemoryFile
from .index_store import IndexStore
from .persistent_store import PersistentStore

logger = logging.getLogger(__name__)

# ── 人格标签 vs 场景标签的分类 ──
_PERSONALITY_TAGS = {
    "playful", "supportive", "energetic", "chatty", "clingy", "funny",
    "sarcastic", "warm", "affectionate", "dramatic", "caring", "teasing",
    "self_deprecating", "emotional", "flirty", "banter", "soft",
    "humorous", "encouraging", "reassuring", "comforting", "lighthearted",
    "serious", "vulnerable", "nostalgic", "grumpy", "moody", "soft_undertone",
    "affectionate_teasing", "body-image-jokes", "food_cravings",
    "food_craving", "self-deprecating", "light_hinglish",
}


class DatasetImporter:
    """数据集导入器 — 将历史对话数据导入为记忆"""

    def __init__(self, index_store: IndexStore, persistent_store: PersistentStore):
        self._index = index_store
        self._persist = persistent_store

    def import_aditi_jsonl(
        self,
        filepath: str,
        session_id_prefix: str = "aditi_init",
    ) -> int:
        """导入 Aditi SFT 格式的 JSONL 文件

        数据格式:
            {"scenario_id": "...", "user_message": "...",
             "context": "Aditi knows...", "assistant_reply": "...",
             "tags": ["playful", "supportive", ...]}

        Returns:
            成功导入的记录数
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        records = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        logger.info(f"Loaded {len(records)} records from {filepath}")

        index_entries: List[IndexEntry] = []
        memory_files: List[MemoryFile] = []
        embedding_docs: List[str] = []

        seen_ids: set = set()
        now = datetime.now()
        time_span_days = 30
        time_step = timedelta(days=time_span_days / max(len(records), 1))

        for i, rec in enumerate(records):
            raw_id = rec.get("scenario_id", uuid.uuid4().hex[:8])
            mem_id = f"{session_id_prefix}_{raw_id}"
            if mem_id in seen_ids:
                mem_id = f"{mem_id}_{i}"
            seen_ids.add(mem_id)

            tags = rec.get("tags", [])
            context = rec.get("context", "")
            user_msg = rec.get("user_message", "")
            reply = rec.get("assistant_reply", "")

            # ── 分离人格标签和场景标签 ──
            personality_tags = [t for t in tags if t.lower() in _PERSONALITY_TAGS]
            scenario_tags = [t for t in tags if t.lower() not in _PERSONALITY_TAGS]

            # ── 生成 summary（语义摘要）──
            if context:
                summary = f"User was {context.lower().replace('aditi knows the user ', '')}. Aditi responded with {personality_tags[0] if personality_tags else 'casual'} tone."
            else:
                summary = f"User said: {user_msg[:80]}. Aditi replied playfully."

            # ── 生成 embedding_text（用于向量检索，包含完整语义）──
            embedding_text = (
                f"{context} "
                f"User: {user_msg} "
                f"Aditi replied: {reply[:150]} "
                f"Style: {', '.join(personality_tags[:3])}. "
                f"Topics: {', '.join(scenario_tags[:3])}."
            )

            # ── 时间 ──
            ts = now - timedelta(days=time_span_days) + time_step * i
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")

            # ── 分类 ──
            category = self._classify_aditi(tags, context)
            sub_category = self._subclassify_aditi(tags, context, user_msg)

            # ── fact_entries（从 context 和 reply 提取事实）──
            fact_list = []
            if context:
                fact_list.append(context)
            # 从 reply 中提取关键事实（简单规则：取第一句话）
            first_sentence = reply.split(".")[0].strip() if reply else ""
            if first_sentence:
                fact_list.append(f"Aditi said: {first_sentence}")

            # ── rel_entries（关系条目）──
            rel_list = []
            if personality_tags:
                rel_list.append(
                    f"Aditi takes a {personality_tags[0]} tone with the user"
                )
            rel_list.append("Aditi and user have a close, casual friendship")

            # ── raw_content ──
            raw_content = (
                f"[Context] {context}\n"
                f"[User] {user_msg}\n"
                f"[Aditi] {reply}"
            )

            # ── IndexEntry ──
            entry = IndexEntry(
                mem_id=mem_id,
                summary=summary,
                embedding_text=embedding_text,
                category=category,
                sub_category=sub_category,
                tags=tags,
                source="aditi",
                speaker="Aditi",
                base_weight=0.6,
                created_at=ts_str,
                last_access=ts_str,
            )
            index_entries.append(entry)
            embedding_docs.append(embedding_text)

            # ── MemoryFile ──
            mf = MemoryFile(
                mem_id=mem_id,
                raw_content=raw_content,
                summary=summary,
                speaker="Aditi",
                category=category,
                sub_category=sub_category,
                tags=",".join(tags),
                source="aditi",
                fact_entries=json.dumps(fact_list, ensure_ascii=False),
                rel_entries=json.dumps(rel_list, ensure_ascii=False),
                session_summary="",
                event_description=context,
                original_date="",
                timestamp=ts_str,
                session_id=session_id_prefix,
                turn_index=i,
                importance=0.6,
            )
            memory_files.append(mf)

        # ── 双写 ──
        self._index.add(index_entries, documents=embedding_docs)
        self._persist.save(memory_files)

        logger.info(
            f"Import complete: {len(index_entries)} index entries, "
            f"{len(memory_files)} memory files"
        )
        return len(index_entries)

    @staticmethod
    def _classify_aditi(tags: List[str], context: str) -> str:
        """L1 分类"""
        tags_lower = [t.lower() for t in tags]
        knowledge_tags = {"math", "science", "history", "geography", "tech", "fact"}
        if any(t in knowledge_tags for t in tags_lower):
            return "知识"
        event_tags = {"bangalore-specific", "commute", "food", "travel", "work", "urban", "outing", "event"}
        if any(t in event_tags for t in tags_lower):
            return "事件"
        return "对话"

    @staticmethod
    def _subclassify_aditi(tags: List[str], context: str, user_msg: str) -> str:
        """L2 分类（简单规则）"""
        combined = " ".join(tags + [context, user_msg]).lower()

        if any(w in combined for w in ["work", "office", "boss", "deadline", "meeting"]):
            return "work"
        if any(w in combined for w in ["family", "mom", "dad", "grandma", "parent"]):
            return "family"
        if any(w in combined for w in ["gym", "walk", "exercise", "run", "leg day"]):
            return "health_fitness"
        if any(w in combined for w in ["food", "eat", "ice cream", "dosa", "coffee", "chai"]):
            return "food"
        if any(w in combined for w in ["metro", "bangalore", "commute", "traffic", "whitefield"]):
            return "bangalore_life"
        if any(w in combined for w in ["sad", "low energy", "anxious", "stressed", "comfort"]):
            return "emotional_support"
        if any(w in combined for w in ["football", "cricket", "liverpool", "match"]):
            return "sports"
        if any(w in combined for w in ["love", "crush", "date", "flirt", "clingy"]):
            return "relationship"
        return "daily_life"
