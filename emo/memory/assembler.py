"""R7: 上下文组装器 — ContextAssembler

Prompt 格式设计（方案 A: 记忆融入 system prompt）:

  [System]
  {persona}

  Things you remember about this person:
  - {memory fact 1}
  - {memory fact 2}

  Recent conversation:
  You: {agent reply}
  User: {user message}

  [User]
  {user message}

设计原则:
  - 记忆作为 system prompt 的一部分自然融入
  - 不触发模型训练格式的输出（无 [Tags] [Context] 等标记）
  - 当前用直接拼接（方案 1），后续升级为做梦时预生成 natural_summary（方案 2）
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .schema import IndexEntry, MemoryFile
from .index_store import IndexStore
from .persistent_store import PersistentStore
from .cache import ActivationCache

logger = logging.getLogger(__name__)


class ContextAssembler:
    """上下文组装器"""

    def __init__(
        self,
        index_store: IndexStore,
        persistent_store: PersistentStore,
        max_memories: int = 5,
        max_buffer_turns: int = 4,
        max_related: int = 3,
    ):
        self._index = index_store
        self._persist = persistent_store
        self._max_memories = max_memories
        self._max_buffer_turns = max_buffer_turns
        self._max_related = max_related

    def assemble(
        self,
        user_message: str,
        persona: str = "",
        buffer_turns: Optional[List[Dict]] = None,
        memories: Optional[List[Tuple[IndexEntry, float]]] = None,
        cache: Optional[ActivationCache] = None,
        now: Optional[datetime] = None,
    ) -> str:
        """组装完整 prompt

        Args:
            user_message: 用户消息
            persona: 角色描述（如 "You are Aditi..."）
            buffer_turns: Buffer 检索到的 turns（list of turn dicts）
            memories: Memory 检索到的 (IndexEntry, score) 列表
            cache: ActivationCache（可选，启用渐进式详情拉取）
            now: 当前时刻（默认系统时间；测试可注入）

        Returns:
            完整 prompt 字符串
        """
        system_parts = []

        # ── 1. Persona ──
        if persona:
            system_parts.append(persona)

        # ── 2. 当前时间锚点（模型推理 "几分钟前" 类问题的 now 基准）──
        system_parts.append(self._current_time_line(now))

        # ── 3. 历史记忆: "Things you remember" ──
        memory_text = self._format_memories(memories, cache)
        if memory_text:
            system_parts.append(memory_text)

        # ── 4. 近期对话: "Recent conversation" ──
        buffer_text = self._format_buffer(buffer_turns)
        if buffer_text:
            system_parts.append(buffer_text)

        # ── 组装 ──
        prompt_parts = []
        if system_parts:
            prompt_parts.append("[System]\n" + "\n\n".join(system_parts))
        prompt_parts.append(f"[User]\n{user_message}")

        return "\n\n".join(prompt_parts)

    def assemble_simple(
        self,
        user_message: str,
        memories: List[Tuple[IndexEntry, float]],
    ) -> str:
        """简化版（向后兼容，无 buffer 无 cache）"""
        return self.assemble(user_message, memories=memories)

    def assemble_messages(
        self,
        user_message: str,
        persona: str = "",
        buffer_turns: Optional[List[Dict]] = None,
        memories: Optional[List[Tuple[IndexEntry, float]]] = None,
        cache: Optional[ActivationCache] = None,
        now: Optional[datetime] = None,
    ) -> List[Dict[str, str]]:
        """组装为 chat messages（配合 tokenizer.apply_chat_template）

        与 assemble() 同一逻辑，但返回结构化消息：
        记忆与近期对话融入 system 角色，用户消息在 user 角色——
        这是 SFT Scored 模型训练/验证时使用的推理路径。
        """
        system_parts = []

        if persona:
            system_parts.append(persona)

        system_parts.append(self._current_time_line(now))

        memory_text = self._format_memories(memories, cache)
        if memory_text:
            system_parts.append(memory_text)

        buffer_text = self._format_buffer(buffer_turns)
        if buffer_text:
            system_parts.append(buffer_text)

        messages = []
        if system_parts:
            messages.append({"role": "system", "content": "\n\n".join(system_parts)})
        messages.append({"role": "user", "content": user_message})
        return messages

    # ── 格式化方法 ──

    def _format_memories(
        self,
        memories: Optional[List[Tuple[IndexEntry, float]]],
        cache: Optional[ActivationCache],
    ) -> str:
        """格式化历史记忆为 "Things you remember" 段落"""
        if not memories:
            return ""

        # 如果有 cache，先注册并处理升级
        if cache:
            cache.register(memories)

        facts = []
        for entry, score in memories[: self._max_memories]:
            mid = entry.mem_id

            # 尝试从 L2 cache 获取详情
            detail_text = None
            date_src = (entry.event_timestamp, entry.original_date)
            if cache:
                if cache.should_upgrade(mid):
                    cache.upgrade(mid)
                if cache.is_detail(mid):
                    l2 = cache.get_l2(mid)
                    if l2:
                        detail_text = self._extract_facts_from_file(l2.memory_file)
                        date_src = (
                            l2.memory_file.event_timestamp,
                            l2.memory_file.original_date,
                        )
                        # 关联记忆摘要注入（做梦建的 related_ids 图谱在线化）
                        related_summaries = [
                            r.summary for r in l2.related_entries[: self._max_related]
                            if r.summary
                        ]
                        if related_summaries:
                            detail_text = (
                                f"{detail_text} "
                                f"(Related: {'; '.join(related_summaries)})"
                            )

            # 如果 L2 没有，从 IndexEntry 的 summary 提取（L1 轻量路径，不查 SQLite）
            if not detail_text:
                detail_text = self._extract_facts_from_entry(entry)

            if detail_text:
                # 时间轴日期前缀：让模型对 when 类问题有据可依
                facts.append(self._date_prefix(*date_src) + detail_text)
                # 效用埋点：记忆被实际注入 prompt，记为一次有效使用
                self._index.record_utility(mid)

        if not facts:
            return ""

        return "Things you remember about this person:\n" + "\n".join(
            f"- {f}" for f in facts
        )

    def _extract_facts_from_entry(self, entry: IndexEntry) -> str:
        """从 IndexEntry 提取事实描述（L1 轻量路径）

        渐进式拉取设计：L1 只使用索引层的 summary，不查 SQLite；
        fact_entries 等详情只在 L2 升级（hits≥2）后通过
        _extract_facts_from_file 提供。
        """
        if entry.summary:
            return entry.summary.strip()
        return ""

    @staticmethod
    def _current_time_line(now: Optional[datetime]) -> str:
        """当前时间行：给模型推理相对时间问题（"几分钟前"）提供 now 基准"""
        n = now or datetime.now()
        return f"Current time: {n.strftime('%Y-%m-%d %H:%M (%A)')}"

    @staticmethod
    def _date_prefix(event_timestamp: str, original_date: str = "") -> str:
        """时间轴日期前缀 "[YYYY-MM-DD] " 或 "[YYYY-MM-DD HH:MM] "（分钟级）

        优先事件时间（做梦 LLM 解析的绝对日期），回退原始日期。
        时刻为 00:00 通常表示"只知日期不知时刻"，此时只显示日期，不伪造精度。
        original_date 可能是 "8 May, 2023" 这类非 ISO 格式，原样使用。
        """
        ts = (event_timestamp or "").strip()
        m = re.match(r"(\d{4}-\d{2}-\d{2})(?:[ T](\d{2}):(\d{2}))?", ts)
        if m:
            if m.group(2) and (m.group(2), m.group(3)) != ("00", "00"):
                return f"[{m.group(1)} {m.group(2)}:{m.group(3)}] "
            return f"[{m.group(1)}] "
        od = (original_date or "").strip()
        if od:
            return f"[{od[:20]}] "
        return ""

    def _extract_facts_from_file(self, mf: MemoryFile) -> str:
        """从 MemoryFile 提取事实描述（detail 模式）

        优先使用 fact_entries（结构化事实），回退到 summary
        """
        facts = mf.get_fact_entries()
        if facts:
            # 取前 2 条 fact，用分号连接
            return "; ".join(facts[:2])

        # 回退: 用 summary
        if mf.summary:
            return mf.summary

        return ""

    def _format_buffer(self, buffer_turns: Optional[List[Dict]]) -> str:
        """格式化近期对话"""
        if not buffer_turns:
            return ""

        # 取最近 N 轮
        recent = buffer_turns[-self._max_buffer_turns:]

        lines = []
        for turn in recent:
            speaker = turn.get("speaker", "unknown")
            text = turn.get("text", "")
            # 映射 speaker 名称
            if speaker == "user":
                lines.append(f"User: {text}")
            elif speaker == "agent":
                lines.append(f"You: {text}")
            else:
                lines.append(f"{speaker}: {text}")

        return "Recent conversation:\n" + "\n".join(lines)
