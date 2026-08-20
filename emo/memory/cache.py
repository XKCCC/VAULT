"""R6: ActivationCache — 两层激活缓存

设计：
  L1 Cache（轻量）: 每次检索到的 IndexEntry + 命中计数
  L2 Cache（详情）: 命中 ≥2 次后从 SQLite 拉取的完整 MemoryFile + 关联记忆

渐进式拉取策略：
  第 1 次命中 → 只用 summary 注入 prompt
  第 2 次命中 → 升级为 detail，拉取 MemoryFile，展示 facts/relations/session
  后续命中 → 直接用 L2 缓存

会话结束后全部清空。
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from .schema import IndexEntry, MemoryFile
from .index_store import IndexStore
from .persistent_store import PersistentStore

logger = logging.getLogger(__name__)


class _L1Entry:
    """L1 缓存条目"""
    __slots__ = ("entry", "hits", "first_hit_turn")

    def __init__(self, entry: IndexEntry, turn: int):
        self.entry = entry
        self.hits = 1
        self.first_hit_turn = turn


class _L2Entry:
    """L2 缓存条目"""
    __slots__ = ("memory_file", "related_entries", "related_loaded", "tokens")

    def __init__(self, memory_file: MemoryFile, tokens: int = 0):
        self.memory_file = memory_file
        self.related_entries: List[IndexEntry] = []
        self.related_loaded = False
        self.tokens = tokens


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数（英文 ~4 chars/token，中文 ~2 chars/token）"""
    return max(len(text) // 3, 1)


def _memory_file_tokens(mf: MemoryFile) -> int:
    """估算一条 MemoryFile 注入 prompt 时的 token 占用"""
    total = _estimate_tokens(mf.summary)
    if mf.fact_entries:
        total += _estimate_tokens(mf.fact_entries)
    return total


class ActivationCache:
    """两层激活缓存（合并原 WorkingMemoryManager 的 token 预算 paging）

    L2 同时受条目数（max_l2_size）与 token 预算（max_tokens）约束，
    超限均按 LRU 淘汰 —— 等价于 MemGPT 的 virtual memory paging。

    Args:
        index_store: 用于拉取 related_ids 对应的 IndexEntry
        persistent_store: 用于按 mem_id 拉取 MemoryFile
        max_l2_size: L2 缓存最大条目数（LRU 淘汰）
        upgrade_threshold: L1 命中多少次后升级到 L2
        max_tokens: L2 详情的 token 预算（原 WorkingMemoryManager 的职责）
    """

    def __init__(
        self,
        index_store: IndexStore,
        persistent_store: PersistentStore,
        max_l2_size: int = 10,
        upgrade_threshold: int = 2,
        max_tokens: int = 2000,
    ):
        self._index = index_store
        self._persist = persistent_store
        self._max_l2 = max_l2_size
        self._threshold = upgrade_threshold
        self._max_tokens = max_tokens

        self._l1: Dict[str, _L1Entry] = {}
        # L2 用 OrderedDict 实现 LRU：最近访问的移到末尾
        self._l2: OrderedDict[str, _L2Entry] = OrderedDict()

        self._current_turn = 0

    @property
    def current_tokens(self) -> int:
        """当前 L2 详情的 token 占用"""
        return sum(e.tokens for e in self._l2.values())

    def next_turn(self) -> None:
        """推进对话轮次（每次用户消息到达时调用）"""
        self._current_turn += 1

    def register(self, entries: List[Tuple[IndexEntry, float]]) -> None:
        """将检索到的记忆注册到 L1 cache

        Args:
            entries: Retriever 返回的 (IndexEntry, score) 列表
        """
        for entry, _score in entries:
            mid = entry.mem_id
            if mid in self._l1:
                self._l1[mid].hits += 1
                logger.debug(f"L1 hit: {mid}, hits={self._l1[mid].hits}")
            else:
                self._l1[mid] = _L1Entry(entry, self._current_turn)

    def should_upgrade(self, mem_id: str) -> bool:
        """判断某条记忆是否应该升级到 L2"""
        l1 = self._l1.get(mem_id)
        if l1 is None:
            return False
        # 已经在 L2 中就不需要再升级
        if mem_id in self._l2:
            return False
        return l1.hits >= self._threshold

    def upgrade(self, mem_id: str) -> Optional[MemoryFile]:
        """将记忆从 L1 升级到 L2（从 SQLite 拉取详情）

        Returns:
            MemoryFile 如果成功拉取，否则 None
        """
        mf = self._persist.get(mem_id)
        if mf is None:
            logger.warning(f"L2 upgrade failed: {mem_id} not in persistent store")
            return None

        l2_entry = _L2Entry(mf, tokens=_memory_file_tokens(mf))
        self._l2[mem_id] = l2_entry

        # LRU 淘汰：先按条目数，再按 token 预算（MemGPT 式 swap out）
        while len(self._l2) > self._max_l2:
            evicted_id, _ = self._l2.popitem(last=False)
            logger.debug(f"L2 evicted (size): {evicted_id}")
        while self.current_tokens > self._max_tokens and len(self._l2) > 1:
            evicted_id, _ = self._l2.popitem(last=False)
            logger.debug(f"L2 evicted (tokens): {evicted_id}")

        # 加载关联记忆
        self._load_related(mem_id)

        logger.info(f"L2 upgrade: {mem_id}")
        return mf

    def _load_related(self, mem_id: str) -> None:
        """加载关联记忆到 L2"""
        l2 = self._l2.get(mem_id)
        if l2 is None or l2.related_loaded:
            return

        l1 = self._l1.get(mem_id)
        if l1 is None:
            return

        related_ids = l1.entry.related_ids
        if not related_ids:
            l2.related_loaded = True
            return

        related = []
        for rid in related_ids:
            entry = self._index.get_by_id(rid)
            if entry:
                related.append(entry)

        l2.related_entries = related
        l2.related_loaded = True
        logger.debug(f"Loaded {len(related)} related memories for {mem_id}")

    def get_l1(self, mem_id: str) -> Optional[IndexEntry]:
        """获取 L1 中的 IndexEntry"""
        l1 = self._l1.get(mem_id)
        return l1.entry if l1 else None

    def get_l2(self, mem_id: str) -> Optional[_L2Entry]:
        """获取 L2 中的详情（同时刷新 LRU 顺序）"""
        if mem_id in self._l2:
            self._l2.move_to_end(mem_id)  # 标记为最近访问
            return self._l2[mem_id]
        return None

    def get_hits(self, mem_id: str) -> int:
        """获取某条记忆的命中次数"""
        l1 = self._l1.get(mem_id)
        return l1.hits if l1 else 0

    def is_detail(self, mem_id: str) -> bool:
        """某条记忆是否已有 L2 详情"""
        return mem_id in self._l2

    def clear(self) -> None:
        """清空所有缓存（会话结束时调用）"""
        self._l1.clear()
        self._l2.clear()
        self._current_turn = 0
        logger.debug("ActivationCache cleared")

    def stats(self) -> dict:
        """返回缓存统计"""
        return {
            "l1_size": len(self._l1),
            "l2_size": len(self._l2),
            "l2_tokens": self.current_tokens,
            "max_tokens": self._max_tokens,
            "current_turn": self._current_turn,
        }
