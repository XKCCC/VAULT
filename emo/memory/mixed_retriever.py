"""MixedRetriever — Buffer + Memory 混合检索

设计:
  - 路径 A: Buffer 检索（即时对话）→ top-3
  - 路径 B: Memory 检索（历史记忆）→ top-1
  - 合并: 3 + 1 = 4 条上下文注入 prompt
  - 比例 3:1，偏向即时对话上下文
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

from .buffer import ConversationBuffer
from .retriever import Retriever
from .schema import IndexEntry

logger = logging.getLogger(__name__)


class MixedRetriever:
    """Buffer + Memory 混合检索器

    Args:
        buffer: ConversationBuffer 实例
        memory_retriever: Retriever 实例（ChromaDB）
        buffer_top_k: Buffer 检索数量（默认 3）
        memory_top_k: Memory 检索数量（默认 1）
    """

    def __init__(
        self,
        buffer: ConversationBuffer,
        memory_retriever: Retriever,
        buffer_top_k: int = 3,
        memory_top_k: int = 1,
    ):
        self._buffer = buffer
        self._memory = memory_retriever
        self._buffer_k = buffer_top_k
        self._memory_k = memory_top_k

    def retrieve(
        self,
        query: str,
        category: Optional[str] = None,
        expand_graph: bool = False,
        graph_decay: float = 0.85,
        graph_extra: int = 5,
        temporal: bool = True,
        hierarchical: bool = False,
    ) -> Tuple[List[Tuple[Dict, float]], List[Tuple[IndexEntry, float]]]:
        """双路检索

        Args:
            query: 用户消息
            category: L1 分类过滤（仅 Memory 侧使用）
            expand_graph: Memory 侧是否沿 related_ids 图扩展
            graph_decay: 图传播衰减
            graph_extra: 图扩展最多追加的邻居数
            temporal: Memory 侧是否启用时间轴召回
            hierarchical: Memory 侧是否启用 L3 层级通道（先簇后点）

        Returns:
            (buffer_results, memory_results) 元组
            buffer_results: List of (turn_dict, score)
            memory_results: List of (IndexEntry, score)
        """
        # 路径 A: Buffer 检索
        buffer_results = self._buffer.search(query, top_k=self._buffer_k)

        # 路径 B: Memory 检索（可选图扩展 + 时间轴并集 + L3 层级通道）
        memory_results = self._memory.retrieve(
            query,
            category=category,
            top_k=self._memory_k,
            expand_graph=expand_graph,
            graph_decay=graph_decay,
            graph_extra=graph_extra,
            temporal=temporal,
            hierarchical=hierarchical,
        )

        logger.info(
            f"Mixed retrieve: buffer={len(buffer_results)}, "
            f"memory={len(memory_results)}"
        )

        return buffer_results, memory_results


def format_mixed_context(
    buffer_results: List[Tuple[Dict, float]],
    memory_results: List[Tuple[IndexEntry, float]],
) -> str:
    """将双路检索结果格式化为 prompt 上下文

    Args:
        buffer_results: Buffer 检索结果 (turn_dict, score)
        memory_results: Memory 检索结果 (IndexEntry, score)

    Returns:
        格式化后的上下文字符串
    """
    parts = []

    # Buffer 部分（即时对话）
    if buffer_results:
        buffer_lines = []
        for turn, score in buffer_results:
            speaker = turn["speaker"].capitalize()
            text = turn["text"]
            buffer_lines.append(f"{speaker}: {text}")

        parts.append("[Recent Conversation]\n" + "\n".join(buffer_lines))

    # Memory 部分（历史记忆）
    if memory_results:
        memory_lines = []
        for entry, score in memory_results:
            memory_lines.append(entry.summary)

        parts.append("[Historical Memory]\n" + "\n".join(f"- {m}" for m in memory_lines))

    return "\n\n".join(parts)
