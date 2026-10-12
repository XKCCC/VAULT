"""M1: 数据模型定义

IndexEntry — 热索引层条目（轻量，存 ChromaDB）
MemoryFile — 持久存储层文件（详细，存 SQLite）

设计原则:
  - 检索永远基于纯语义相似度，weight/时间/频率不参与排序
  - weight 衰减只在做梦时用于存储管理（压缩/融合/淘汰）
  - 持久存储层永不删除，按 mem_id 随时可拉取详情
"""

from __future__ import annotations

import uuid
import math
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class IndexEntry:
    """热索引层条目 — 用于 ANN 快速检索

    每条记忆在向量库中的轻量表示。
    检索基于纯语义相似度（cosine similarity on embedding_text）。
    """

    # ── 核心内容 ──
    mem_id: str                         # 唯一标识
    summary: str                        # 语义摘要（用于 prompt 注入 + 展示）
    embedding_text: str = ""            # embedding 专用文本（可比 summary 更长更丰富）
    category: str = ""                  # L1: 知识/事件/对话
    sub_category: str = ""              # L2: 工作/亲密/开心/社交...
    tags: List[str] = field(default_factory=list)
    source: str = "interaction"         # locomo / aditi / interaction
    speaker: str = ""                   # 说话人

    # ── 时间 ──
    original_date: str = ""             # 原始日期（如 "8 May, 2023"）
    event_timestamp: str = ""           # 事件发生时间（做梦时 LLM 提取，YYYY-MM-DD HH:MM:SS）
                                        # 用于时间轴对齐与 prompt 日期前缀
    created_at: str = field(default_factory=_now_str)
    last_access: str = field(default_factory=_now_str)

    # ── 做梦阶段使用（不影响检索排序）──
    base_weight: float = 0.5            # 初始重要性（做梦时调整）
    access_count: int = 0               # 被检索总次数
    utility_count: float = 0.0          # 检索后帮助成功的次数
    related_ids: List[str] = field(default_factory=list)  # 关联记忆（做梦时建立）
    superseded_by: str = ""             # 被哪条新记忆取代；非空则检索排除

    def current_weight(self, now: Optional[datetime] = None) -> float:
        """计算当前有效权重（Ebbinghaus 衰减，不归零）

        注意：此方法仅在做梦阶段用于存储管理，不影响检索排序。
        """
        if now is None:
            now = datetime.now()
        created = datetime.strptime(self.created_at, "%Y-%m-%d %H:%M:%S")
        days_idle = max((now - created).days, 0)

        beta = 0.8 if self.base_weight >= 0.7 else 1.2
        lambda_i = 0.1 * math.exp(-0.5 * self.base_weight)
        decay = math.exp(-lambda_i * (days_idle ** beta))
        access_boost = self.access_count / (1 + self.access_count)

        w = self.base_weight * decay * (0.5 + 0.5 * access_boost)
        return max(w, 0.01)

    def to_metadata(self) -> dict:
        """转为 ChromaDB metadata dict（只支持 str/int/float/bool）"""
        return {
            "mem_id": self.mem_id,
            "summary": self.summary,
            "embedding_text": self.embedding_text,
            "category": self.category,
            "sub_category": self.sub_category,
            "tags": ",".join(self.tags),
            "source": self.source,
            "speaker": self.speaker,
            "original_date": self.original_date,
            "event_timestamp": self.event_timestamp,
            "related_ids": ",".join(self.related_ids),
            "superseded_by": self.superseded_by,
            "base_weight": self.base_weight,
            "access_count": self.access_count,
            "utility_count": self.utility_count,
            "created_at": self.created_at,
            "last_access": self.last_access,
        }

    @classmethod
    def from_metadata(cls, meta: dict) -> "IndexEntry":
        """从 ChromaDB metadata dict 还原"""
        return cls(
            mem_id=meta["mem_id"],
            summary=meta["summary"],
            embedding_text=meta.get("embedding_text", ""),
            category=meta.get("category", ""),
            sub_category=meta.get("sub_category", ""),
            tags=meta.get("tags", "").split(",") if meta.get("tags") else [],
            source=meta.get("source", "interaction"),
            speaker=meta.get("speaker", ""),
            original_date=meta.get("original_date", ""),
            event_timestamp=meta.get("event_timestamp", ""),
            related_ids=meta.get("related_ids", "").split(",") if meta.get("related_ids") else [],
            superseded_by=meta.get("superseded_by", ""),
            base_weight=meta.get("base_weight", 0.5),
            access_count=meta.get("access_count", 0),
            utility_count=meta.get("utility_count", 0.0),
            created_at=meta.get("created_at", _now_str()),
            last_access=meta.get("last_access", _now_str()),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MemoryFile:
    """持久存储层完整记忆文件 — 存 SQLite，永不删除

    包含原始内容、双视角条目（事实+关系）、完整的会话上下文。
    即使热索引层的 IndexEntry 已经"沉底"，这里的数据始终可以通过 mem_id 拉取。
    """

    mem_id: str

    # ── 核心内容 ──
    raw_content: str                    # 完整原始内容（Skill 按 ID 拉取时返回）
    summary: str = ""                   # 语义摘要
    speaker: str = ""                   # 说话人
    category: str = ""                  # L1 分类
    sub_category: str = ""              # L2 分类
    tags: str = ""                      # 逗号分隔标签
    source: str = "interaction"         # 数据来源

    # ── 双视角条目（StructMem 式）──
    fact_entries: str = ""              # 事实条目 JSON list，如 ["Caroline attended..."]
    rel_entries: str = ""               # 关系条目 JSON list，如 ["Caroline shared with Melanie"]

    # ── A-MEM 式属性 ──
    keywords: str = ""                  # 关键词 JSON list（LLM 提取的核心概念）
    context_description: str = ""       # 上下文描述（LLM 生成的语义理解）
    box_id: str = ""                    # Box 聚类 ID（共享相似 context_description 的记忆归为同一 Box）

    # ── 上下文信息 ──
    session_summary: str = ""           # 所属 session 的整体摘要
    event_description: str = ""         # 事件描述

    # ── 时间与定位 ──
    original_date: str = ""             # 原始日期字符串（如 "8 May, 2023"）
    timestamp: str = field(default_factory=_now_str)  # 导入时间
    session_id: str = ""                # 所属会话 ID
    turn_index: int = 0                 # 在会话中的轮次位置

    # ── 时序信息（LLM 提取）──
    temporal_context: str = ""          # 时序上下文（如 "2023年夏天，大约6-7月"、"大学期间"）
    event_sequence: str = ""            # 事件时间线位置（如 "在换工作之后，搬家之前"）
    event_timestamp: str = ""           # 事件发生时间（用于衰减计算，格式 YYYY-MM-DD HH:MM:SS）
                                        # 如无明确时间，由 LLM 推断或 fallback 到 timestamp

    # ── 做梦阶段使用 ──
    importance: float = 0.5             # 初始重要性（做梦时调整）
    status: str = "raw"                 # raw / dreamed / settled
    label1: str = ""                    # 大类标签（对话/知识/看法/事实/能力）
    label2: str = ""                    # 次级标签（兴趣/天文/地理/历史/食物...）
    related_ids: str = ""               # 关联记忆 ID（逗号分隔，与 ChromaDB 同步）
    superseded_by: str = ""             # 被哪条新记忆取代（mem_id）。非空即失效——
                                        # 检索默认排除，但保留供时间轴/审计追溯

    def get_fact_entries(self) -> List[str]:
        """反序列化 fact_entries"""
        if not self.fact_entries:
            return []
        try:
            return json.loads(self.fact_entries)
        except json.JSONDecodeError:
            return []

    def get_rel_entries(self) -> List[str]:
        """反序列化 rel_entries"""
        if not self.rel_entries:
            return []
        try:
            return json.loads(self.rel_entries)
        except json.JSONDecodeError:
            return []

    def get_keywords(self) -> List[str]:
        """反序列化 keywords"""
        if not self.keywords:
            return []
        try:
            return json.loads(self.keywords)
        except json.JSONDecodeError:
            return []

    def to_dict(self) -> dict:
        return asdict(self)
