"""EMO Memory System — Phase 3.1 最小闭环

模块:
    schema:           M1 — 数据模型 (IndexEntry + MemoryFile)
    index_store:      M5 — 热索引层 (ChromaDB)
    persistent_store: M6 — 持久存储层 (SQLite)
    importer:         M8 — 数据集导入
    retriever:        R1+R2 — 编码 + ANN 检索
    assembler:        R7 — Prompt 组装
"""

import os
# HuggingFace 镜像（必须在 sentence-transformers/chromadb 导入之前设置）
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from .schema import IndexEntry, MemoryFile
from .index_store import IndexStore
from .persistent_store import PersistentStore
from .importer import DatasetImporter
from .retriever import Retriever
from .assembler import ContextAssembler
from .cache import ActivationCache
from .buffer import ConversationBuffer
from .mixed_retriever import MixedRetriever, format_mixed_context
from .dreamer import DreamOrchestrator
from .tag_classifier import TagClassifier
from .unknown_topics import UnknownTopics
from .temporal import resolve_temporal_range, weekday_name

__all__ = [
    "IndexEntry",
    "MemoryFile",
    "IndexStore",
    "PersistentStore",
    "DatasetImporter",
    "Retriever",
    "ContextAssembler",
    "ActivationCache",
    "ConversationBuffer",
    "MixedRetriever",
    "format_mixed_context",
    "DreamOrchestrator",
    "TagClassifier",
    "UnknownTopics",
    "resolve_temporal_range",
    "weekday_name",
]
