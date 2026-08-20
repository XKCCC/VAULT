"""M5: 热索引层 — 基于 ChromaDB 的向量存储

职责:
    - 存储 IndexEntry（轻量索引条目）
    - ANN 检索（按语义相似度 + weight 过滤）
    - CRUD 操作

使用本地 embedding 模型（all-MiniLM-L6-v2, 384维），
存放在 emo/models/all-MiniLM-L6-v2/，无需联网。
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from .schema import IndexEntry

logger = logging.getLogger(__name__)

# 默认本地模型路径
_DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parent.parent / "models" / "all-MiniLM-L6-v2")


class IndexStore:
    """热索引层 — ChromaDB 向量存储"""

    def __init__(
        self,
        collection_name: str = "emo_memories",
        persist_dir: Optional[str] = None,
        embedding_model_path: Optional[str] = None,
        embedding_fn=None,
    ):
        """
        Args:
            collection_name: ChromaDB collection 名称
            persist_dir: 持久化目录路径。None 则使用内存模式（测试用）
            embedding_model_path: 本地 embedding 模型路径。
                默认使用 emo/models/all-MiniLM-L6-v2
            embedding_fn: 可选的共享 embedding function（如
                SentenceTransformerEmbeddingFunction 实例）。批量评测建多个库时
                传入同一个实例，避免每个库重复加载模型（CPU 上每次约 2 分钟）。
                护栏打标仍按 embedding_model_path 记录。
        """
        model_path = embedding_model_path or _DEFAULT_MODEL_PATH

        if persist_dir:
            self._client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            self._client = chromadb.EphemeralClient(
                settings=Settings(anonymized_telemetry=False),
            )

        # 使用本地 embedding 模型（不走 HuggingFace 下载）
        if embedding_fn is not None:
            self._embedding_fn = embedding_fn
            logger.info(f"使用共享 embedding function（模型: {model_path}）")
        elif os.path.isdir(model_path):
            self._embedding_fn = SentenceTransformerEmbeddingFunction(
                model_name=model_path,
            )
            logger.info(f"使用本地 embedding 模型: {model_path}")
        else:
            self._embedding_fn = None
            logger.warning(
                f"本地模型不存在: {model_path}，将使用 ChromaDB 默认（可能需要联网）"
            )

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # 使用余弦相似度
            embedding_function=self._embedding_fn,
        )

        # embedding 模型一致性护栏：新库打标（创建时），有标且不匹配则报错。
        # 无标旧库（legacy）不动——避免错误补标。
        model_tag = os.path.basename(model_path.rstrip("/"))
        meta = dict(self._collection.metadata or {})
        stamped = meta.get("embed_model")
        if stamped and stamped != model_tag:
            raise ValueError(
                f"索引库 embedding 不匹配：该库由 {stamped} 构建，当前加载 {model_tag}。"
                f"请用原模型打开或重建索引（否则检索结果是垃圾向量）。"
            )
        if not stamped and self._collection.count() == 0:
            try:
                meta["embed_model"] = model_tag
                self._collection.modify(metadata=meta)
            except Exception:
                pass

        # 读路径不写库：访问/效用计数先累积在内存，做梦时 flush_stats 批量回写
        self._pending_access: Dict[str, int] = {}
        self._pending_utility: Dict[str, int] = {}

        # query 编码缓存（LRU 256）：高频/重复 query 免重编码
        from collections import OrderedDict
        self._qcache: OrderedDict[str, list] = OrderedDict()
        self._qcache_max = 256

        print(f"  [IndexStore] collection ready, count={self._collection.count()}")
        logger.info(
            f"IndexStore initialized: collection='{collection_name}', "
            f"count={self._collection.count()}"
        )

    def _query_embedding(self, query: str) -> list:
        """query 编码（带 LRU 缓存）"""
        emb = self._qcache.get(query)
        if emb is not None:
            self._qcache.move_to_end(query)
            return emb
        emb = self._embedding_fn([query])[0]
        self._qcache[query] = emb
        self._qcache.move_to_end(query)
        while len(self._qcache) > self._qcache_max:
            self._qcache.popitem(last=False)
        return emb

    def get_embeddings(self, mem_ids: List[str]) -> Dict[str, list]:
        """按 mem_id 取 embedding 向量（MMR 去重的成对相似度计算用）"""
        if not mem_ids:
            return {}
        results = self._collection.get(ids=mem_ids, include=["embeddings"])
        out = {}
        embs = results.get("embeddings")
        if embs is None:
            return out
        for i, mid in enumerate(results["ids"]):
            out[mid] = embs[i]
        return out

    # ── 写入 ──

    def add(self, entries: List[IndexEntry], documents: Optional[List[str]] = None) -> None:
        """批量添加索引条目

        Args:
            entries: IndexEntry 列表
            documents: ChromaDB 用于生成 embedding 的文本列表。
                       如果为 None，默认使用 summary。
        """
        if not entries:
            return

        if documents is None:
            # 优先使用 embedding_text，没有则用 summary + tags
            documents = [
                e.embedding_text if e.embedding_text
                else f"{e.summary} {' '.join(e.tags)}"
                for e in entries
            ]

        ids = [e.mem_id for e in entries]
        metadatas = [e.to_metadata() for e in entries]

        # 分块写入：chroma rust 后端单次 add 上限 5461 条
        # （LifeBench 单用户 1.5 万条，一次性写入会被拒）
        _BATCH = 2000
        for start in range(0, len(entries), _BATCH):
            end = start + _BATCH
            self._collection.add(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )
        logger.info(f"Added {len(entries)} entries to index (total: {self._collection.count()})")

    # ── 检索 ──

    def search(
        self,
        query: str,
        top_k: int = 10,
        category: Optional[str] = None,
        min_weight: float = 0.0,
        include_superseded: bool = False,
    ) -> List[Tuple[IndexEntry, float]]:
        """语义检索 top-K

        Args:
            query: 查询文本
            top_k: 返回数量
            category: L1 分类过滤（None 则不过滤）
            min_weight: 最低权重阈值（低于此值的"沉底"记忆不返回）

        Returns:
            List of (IndexEntry, similarity_score) tuples，按分数降序
        """
        where_filter = None
        if category and min_weight > 0:
            where_filter = {
                "$and": [
                    {"category": {"$eq": category}},
                    {"base_weight": {"$gte": min_weight}},
                ]
            }
        elif category:
            where_filter = {"category": {"$eq": category}}
        elif min_weight > 0:
            where_filter = {"base_weight": {"$gte": min_weight}}

        # ChromaDB 需要多请求一些：后面还要按 current_weight / superseded 过滤
        fetch_k = top_k * 3 if min_weight > 0 else top_k * 2

        results = self._collection.query(
            query_embeddings=[self._query_embedding(query)],
            n_results=min(fetch_k, max(self._collection.count(), 1)),
            where=where_filter if where_filter else None,
        )

        if not results["ids"] or not results["ids"][0]:
            logger.debug(f"ChromaDB returned 0 results (where={where_filter})")
            return []

        logger.debug(
            f"ChromaDB returned {len(results['ids'][0])} candidates, "
            f"distances: {results.get('distances', [['N/A']])[0][:3]}..."
        )

        candidates = []
        now = datetime.now()
        for i, mem_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i] if results.get("distances") else 0
            similarity = 1.0 - distance  # cosine distance → similarity

            entry = IndexEntry.from_metadata(meta)

            # 被取代的记忆默认不可检索（保留在库中供时间轴/审计追溯）
            if entry.superseded_by and not include_superseded:
                continue

            # 按 current_weight 过滤
            cw = entry.current_weight(now)
            if cw < min_weight:
                logger.debug(f"  {mem_id}: weight {cw:.4f} < {min_weight}, skipped")
                continue

            candidates.append((entry, similarity))

        # 按相似度降序，取 top_k
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    # ── 更新 ──

    def update_entry(self, entry: IndexEntry) -> None:
        """更新已有条目的 metadata（不重新生成 embedding）"""
        self._collection.update(
            ids=[entry.mem_id],
            metadatas=[entry.to_metadata()],
        )

    def rescore(self, query: str, mem_ids: List[str]) -> Dict[str, float]:
        """对指定 mem_id 计算与 query 的余弦相似度（图扩展候选的重打分）

        Args:
            query: 查询文本
            mem_ids: 候选记忆 ID 列表

        Returns:
            {mem_id: cosine_similarity}
        """
        if not mem_ids or self._embedding_fn is None:
            return {}

        import numpy as np

        results = self._collection.get(ids=mem_ids, include=["embeddings"])
        if not results["ids"]:
            return {}

        q = np.array(self._query_embedding(query), dtype=np.float32)
        q_norm = float(np.linalg.norm(q))

        scores: Dict[str, float] = {}
        embeddings = results.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            return {}
        for i, mem_id in enumerate(results["ids"]):
            v = np.array(embeddings[i], dtype=np.float32)
            denom = q_norm * float(np.linalg.norm(v))
            scores[mem_id] = float(np.dot(q, v) / denom) if denom > 0 else 0.0
        return scores

    def record_access(self, mem_id: str) -> None:
        """读路径只记内存计数，不写库（做梦时 flush_stats 批量回写）"""
        self._pending_access[mem_id] = self._pending_access.get(mem_id, 0) + 1

    def record_utility(self, mem_id: str) -> None:
        """记录一次实际使用（记忆被注入 prompt），内存累积，flush 时回写"""
        self._pending_utility[mem_id] = self._pending_utility.get(mem_id, 0) + 1

    def flush_stats(self) -> Dict[str, int]:
        """批量回写访问/效用计数到 ChromaDB

        频率信号只进 access_count / utility_count，
        不再调整 base_weight（衰减公式中已有 access_boost 项，
        之前的 +0.05 永久膨胀会污染衰减计算）。

        Returns:
            {"access": 回写条数, "utility": 回写条数}
        """
        if not self._pending_access and not self._pending_utility:
            return {"access": 0, "utility": 0}

        all_ids = list(set(self._pending_access) | set(self._pending_utility))
        results = self._collection.get(ids=all_ids)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        n_access = 0
        n_utility = 0
        for i, mem_id in enumerate(results["ids"]):
            meta = results["metadatas"][i]
            entry = IndexEntry.from_metadata(meta)

            hits = self._pending_access.get(mem_id, 0)
            uses = self._pending_utility.get(mem_id, 0)
            if hits:
                entry.access_count += hits
                entry.last_access = now_str
                n_access += 1
            if uses:
                entry.utility_count += uses
                n_utility += 1
            if hits or uses:
                self.update_entry(entry)

        self._pending_access.clear()
        self._pending_utility.clear()
        logger.info(f"Flushed stats: {n_access} access, {n_utility} utility")
        return {"access": n_access, "utility": n_utility}

    def update_access(self, mem_id: str) -> None:
        """已废弃：保留兼容旧调用，等价于 record_access（不再立即写库）"""
        self.record_access(mem_id)

    # ── 删除 ──

    def delete(self, mem_ids: List[str]) -> None:
        """从热索引层移除（持久存储层不受影响）"""
        if mem_ids:
            self._collection.delete(ids=mem_ids)
            logger.info(f"Deleted {len(mem_ids)} entries from index")

    # ── 工具方法 ──

    def count(self) -> int:
        return self._collection.count()

    def get_by_id(self, mem_id: str) -> Optional[IndexEntry]:
        """按 ID 获取单条索引"""
        results = self._collection.get(ids=[mem_id])
        if not results["ids"]:
            return None
        return IndexEntry.from_metadata(results["metadatas"][0])

    def get_all_ids(self) -> List[str]:
        """获取所有索引 ID"""
        results = self._collection.get()
        return results["ids"] if results["ids"] else []

    def get_all_entries(self) -> List[IndexEntry]:
        """获取所有 IndexEntry（用于做梦时的图分析）"""
        results = self._collection.get()
        if not results["ids"]:
            return []
        return [
            IndexEntry.from_metadata(meta)
            for meta in results["metadatas"]
        ]
