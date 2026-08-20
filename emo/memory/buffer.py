"""ConversationBuffer — 对话工作记忆缓冲区

设计:
  - 内存中保存当前会话的所有 turn
  - Faiss (语义) + BM25 (关键词) 双路召回
  - token 上限管理，接近上限时触发后台压缩
  - 压缩: 低分 turn 持久化到 SQLite (status="raw")，从 buffer 移除

每个 turn 的结构:
  {
    "turn_id": int,
    "speaker": "user" / "agent",
    "text": str,
    "timestamp": str,
    "access_count": int,
    "embedding": np.ndarray,     # Faiss 向量
  }
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _token_estimate(text: str) -> int:
    """粗略估算 token 数（英文 ~4 chars/token，中文 ~2 chars/token）"""
    return max(len(text) // 3, 1)


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class ConversationBuffer:
    """对话工作记忆缓冲区

    Args:
        embedding_model: sentence-transformers 模型（用于生成 turn embedding）
        max_tokens: buffer 的 token 上限（默认 6000）
        compress_ratio: 触发压缩的容量比例（默认 0.8 = 80%）
        compress_callback: 压缩时的回调函数，接收要持久化的 turns 列表
    """

    def __init__(
        self,
        embedding_model,
        max_tokens: int = 6000,
        compress_ratio: float = 0.8,
        compress_callback: Optional[Callable] = None,
    ):
        import faiss

        self._model = embedding_model
        self._max_tokens = max_tokens
        self._compress_ratio = compress_ratio
        self._compress_callback = compress_callback

        # turn 数据
        self._turns: List[Dict] = []
        self._total_tokens = 0
        self._next_id = 0

        # Faiss 索引（延迟初始化，等第一条 turn 来了才知道维度）
        self._faiss_index: Optional[faiss.IndexFlatIP] = None
        self._dim = 0

        # BM25 索引（每次 search 时重建，因为 buffer 是动态的）
        self._bm25_corpus: List[List[str]] = []

        # 结构锁：_turns/_faiss_index/_bm25_corpus 三者必须一致，
        # search 与后台压缩并发时不得读到换了一半的状态
        self._lock = threading.RLock()

        # 压缩锁（防止并发压缩）
        self._compressing = False

    def add_turn(self, speaker: str, text: str) -> int:
        """添加一条对话 turn

        Args:
            speaker: "user" 或 "agent"
            text: 对话文本

        Returns:
            turn_id
        """
        # embedding 计算在锁外（慢操作，不持锁）
        embedding = self._model.encode(text, normalize_embeddings=True)

        with self._lock:
            turn_id = self._next_id
            self._next_id += 1

            # 初始化 Faiss 索引
            if self._faiss_index is None:
                import faiss
                self._dim = len(embedding)
                self._faiss_index = faiss.IndexFlatIP(self._dim)

            turn = {
                "turn_id": turn_id,
                "speaker": speaker,
                "text": text,
                "timestamp": _now_str(),
                "access_count": 0,
                "embedding": embedding,
            }

            self._turns.append(turn)
            self._faiss_index.add(embedding.reshape(1, -1).astype(np.float32))
            self._total_tokens += _token_estimate(text)

            # BM25 分词（简单空格分词）
            self._bm25_corpus.append(text.lower().split())

            logger.debug(
                f"Buffer add turn {turn_id}: [{speaker}] {text[:50]}... "
                f"(tokens: {self._total_tokens}/{self._max_tokens})"
            )

            # 检查是否需要压缩
            if self._total_tokens > self._max_tokens * self._compress_ratio:
                self._trigger_compression()

            return turn_id

    def search(
        self, query: str, top_k: int = 5
    ) -> List[Tuple[Dict, float]]:
        """双路检索: Faiss (语义) + BM25 (关键词)

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            List of (turn_dict, combined_score) tuples
        """
        # query embedding 在锁外（慢操作，不持锁）
        query_emb = self._model.encode(query, normalize_embeddings=True)

        # 整个检索过程持锁：_turns/_faiss_index/_bm25_corpus 必须读到一致快照，
        # 否则后台压缩换列表的瞬间会出现 idx 越界（IndexError）
        with self._lock:
            if not self._turns:
                return []

            # ── Faiss 语义检索 ──
            query_vec = query_emb.reshape(1, -1).astype(np.float32)
            faiss_k = min(top_k * 2, len(self._turns))
            faiss_scores, faiss_indices = self._faiss_index.search(query_vec, faiss_k)

            faiss_results = {}
            for score, idx in zip(faiss_scores[0], faiss_indices[0]):
                if idx >= 0 and idx < len(self._turns):
                    faiss_results[idx] = float(score)

            # ── BM25 关键词检索 ──
            bm25_results = {}
            if self._bm25_corpus:
                from rank_bm25 import BM25Okapi
                bm25 = BM25Okapi(self._bm25_corpus)
                query_tokens = query.lower().split()
                bm25_scores = bm25.get_scores(query_tokens)

                # 取 top-K
                bm25_top_idx = np.argsort(bm25_scores)[::-1][:faiss_k]
                for idx in bm25_top_idx:
                    if bm25_scores[idx] > 0:
                        bm25_results[int(idx)] = float(bm25_scores[idx])

            # ── 融合分数（RRF: Reciprocal Rank Fusion）──
            combined = {}
            for idx in set(faiss_results.keys()) | set(bm25_results.keys()):
                # RRF: score = sum(1 / (k + rank))
                score = 0.0
                if idx in faiss_results:
                    # faiss_scores 已按降序排列，用排名
                    rank = sorted(faiss_results.keys(),
                                  key=lambda x: faiss_results[x], reverse=True).index(idx)
                    score += 1.0 / (60 + rank)  # k=60 是 RRF 标准参数
                if idx in bm25_results:
                    rank = sorted(bm25_results.keys(),
                                  key=lambda x: bm25_results[x], reverse=True).index(idx)
                    score += 1.0 / (60 + rank)
                combined[idx] = score

            # 排序取 top-K
            sorted_indices = sorted(combined.keys(), key=lambda x: combined[x], reverse=True)
            results = []
            for idx in sorted_indices[:top_k]:
                turn = self._turns[idx]
                turn["access_count"] += 1  # 记录访问
                results.append((turn, combined[idx]))

            return results

    def _trigger_compression(self) -> None:
        """触发后台压缩（异步）"""
        if self._compressing:
            return

        self._compressing = True
        thread = threading.Thread(target=self._do_compression, daemon=True)
        thread.start()

    def _do_compression(self) -> None:
        """执行压缩: 低分 turn 移出 buffer

        锁策略：打分/持久化回调在锁外用快照做（慢操作不阻塞 search/add），
        只有最后交换结构时持锁。压缩期间新 add 的 turn 在快照之外，
        必须无条件保留——否则会被静默丢弃且从未持久化（数据丢失）。
        """
        try:
            logger.info(
                f"Buffer compression triggered: "
                f"{self._total_tokens}/{self._max_tokens} tokens"
            )

            # 快照（持锁，保证打分的 turn 集合自洽）
            with self._lock:
                snapshot = list(self._turns)

            # 给每个 turn 打分: access_count * 0.6 + recency * 0.4
            now = datetime.now()
            scored = []
            for i, turn in enumerate(snapshot):
                ts = datetime.strptime(turn["timestamp"], "%Y-%m-%d %H:%M:%S")
                age_minutes = max((now - ts).total_seconds() / 60, 0)
                recency = 1.0 / (1.0 + 0.01 * age_minutes)
                score = turn["access_count"] * 0.6 + recency * 0.4
                scored.append((i, score, turn))

            # 按分数排序，移除底部 50%
            scored.sort(key=lambda x: x[1])
            n_remove = len(scored) // 2
            to_remove = scored[:n_remove]
            to_keep = scored[n_remove:]

            # 收集要持久化的 turns
            persist_turns = []
            for _, _, turn in to_remove:
                persist_turns.append({
                    "speaker": turn["speaker"],
                    "text": turn["text"],
                    "timestamp": turn["timestamp"],
                    "turn_id": turn["turn_id"],
                })

            # 回调: 持久化到 SQLite（锁外，慢 IO 不阻塞检索）
            if self._compress_callback and persist_turns:
                self._compress_callback(persist_turns)
                logger.info(f"Compressed {len(persist_turns)} turns to persistent store")

            # 交换结构（持锁，与 search/add 互斥）
            with self._lock:
                # 压缩期间新进来的 turn 原样保留在尾部
                new_since_snapshot = self._turns[len(snapshot):]
                keep_turns = [turn for _, _, turn in sorted(to_keep, key=lambda x: x[0])]
                self._rebuild_buffer(keep_turns + list(new_since_snapshot))

            logger.info(
                f"Compression done: kept {len(self._turns)} turns, "
                f"{self._total_tokens}/{self._max_tokens} tokens"
            )

        finally:
            self._compressing = False

    def _rebuild_buffer(self, new_turns: List[Dict]) -> None:
        """重建 buffer（调用方须已持锁）"""
        import faiss

        # 重建 Faiss
        self._faiss_index = faiss.IndexFlatIP(self._dim)
        for turn in new_turns:
            self._faiss_index.add(turn["embedding"].reshape(1, -1).astype(np.float32))

        # 重建 BM25
        self._bm25_corpus = [t["text"].lower().split() for t in new_turns]

        # 更新状态
        self._turns = list(new_turns)
        self._total_tokens = sum(_token_estimate(t["text"]) for t in new_turns)

    def get_all_turns(self) -> List[Dict]:
        """获取 buffer 中所有 turns"""
        with self._lock:
            return self._turns.copy()

    def clear(self) -> None:
        """清空 buffer"""
        import faiss
        with self._lock:
            self._turns.clear()
            self._bm25_corpus.clear()
            self._total_tokens = 0
            self._next_id = 0
            if self._dim > 0:
                self._faiss_index = faiss.IndexFlatIP(self._dim)

    def flush(self) -> int:
        """把 buffer 中所有剩余 turn 全部持久化并清空

        压缩只淘汰低分 turn，高分 turn（被反复检索命中的）会一直留在
        内存；做梦前/会话结束时不 flush，这些 turn 永远不会进入长程记忆。

        Returns:
            持久化的 turn 数
        """
        # 等进行中的后台压缩结束，避免与压缩回调交错写
        for _ in range(100):
            if not self._compressing:
                break
            time.sleep(0.1)

        # 持锁段：收集 turn + 清空必须原子——否则 flush 期间新 add 的 turn
        # 会被 clear 抹掉且从未持久化（与服务端 /chat 并发时的真实路径）
        with self._lock:
            if not self._turns:
                return 0

            persist_turns = [
                {
                    "speaker": t["speaker"],
                    "text": t["text"],
                    "timestamp": t["timestamp"],
                    "turn_id": t["turn_id"],
                }
                for t in self._turns
            ]
            if self._compress_callback and persist_turns:
                self._compress_callback(persist_turns)

            n = len(persist_turns)
            # 保留 _next_id：clear 会归零，否则同 session 后续 turn 复用旧 id，
            # 落盘 mem_id 会与刚 flush 的记录撞库（INSERT OR REPLACE 覆盖）
            next_id = self._next_id
            self.clear()
            self._next_id = next_id

        logger.info(f"Buffer flushed: {n} turns persisted")
        return n

    def stats(self) -> dict:
        """返回 buffer 统计"""
        with self._lock:
            return {
                "turns": len(self._turns),
                "tokens": self._total_tokens,
                "max_tokens": self._max_tokens,
                "utilization": f"{self._total_tokens / self._max_tokens * 100:.0f}%",
                "compressing": self._compressing,
            }
