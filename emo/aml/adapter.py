"""AML 记忆适配层核心：每个 user_id 一座独立记忆库（chroma + sqlite + retriever + dreamer）。

Add（契约要求同步可检索）：
  messages → 原始条目 status=raw 持久化 + 立即建索引（原始形态即可被 Search 命中）
  → 后台做梦队列把 raw 条目结构化/建链/supersede（完成后检索侧自动升级为做梦态内容）
Search：
  平台 query 原文 → retriever（bge-m3 + CE 精排 + 图扩展）→ 按 mem_id 回取
  raw_content 全文（全文保真注入是我们的核心主张，不要返回摘要）→ data[] 按分排序
"""
import asyncio
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from openai import OpenAI

from .config import AMLConfig

logger = logging.getLogger("aml.adapter")


@dataclass
class _Library:
    index: object          # IndexStore
    persist: object        # PersistentStore
    retriever: object      # Retriever
    dreamer: object        # Dreamer
    lock: threading.Lock = field(default_factory=threading.Lock)
    dirty: bool = False    # 有 raw 条目待做梦
    last_add_ts: float = 0.0


class AMLMemoryAdapter:
    def __init__(self, cfg: Optional[AMLConfig] = None):
        import sys
        emo_dir = Path(__file__).resolve().parent.parent
        if str(emo_dir) not in sys.path:
            sys.path.insert(0, str(emo_dir))
        if str(emo_dir / "scripts") not in sys.path:
            sys.path.insert(0, str(emo_dir / "scripts"))

        from bench_utils import get_embedding_fn  # 共享 EF（显式 device）
        from memory.index_store import IndexStore
        from memory.persistent_store import PersistentStore
        from memory.retriever import Retriever
        from memory.dreamer import DreamOrchestrator

        self._cfg = cfg or AMLConfig()
        self._classes = (IndexStore, PersistentStore, Retriever, DreamOrchestrator)
        self._ef = get_embedding_fn(self._cfg.embed_model)
        self._reranker = None  # 懒加载
        self._libs: Dict[str, _Library] = {}
        self._registry_lock = threading.Lock()

        self._llm = OpenAI(
            base_url=self._cfg.llm_base,
            api_key=__import__("os").environ[self._cfg.llm_key_env],
        )

        root = Path(self._cfg.root)
        (root / "chroma").mkdir(parents=True, exist_ok=True)
        (root / "sqlite").mkdir(parents=True, exist_ok=True)

        # 后台做梦调度
        self._sched_lock = threading.Lock()
        self._sched_cond = threading.Condition(self._sched_lock)
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._dream_loop, daemon=True)
        self._worker.start()

    @staticmethod
    def _safe_uid(user_id: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "_", user_id)

    def _get_lib(self, user_id: str) -> _Library:
        with self._registry_lock:
            lib = self._libs.get(user_id)
            if lib is None:
                IndexStore, PersistentStore, Retriever, Dreamer = self._classes
                uid = self._safe_uid(user_id)
                index = IndexStore(
                    persist_dir=str(Path(self._cfg.root) / "chroma" / uid),
                    embedding_model_path=self._cfg.embed_model,
                    embedding_fn=self._ef,
                )
                persist = PersistentStore(
                    db_path=str(Path(self._cfg.root) / "sqlite" / f"{uid}.db")
                )
                retriever = Retriever(index, persistent_store=persist)
                dreamer = Dreamer(index, persist, self._llm, self._cfg.llm_model)
                lib = _Library(index, persist, retriever, dreamer)
                self._libs[user_id] = lib
                logger.info("新建记忆库 user_id=%s", user_id)
            return lib

    def _get_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            xe = CrossEncoder(self._cfg.rerank_model)

            def reranker(q, pairs):
                if not pairs:
                    return pairs
                scores = xe.predict([(q, (e.summary or "")) for e, _ in pairs])
                order = sorted(range(len(pairs)), key=lambda i: float(scores[i]), reverse=True)
                return [(pairs[i][0], float(scores[i])) for i in order]

            self._reranker = reranker
        return self._reranker

    # ── Add ──

    def add(self, user_id: str, session_id: str, messages: List[dict]) -> int:
        """写入并立即可检索；返回写入条数。做梦在后台队列补齐。"""
        from memory.schema import IndexEntry, MemoryFile

        lib = self._get_lib(user_id)
        mfs, entries = [], []
        for i, msg in enumerate(messages):
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            mid = f"{self._safe_uid(user_id)}_{session_id}_{i}_{uuid4().hex[:8]}"
            ts_ms = msg.get("timestamp")
            date_str = (
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts_ms / 1000))
                if ts_ms
                else ""
            )
            speaker = msg.get("role", "")
            mfs.append(
                MemoryFile(
                    mem_id=mid,
                    raw_content=content,
                    summary=content[:500],
                    speaker=speaker,
                    source="aml",
                    original_date=date_str,
                    event_timestamp=date_str,
                    session_id=session_id,
                    turn_index=i,
                    status="raw",
                )
            )
            entries.append(
                IndexEntry(
                    mem_id=mid,
                    summary=content[:500],
                    embedding_text=content,
                    source="aml",
                    speaker=speaker,
                    original_date=date_str,
                )
            )
        if not mfs:
            return 0
        with lib.lock:
            lib.persist.save(mfs)
            lib.index.add(entries)
            lib.dirty = True
            lib.last_add_ts = time.time()
        with self._sched_lock:
            self._sched_cond.notify_all()
        return len(mfs)

    # ── Search ──

    def search(self, user_id: str, query: str, top_k: int) -> List[dict]:
        lib = self._get_lib(user_id)
        k = max(1, min(int(top_k or 10), self._cfg.max_top_k))
        with lib.lock:
            hits = lib.retriever.retrieve(
                query,
                top_k=k,
                expand_graph=True,
                graph_extra=self._cfg.graph_extra,
                reranker=self._get_reranker(),
                include_superseded=False,
            )
            out = []
            for entry, score in hits[:k]:
                mf = lib.persist.get(entry.mem_id)
                content = (mf.raw_content if mf else "") or entry.summary
                if not content:
                    continue
                item = {
                    "id": entry.mem_id,
                    "content": content,
                    "score": float(score),
                }
                if mf and mf.timestamp:
                    item["created_at"] = mf.timestamp
                out.append(item)
        return out

    # ── 后台做梦 ──

    def _dream_loop(self):
        """巡视各库：有 raw 条目且距上次写入 >15s（写入相告一段落）就做一轮梦。"""
        while not self._stop.is_set():
            with self._sched_lock:
                self._sched_cond.wait(timeout=5)
            for uid, lib in list(self._libs.items()):
                if not lib.dirty or time.time() - lib.last_add_ts < 15:
                    continue
                with lib.lock:
                    lib.dirty = False
                try:
                    stats = asyncio.run(
                        lib.dreamer.structure_and_link_memories_async(
                            top_k=10, batch_size=self._cfg.dream_batch
                        )
                    )
                    logger.info("做梦完成 user_id=%s stats=%s", uid, stats)
                except Exception:
                    logger.exception("做梦失败 user_id=%s", uid)
                    with lib.lock:
                        lib.dirty = True  # 下轮重试

    def close(self):
        self._stop.set()
        with self._sched_lock:
            self._sched_cond.notify_all()
