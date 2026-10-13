"""AML Add/Search 适配层：每 user_id 一座独立库；Add 同步可检索；后台做梦+话题迁移；
Search 走 vault2lib CSR 排序扩展到 top_k（接口层截断，不动冻结答题常数），返回 raw 全文。

契约（agentmemories.ai/api-guide）：
- POST Add 返回前必须可检索；POST Search 按相关性排序返回 [{id, content, score?}]。
"""
import asyncio
import logging
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

import numpy as np
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vault2lib import constants as C  # noqa: E402
from vault2lib import retrieval as _retrieval  # noqa: E402
from vault2lib.retrieval import fetch_full_lines, get_env, get_xe  # noqa: E402

from .config import AMLConfig  # noqa: E402

logger = logging.getLogger("aml.adapter")


@dataclass
class _Library:
    lock: threading.Lock = field(default_factory=threading.Lock)
    dirty: bool = False
    last_add_ts: float = 0.0


_EF = {}


def get_embedding_fn_shared(path):
    if path not in _EF:
        from bench_utils import get_embedding_fn
        _EF[path] = get_embedding_fn(path)
    return _EF[path]


class AMLMemoryAdapter:
    def __init__(self, cfg: Optional[AMLConfig] = None):
        from memory.dreamer import DreamOrchestrator
        from memory.index_store import IndexStore
        from memory.persistent_store import PersistentStore

        self._cfg = cfg or AMLConfig()
        self._classes = (IndexStore, PersistentStore, DreamOrchestrator)
        self._ef = get_embedding_fn_shared(self._cfg.embed_model)
        self._libs: Dict[str, _Library] = {}
        self._stores: Dict[str, dict] = {}
        self._registry_lock = threading.Lock()
        self._llm = OpenAI(base_url=self._cfg.llm_base,
                           api_key=__import__("os").environ[self._cfg.llm_key_env])
        root = Path(self._cfg.root)
        (root / "chroma").mkdir(parents=True, exist_ok=True)
        (root / "sqlite").mkdir(parents=True, exist_ok=True)
        self._sched_lock = threading.Lock()
        self._sched_cond = threading.Condition(self._sched_lock)
        self._stop = threading.Event()
        threading.Thread(target=self._dream_loop, daemon=True).start()

    @staticmethod
    def _safe_uid(user_id: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "_", user_id)

    def _paths(self, user_id: str):
        uid = self._safe_uid(user_id)
        root = Path(self._cfg.root)
        return uid, root / "chroma" / uid, root / "sqlite"

    @staticmethod
    def _invalidate_env(sqlite_dir: str, uid: str):
        """Add/迁移后 vault2lib 的三处缓存（env/raw/bm25）必须失效，否则 Search 读旧池。"""
        key = (str(sqlite_dir), uid)
        for cache in (_retrieval._ENV, _retrieval._RAW, _retrieval._BM25):
            cache.pop(key, None)

    def _get_stores(self, user_id: str):
        with self._registry_lock:
            st = self._stores.get(user_id)
            if st is None:
                IndexStore, PersistentStore, Dreamer = self._classes
                uid, chroma, sqlite_dir = self._paths(user_id)
                index = IndexStore(persist_dir=str(chroma),
                                   embedding_model_path=self._cfg.embed_model,
                                   embedding_fn=self._ef)
                persist = PersistentStore(db_path=str(sqlite_dir / f"{uid}.db"))
                dreamer = Dreamer(index, persist, self._llm, self._cfg.llm_model)
                st = {"uid": uid, "sqlite_dir": str(sqlite_dir), "index": index,
                      "persist": persist, "dreamer": dreamer, "lib": _Library()}
                self._stores[user_id] = st
                self._libs[user_id] = st["lib"]
                logger.info("新建记忆库 user_id=%s", user_id)
            return st

    # ── Add ──

    def add(self, user_id: str, session_id: str, messages: List[dict]) -> int:
        from memory.schema import IndexEntry, MemoryFile
        st = self._get_stores(user_id)
        mfs, entries = [], []
        for i, msg in enumerate(messages):
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            mid = f"{st['uid']}_{session_id}_{i}_{uuid4().hex[:8]}"
            ts_ms = msg.get("timestamp")
            date_str = (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts_ms / 1000))
                        if ts_ms else "")
            speaker = msg.get("role", "")
            mfs.append(MemoryFile(mem_id=mid, raw_content=content, summary=content[:500],
                                  speaker=speaker, source="aml", original_date=date_str,
                                  event_timestamp=date_str, session_id=session_id,
                                  turn_index=i, status="raw"))
            entries.append(IndexEntry(mem_id=mid, summary=content[:500],
                                      embedding_text=content, source="aml",
                                      speaker=speaker, original_date=date_str))
        if not mfs:
            return 0
        with st["lib"].lock:
            st["persist"].save(mfs)
            st["index"].add(entries)
            self._invalidate_env(st["sqlite_dir"], st["uid"])
            st["lib"].dirty = True
            st["lib"].last_add_ts = time.time()
        with self._sched_lock:
            self._sched_cond.notify_all()
        return len(mfs)

    # ── Search（CSR 排序扩展到 top_k）──

    def search(self, user_id: str, query: str, top_k: int) -> List[dict]:
        st = self._get_stores(user_id)
        k = max(1, min(int(top_k or 10), self._cfg.max_top_k))
        env = get_env(st["sqlite_dir"], st["uid"])
        if not env["mem_ids"]:
            return []
        qv = np.array(env["embed"]([query]))
        qv = qv / np.linalg.norm(qv, axis=1, keepdims=True)
        cand = []
        if env["cl_v"] is not None:
            sims = (env["cl_v"] @ qv.T).flatten()
            top_c = set(sims.argsort()[::-1][:C.N_CLUSTERS].tolist())
            cand = list({m for ci in top_c for m in env["clusters"][ci]["members"]
                         if m in env["mems"]})
        cand_set = set(cand)
        rest_idx = [env["midx"][m] for m in env["mem_ids"] if m not in cand_set]
        if rest_idx:
            s3 = (env["mv"][rest_idx] @ qv.T).flatten()
            order = sorted(range(len(rest_idx)), key=lambda i: float(s3[i]),
                           reverse=True)[:150]
            cand += [env["mem_ids"][rest_idx[i]] for i in order]
        if not cand:
            return []
        scores = get_xe().predict([(query, env["mems"][m][1]) for m in cand])
        s_rank = sorted(range(len(cand)), key=lambda i: float(scores[i]), reverse=True)
        candM = [cand[i] for i in s_rank[:max(k, C.RRF_CAND)]]
        raw_lines, miss = fetch_full_lines(env["con_path"], candM)
        valid = [m for m in candM if m not in set(miss)]
        if not valid:
            return []
        r_scores = get_xe().predict([(query, l) for l in raw_lines])
        r_rank = sorted(range(len(valid)), key=lambda i: float(r_scores[i]), reverse=True)
        s_pos = {m: rk for rk, m in enumerate(valid)}
        r_pos = {valid[i]: rk for rk, i in enumerate(r_rank)}
        fused = sorted(valid, key=lambda m: 1.0 / (C.RRF_K + s_pos[m] + 1)
                       + 1.0 / (C.RRF_K + r_pos[m] + 1), reverse=True)
        line_of = dict(zip(valid, raw_lines))
        return [{"id": m, "content": line_of[m], "score": 1.0 / (C.RRF_K + rk + 1)}
                for rk, m in enumerate(fused[:k])]

    # ── 后台做梦 + 话题迁移 ──

    def _dream_loop(self):
        while not self._stop.is_set():
            with self._sched_lock:
                self._sched_cond.wait(timeout=5)
            for uid, lib in list(self._libs.items()):
                if not lib.dirty or time.time() - lib.last_add_ts < self._cfg.quiet_s:
                    continue
                with lib.lock:
                    lib.dirty = False
                st = self._stores[uid]
                try:
                    stats = asyncio.run(
                        st["dreamer"].structure_and_link_memories_async(
                            batch_size=self._cfg.dream_batch))
                    stats["fused"] = st["dreamer"].fuse_clusters()
                    logger.info("做梦完成 user=%s stats=%s", uid, stats)
                except Exception:  # noqa: BLE001
                    logger.exception("做梦失败 user=%s", uid)
                    with lib.lock:
                        lib.dirty = True
                    continue
                try:
                    subprocess.run(
                        [sys.executable,
                         str(REPO_ROOT / "emo" / "scripts" / "csr_topic_migrate.py"),
                         st["uid"], "--dir", st["sqlite_dir"], "--model", self._cfg.llm_model],
                        check=True, cwd=str(REPO_ROOT))
                    self._invalidate_env(st["sqlite_dir"], st["uid"])
                except Exception:  # noqa: BLE001
                    logger.exception("话题迁移失败 user=%s", uid)

    def close(self):
        self._stop.set()
        with self._sched_lock:
            self._sched_cond.notify_all()
