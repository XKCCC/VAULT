"""AML Add/Search 适配层：每 user_id 一座独立库；Add 同步可检索；后台做梦+话题迁移；
Search 走 vault2lib CSR 排序扩展到 top_k（接口层截断，不动冻结答题常数），返回 raw 全文。

契约（agentmemories.ai/api-guide）：
- POST Add 返回前必须可检索；POST Search 按相关性排序返回 [{id, content, score?}]。
"""
import asyncio
import logging
import os
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
from openai import AsyncOpenAI, OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vault2lib import constants as C  # noqa: E402
from vault2lib import retrieval as _retrieval  # noqa: E402
from vault2lib.retrieval import (fetch_full_lines, gap_supplements,  # noqa: E402
                               get_env, get_xe, session_merge)

from .config import AMLConfig  # noqa: E402

logger = logging.getLogger("aml.adapter")


@dataclass
class _Library:
    lock: threading.Lock = field(default_factory=threading.Lock)
    dirty: bool = False
    last_add_ts: float = 0.0
    consolidating: bool = False
    cond: threading.Condition = field(default_factory=threading.Condition)

    def __post_init__(self):
        # Condition 必须共享 lib.lock，否则 cond.wait() 报 un-acquired lock
        self.cond = threading.Condition(self.lock)


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
        self._llm_async = AsyncOpenAI(base_url=self._cfg.llm_base,
                                      api_key=os.environ[self._cfg.llm_key_env])
        root = Path(self._cfg.root)
        (root / "chroma").mkdir(parents=True, exist_ok=True)
        (root / "sqlite").mkdir(parents=True, exist_ok=True)
        self._sched_lock = threading.Lock()
        self._last_search_ts = 0.0
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

    @staticmethod
    def _ensure_topic_tables(st):
        """建库即建话题表（DDL 同 csr_topic_migrate）；否则未做梦新库 Search 撞缺表 500。"""
        import sqlite3 as _sql
        con = _sql.connect(str(Path(st["sqlite_dir"]) / f"{st['uid']}.db"))
        con.execute("""CREATE TABLE IF NOT EXISTS topic_cluster (
            topic_id TEXT PRIMARY KEY, name TEXT, narrative TEXT,
            timeline_json TEXT DEFAULT '[]', n_members INT DEFAULT 0)""")
        con.execute("""CREATE TABLE IF NOT EXISTS topic_member (
            topic_id TEXT NOT NULL, mem_id TEXT NOT NULL,
            PRIMARY KEY (topic_id, mem_id))""")
        con.commit()
        con.close()

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
                self._ensure_topic_tables(st)
                self._stores[user_id] = st
                self._libs[user_id] = st["lib"]
                logger.info("新建记忆库 user_id=%s", user_id)
            return st

    # ── Add ──

    def add(self, user_id: str, session_id: str, messages: List[dict],
            request_id: str = "") -> int:
        from memory.schema import IndexEntry, MemoryFile
        st = self._get_stores(user_id)
        if request_id:
            import sqlite3 as _sql
            _con = _sql.connect(str(Path(st["sqlite_dir"]) / f"{st['uid']}.db"))
            _con.execute("CREATE TABLE IF NOT EXISTS aml_req "
                         "(request_id TEXT PRIMARY KEY, ts REAL)")
            _cur = _con.execute("INSERT OR IGNORE INTO aml_req VALUES (?, ?)",
                                (request_id, time.time()))
            _con.commit()
            _dup = _cur.rowcount == 0
            _con.close()
            if _dup:
                logger.info("幂等跳过重复 request_id user=%s req=%s", st["uid"], request_id)
                return -1
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

    def _kick_catchup(self, st, env):
        """Search 撞上"尚无话题簇+记忆量够"的库：立即后台补一次做梦+迁移（不等写静默窗）。
        防连续交错 add/search 永不开 quiet 窗 → 簇级路由全程空转退化成稠密单路。"""
        lib = st["lib"]
        if env["cl_v"] is not None or len(env["mem_ids"]) < 8:
            return
        now = time.time()
        with lib.lock:
            if now - getattr(lib, "kick_ts", 0.0) < 90.0:
                return
            lib.kick_ts = now
            lib.dirty = True
            lib.last_add_ts = 0.0
        with self._sched_lock:
            self._sched_cond.notify_all()
        logger.info("search 驱动追赶触发 user=%s mems=%d", st["uid"], len(env["mem_ids"]))

    # ── Search（CSR 排序扩展到 top_k）──

    @staticmethod
    def _wrap_line(env, mid, text):
        sp = env["speakers"].get(mid, "") or "user"
        ts = env["ets"].get(mid, "")
        return f"ROLE: {sp}\n" + (f"TIMESTAMP: {ts}\n" if ts else "") + f"TEXT: {text}"

    def search(self, user_id: str, query: str, top_k: int) -> List[dict]:
        t0 = time.time()
        self._last_search_ts = t0
        st = self._get_stores(user_id)
        k = max(1, min(int(top_k or 10), self._cfg.max_top_k))
        t_env0 = time.time()
        env = get_env(st["sqlite_dir"], st["uid"])
        t_env = time.time() - t_env0
        if not env["mem_ids"]:
            return []
        self._kick_catchup(st, env)
        lib = st["lib"]
        if env["cl_v"] is None and len(env["mem_ids"]) >= 8:
            # 平台 I/O 超时 1800s：search 可有界等待本库整合完成（默认帽 600s），
            # 等到即切完整 CSR 档；等不到回当前最优档，不超帽。
            wait_cap = float(os.environ.get("AML_SEARCH_WAIT_S", "600"))
            deadline = time.time() + wait_cap
            waited = False
            with lib.lock:
                while (lib.dirty or lib.consolidating) and time.time() < deadline:
                    lib.cond.wait(timeout=min(5.0, max(0.1, deadline - time.time())))
                    waited = True
            if waited:
                logger.info("search 等待整合结束 user=%s 用时=%.1fs", st["uid"],
                            min(wait_cap, time.time() - (deadline - wait_cap)))
                self._invalidate_env(st["sqlite_dir"], st["uid"])
                env = get_env(st["sqlite_dir"], st["uid"])
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
        memb = {m for c in env["clusters"] for m in c["members"]}
        logger.info("SEARCH-TIME user=%s env=%.1fs total=%.1fs mems=%d clusters=%d",
                    st["uid"], t_env, time.time() - t0, len(env["mem_ids"]), len(env["clusters"]))
        top = fused[:10]
        share = (sum(1 for m in top if m in memb) / len(top)) if top else 0.0
        logger.info("SEARCH-PATH user=%s mems=%d clusters=%d path=%s top10_cluster_share=%.2f",
                    st["uid"], len(env["mem_ids"]), len(env["clusters"]),
                    "csr" if env["cl_v"] is not None else "dense_only", share)
        score_of = {m: 1.0 / (C.RRF_K + rk + 1) for rk, m in enumerate(fused)}
        ids = fused[:k]
        if os.environ.get("AML_READ_SHAPING", "1") == "1" and ids:
            # headline 两件套同步进 AML 路径，预算中性：补充条+保留条 = top_k 上限
            lines = [line_of[m] for m in ids]
            merged = session_merge(st["sqlite_dir"], st["uid"], ids, lines)
            mid_of = []
            for ml in merged:
                mid_of.append(next((m for m in ids if ml.startswith(line_of[m])), ids[0]))
            supps: List[str] = []
            if merged and os.environ.get("AML_GAP_SUPP", "1") == "1":
                async def _sup():
                    sem = asyncio.Semaphore(4)
                    return await gap_supplements(self._llm_async, sem, st["sqlite_dir"],
                                                 st["uid"], query, merged)
                try:
                    res = asyncio.run(_sup())
                    if isinstance(res, tuple):
                        supps = list(res[0]) if res else []
                    else:
                        supps = list(res) if res else []
                except Exception:  # noqa: BLE001
                    logger.exception("gap_supplements 失败；不带补充继续")
                    supps = []
            room = max(0, k - len(supps))
            items = [{"id": mid, "content": self._wrap_line(env, mid, ln),
                      "score": score_of.get(mid, 0.0),
                      "created_at": env["ets"].get(mid, "")}
                     for mid, ln in zip(mid_of[:room], merged[:room])]
            floor = min(score_of.values(), default=0.0)
            for si, sp in enumerate(supps):
                src = next((m for m in fused if sp.startswith(line_of.get(m, "x")[:40])), None)
                items.append({"id": f"gap_{si:03d}", "score": floor,
                              "content": self._wrap_line(env, src, sp) if src else sp,
                              "created_at": env["ets"].get(src, "") if src else ""})
            logger.info("SEARCH-SHAPED user=%s merged=%d supps=%d total=%d cap=%d",
                        st["uid"], len(merged), len(supps), len(items), k)
            return items
        return [{"id": m, "content": self._wrap_line(env, m, line_of[m]), "score": score_of[m],
                 "created_at": env["ets"].get(m, "")} for m in ids]

    # ── 后台做梦 + 话题迁移 ──

    def _dream_one(self, uid: str):
        st = self._stores.get(uid)
        if st is None:
            return
        lib = st["lib"]
        with lib.lock:
            lib.consolidating = True
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
            return
        try:
            _env = dict(os.environ)
            _env["CUDA_VISIBLE_DEVICES"] = "1"
            subprocess.run(
                [sys.executable,
                 str(REPO_ROOT / "emo" / "scripts" / "csr_topic_migrate.py"),
                 st["uid"], "--dir", st["sqlite_dir"], "--model", self._cfg.llm_model],
                check=True, cwd=str(REPO_ROOT), env=_env)
            self._invalidate_env(st["sqlite_dir"], st["uid"])
            logger.info("话题迁移完成 user=%s", uid)
        except Exception:  # noqa: BLE001
            logger.exception("话题迁移失败 user=%s", uid)
        finally:
            with lib.lock:
                lib.consolidating = False
                lib.cond.notify_all()

    def _dream_loop(self):
        # 相位边界（add 阶段结束）会让所有库同时 dirty；串行逐库会把簇新鲜度
        # 拖到几十分钟后——并行 worker + search 踢过的库插队，把追赶时间压到分钟级。
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=int(os.environ.get("AML_DREAM_WORKERS", "4")),
                                   thread_name_prefix="amldream")
        while not self._stop.is_set():
            with self._sched_lock:
                self._sched_cond.wait(timeout=5)
            pause_s = float(os.environ.get("AML_PAUSE_ON_SEARCH_S", "90"))
            if time.time() - self._last_search_ts < pause_s:
                continue
            jobs = []
            for uid, lib in list(self._libs.items()):
                if not lib.dirty or time.time() - lib.last_add_ts < self._cfg.quiet_s:
                    continue
                with lib.lock:
                    lib.dirty = False
                    lib.consolidating = True
                jobs.append((getattr(lib, "kick_ts", 0.0), uid))
            jobs.sort(key=lambda x: -x[0])
            for _kick, uid in jobs:
                pool.submit(self._dream_one, uid)

    def close(self):
        self._stop.set()
        with self._sched_lock:
            self._sched_cond.notify_all()
