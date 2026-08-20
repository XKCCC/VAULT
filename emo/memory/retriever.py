"""R1+R2: 检索器 — 纯语义 ANN 检索 + 图扩展

职责:
    - 接收用户消息
    - 在 IndexStore 中按语义相似度检索相关记忆（种子）
    - 可选：沿 related_ids 图扩展一跳，重打分后补充返回
    - 返回 top-K 记忆

设计原则:
    种子检索永远基于纯语义相似度（cosine similarity）。
    时间衰减、权重、频率等因素只在「做梦」阶段使用，
    用于决定记忆的压缩/融合/淘汰，不影响检索排序。

图扩展（P1）:
    做梦时建立的 related_ids（关联图谱）在检索时一跳扩展：
    种子 top-k → 收集种子的 related_ids → 对候选用 query 重打分（rescore）
    → 混合分 = graph_alpha * 语义分 + (1-graph_alpha) * 图传播分
    → 取 top graph_extra 追加在种子之后返回。
    解决多跳问题：两条证据各自语义分不高但互为关联时，
    第一条命中可把第二条带出来。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from .schema import IndexEntry
from .index_store import IndexStore

logger = logging.getLogger(__name__)


class Retriever:
    """记忆检索器 — 纯语义 ANN 检索 + 时间轴并集 + 可选图扩展

    Args:
        index_store: 热索引层
        top_k: 种子检索数量
        persistent_store: 持久层（可选）。提供后启用时间轴召回：
            查询含时间意图时按 event_timestamp 范围补充候选。
    """

    def __init__(
        self,
        index_store: IndexStore,
        top_k: int = 5,
        persistent_store=None,
    ):
        self._store = index_store
        self._top_k = top_k
        self._persist = persistent_store

    def retrieve(
        self,
        query: str,
        category: Optional[str] = None,
        top_k: Optional[int] = None,
        expand_graph: bool = False,
        graph_decay: float = 0.85,
        graph_extra: int = 5,
        graph_alpha: float = 0.6,
        temporal: bool = True,
        temporal_extra: int = 5,
        query_rewriter=None,
        reranker=None,
        hierarchical: bool = False,
        mmr: bool = False,
        mmr_lambda: float = 0.7,
        temporal_now=None,
        include_superseded: bool = False,
    ) -> List[Tuple[IndexEntry, float]]:
        """纯语义检索（+ 时间轴并集 + 可选图扩展）

        Args:
            query: 用户消息
            category: L1 分类过滤（可选，Phase 3.2 用）
            top_k: 种子返回数量
            expand_graph: 是否沿 related_ids 一跳扩展
            graph_decay: 图传播衰减（邻居图分 = 种子分 * decay）
            graph_extra: 图扩展最多追加多少条邻居
            graph_alpha: 混合分中语义分的权重（1-alpha 为图分权重）
            temporal: 是否启用时间轴召回（解析查询时间表达 → 范围查询并集）
            temporal_extra: 时间轴最多追加多少条候选
            query_rewriter: 可选的 query 改写函数（HyDE 等）。
                改写文本只用于语义编码；原 query 仍用于时间解析——
                改写会毁掉 "上周/yesterday" 这类时间表达。
            reranker: 可选精排函数 (query, [(entry, score)]) -> [(entry, score)]，
                取语义 top-2k 种子重排后保留 top-k（cross-encoder 精排）。
            hierarchical: 是否启用 L3 层级通道（先簇后点）。
            mmr: 是否对种子做 MMR 多样性去重（惩罚与已选条目近义的候选，
                防止同一事实的 turn/session/L3 多粒度形态挤占 top-k 槽位）。
            mmr_lambda: MMR 相关性权重（1-lambda 为多样性惩罚权重）。
            temporal_now: 时间解析的 "now" 锚点（默认系统当前时间；
                评测 2023 年数据集时传对话末 session 日期）。

        Returns:
            List of (IndexEntry, similarity_score) tuples。
            种子在前（按语义分降序），时间轴候选居中，图扩展邻居在后。
        """
        k = top_k or self._top_k
        encode_query = query_rewriter(query) if query_rewriter else query
        pool = k * 2 if (reranker or mmr) else k

        results = self._store.search(
            query=encode_query,
            top_k=pool,
            category=category,
            min_weight=0.0,  # 不按 weight 过滤，所有记忆都可检索
            include_superseded=include_superseded,
        )

        if not results:
            logger.debug(f"No memories found for query: {query[:50]}...")
            return []

        if reranker:
            seeds = reranker(encode_query, results[:pool])[:k]
        elif mmr:
            seeds = self._mmr_select(results[:pool], k, mmr_lambda)
        else:
            seeds = results[:k]
        final = list(seeds)

        # L3 层级通道：先命中主题簇，再展开簇内成员（与直接 top-k 互补）
        if hierarchical:
            final.extend(self._retrieve_l3_channel(encode_query, seeds))

        # 时间轴并集：查询含时间意图时按 event_timestamp 范围补充
        # （只增不替——事实/偏好类记忆不受时间门控）
        if temporal and self._persist is not None:
            final.extend(self._retrieve_by_time(query, seeds, temporal_extra, now=temporal_now))

        if expand_graph:
            neighbors = self._expand_with_graph(
                encode_query, seeds, graph_decay, graph_extra, graph_alpha
            )
            final.extend(neighbors)

        logger.info(
            f"Retrieved {len(final)} memories for query: {query[:50]}... "
            f"(seeds={len(seeds)}, graph_added={len(final) - len(seeds)}, "
            f"scores: {[f'{s:.3f}' for _, s in final]})"
        )

        # 记录访问（内存计数，不写库；做梦时 flush_stats 批量回写，
        # 数据供做梦模块使用：频率统计、效用追踪、衰减计算）
        for entry, _ in final:
            self._store.record_access(entry.mem_id)

        return final

    def retrieve_multi(
        self,
        query: str,
        top_k: Optional[int] = None,
        rrf_k: int = 60,
        candidate_pool: int = 30,
        dense_pool: int = 20,
        graph_decay: float = 0.85,
        l3_top: int = 2,
        l3_hit_threshold: float = 0.55,
        l3_member_top: int = 3,
        l3_cards: int = 2,
        temporal: bool = True,
        temporal_extra: int = 5,
        temporal_now=None,
        reranker=None,
        channel_weights: Optional[Dict[str, float]] = None,
        l3_approx: bool = False,
    ) -> List[Tuple[IndexEntry, float]]:
        """多路召回 + RRF 融合

        四路独立召回各产排名，RRF 融合后精排：
          C1 dense    ：query 语义 top-dense_pool
          C2 graph    ：dense 种子的一跳邻居（图传播分排名）
          C3 L3 簇    ：命中 L3（label1=semantic）→ 精确展开簇成员
                        （成员连接在做梦 Step4 持久化于 L3 的 related_ids；
                        旧库未回补时退化为 L3 摘要二次查询的近似带块）
          C4 temporal ：查询时间表达 → event_timestamp 范围查询

        L3 本体不占 top-k 竞争位——作为"主题背景卡"追加在最后注入
        （偏好/模糊查询的带路者定位，避免模糊洞察稀释证据焦点）。

        融合：rrf_score(m) = Σ_路 w_c / (rrf_k + rank_c(m))，取 candidate_pool
        候选交 reranker 精排留 top-k；无 reranker 则按 RRF 分取 top-k。
        """
        k = top_k or self._top_k
        w = {"dense": 1.0, "graph": 1.0, "l3": 1.0, "temporal": 1.0}
        if channel_weights:
            w.update(channel_weights)

        # ── C1 dense ──
        dense = self._store.search(query=query, top_k=dense_pool, min_weight=0.0)
        if not dense:
            return []
        entry_by_id = {e.mem_id: e for e, _ in dense}
        channels: Dict[str, List[Tuple[str, float]]] = {
            "dense": [(e.mem_id, s) for e, s in dense]
        }
        seeds = dense[:k]

        # ── C2 graph：种子一跳邻居，图传播分排名 ──
        graph_score: Dict[str, float] = {}
        for entry, score in seeds:
            for rid in entry.related_ids:
                if not rid:
                    continue
                gs = score * graph_decay
                if gs > graph_score.get(rid, 0.0):
                    graph_score[rid] = gs
        channels["graph"] = sorted(graph_score.items(), key=lambda x: x[1], reverse=True)

        # ── C3 L3 簇：精确成员展开（无持久化成员时退化为摘要二次查询）──
        l3_hits = self._store.search(
            query=query, top_k=l3_top, category="semantic", min_weight=0.0
        )
        l3_member_votes: List[Tuple[str, float]] = []
        l3_card_list: List[Tuple[IndexEntry, float]] = []
        for l3e, l3s in l3_hits:
            if l3s < l3_hit_threshold:
                continue
            l3_card_list.append((l3e, l3s))
            member_ids = [rid for rid in (l3e.related_ids or []) if rid]
            if member_ids and not l3_approx:
                sem = self._store.rescore(query, member_ids)
                ranked = sorted(sem.items(), key=lambda x: x[1], reverse=True)[:l3_member_top]
            else:
                ranked = [
                    (e.mem_id, s * l3s * 0.8)
                    for e, s in self._store.search(
                        query=l3e.summary, top_k=l3_member_top, min_weight=0.0
                    )
                    if e.category != "semantic"
                ]
            l3_member_votes.extend(ranked)
        seen_l3: set = set()
        l3_ranked: List[Tuple[str, float]] = []
        for mid, s in sorted(l3_member_votes, key=lambda x: x[1], reverse=True):
            if mid in seen_l3:
                continue
            seen_l3.add(mid)
            l3_ranked.append((mid, s))
        channels["l3"] = l3_ranked

        # ── C4 temporal ──
        if temporal and self._persist is not None:
            t_hits = self._retrieve_by_time(query, seeds, temporal_extra, now=temporal_now)
            channels["temporal"] = [(e.mem_id, s) for e, s in t_hits]
            for e, _ in t_hits:
                entry_by_id.setdefault(e.mem_id, e)

        # ── RRF 融合（L3/semantic 条目一律剔除出竞争，只作背景卡）──
        card_ids = {e.mem_id for e, _ in l3_card_list}
        rrf: Dict[str, float] = {}
        for ch, ranked in channels.items():
            for rank, (mid, _) in enumerate(ranked):
                if mid in card_ids:
                    continue
                rrf[mid] = rrf.get(mid, 0.0) + w.get(ch, 1.0) / (rrf_k + rank + 1)
        fused_order = sorted(rrf.items(), key=lambda x: x[1], reverse=True)

        for mid, _ in fused_order:
            if mid not in entry_by_id:
                e = self._store.get_by_id(mid)
                if e is not None:
                    entry_by_id[mid] = e
        pairs = []
        for mid, s in fused_order:
            e = entry_by_id.get(mid)
            if e is None or e.category == "semantic":
                continue  # L3 不占 top-k 槽位
            pairs.append((e, s))
            if len(pairs) >= candidate_pool:
                break

        if reranker and pairs:
            pairs = reranker(query, pairs)[:k]
        else:
            pairs = pairs[:k]

        final = pairs + l3_card_list[:l3_cards]
        logger.info(
            f"Multi-channel retrieve: query='{query[:40]}...' "
            f"dense={len(channels['dense'])} graph={len(channels['graph'])} "
            f"l3={len(channels['l3'])} temporal={len(channels.get('temporal', []))} "
            f"→ fused={len(pairs)} +cards={len(l3_card_list[:l3_cards])}"
        )
        for entry, _ in final:
            self._store.record_access(entry.mem_id)
        return final

    def _mmr_select(
        self,
        candidates: List[Tuple[IndexEntry, float]],
        k: int,
        lam: float = 0.7,
    ) -> List[Tuple[IndexEntry, float]]:
        """MMR 多样性选择：λ·相关性 − (1−λ)·与已选条目的最大相似度

        解决的问题：同一事实的 turn 级 observation / session 摘要 / L3 簇
        多粒度形态近义重复，会挤占 top-k 槽位把其他证据挤出去。
        """
        import numpy as np

        if len(candidates) <= k:
            return candidates

        ids = [e.mem_id for e, _ in candidates]
        embs = self._store.get_embeddings(ids)
        if len(embs) < len(ids):  # 有缺 embedding 的则退化为直通
            return candidates[:k]

        def _cos(a, b):
            a = np.asarray(a, dtype=np.float32)
            b = np.asarray(b, dtype=np.float32)
            d = float(np.linalg.norm(a) * np.linalg.norm(b))
            return float(np.dot(a, b) / d) if d > 0 else 0.0

        selected = [0]  # top-1 永远入选
        remaining = list(range(1, len(candidates)))
        while remaining and len(selected) < k:
            best_i, best_val = None, -1e18
            for i in remaining:
                rel = candidates[i][1]
                max_sim = max(_cos(embs[ids[i]], embs[ids[j]]) for j in selected)
                val = lam * rel - (1.0 - lam) * max_sim
                if val > best_val:
                    best_val, best_i = val, i
            selected.append(best_i)
            remaining.remove(best_i)

        return [candidates[i] for i in selected]

    def _retrieve_l3_channel(
        self,
        encode_query: str,
        seeds: List[Tuple[IndexEntry, float]],
        l3_top: int = 2,
        member_top: int = 3,
        hit_threshold: float = 0.55,
    ) -> List[Tuple[IndexEntry, float]]:
        """L3 层级通道：先命中主题簇（label1=semantic），再展开簇内成员

        L3 簇是对 settled 记忆的主题抽象（做梦 Step4 融合）。先按主题
        命中可扩大召回面——成员记忆可能直接语义检索进不了 top-k，
        但经由其所属主题簇可以带回。命中的 L3 本身也作为主题级上下文注入。
        """
        seed_ids = {e.mem_id for e, _ in seeds}
        out: List[Tuple[IndexEntry, float]] = []

        l3_hits = self._store.search(
            query=encode_query, top_k=l3_top, category="semantic", min_weight=0.0
        )
        for l3e, l3s in l3_hits:
            if l3s < hit_threshold or l3e.mem_id in seed_ids:
                continue
            # 主题命中：展开簇内成员（L3 摘要作为二次查询）
            members = self._store.search(query=l3e.summary, top_k=member_top, min_weight=0.0)
            for e2, s2 in members:
                if e2.mem_id in seed_ids or e2.mem_id == l3e.mem_id or e2.category == "semantic":
                    continue
                seed_ids.add(e2.mem_id)
                out.append((e2, s2 * l3s * 0.8))
            # L3 簇本身作为主题级上下文注入
            out.append((l3e, l3s * 0.85))
            seed_ids.add(l3e.mem_id)

        out.sort(key=lambda x: x[1], reverse=True)
        return out

    def _retrieve_by_time(
        self,
        query: str,
        seeds: List[Tuple[IndexEntry, float]],
        extra: int = 5,
        now=None,
    ) -> List[Tuple[IndexEntry, float]]:
        """时间轴召回：解析查询中的时间表达 → event_timestamp 范围查询

        解决的问题：chatbot 长期运行后，"上周末/in July" 这类表达对应
        大量不同日期的记忆，纯语义无法区分。按绝对区间补充候选，
        与语义种子求并集（不替换语义排序，事实/偏好不受时间门控）。

        已被衰减移出索引的记忆也可经时间轴带回
        （"遗忘的是索引精度，不是记忆本身"）。
        """
        from .temporal import resolve_temporal_range

        span = resolve_temporal_range(query, now=now)
        if not span:
            return []

        mfs = self._persist.get_by_event_time_range(span[0], span[1], limit=50)
        if not mfs:
            return []

        seed_ids = {e.mem_id for e, _ in seeds}
        cand_ids = [mf.mem_id for mf in mfs if mf.mem_id not in seed_ids]
        if not cand_ids:
            return []

        sem = self._store.rescore(query, cand_ids)
        mf_by_id = {mf.mem_id: mf for mf in mfs}
        hits = []
        for mid in cand_ids:
            entry = self._store.get_by_id(mid)
            if entry is None:
                mf = mf_by_id[mid]
                entry = IndexEntry(
                    mem_id=mf.mem_id,
                    summary=mf.summary,
                    category=mf.label1 or mf.category,
                    sub_category=mf.label2 or mf.sub_category,
                    source=mf.source,
                    speaker=mf.speaker,
                    original_date=mf.original_date,
                    event_timestamp=mf.event_timestamp,
                )
            hits.append((entry, sem.get(mid, 0.0)))

        hits.sort(key=lambda x: x[1], reverse=True)
        logger.info(
            f"Temporal recall: '{query[:40]}' range [{span[0]} ~ {span[1]}) "
            f"→ {len(mfs)} in range, {len(hits[:extra])} added"
        )
        return hits[:extra]

    def retrieve_with_feedback(
        self,
        query: str,
        category: Optional[str] = None,
        top_k: Optional[int] = None,
        expand_graph: bool = False,
        graph_decay: float = 0.85,
        graph_extra: int = 5,
        feedback_docs: int = 2,
    ) -> List[Tuple[IndexEntry, float]]:
        """PRF 伪相关反馈两轮检索（零 LLM 成本）

        第一轮：正常检索（可含图扩展）
        第二轮：query 扩写 = 原 query（×2 加权）+ 第一轮 top 证据摘要
                 → 带出初始 query 表达不出的证据（多跳桥接实体）
        合并：第一轮结果保持原位，第二轮新条目按分数追加（去重）

        Args:
            feedback_docs: 用第一轮前几条摘要做反馈扩写

        Returns:
            合并去重后的 (IndexEntry, score) 列表
        """
        round1 = self.retrieve(
            query,
            category=category,
            top_k=top_k,
            expand_graph=expand_graph,
            graph_decay=graph_decay,
            graph_extra=graph_extra,
        )
        if not round1:
            return []

        # 反馈扩写：原 query 重复两次保持权重，追加 top 证据摘要
        feedback = " ".join(e.summary for e, _ in round1[:feedback_docs] if e.summary)
        if not feedback:
            return round1

        query2 = f"{query} {query} {feedback}"
        round2 = self.retrieve(
            query2,
            category=category,
            top_k=top_k,
            expand_graph=expand_graph,
            graph_decay=graph_decay,
            graph_extra=graph_extra,
        )

        seen = {e.mem_id for e, _ in round1}
        merged = list(round1)
        added = 0
        for entry, score in round2:
            if entry.mem_id not in seen:
                merged.append((entry, score))
                seen.add(entry.mem_id)
                added += 1

        logger.info(
            f"PRF round2 added {added} new memories "
            f"(feedback from top-{feedback_docs})"
        )
        return merged

    def _expand_with_graph(
        self,
        query: str,
        seeds: List[Tuple[IndexEntry, float]],
        decay: float,
        extra: int,
        alpha: float,
    ) -> List[Tuple[IndexEntry, float]]:
        """沿 related_ids 一跳扩展并重打分

        图传播分：邻居 = max(种子语义分 * decay)（多种子可达取最大）
        混合分：alpha * 邻居自身语义分 + (1-alpha) * 图传播分
        """
        seed_ids = {e.mem_id for e, _ in seeds}

        # 收集候选邻居及其图传播分
        graph_score: Dict[str, float] = {}
        for entry, score in seeds:
            for rid in entry.related_ids:
                if rid in seed_ids:
                    continue
                gs = score * decay
                if gs > graph_score.get(rid, 0.0):
                    graph_score[rid] = gs

        if not graph_score:
            return []

        # 候选重打分（邻居自身与 query 的语义相似度）
        cand_ids = list(graph_score.keys())
        sem_scores = self._store.rescore(query, cand_ids)

        blended = []
        for rid in cand_ids:
            s_sem = sem_scores.get(rid, 0.0)
            s_graph = graph_score[rid]
            blended.append((rid, alpha * s_sem + (1.0 - alpha) * s_graph))
        blended.sort(key=lambda x: x[1], reverse=True)

        neighbors = []
        for rid, score in blended[:extra]:
            entry = self._store.get_by_id(rid)
            if entry is not None:
                neighbors.append((entry, score))

        logger.info(
            f"Graph expansion: {len(graph_score)} candidates "
            f"→ {len(neighbors)} added"
        )
        return neighbors
