#!/usr/bin/env python3
"""LongMemEval 检索真实口径分析（turn 级召回 + NDCG@10）

动机（2026-08-11）：全量跑出的 Recall@10 = 0.953 是 **session 级命中率**
（证据 session 的任一条 turn 进 top-10 即算覆盖），与论文表 3 的 turn 级 NDCG
不同单位，不能直接比。本脚本复用与全量完全相同的检索路径（bge-m3 + 精排 +
时间并集），按 turn 上的 has_answer 标签计算：

  - session_recall@10  ：复核口径（应与全量结果一致）
  - turn_recall@10     ：严格 top-10 内 has_answer turn 的覆盖率
  - NDCG@10            ：has_answer 作为 0/1 相关性分级（与论文表 3 同口径）
  - extra_inflation    ：时间并集导致实际返回 >10 条的题占比（口径膨胀量）

只跑检索不答题，零 API 成本。复用已建库；缺库（如前 50 题被清理过）默认跳过，
--allow-import 才会重导（GPU 上 ~1 min/题，CPU 勿开）。

用法：
  python emo/scripts/analyze_lme_retrieval.py            # 分析全部已建库
  python emo/scripts/analyze_lme_retrieval.py --limit 5  # 冒烟
"""

import argparse
import json
import logging
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EMO_DIR = PROJECT_ROOT / "emo"
sys.path.insert(0, str(EMO_DIR))
sys.path.insert(0, str(EMO_DIR / "scripts"))

from memory.index_store import IndexStore
from memory.persistent_store import PersistentStore
from memory.longmemeval_loader import LongMemEvalLoader, parse_lme_datetime
from memory.retriever import Retriever

from bench_utils import (
    DEFAULT_EMBED_MODEL, DASHSCOPE_BASE,
    add_retrieval_args, create_client, get_embedding_fn, make_retrieve_fn,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
for name in ["chromadb", "httpx", "sentence_transformers", "openai", "httpcore"]:
    logging.getLogger(name).setLevel(logging.WARNING)
logger = logging.getLogger("analyze_lme_retrieval")


def dcg(rels):
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def analyze(args):
    data = json.load(open(args.data_file, encoding="utf-8"))
    if args.limit:
        data = data[: args.limit]

    # client_info 只为满足 make_retrieve_fn 签名（reranker 是本地模型；
    # query_rewrite/iter2 关闭时不会发 API 调用）
    client_info = create_client(args)
    embed_fn = get_embedding_fn(args.embed_model)

    per_q = []
    skipped_missing = []

    for idx, inst in enumerate(data):
        qid = inst["question_id"]
        variant = Path(args.data_file).stem
        chroma_dir = str(Path(args.chroma_dir) / variant / qid)
        sqlite_path = str(Path(args.sqlite_dir) / variant / f"{qid}.db")

        if not Path(chroma_dir).exists() or not Path(sqlite_path).exists():
            if not args.allow_import:
                skipped_missing.append(qid)
                continue
            Path(chroma_dir).mkdir(parents=True, exist_ok=True)
            Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)

        index_store = IndexStore(persist_dir=chroma_dir, embedding_model_path=args.embed_model,
                                 embedding_fn=embed_fn)
        persistent_store = PersistentStore(db_path=sqlite_path)
        if index_store.count() == 0:
            if not args.allow_import:
                skipped_missing.append(qid)
                continue
            n = LongMemEvalLoader(index_store, persistent_store, session_id_prefix=qid) \
                .load_instance(inst)
            logger.info(f"[{idx+1}/{len(data)}] {qid} 重导 {n} 条")

        retriever = Retriever(index_store, top_k=args.top_k, persistent_store=persistent_store)
        retrieve = make_retrieve_fn(args, retriever, client_info)

        temporal_now = None
        parsed = parse_lme_datetime(inst.get("question_date", ""))
        if parsed:
            from datetime import datetime as _dt
            temporal_now = _dt.strptime(parsed, "%Y-%m-%d %H:%M:%S")

        pairs = retrieve(inst["question"], temporal_now=temporal_now)
        strict = pairs[: args.top_k]           # 严格 top-K（时间并集追加在其后）
        extra_n = max(0, len(pairs) - args.top_k)

        # turn 级真值：has_answer 标签（loader 导入时已打进 tags）
        total_rel = sum(
            1 for s in inst["haystack_sessions"] for t in s if t.get("has_answer")
        )
        rels = [1 if "has_answer" in e.tags else 0 for e, _ in strict]

        # session 级（复核口径）
        ans_sessions = set(inst.get("answer_session_ids", []))
        retrieved_sessions = {
            t for e, _ in strict for t in e.tags if t in ans_sessions
        }

        row = {
            "question_id": qid,
            "question_type": inst["question_type"],
            "n_returned": len(pairs),
            "extra_n": extra_n,
            "total_relevant_turns": total_rel,
            "turn_hits@10": sum(rels),
            "ndcg@10": (dcg(rels) / dcg([1] * min(total_rel, args.top_k))) if total_rel else None,
            "turn_recall@10": (sum(rels) / total_rel) if total_rel else None,
            "session_recall@10": (len(retrieved_sessions) / len(ans_sessions)) if ans_sessions else None,
        }
        per_q.append(row)
        if (idx + 1) % 20 == 0:
            logger.info(f"[{idx+1}/{len(data)}] ...")

    # ── 汇总 ──
    agg = defaultdict(lambda: defaultdict(list))
    for r in per_q:
        for k in ("session_recall@10", "turn_recall@10", "ndcg@10"):
            if r[k] is not None:
                agg[r["question_type"]][k].append(r[k])
        agg[r["question_type"]]["extra_ratio"].append(1.0 if r["extra_n"] > 0 else 0.0)
        agg[r["question_type"]]["n"].append(1)

    print(f"\n{'='*78}\n  LongMemEval 检索真实口径（严格 top-{args.top_k}，rerank={'on' if args.rerank else 'off'}）\n{'='*78}")
    print(f"  {'type':28s} {'sess_recall':>11s} {'turn_recall':>11s} {'NDCG@10':>9s} {'extra题占比':>10s} {'n':>4s}")
    overall = defaultdict(list)
    for qt in sorted(agg):
        a = agg[qt]
        line = f"  {qt:28s}"
        for k in ("session_recall@10", "turn_recall@10", "ndcg@10"):
            v = a[k]
            line += f" {sum(v)/len(v):>11.3f}" if v else f" {'-':>11s}"
            overall[k].extend(a[k])
        line += f" {sum(a['extra_ratio'])/len(a['extra_ratio']):>10.1%} {len(a['n']):>4d}"
        print(line)
        overall["extra_ratio"].extend(a["extra_ratio"])
    line = f"  {'Overall':28s}"
    for k in ("session_recall@10", "turn_recall@10", "ndcg@10"):
        v = overall[k]
        line += f" {sum(v)/len(v):>11.3f}" if v else f" {'-':>11s}"
    line += f" {sum(overall['extra_ratio'])/len(overall['extra_ratio']):>10.1%} {len(per_q):>4d}"
    print(line)
    if skipped_missing:
        print(f"\n  ⚠️ 跳过 {len(skipped_missing)} 个未建库实例（--allow-import 可重导）："
              f"单会话类占比高时注意品类 n 缩水")

    out = {"config": {"top_k": args.top_k, "rerank": args.rerank},
           "skipped_missing": skipped_missing, "per_question": per_q}
    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out_file, "w"), indent=2, ensure_ascii=False)
    print(f"\n明细保存: {args.out_file}")


def parse_args():
    p = argparse.ArgumentParser(description="LongMemEval 检索真实口径分析")
    p.add_argument("--data-file",
                   default=str(PROJECT_ROOT / "LongMemEval" / "data" / "longmemeval_s_cleaned.json"))
    p.add_argument("--out-file",
                   default=str(PROJECT_ROOT / "outputs" / "longmemeval" / "retrieval_analysis.json"))
    p.add_argument("--model", default="dashscope/qwen-plus")
    p.add_argument("--api-base", default=DASHSCOPE_BASE)
    p.add_argument("--api-key", default="")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--allow-import", action="store_true", help="缺库时重导（GPU 才开）")
    p.add_argument("--chroma-dir", default=str(EMO_DIR / "memory" / "lme_chroma"))
    p.add_argument("--sqlite-dir", default=str(EMO_DIR / "memory" / "lme_sqlite"))
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    # make_retrieve_fn 需要的开关（默认对齐全量配置：rerank on，其余 off）
    p.add_argument("--rerank", action="store_true", default=True)
    p.add_argument("--rerank-model", default=str(EMO_DIR / "models" / "bge-reranker-v2-m3"))
    p.add_argument("--graph-expand", action="store_true", default=False)
    p.add_argument("--graph-extra", type=int, default=5)
    p.add_argument("--graph-decay", type=float, default=0.85)
    p.add_argument("--query-rewrite", action="store_true", default=False)
    p.add_argument("--iter2", action="store_true", default=False)
    p.add_argument("--hier", action="store_true", default=False)
    p.add_argument("--mmr", action="store_true", default=False)
    p.add_argument("--no-l3", action="store_true", default=False)
    p.add_argument("--local", action="store_true", default=False)
    p.add_argument("--base-model", default="")
    p.add_argument("--adapter", default="")
    return p.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
