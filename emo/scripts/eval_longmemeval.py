#!/usr/bin/env python3
"""LongMemEval Benchmark Evaluation for EMO Memory System

流程：
  1. 加载 LongMemEval 数据（默认 S 版：500 题，每题 ~115k token 独立 haystack）
  2. 每题一个独立记忆库：haystack sessions 导入 EMO（chroma_dir/<qid>）
  3. 检索记忆（全部开关与 eval_locomo 对齐）→ 组装 prompt（带 question_date
     时间锚点）→ 调模型答题
  4. 召回率：session 级（官方口径 answer_session_ids）
  5. 准确率由 judge_longmemeval.py 用官方 prompt 裁判（本脚本不判分）

用法：
  # 快速冒烟（3 题）
  python emo/scripts/eval_longmemeval.py --model dashscope/qwen-plus --limit 3

  # 全量（500 题，bge-m3 + 精排）
  python emo/scripts/eval_longmemeval.py --model dashscope/qwen-plus --rerank

  # oracle 版（只含证据 session，验证检索上限）
  python emo/scripts/eval_longmemeval.py --model dashscope/qwen-plus \
      --data-file LongMemEval/data/longmemeval_oracle.json

  # 断点续跑：重跑同命令即可（按 question_id 跳过已完成）
"""

import argparse
import json
import logging
import os
import re
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
    add_retrieval_args, create_client, generate_answer, get_embedding_fn,
    make_retrieve_fn,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
for name in ["chromadb", "httpx", "sentence_transformers", "openai", "httpcore"]:
    logging.getLogger(name).setLevel(logging.WARNING)
logger = logging.getLogger("eval_longmemeval")


# 官方 reader prompt 的记忆版：History Chats 换成检索到的记忆条目，
# 保留 Current Date 锚点（LongMemEval 时间推理题依赖它）
QA_PROMPT = (
    "I will give you several relevant memory entries retrieved from the history chats "
    "between you and a user. Please answer the question based on the relevant memory entries.\n\n"
    "Memory Entries:\n\n{memories}\n\n"
    "Current Date: {question_date}\n"
    "Question: {question}\n"
    "Answer:"
)

NO_MEMORY_PREFIX = "(No relevant memories were retrieved.)\n\n"


def format_memories(memories: list, persistent_store=None) -> str:
    """组装答题记忆：默认注入 sqlite 里的 raw_content 全文。

    ⚠️ 2026-08-12 修复：此前只注入 entry.summary（loader 截断 200 字符），
    LME 金标 turn 中位 292 字符（assistant 中位 1225，98% 被截）→ reader
    看不见针，答题分系统性失真。全文从 persistent_store 按 mem_id 拉取，
    检索路径不变（证据集不变，新旧结果可配对）。
    """
    if not memories:
        return NO_MEMORY_PREFIX
    full_by_id = {}
    if persistent_store is not None:
        full_by_id = {
            mf.mem_id: mf.raw_content
            for mf in persistent_store.get_batch([e.mem_id for e, _ in memories])
            if mf.raw_content
        }
    lines = []
    for entry, _ in memories:
        if entry.category == "semantic":
            # L3 背景卡：raw_content 是占位符，洞察正文在 summary
            lines.append(f"- [主题洞察] {entry.summary}")
        else:
            lines.append(f"- {full_by_id.get(entry.mem_id) or entry.summary}")
    return "\n\n".join(lines)


def compute_recall(memories: list, answer_session_ids: list, haystack_session_ids: list = None):
    """session 级召回：检索到的记忆覆盖了 answer_session_ids 的几分之几

    无可统计证据（如 abstention 题）返回 None，不进召回统计。
    做梦后 tags 可能被语义标签替换（dreamer 重建条目时），需从 mem_id
    解析 session 序号兜底（2026-08-11 实锤：做梦库按 tags 算召回全 0）。
    """
    if not answer_session_ids:
        return None
    ans = set(answer_session_ids)
    retrieved_sessions = set()
    for entry, _ in memories:
        for tag in entry.tags:
            if tag in ans:
                retrieved_sessions.add(tag)
        if haystack_session_ids:
            m = re.search(r"_s(\d+)_t\d+$", entry.mem_id)
            if m:
                si = int(m.group(1))
                if si < len(haystack_session_ids) and haystack_session_ids[si] in ans:
                    retrieved_sessions.add(haystack_session_ids[si])
    return len(retrieved_sessions) / len(answer_session_ids)


def _retrieved_sessions(memories: list, inst: dict) -> set:
    """检索结果覆盖的 haystack session 集合（tags 优先，mem_id 解析兜底）"""
    sids = inst.get("haystack_session_ids", [])
    valid = set(sids)
    out = set()
    for e, _ in memories:
        for t in e.tags:
            if t in valid:
                out.add(t)
        m = re.search(r"_s(\d+)_t\d+$", e.mem_id)
        if m:
            si = int(m.group(1))
            if si < len(sids):
                out.add(sids[si])
    return out


def evaluate(args):
    logger.info(f"加载数据: {args.data_file}")
    data = json.load(open(args.data_file, encoding="utf-8"))

    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 断点续跑
    if out_file.exists() and not args.overwrite:
        done = {r["question_id"]: r for r in json.load(open(out_file))}
        logger.info(f"加载已有结果: {len(done)} 题")
    else:
        done = {}

    client_info = create_client(args)
    logger.info(f"模型: {args.model}, top-k: {args.top_k}")

    # 共享 embedding 模型：500 个库只加载一次
    embed_fn = get_embedding_fn(args.embed_model)

    if args.question_idx is not None:
        instances = [data[args.question_idx]]
    else:
        pool = data
        if args.stratified:
            # 分层抽样：每个题型等距取 N 题。数据文件按类型扎堆排序，
            # --limit 50 会全是 single-session-user（2026-08-11 查实），
            # 做梦消融必须分层才能保证品类可归因
            by_type = defaultdict(list)
            for inst in data:
                by_type[inst["question_type"]].append(inst)
            pool = []
            for t in sorted(by_type):
                items = by_type[t]
                step = max(1, len(items) // args.stratified)
                pool.extend(items[::step][: args.stratified])
            logger.info(
                f"分层抽样: 每类 {args.stratified} 题 × {len(by_type)} 类 = {len(pool)} 题"
            )
        if args.shard:
            # 多进程并行：--shard k/N 跑第 k 片（在已选池内按序号取模分片）
            # 每片必须配独立 --out-file（断点状态按文件存，同文件并发写会互相覆盖）
            k, n = (int(x) for x in args.shard.split("/"))
            instances = [inst for i, inst in enumerate(pool) if i % n == k]
            logger.info(f"分片 {k}/{n}: {len(instances)} 题")
        elif args.limit:
            instances = pool[: args.limit]
        else:
            instances = pool

    stats = defaultdict(lambda: {"total": 0, "recall_sum": 0.0, "recall_n": 0})
    # 已有结果也计入统计
    for r in done.values():
        qt = r["question_type"]
        stats[qt]["total"] += 1
        if r.get("recall") is not None:
            stats[qt]["recall_sum"] += r["recall"]
            stats[qt]["recall_n"] += 1

    for idx, inst in enumerate(instances):
        qid = inst["question_id"]
        if qid in done and not args.overwrite:
            continue

        qtype = inst["question_type"]
        question = inst["question"]
        is_abs = "_abs" in qid

        logger.info(
            f"\n[{idx+1}/{len(instances)}] {qid} [{qtype}{'/abs' if is_abs else ''}] "
            f"{question[:70]}"
        )

        # ── 构建/加载记忆库（每题独立；按数据文件变体分命名空间，
        #    oracle 与 S 的 question_id 相同，不隔离会互相污染）──
        variant = Path(args.data_file).stem
        chroma_dir = str(Path(args.chroma_dir) / variant / qid)
        sqlite_path = str(Path(args.sqlite_dir) / variant / f"{qid}.db")
        Path(chroma_dir).mkdir(parents=True, exist_ok=True)
        Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)

        if args.force_reimport:
            import shutil
            shutil.rmtree(chroma_dir, ignore_errors=True)
            if Path(sqlite_path).exists():
                Path(sqlite_path).unlink()

        index_store = IndexStore(persist_dir=chroma_dir, embedding_model_path=args.embed_model,
                                 embedding_fn=embed_fn)
        persistent_store = PersistentStore(db_path=sqlite_path)

        if index_store.count() == 0 or args.force_reimport:
            n = LongMemEvalLoader(index_store, persistent_store, session_id_prefix=qid) \
                .load_instance(inst)
            logger.info(f"  导入 {n} 条记忆（{len(inst['haystack_sessions'])} sessions）")
        else:
            logger.info(f"  记忆已存在: {index_store.count()} 条")

        # ── 做梦（可选；默认关——S 版全量 24.7 万 turn，做梦成本过高）──
        if args.dream:
            from memory.dreamer import DreamOrchestrator
            import asyncio
            steps = set(int(s) for s in args.dream_steps.split(","))
            dreamer = DreamOrchestrator(
                index_store, persistent_store, client_info[0], llm_model=client_info[1]
            )
            dream_stats = {}
            if 1 in steps or 2 in steps:
                # 合并异步版：每条一次调用同时完成 结构化+关联+supersede
                # （分离路径的 generate_links 不做 supersede！2026-08-11 查实），
                # batch_size 路并发 —— 比分离串行快约 10x
                dream_stats.update(asyncio.run(
                    dreamer.structure_and_link_memories_async(batch_size=args.dream_batch)
                ))
            if 4 in steps:
                dream_stats["fused"] = dreamer.fuse_clusters()
            logger.info(f"  做梦完成: {dream_stats}")

        # persistent_store 必传：启用时间轴并集召回（eval_locomo 没传是已知盲区）
        retriever = Retriever(index_store, top_k=args.top_k, persistent_store=persistent_store)
        retrieve = make_retrieve_fn(args, retriever, client_info)

        # ── 时间锚点：该题的 "now" = question_date ──
        temporal_now = None
        parsed = parse_lme_datetime(inst.get("question_date", ""))
        if parsed:
            from datetime import datetime as _dt
            temporal_now = _dt.strptime(parsed, "%Y-%m-%d %H:%M:%S")

        # ── 检索 → 答题 ──
        memories = retrieve(question, temporal_now=temporal_now)
        logger.info(f"  检索到 {len(memories)} 条记忆")

        prompt = QA_PROMPT.format(
            memories=format_memories(memories, persistent_store),
            question_date=inst.get("question_date", ""),
            question=question,
        )
        hypothesis = generate_answer(client_info, prompt, max_tokens=args.max_tokens)
        logger.info(f"  回答: {hypothesis[:80]}")

        recall = compute_recall(memories, inst.get("answer_session_ids", []),
                                inst.get("haystack_session_ids"))

        record = {
            "question_id": qid,
            "question_type": qtype,
            "question": question,
            "answer": inst["answer"],
            "question_date": inst.get("question_date", ""),
            "hypothesis": hypothesis,
            "recall": round(recall, 4) if recall is not None else None,
            "retrieved_sessions": sorted(_retrieved_sessions(memories, inst)),
            # 原始检索 mem_id 全量落盘：任何新指标口径（turn 级/NDCG/未来的）
            # 都能离线重算，无需重放检索（2026-08-11 口径错配的教训）
            "retrieved_ids": [e.mem_id for e, _ in memories],
        }
        done[qid] = record

        stats[qtype]["total"] += 1
        if recall is not None:
            stats[qtype]["recall_sum"] += recall
            stats[qtype]["recall_n"] += 1

        # 每题落盘（中断只损失当前题）
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(list(done.values()), f, indent=2, ensure_ascii=False)

    # ── 汇总 ──
    print(f"\n{'='*60}\n  LongMemEval Retrieval Summary: {args.out_file}\n{'='*60}")
    total_r, total_n = 0.0, 0
    for qt in sorted(stats):
        s = stats[qt]
        if s["recall_n"]:
            avg = s["recall_sum"] / s["recall_n"]
            print(f"  {qt:28s}: Recall@K = {avg:.3f}  (n={s['total']}, recall_n={s['recall_n']})")
            total_r += s["recall_sum"]
            total_n += s["recall_n"]
        else:
            print(f"  {qt:28s}: (无召回口径)  (n={s['total']})")
    if total_n:
        print(f"\n  {'Overall':28s}: Recall@K = {total_r/total_n:.3f}  (recall_n={total_n})")
    print("\n下一步: python emo/scripts/judge_longmemeval.py --file", out_file)


def parse_args():
    parser = argparse.ArgumentParser(description="LongMemEval Evaluation for EMO Memory System")
    parser.add_argument("--data-file",
                        default=str(PROJECT_ROOT / "LongMemEval" / "data" / "longmemeval_s_cleaned.json"))
    parser.add_argument("--out-file",
                        default=str(PROJECT_ROOT / "outputs" / "longmemeval" / "emo_lme_s_results.json"))
    parser.add_argument("--model", default="dashscope/qwen-plus")
    parser.add_argument("--api-base", default=DASHSCOPE_BASE)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--max-tokens", type=int, default=256)

    add_retrieval_args(parser)

    parser.add_argument("--question-idx", type=int, default=None, help="只跑指定索引的题")
    parser.add_argument("--shard", default=None,
                        help="多卡并行分片，格式 k/N（跑 index%%N==k 的题；每片需独立 --out-file）")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题（冒烟用）")
    parser.add_argument("--stratified", type=int, default=None,
                        help="分层抽样：每个题型等距取 N 题（做梦消融用；数据按类型扎堆，"
                             "勿用 --limit 代替）")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--base-model", default=str(PROJECT_ROOT / "Qwen2.5-7B-Instruct"))
    parser.add_argument("--adapter", default="")
    parser.add_argument("--chroma-dir", default=str(EMO_DIR / "memory" / "lme_chroma"))
    parser.add_argument("--sqlite-dir", default=str(EMO_DIR / "memory" / "lme_sqlite"))
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--force-reimport", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dream", action="store_true",
                        help="导入后做梦（成本高，建议配合 --limit 做消融）")
    parser.add_argument("--dream-steps", default="1,2,4")
    parser.add_argument("--dream-batch", type=int, default=5,
                        help="做梦并发路数（合并异步调用；API 延迟主导时建议 5-10）")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    evaluate(args)
