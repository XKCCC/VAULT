#!/usr/bin/env python3
"""LifeBench Benchmark Evaluation for EMO Memory System

数据源：LifeBench-memory/life_bench_data/locomo_format/our_en.json
  10 个虚拟用户 × 一年手机痕迹（每人 ~364 日级 session、~1.5 万 turn），2003 题分 5 类：
  0=IE 信息抽取(718)  1=MR 多跳推理(597)  2=ND 非陈述性记忆(429)
  3=TKU 时序与知识更新(229)  4=UA 不可回答(30)

流程（对齐 eval_locomo.py）：
  1. 逐用户导入记忆库（chroma_dir/<sample_id>），断点续跑
  2. 检索（开关与 eval_locomo 对齐）→ 答题
  3. 免费指标：选择题字母匹配 acc / 简答题 F1 / UA 拒答检出率 / evidence 召回
  4. 判分准确率由 judge_lifebench.py 输出

用法：
  # 冒烟（第 0 个用户，20 题）
  python emo/scripts/eval_lifebench.py --model dashscope/qwen-plus --sample-idx 0 --limit 20

  # 全量
  python emo/scripts/eval_lifebench.py --model dashscope/qwen-plus --rerank

  # 中文版数据
  python emo/scripts/eval_lifebench.py --data-file LifeBench-memory/life_bench_data/locomo_format/our.json
"""

import argparse
import json
import logging
import os
import re
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path

if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EMO_DIR = PROJECT_ROOT / "emo"
sys.path.insert(0, str(EMO_DIR))
sys.path.insert(0, str(EMO_DIR / "scripts"))

from memory.index_store import IndexStore
from memory.persistent_store import PersistentStore
from memory.lifebench_loader import LifeBenchLoader, parse_lifebench_date
from memory.retriever import Retriever

from bench_utils import (
    DEFAULT_EMBED_MODEL, DASHSCOPE_BASE,
    add_retrieval_args, create_client, generate_answer, get_embedding_fn,
    make_retrieve_fn,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
for name in ["chromadb", "httpx", "sentence_transformers", "openai", "httpcore"]:
    logging.getLogger(name).setLevel(logging.WARNING)
logger = logging.getLogger("eval_lifebench")


CAT_NAMES = {
    "0": "IE",
    "1": "MR",
    "2": "ND",
    "3": "TKU",
    "4": "UA",
}

# 记忆上下文 + 指令（v2 风格：日期前缀说明 + 矛盾取最新 + 简答约束）
QA_PROMPT = (
    "You are answering a question about a user's life based on the retrieved memories above. "
    "The memories are the user's digital traces (messages, calls, calendar, photos, notes, "
    "health records, assistant chats) collected over one year. "
    "Each memory begins with its date in parentheses.\n\n"
    "Instructions:\n"
    "1. Base your answer ONLY on the memories above. Use exact words from them when possible.\n"
    "2. Pay attention to the dates: if memories conflict, the most recent memory wins.\n"
    "3. Answer concisely (a word, a phrase, or one short sentence). No explanation.\n\n"
    "Current date: {now}\n"
    "Question: {question}\n"
    "Answer:"
)

QA_PROMPT_CHOICE_SUFFIX = (
    " The question provides options; answer with the option letter only (A, B, C, or D)."
)

MEMORY_CONTEXT_HEADER = (
    "Below are relevant memories from the digital traces of {speaker_a} "
    "(collected over one year, each prefixed with its date):\n\n"
)

NO_MEMORY_PROMPT = (
    "You are given a question about a user's life, but no relevant memories were found.\n"
    "If you cannot answer based on the information provided, write 'Unable to answer'.\n\n"
)

_CHOICE_RE = re.compile(r"\sA:\s.+\sB:\s.+\sC:", re.DOTALL)
_LETTER_RE = re.compile(r"\b([A-E])\b")


def is_choice_question(question: str) -> bool:
    """从题干检测选择题（不看金标）：选项以 'A: ... B: ... C: ...' 内嵌"""
    return bool(_CHOICE_RE.search(question))


def extract_letter(prediction: str) -> str:
    """从模型输出提取选项字母"""
    p = prediction.strip()
    if len(p) <= 2 and p.upper() in "ABCDE":
        return p.upper()
    m = _LETTER_RE.search(p)
    return m.group(1) if m else ""


def normalize_answer(s):
    s = str(s).replace(",", "")
    s = re.sub(r'\b(a|an|the|and)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    return ' '.join(s.lower().split())


def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def evaluate(args):
    logger.info(f"加载数据: {args.data_file}")
    data = json.load(open(args.data_file, encoding="utf-8"))

    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    model_key = f"emo_{args.model.replace('/', '_')}_top{args.top_k}"
    if args.rerank:
        model_key += "_rr"
    if args.iter2:
        model_key += "_ir"
    if args.mmr:
        model_key += "_mmr"
    if args.hier:
        model_key += "_hier"
    if args.graph_expand:
        model_key += f"_graph{args.graph_extra}"
    if getattr(args, "multi_channel", False):
        model_key += "_multi"
    prediction_key = f"{model_key}_prediction"

    if out_file.exists() and not args.overwrite:
        out_samples = {d["sample_id"]: d for d in json.load(open(out_file))}
        logger.info(f"加载已有结果: {len(out_samples)} users")
    else:
        out_samples = {}

    client_info = create_client(args)
    logger.info(f"模型: {args.model}, top-k: {args.top_k}")

    # 共享 embedding 模型：10 个用户库只加载一次
    embed_fn = get_embedding_fn(args.embed_model)

    samples = [data[args.sample_idx]] if args.sample_idx is not None else data

    total_stats = defaultdict(lambda: {"total": 0, "score_sum": 0.0, "recall_sum": 0.0, "recall_n": 0})

    for sample_idx, sample in enumerate(samples):
        sample_id = str(sample["sample_id"])
        conv = sample["conversation"]
        qas = sample["qa"]
        speaker_a = conv.get("speaker_a", sample_id)

        # 时间锚点：该用户的 "now" = 最后一个 session 的日期
        temporal_now = None
        last_date = LifeBenchLoader.get_last_session_date(conv)
        parsed = parse_lifebench_date(last_date)
        if parsed:
            from datetime import datetime as _dt
            temporal_now = _dt.strptime(parsed, "%Y-%m-%d %H:%M:%S")

        logger.info(
            f"\n{'='*60}\n[{sample_idx+1}/{len(samples)}] {sample_id} "
            f"({len(qas)} questions, now={last_date})\n{'='*60}"
        )

        # ── 构建/加载记忆库（按数据文件变体分命名空间，
        #    防止夹具/中英文版的同名 sample 互相污染）──
        variant = Path(args.data_file).stem
        chroma_dir = str(Path(args.chroma_dir) / variant / sample_id)
        sqlite_path = str(Path(args.sqlite_dir) / variant / f"{sample_id}.db")
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
            print(f"  [导入] {sample_id} 的一年痕迹...", flush=True)
            n = LifeBenchLoader(index_store, persistent_store, session_id_prefix=sample_id) \
                .load_conversation(conv)
            print(f"  [导入] 完成: {n} 条记忆", flush=True)
        else:
            print(f"  [导入] 记忆已存在: {index_store.count()} 条", flush=True)

        # ── 做梦（可选；15k+ 条/用户，成本高，默认关）──
        if args.dream:
            from memory.dreamer import DreamOrchestrator
            import asyncio
            steps = set(int(s) for s in args.dream_steps.split(","))
            print(f"  [做梦] steps={sorted(steps)} batch={args.dream_batch}...", flush=True)
            dreamer = DreamOrchestrator(
                index_store, persistent_store, client_info[0], llm_model=client_info[1]
            )
            dream_stats = {}
            if 1 in steps or 2 in steps:
                # 合并异步版：每条一次调用同时完成 结构化+关联+supersede
                # （分离路径的 generate_links 不做 supersede），
                # batch_size 路并发 —— 比分离串行快约 10x
                dream_stats.update(asyncio.run(
                    dreamer.structure_and_link_memories_async(batch_size=args.dream_batch)
                ))
            if 4 in steps:
                dream_stats["fused"] = dreamer.fuse_clusters()
            print(f"  [做梦] 完成: {dream_stats}", flush=True)

        retriever = Retriever(index_store, top_k=args.top_k, persistent_store=persistent_store)
        retrieve = make_retrieve_fn(args, retriever, client_info)

        # evidence → dia_id 映射预检（统计可映射覆盖率）
        all_dia_ids = {
            dialog.get("dia_id", "")
            for k, v in conv.items()
            if k.startswith("session_") and "date_time" not in k
            for dialog in v
        }

        # ── 初始化/恢复输出 ──
        if sample_id in out_samples and not args.overwrite:
            out_data = out_samples[sample_id]
            for qa in out_data["qa"]:
                if prediction_key + "_score" in qa:
                    cat = qa["category"]
                    total_stats[cat]["total"] += 1
                    total_stats[cat]["score_sum"] += qa[prediction_key + "_score"]
                    if qa.get(prediction_key + "_recall") is not None:
                        total_stats[cat]["recall_sum"] += qa[prediction_key + "_recall"]
                        total_stats[cat]["recall_n"] += 1
            logger.info("已有部分结果，继续评估...")
        else:
            out_data = {"sample_id": sample_id, "qa": [dict(qa) for qa in qas]}

        # ── 逐题评估 ──
        total_qa = len(out_data["qa"])
        answered = 0
        eval_limit = args.limit if args.limit else total_qa
        print(f"  [评估] 开始 ({min(eval_limit, total_qa)}/{total_qa} 题)...", flush=True)

        for i, qa in enumerate(out_data["qa"]):
            if answered >= eval_limit:
                break
            if prediction_key in qa and not args.overwrite:
                continue

            question = qa["question"]
            gold = str(qa.get("answer", ""))
            category = str(qa["category"])
            evidence = qa.get("evidence", [])
            choice = is_choice_question(question)

            memories = retrieve(question, temporal_now=temporal_now)

            # ── 组装 prompt（全文注入：summary 截断 200 字符会致盲 reader；
            #    检索路径不变，新旧可配对）──
            parts = []
            if memories:
                full_by_id = {
                    mf.mem_id: mf.raw_content
                    for mf in persistent_store.get_batch([e.mem_id for e, _ in memories])
                    if mf.raw_content
                }
                lines = [MEMORY_CONTEXT_HEADER.format(speaker_a=speaker_a)]
                for e, _ in memories:
                    if e.category == "semantic":
                        # L3 背景卡：raw_content 是占位符，洞察正文在 summary
                        lines.append(f"- [主题洞察] {e.summary}")
                    else:
                        lines.append(f"- {full_by_id.get(e.mem_id) or e.summary}")
                parts.append("\n\n".join(lines))
            else:
                parts.append(NO_MEMORY_PROMPT)
            template = QA_PROMPT + (QA_PROMPT_CHOICE_SUFFIX if choice else "")
            parts.append(template.format(now=last_date, question=question))
            prompt = "\n\n".join(parts)

            raw_output = generate_answer(client_info, prompt, max_tokens=args.max_tokens)
            prediction = raw_output.strip()

            # ── 免费指标 ──
            if choice:
                score = 1.0 if extract_letter(prediction) == gold.strip().upper() else 0.0
            elif category == "4":
                pred_lower = prediction.lower()
                score = 1.0 if ("unable to answer" in pred_lower or "not in memory" in pred_lower
                                or "no information" in pred_lower or "not mentioned" in pred_lower) else 0.0
            else:
                score = f1_score(prediction, gold)

            # ── 召回（evidence 后缀映射；不可映射的从分母剔除）──
            recall = None
            if evidence:
                mapped, _unmapped = LifeBenchLoader.match_evidence(evidence, all_dia_ids)
                if mapped:
                    retrieved_dia = {
                        tag for e, _ in memories for tag in e.tags if "-" in tag and "_" in tag
                    }
                    hits = sum(
                        1 for ev, dias in mapped.items() if retrieved_dia & dias
                    )
                    recall = hits / len(mapped)

            qa[prediction_key] = prediction
            qa[prediction_key + "_score"] = round(score, 4)
            qa[prediction_key + "_choice"] = choice
            if recall is not None:
                qa[prediction_key + "_recall"] = round(recall, 4)
            qa[prediction_key + "_retrieved_ids"] = [e.mem_id for e, _ in memories]

            total_stats[category]["total"] += 1
            total_stats[category]["score_sum"] += score
            if recall is not None:
                total_stats[category]["recall_sum"] += recall
                total_stats[category]["recall_n"] += 1

            answered += 1
            if answered % 20 == 0:
                done_score = sum(s["score_sum"] for s in total_stats.values())
                done_n = sum(s["total"] for s in total_stats.values())
                print(f"    进度 {answered} 题, 累计均分 {done_score/max(done_n,1):.3f}", flush=True)

        out_samples[sample_id] = out_data
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(list(out_samples.values()), f, indent=2, ensure_ascii=False)
        logger.info(f"✅ {sample_id} 完成: {answered} 题, 已保存")

    # ── 汇总 ──
    print(f"\n{'='*60}\n  LifeBench Results: {model_key}\n{'='*60}")
    total_score = total_count = 0
    total_recall, recall_n = 0.0, 0
    for cat in ["0", "1", "2", "3", "4"]:
        s = total_stats.get(cat)
        if not s or s["total"] == 0:
            continue
        avg = s["score_sum"] / s["total"]
        line = f"  Cat {cat} ({CAT_NAMES[cat]:4s}): score = {avg:.3f}"
        if s["recall_n"]:
            line += f"  Recall@K = {s['recall_sum']/s['recall_n']:.3f} (recall_n={s['recall_n']})"
        line += f"  (n={s['total']})"
        print(line)
        total_score += s["score_sum"]
        total_count += s["total"]
        total_recall += s["recall_sum"]
        recall_n += s["recall_n"]
    if total_count:
        print(f"\n  Overall: score = {total_score/total_count:.3f} (n={total_count})", )
        if recall_n:
            print(f"           Recall@K = {total_recall/recall_n:.3f} (recall_n={recall_n})")
    print("  注: score 为免费指标（选择=字母匹配, 简答=F1, UA=拒答检出）；"
          "论文口径的判分准确率请跑 judge_lifebench.py")
    print("\n下一步: python emo/scripts/judge_lifebench.py --file", out_file,
          "--prediction-key", prediction_key)


def parse_args():
    parser = argparse.ArgumentParser(description="LifeBench Evaluation for EMO Memory System")
    parser.add_argument("--data-file",
                        default=str(PROJECT_ROOT / "LifeBench-memory" / "life_bench_data"
                                    / "locomo_format" / "our_en.json"))
    parser.add_argument("--out-file",
                        default=str(PROJECT_ROOT / "outputs" / "lifebench" / "emo_lifebench_results.json"))
    parser.add_argument("--model", default="dashscope/qwen-plus")
    parser.add_argument("--api-base", default=DASHSCOPE_BASE)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--max-tokens", type=int, default=128)

    add_retrieval_args(parser)

    parser.add_argument("--sample-idx", type=int, default=None, help="只跑指定索引的用户")
    parser.add_argument("--limit", type=int, default=None, help="每个用户最多评多少题")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--base-model", default=str(PROJECT_ROOT / "Qwen2.5-7B-Instruct"))
    parser.add_argument("--adapter", default="")
    parser.add_argument("--chroma-dir", default=str(EMO_DIR / "memory" / "lifebench_chroma"))
    parser.add_argument("--sqlite-dir", default=str(EMO_DIR / "memory" / "lifebench_sqlite"))
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--force-reimport", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dream", action="store_true")
    parser.add_argument("--dream-steps", default="1,2,4")
    parser.add_argument("--dream-batch", type=int, default=5,
                        help="做梦并发路数（合并异步调用；API 延迟主导时建议 5-10）")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    evaluate(args)
