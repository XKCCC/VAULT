#!/usr/bin/env python3
"""LoCoMo 全文重答（修复 turn 无 observation 时 summary 截断 120 字符的致盲问题）

与 reanswer_lme_fulltext.py 同原理：不重检索，读已有结果的 retrieved_ids，
从 sqlite raw_content 重建完整注入（turn → "(date) speaker: 全文"；session/L3 → 原文），
prompt 模板逐字复刻 eval_locomo.py（v2 口径），F1 评分函数与官方 task_eval 同语义。

用法：
  python emo/scripts/reanswer_locomo_fulltext.py                # Full 最佳组合全量
  python emo/scripts/reanswer_locomo_fulltext.py --limit 20     # 冒烟
  python emo/scripts/reanswer_locomo_fulltext.py \
    --results-file outputs/locomo/ablation_b1_mmr.json \
    --out-file outputs/locomo/ablation_b1_mmr_fulltext.json     # 复核 B 类消融
"""

import argparse
import asyncio
import json
import os
import re
import sqlite3
import string
import time
from collections import Counter, defaultdict
from pathlib import Path

from cost_log import log_cost
from openai import AsyncOpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# ── 以下常量逐字复制自 eval_locomo.py（v2 prompt 口径）──
MEMORY_CONTEXT_HEADER = (
    "Below are relevant memories from a conversation between {speaker_a} and {speaker_b} "
    "that took place over multiple sessions:\n\n"
)
NO_MEMORY_PROMPT = (
    "You are given a question about a conversation, but no relevant memories were found.\n"
    "If you cannot answer based on the information provided, write 'No information available'.\n\n"
)
QA_PROMPT_V2 = (
    "You are answering a question about a past conversation based on the retrieved memories above.\n"
    "Each memory begins with its date in parentheses, e.g. \"(1:56 pm on 8 May, 2023)\".\n\n"
    "Instructions:\n"
    "1. Base your answer ONLY on the memories above. Use exact words from them when possible.\n"
    "2. Pay attention to the dates: if memories conflict, the most recent memory wins.\n"
    "3. If a memory contains relative time (\"yesterday\", \"last week\"), convert it to an absolute date "
    "using that memory's own date before answering.\n"
    "4. Answer in 1-5 words. Be specific. No explanation, no full sentences.\n\n"
    "Examples:\n"
    "Q: What did Caroline research? A: counseling and mental health\n"
    "Q: What is Caroline's identity? A: trans woman\n\n"
    "Question: {question}\n"
    "Answer:"
)
QA_PROMPT_CAT2_V2 = (
    "You are answering a time-related question about a past conversation based on the retrieved memories above.\n"
    "Each memory begins with its date in parentheses, e.g. \"(1:56 pm on 8 May, 2023)\".\n\n"
    "Instructions:\n"
    "1. Find the memory that mentions the event, and use ITS date as the answer.\n"
    "2. If the memory text contains relative time (\"yesterday\", \"last week\", \"two days ago\"), "
    "compute the absolute date from the memory's own date. Example: a memory dated \"May 8, 2023\" says "
    "\"I adopted a puppy yesterday\" → the adoption was May 7, 2023.\n"
    "3. Answer with a SPECIFIC date (e.g., \"May 7, 2023\", \"July 2023\"). If the question asks HOW LONG "
    "or a duration, answer with the duration (e.g., \"4 years\", \"2 months\").\n"
    "4. NEVER answer with relative time (\"last week\", \"yesterday\"). Answer in 1-5 words.\n\n"
    "Examples:\n"
    "Q: When did Caroline go to the support group? A: May 7, 2023\n"
    "Q: When did Melanie run a charity race? A: August 12, 2023\n"
    "Q: How long has Jon owned his car? A: 4 years\n\n"
    "Question: {question}\n"
    "Answer:"
)
QA_PROMPT_CAT5 = (
    "Based on the above memories, select the correct answer.\n\n"
    "Question: {question}\n"
    "Select: (a) {option_a} (b) {option_b}\n"
    "Answer (a or b):"
)

# cat3 两阶段 CoT v2：有相关事实必须尽力推理，推不出才拒答
# （v1 拒答指令太激进，会把成功的世界知识推理否决掉）
QA_PROMPT_CAT3_COT = (
    "You are answering a question about a past conversation based on the retrieved memories above.\n"
    "Each memory begins with its date in parentheses.\n"
    "Some questions require combining a memory fact with world knowledge. Example: the memory says "
    "\"Alice bought an iPad Pro last week\"; the question asks what charging cable it uses — you must "
    "first recall \"iPad Pro\" from the memory, then use world knowledge: \"iPad Pro uses USB-C\".\n\n"
    "Instructions:\n"
    "1. First, quote the memory fact(s) relevant to the question (one line).\n"
    "2. Then reason step by step, combining the memory fact with world knowledge (1-3 sentences).\n"
    "3. If the memories contain ANY related facts, you MUST make your best inference from them — "
    "answer even when uncertain; a well-reasoned guess is far better than a refusal. "
    "Only answer \"Not mentioned in memories\" when the memories contain NOTHING relevant at all.\n"
    "4. On the LAST line, give your final short answer (1-5 words) exactly in this format: Answer: X\n\n"
    "Question: {question}\n"
)

# v3：v2 + MemPro locomo prompt 的答案形态条款（复数全列/why 双因/
# 限定词锚定），放开 1-5 词硬约束。仅动 cat1/4 默认分支，cat3 独立 prompt 不变。
QA_PROMPT_V3 = (
    "You are answering a question about a past conversation based on the retrieved memories above.\n"
    "Each memory begins with its date in parentheses, e.g. \"(1:56 pm on 8 May, 2023)\".\n\n"
    "Instructions:\n"
    "1. Base your answer ONLY on the memories above. Use exact words from them when possible.\n"
    "2. Pay attention to the dates: if memories conflict, the most recent memory wins.\n"
    "3. If a memory contains relative time (\"yesterday\", \"last week\"), convert it to an absolute date "
    "using that memory's own date before answering.\n"
    "4. If the question asks for plural items, examples, events, values, or reasons, include ALL directly "
    "relevant specific items from the memories, separated by commas — do not pick just one.\n"
    "5. For \"why\" questions, keep every direct reason given in the memories (after words like "
    "\"because\", \"since\", \"wanted\", \"dreaming of\") — include both the long-term motivation and "
    "the triggering event if both appear.\n"
    "6. If the question contains a qualifier (\"through\", \"by\", \"after\", \"on\", \"about\"...), answer "
    "only the fact tied to that qualifier, not nearby background facts.\n"
    "7. Answer with a short phrase (a few more words are fine when listing multiple items). "
    "No explanation, no full sentences.\n\n"
    "Examples:\n"
    "Q: What did Caroline research? A: counseling and mental health\n"
    "Q: What is Caroline's identity? A: trans woman\n"
    "Q: What hobbies does Caroline enjoy? A: painting, hiking, yoga\n\n"
    "Question: {question}\n"
    "Answer:"
)

# ── 评分函数（与 eval_locomo 内置 fallback 同语义，即官方 task_eval 逻辑）──
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
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)

def f1_multi(prediction, ground_truth):
    preds = [p.strip() for p in str(prediction).split(',')]
    golds = [g.strip() for g in str(ground_truth).split(',')]
    return sum(max(f1_score(p, g) for p in preds) for g in golds) / len(golds)

def score_answer(prediction: str, answer, category: int) -> float:
    if category == 5:
        pl = prediction.lower()
        return 1.0 if ("no information available" in pl or "not mentioned" in pl) else 0.0
    if category == 3:
        answer = str(answer).split(";")[0].strip()
    if category == 1:
        return f1_multi(prediction, answer)
    return f1_score(prediction, str(answer))


def full_line_from_raw(raw: str, summary: str = "") -> str:
    """raw_content → 完整注入行（重建 loader 的 "(date) ..." 前缀格式，全文不截断）

    L3 条目的 raw_content 是 "L3 insight from..." 占位符，正文在 summary。
    """
    if raw.startswith("L3 insight") and summary:
        return f"[主题洞察] {summary.strip()}"
    m = re.match(r"DATE: (.*?)\nSPEAKER: (.*?)\nDIA_ID: .*?\nTEXT: (.*)", raw, re.DOTALL)
    if m:
        return f"({m.group(1)}) {m.group(2)}: {m.group(3).strip()}"
    m = re.match(r"SESSION: .*?\nDATE: (.*?)\nSUMMARY: (.*)", raw, re.DOTALL)
    if m:
        return f"({m.group(1)}) {m.group(2).strip()}"
    return raw.strip()


def fetch_full_lines(db_path: str, mem_ids: list) -> list:
    if not Path(db_path).exists():
        return []
    con = sqlite3.connect(db_path)
    out = []
    for mid in mem_ids:
        row = con.execute(
            "SELECT raw_content, summary FROM memory_files WHERE mem_id=?", (mid,)
        ).fetchone()
        if row and row[0]:
            out.append(full_line_from_raw(row[0], row[1] or ""))
    con.close()
    return out


def _maybe_temperature(model: str):
    """O 系列/gpt-5 推理模型不支持 temperature 参数（litellm 会 400）"""
    return {} if re.match(r"^(o\d|gpt-5|.*o4-mini)", model) else {"temperature": 0.0}


async def reanswer(args):
    samples = json.load(open(args.results_file))
    data = {s["sample_id"]: s for s in json.load(open(args.data_file))}

    # 自动探测 prediction key
    q0 = samples[0]["qa"][0]
    pk = args.prediction_key or next(k for k in q0 if k.endswith("_prediction"))
    print(f"prediction key: {pk}")

    client = AsyncOpenAI(
        api_key=os.environ[args.api_key_env],
        base_url=args.base_url or DASHSCOPE_BASE,
        timeout=120,
    )
    model = args.model.replace("dashscope/", "")
    extra = {"enable_thinking": False} if "qwen3.7" in model else None
    sem = asyncio.Semaphore(args.concurrency)

    async def one(db_path, sa, sb, qa):
        mem_ids = qa.get(pk + "_retrieved_ids") or []
        lines = fetch_full_lines(db_path, mem_ids)
        parts = []
        if lines:
            parts.append(MEMORY_CONTEXT_HEADER.format(speaker_a=sa, speaker_b=sb)
                          + "\n".join(f"- {l}" for l in lines))
        else:
            parts.append(NO_MEMORY_PROMPT)
        cat = qa["category"]
        cot = args.cat3_cot and cat == 3
        if cat == 5:
            adv = qa.get("adversarial_answer", qa.get("answer", ""))
            # 原 eval 选项顺序随机；判分只查 not-mentioned 关键词，固定顺序语义等价
            options = {"a": adv, "b": "Not mentioned in the conversation"}
            parts.append(QA_PROMPT_CAT5.format(
                question=qa["question"], option_a=options["a"], option_b=options["b"]))
        elif cat == 2:
            parts.append(QA_PROMPT_CAT2_V2.format(question=qa["question"]))
        elif cot:
            parts.append(QA_PROMPT_CAT3_COT.format(question=qa["question"]))
        else:
            main_prompt = QA_PROMPT_V3 if args.prompt_version == "v3" else QA_PROMPT_V2
            parts.append(main_prompt.format(question=qa["question"]))
        prompt = "\n\n".join(parts)
        async with sem:
            for attempt in range(3):
                try:
                    kw = {"extra_body": extra} if extra else {}
                    t0 = time.time()
                    resp = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=512 if cot else 128, **_maybe_temperature(model), **kw,
                    )
                    log_cost("reanswer_locomo", model, resp, time.time() - t0)
                    out = resp.choices[0].message.content.strip()
                    break
                except Exception as e:
                    if attempt == 2:
                        out = f"__ERROR__: {e}"
                    else:
                        await asyncio.sleep(2 ** attempt * 3)
        if cot:
            qa[pk + "_cot_trace"] = out  # 完整推理链存档，评分只取末行短答
            m = re.search(r"Answer:\s*(.+?)\s*$", out, re.MULTILINE)
            out = m.group(1) if m else out.split("\n")[-1].strip()
        if cat == 5:
            ol = out.lower().strip().strip("()")
            if ol == "a":
                out = adv
            elif ol == "b":
                out = "Not mentioned in the conversation"
        return out

    tasks, index = [], []
    for sample in samples:
        sid = sample["sample_id"]
        conv = data.get(sid, {}).get("conversation", {})
        sa, sb = conv.get("speaker_a", ""), conv.get("speaker_b", "")
        db_path = str(Path(args.sqlite_dir) / f"{sid}.db")
        qas = sample["qa"][: args.limit] if args.limit else sample["qa"]
        for qa in qas:
            if args.category is not None and qa["category"] != args.category:
                continue
            tasks.append(one(db_path, sa, sb, qa))
            index.append(qa)

    print(f"共 {len(tasks)} 题待重答...", flush=True)
    results = await asyncio.gather(*tasks)
    n_err = 0
    for qa, pred in zip(index, results):
        if str(pred).startswith("__ERROR__"):
            n_err += 1
        qa[pk] = pred
        qa[pk + "_f1"] = score_answer(pred, qa.get("answer", ""), qa["category"])
        qa[pk + "_fulltext"] = True

    # 分题型汇总
    agg = defaultdict(lambda: [0.0, 0])
    for qa in index:
        agg[qa["category"]][0] += qa[pk + "_f1"]
        agg[qa["category"]][1] += 1
    names = {1: "Multi-hop", 2: "Temporal", 3: "Open-domain", 4: "Single-hop", 5: "Adversarial"}
    tf = tn = 0
    print("\n== 全文口径 F1 ==")
    for c in sorted(agg):
        f, n = agg[c]
        print(f"  cat{c} {names.get(c, '?'):12s}: n={n:4d}  F1={f/n:.4f}")
        tf += f; tn += n
    print(f"  Overall: F1={tf/tn:.4f}  (错误 {n_err})")

    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    json.dump(samples, open(args.out_file, "w"), ensure_ascii=False, indent=2)
    print(f"→ {args.out_file}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-file",
                   default=str(PROJECT_ROOT / "outputs/locomo/locomo_bge_v2_rr_results.json"))
    p.add_argument("--data-file",
                   default=str(PROJECT_ROOT / "locomo/data/locomo10.json"))
    p.add_argument("--out-file",
                   default=str(PROJECT_ROOT / "outputs/locomo/locomo_bge_v2_rr_fulltext_results.json"))
    p.add_argument("--sqlite-dir", default=str(PROJECT_ROOT / "emo/memory/locomo_sqlite_bge"))
    p.add_argument("--prediction-key", default=None)
    p.add_argument("--model", default="dashscope/qwen-plus")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--base-url", default=None, help="自定义 API 端点（如 litellm 代理）")
    p.add_argument("--api-key-env", default="DASHSCOPE_API_KEY", help="API key 的环境变量名")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--category", type=int, default=None, help="只跑指定品类（如 3=open-domain）")
    p.add_argument("--prompt-version", choices=["v2", "v3"], default="v2",
                   help="默认分支（cat1/4）prompt 版本：v3=MemPro 形态条款移植")
    p.add_argument("--cat3-cot", action="store_true", help="cat3 用两阶段 CoT prompt（先引记忆再世界知识推理，末行 Answer: X 短答）")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(reanswer(parse_args()))
