#!/usr/bin/env python3
"""LifeBench 全文重答（修复 summary 截断 200 字符的致盲 bug）

与 reanswer_lme_fulltext.py 同原理：不重检索，读已有结果的 retrieved_ids，
从 sqlite 拉 raw_content 全文，按 eval_lifebench.py 的 prompt 结构逐字复刻重答。
证据集与基线逐题一致，配对分析成立。纯 API，无 GPU 可跑。

用法：
  python emo/scripts/reanswer_lifebench_fulltext.py --sample-idx 0 --limit-qa 5   # 冒烟
  python emo/scripts/reanswer_lifebench_fulltext.py                              # 全量
裁判：
  python emo/scripts/judge_lifebench.py \
    --file outputs/lifebench/emo_lifebench_en_fulltext_results.json \
    --prediction-key emo_dashscope_qwen-plus_top10_rr_prediction
"""

import argparse
import asyncio
import json
import os
import re
import sqlite3
from pathlib import Path

from openai import AsyncOpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
PREDICTION_KEY = "emo_dashscope_qwen-plus_top10_rr_prediction"  # 默认；可用 --prediction-key 覆盖

# ── 以下常量/函数逐字复制自 eval_lifebench.py（保持 prompt 完全同构）──
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


def is_choice_question(question: str) -> bool:
    return bool(_CHOICE_RE.search(question))


def last_session_date(conv: dict) -> str:
    nums = [int(k.split("_")[1]) for k in conv
            if k.startswith("session_") and "date_time" not in k]
    return conv.get(f"session_{max(nums)}_date_time", "") if nums else ""


def fetch_full_texts(db_path: str, mem_ids: list) -> list:
    """L3 条目（raw_content 为占位符）改用 summary（洞察正文）"""
    if not Path(db_path).exists():
        return []
    con = sqlite3.connect(db_path)
    out = []
    for mid in mem_ids:
        row = con.execute(
            "SELECT raw_content, summary FROM memory_files WHERE mem_id=?", (mid,)
        ).fetchone()
        if row and row[0]:
            raw, summ = row
            if raw.startswith("L3 insight") and summ:
                out.append(f"[主题洞察] {summ}")
            else:
                out.append(raw)
    con.close()
    return out


def _maybe_temperature(model: str):
    """O 系列/gpt-5 推理模型不支持 temperature 参数（litellm 会 400）"""
    return {} if re.match(r"^(o\d|gpt-5|.*o4-mini)", model) else {"temperature": 0.0}


async def reanswer(args):
    pk = args.prediction_key
    samples = json.load(open(args.results_file))
    data = {str(s["sample_id"]): s for s in json.load(open(args.data_file))}
    if args.sample_idx is not None:
        samples = [samples[args.sample_idx]]

    client = AsyncOpenAI(
        api_key=os.environ[args.api_key_env],
        base_url=args.base_url or DASHSCOPE_BASE,
        timeout=180,
    )
    model = args.model.replace("dashscope/", "")
    extra = {"enable_thinking": False} if "qwen3.7" in model else None
    sem = asyncio.Semaphore(args.concurrency)

    async def one(db_path, speaker_a, now, qa):
        mem_ids = qa.get(pk + "_retrieved_ids") or []
        texts = fetch_full_texts(db_path, mem_ids)
        parts = []
        if texts:
            lines = [MEMORY_CONTEXT_HEADER.format(speaker_a=speaker_a)]
            lines.extend(f"- {t}" for t in texts)
            parts.append("\n\n".join(lines))
        else:
            parts.append(NO_MEMORY_PROMPT)
        template = QA_PROMPT + (QA_PROMPT_CHOICE_SUFFIX if is_choice_question(qa["question"]) else "")
        parts.append(template.format(now=now, question=qa["question"]))
        prompt = "\n\n".join(parts)
        async with sem:
            for attempt in range(3):
                try:
                    kw = {"extra_body": extra} if extra else {}
                    resp = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=1024, **_maybe_temperature(model), **kw,
                    )
                    return resp.choices[0].message.content.strip()
                except Exception as e:
                    if attempt == 2:
                        return f"__ERROR__: {e}"
                    await asyncio.sleep(2 ** attempt * 3)

    tasks, index = [], []
    for sample in samples:
        sid = str(sample["sample_id"])
        src = data.get(sid, {})
        conv = src.get("conversation", {})
        speaker_a = conv.get("speaker_a", sid)
        now = last_session_date(conv)
        db_path = str(Path(args.sqlite_dir) / Path(args.data_file).stem / f"{sid}.db")
        qas = sample["qa"][: args.limit_qa] if args.limit_qa else sample["qa"]
        for qa in qas:
            tasks.append(one(db_path, speaker_a, now, qa))
            index.append(qa)

    print(f"共 {len(tasks)} 题待重答...", flush=True)
    results = await asyncio.gather(*tasks)
    n_err = 0
    for qa, pred in zip(index, results):
        if str(pred).startswith("__ERROR__"):
            n_err += 1
        qa[pk] = pred
        qa[pk + "_fulltext"] = True
        # 旧分数是截断答案的判分，留着会被误读——权威判分以重判后的 detail 文件为准
        qa.pop(pk + "_score", None)
    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    json.dump(samples, open(args.out_file, "w"), ensure_ascii=False, indent=2)
    print(f"完成（错误 {n_err}）→ {args.out_file}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-file",
                   default=str(PROJECT_ROOT / "outputs/lifebench/emo_lifebench_en_results.json"))
    p.add_argument("--data-file",
                   default=str(PROJECT_ROOT / "LifeBench-memory/life_bench_data/locomo_format/our_en.json"))
    p.add_argument("--out-file",
                   default=str(PROJECT_ROOT / "outputs/lifebench/emo_lifebench_en_fulltext_results.json"))
    p.add_argument("--sqlite-dir", default=str(PROJECT_ROOT / "emo/memory/lifebench_sqlite"))
    p.add_argument("--model", default="dashscope/qwen-plus")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--base-url", default=None, help="自定义 API 端点（如 litellm 代理）")
    p.add_argument("--api-key-env", default="DASHSCOPE_API_KEY", help="API key 的环境变量名")
    p.add_argument("--sample-idx", type=int, default=None)
    p.add_argument("--limit-qa", type=int, default=None)
    p.add_argument("--prediction-key", default=PREDICTION_KEY)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(reanswer(parse_args()))
