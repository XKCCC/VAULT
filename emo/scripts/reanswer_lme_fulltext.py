#!/usr/bin/env python3
"""LongMemEval 全文重答（修复 summary 截断 200 字符的致盲问题）

背景：loader 的 summary = text[:200]，而 LME 金标 turn
中位 292 字符、assistant 金标中位 1225（98% 被截断）。答题 prompt 只注入
summary → reader 看不见针。本脚本不重检索（直接读主结果的 retrieved_ids），
从 sqlite 拉 raw_content 全文重新答题——检索证据集与主结果逐题一致，
配对分析（McNemar）因此成立。

用法：
  python emo/scripts/reanswer_lme_fulltext.py                      # 全量 qwen-plus
  python emo/scripts/reanswer_lme_fulltext.py --limit 5            # 冒烟
  python emo/scripts/reanswer_lme_fulltext.py --model dashscope/qwen3.7-max \
      --out-file outputs/longmemeval/emo_lme_s_fulltext_qwen37max_results.json
"""

import argparse
import asyncio
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

from cost_log import log_cost
from openai import AsyncOpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 与 eval_longmemeval.QA_PROMPT 逐字一致（只把记忆内容从截断 summary 换成全文）
QA_PROMPT = (
    "I will give you several relevant memory entries retrieved from the history chats "
    "between you and a user. Please answer the question based on the relevant memory entries.\n\n"
    "Memory Entries:\n\n{memories}\n\n"
    "Current Date: {question_date}\n"
    "Question: {question}\n"
    "Answer:"
)


def fetch_full_texts(db_path: str, mem_ids: list) -> list:
    """按 retrieved_ids 顺序拉全文；缺失的 mem_id 跳过。

    L3 条目（raw_content 为 "L3 insight from..." 占位符）改用 summary（洞察正文）。
    """
    if not Path(db_path).exists():
        return []
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    out = []
    for mid in mem_ids:
        row = cur.execute(
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
    records = json.load(open(args.results_file))
    if args.limit:
        records = records[: args.limit]

    client = AsyncOpenAI(
        api_key=os.environ[args.api_key_env],
        base_url=args.base_url or DASHSCOPE_BASE,
        timeout=180,
    )
    model = args.model.replace("dashscope/", "")
    extra = {"enable_thinking": False} if "qwen3.7" in model else None
    sem = asyncio.Semaphore(args.concurrency)

    async def one(rec):
        qid = rec["question_id"]
        db = str(Path(args.sqlite_dir) / "longmemeval_s_cleaned" / f"{qid}.db")
        texts = fetch_full_texts(db, rec.get("retrieved_ids", []))
        memories = "\n\n".join(f"- {t}" for t in texts) if texts else "(No relevant memories were retrieved.)\n"
        prompt = QA_PROMPT.format(
            memories=memories,
            question_date=rec.get("question_date", ""),
            question=rec["question"],
        )
        async with sem:
            for attempt in range(3):
                try:
                    kw = {"extra_body": extra} if extra else {}
                    t0 = time.time()
                    resp = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=1024,
                        **_maybe_temperature(model),
                        **kw,
                    )
                    log_cost("reanswer_lme", model, resp, time.time() - t0)
                    hyp = resp.choices[0].message.content.strip()
                    break
                except Exception as e:
                    if attempt == 2:
                        hyp = f"__ERROR__: {e}"
                    else:
                        await asyncio.sleep(2 ** attempt * 3)
        out = dict(rec)
        out["hypothesis"] = hyp
        out["answer_mode"] = "fulltext_raw_content"
        return out

    results = await asyncio.gather(*(one(r) for r in records))
    n_err = sum(1 for r in results if str(r["hypothesis"]).startswith("__ERROR__"))
    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out_file, "w"), ensure_ascii=False, indent=2)
    print(f"完成 {len(results)} 题（错误 {n_err}）→ {args.out_file}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-file",
                   default=str(PROJECT_ROOT / "outputs/longmemeval/emo_lme_s_results.json"))
    p.add_argument("--out-file",
                   default=str(PROJECT_ROOT / "outputs/longmemeval/emo_lme_s_fulltext_results.json"))
    p.add_argument("--sqlite-dir", default=str(PROJECT_ROOT / "emo/memory/lme_sqlite"))
    p.add_argument("--model", default="dashscope/qwen-plus")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--base-url", default=None, help="自定义 API 端点（如 litellm 代理）")
    p.add_argument("--api-key-env", default="DASHSCOPE_API_KEY", help="API key 的环境变量名")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(reanswer(parse_args()))
