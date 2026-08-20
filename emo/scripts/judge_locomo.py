#!/usr/bin/env python3
"""LLM-judge 打分：对已有 LoCoMo 预测结果做 CORRECT/WRONG 裁判评分

裁判: qwen3.7-max（dashscope，统一裁判）
prompt: mem0 memory-benchmarks 官方 judge prompt（cats 1-4，二元判定）

⚠️ 仅作水位参考：裁判型号与 Mem0 论文（gpt-4o-mini）不同，
跨表数字不可直接对位，只看我们自己各配置间的相对高低。

用法:
  python emo/scripts/judge_locomo.py \
      --files outputs/locomo/locomo_graph_full.json outputs/locomo/locomo_3b_results.json
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from cost_log import log_cost

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "memory-benchmarks"))

from openai import AsyncOpenAI
from benchmarks.locomo.prompts import JUDGE_SYSTEM_PROMPT, get_judge_prompt, preprocess_answer

DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
CAT_NAMES = {1: "Multi-hop", 2: "Temporal", 3: "Open-domain", 4: "Single-hop"}


async def judge_one(client, sem, category, question, gold, prediction):
    gold = preprocess_answer(category, str(gold))
    prompt = get_judge_prompt(category, question, gold, prediction)
    # O 系列/gpt-5 推理模型：不发 temperature、用 max_completion_tokens（推理 token 占预算）
    reasoning = bool(re.match(r"^(o\d|gpt-5)", JUDGE_MODEL))
    async with sem:
        for attempt in range(8):
            try:
                kw = {}
                if reasoning:
                    kw["max_completion_tokens"] = 1024
                else:
                    kw["temperature"] = 0
                    kw["max_tokens"] = 512
                    if "qwen3.7" in JUDGE_MODEL:
                        # qwen3.7-max 是推理模型：关闭思考（二元判定不需要），
                        # 否则推理 token 会吃掉 max_tokens 预算导致 verdict 截断
                        kw["extra_body"] = {"enable_thinking": False}
                t0 = time.time()
                r = await client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    **kw,
                )
                log_cost("judge_locomo", JUDGE_MODEL, r, time.time() - t0)
                txt = r.choices[0].message.content.strip()
                m = re.search(r'"label"\s*:\s*"(CORRECT|WRONG)"', txt, re.IGNORECASE)
                if not m:
                    m = re.search(r"\b(CORRECT|WRONG)\b", txt)
                if m:
                    return m.group(1).upper() == "CORRECT"
            except Exception as e:
                err = str(e)
                transient = any(k in err for k in ("402", "429", "500", "502", "503", "504", "timeout", "Connection"))
                if attempt < 7:
                    await asyncio.sleep(25 if transient else 2 ** attempt)
                else:
                    print(f"  judge failed: {err[:100]}", flush=True)
    return None


async def judge_file(client, path: Path, sem, out_records):
    data = json.load(open(path))
    stats = defaultdict(lambda: {"correct": 0, "wrong": 0, "fail": 0})

    tasks = []
    for s in data:
        keys = [k for k in s["qa"][0] if k.endswith("_prediction") and not k.endswith(("_f1", "_recall", "_retrieved_ids"))]
        if not keys:
            continue
        pk = keys[0]
        for qa in s["qa"]:
            cat = qa.get("category")
            pred = qa.get(pk)
            if cat not in (1, 2, 3, 4) or pred is None:
                continue
            tasks.append((cat, qa["question"], str(qa.get("answer", "")), str(pred)))

    print(f"  {path.name}: {len(tasks)} 题待裁判", flush=True)
    results = await asyncio.gather(*[
        judge_one(client, sem, c, q, g, p) for c, q, g, p in tasks
    ])

    for (cat, *_), ok in zip(tasks, results):
        if ok is None:
            stats[cat]["fail"] += 1
        elif ok:
            stats[cat]["correct"] += 1
        else:
            stats[cat]["wrong"] += 1

    total_c = sum(v["correct"] for v in stats.values())
    total_w = sum(v["wrong"] for v in stats.values())
    record = {"file": str(path), "judge": JUDGE_MODEL, "by_category": {}, "overall": {}}
    for cat in (4, 1, 2, 3):
        v = stats.get(cat)
        if not v:
            continue
        n = v["correct"] + v["wrong"]
        acc = v["correct"] / n if n else 0
        record["by_category"][CAT_NAMES[cat]] = {"j": round(acc * 100, 2), "n": n, "judge_fail": v["fail"]}
    n_all = total_c + total_w
    record["overall"] = {"j": round(total_c / n_all * 100, 2) if n_all else 0, "n": n_all}
    out_records.append(record)
    return record


async def amain():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--judge-model", default="qwen3.7-max")
    ap.add_argument("--judge-base-url", default=None,
                    help="指定裁判 API 端点（如 litellm 代理 https://www.litellm.org/），key 读 OPENAI_API_KEY")
    ap.add_argument("--concurrency", type=int, default=10)
    args = ap.parse_args()

    global JUDGE_MODEL
    JUDGE_MODEL = args.judge_model

    if args.judge_base_url:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            sys.exit("错误：未检测到 OPENAI_API_KEY 环境变量。请先执行：\n"
                     "  export OPENAI_API_KEY=\"sk-...\"   # litellm 代理 key")
        client = AsyncOpenAI(
            api_key=key,
            base_url=args.judge_base_url,
            timeout=120,
        )
    else:
        key = os.environ.get("DASHSCOPE_API_KEY")
        if not key:
            sys.exit("错误：未检测到 DASHSCOPE_API_KEY 环境变量。")
        client = AsyncOpenAI(
            api_key=key,
            base_url=DASHSCOPE_BASE,
            timeout=120,
        )
    sem = asyncio.Semaphore(args.concurrency)

    records = []
    for f in args.files:
        rec = await judge_file(client, Path(f), sem, records)
        print(f"  → {Path(f).name}: overall J = {rec['overall']['j']} (n={rec['overall']['n']})", flush=True)

    out_path = ROOT / "outputs" / "locomo" / "judge_results.json"
    # 合并而非覆盖（2026-08-12 事故：单次运行覆盖了历史全部条目）——
    # 按 (file, judge) 复合键替换：同一文件不同裁判各留一条（2026-08-14 o4-mini 覆盖 qwen 条目事故）
    if out_path.exists():
        try:
            old = json.load(open(out_path))
        except Exception:
            old = []
        old_map = {(r.get("file"), r.get("judge")): r for r in old if isinstance(r, dict) and r.get("file")}
        for rec in records:
            old_map[(rec["file"], rec.get("judge"))] = rec
        records = list(old_map.values())
    json.dump(records, open(out_path, "w"), indent=2, ensure_ascii=False)

    print(f"\n{'='*60}\n  LLM-judge 汇总 (judge={JUDGE_MODEL})\n{'='*60}")
    for rec in records:
        if "n" not in rec.get("overall", {}):
            continue  # 历史重建条目可能缺 n（2026-08-12 覆盖事故），跳过汇总打印
        print(f"\n  {Path(rec['file']).name}")
        for cat, v in rec["by_category"].items():
            print(f"    {cat:12s}: J = {v['j']:6.2f}  (n={v['n']})")
        print(f"    {'Overall':12s}: J = {rec['overall']['j']:6.2f}  (n={rec['overall']['n']})")
    print(f"\n结果保存: {out_path}")


if __name__ == "__main__":
    asyncio.run(amain())
