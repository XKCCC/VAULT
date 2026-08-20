#!/usr/bin/env python3
"""LifeBench LLM-judge：分类裁判（选择=字母匹配，简答=yes/no 判定，UA=拒答判定）

论文口径：LoCoMo 格式 + LLM-as-judge（得分=答对比例），基座与裁判 GPT-5.1-Mini。
我们用 qwen3.7-max 裁判（推理模型须 enable_thinking=False），跨表对位仅作水位参考。

判分规则：
  - 选择题（gold 为字母）：模型输出提取字母精确匹配，不过 judge（确定性、免费）
  - 简答题（cat 0-3 文本答案）：yes/no 判准（LongMemEval anscheck 标准模板）
  - UA（cat 4 文本答案）：判模型是否正确识别"不可答"

用法:
  python emo/scripts/judge_lifebench.py \
      --file outputs/lifebench/emo_lifebench_results.json \
      --prediction-key emo_dashscope_qwen-plus_top10_prediction
"""

import argparse
import asyncio
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from openai import AsyncOpenAI

DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
JUDGE_MODEL = "qwen3.7-max"

CAT_NAMES = {"0": "IE", "1": "MR", "2": "ND", "3": "TKU", "4": "UA"}

# LongMemEval anscheck 标准模板（与 judge_longmemeval 同源）
ANSCHECK = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response is equivalent to the correct answer or contains all the intermediate "
    "steps to get the correct answer, you should also answer yes. If the response only "
    "contains a subset of the information required by the answer, answer no. \n\n"
    "Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
    "Is the model response correct? Answer yes or no only."
)

ABSTENTION = (
    "I will give you an unanswerable question and a response from a model. "
    "Please answer yes if the model correctly identifies the question as unanswerable "
    "(e.g., says the information is not available, not in memory, or unable to answer). "
    "Otherwise, answer no.\n\n"
    "Question: {question}\n\nModel Response: {response}\n\n"
    "Does the model correctly identify the question as unanswerable? Answer yes or no only."
)

_LETTER_RE = re.compile(r"\b([A-E])\b")


def extract_letter(prediction: str) -> str:
    p = prediction.strip()
    if len(p) <= 2 and p.upper() in "ABCDE":
        return p.upper()
    m = _LETTER_RE.search(p)
    return m.group(1) if m else ""


async def judge_one(client, sem, question, gold, prediction, ua):
    template = ABSTENTION if ua else ANSCHECK
    prompt = (template.format(question=question, response=prediction) if ua
              else template.format(question=question, answer=gold, response=prediction))
    async with sem:
        for attempt in range(3):
            try:
                r = await client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    extra_body={"enable_thinking": False},
                    max_tokens=16,
                )
                return "yes" in r.choices[0].message.content.strip().lower()
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                else:
                    print(f"  judge failed: {str(e)[:100]}", flush=True)
    return None


async def amain():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--prediction-key", required=True,
                    help="eval_lifebench.py 结束时打印的 prediction key")
    ap.add_argument("--judge-model", default="qwen3.7-max")
    ap.add_argument("--concurrency", type=int, default=10)
    args = ap.parse_args()

    global JUDGE_MODEL
    JUDGE_MODEL = args.judge_model

    data = json.load(open(args.file))
    pk = args.prediction_key

    # 选择题不过 judge：先确定性判掉；judge 任务只收简答+UA 文本题
    stats = defaultdict(lambda: {"correct": 0, "wrong": 0, "fail": 0})
    judge_tasks = []  # (cat, question, gold, prediction, ua)
    detail = {}       # question -> bool（配对差分分析依赖）
    for sample in data:
        for qa in sample["qa"]:
            pred = qa.get(pk)
            if pred is None:
                continue
            cat = str(qa["category"])
            gold = str(qa.get("answer", ""))
            if qa.get(pk + "_choice"):
                ok = extract_letter(str(pred)) == gold.strip().upper()
                stats[cat]["correct" if ok else "wrong"] += 1
                detail[qa["question"]] = ok
            else:
                judge_tasks.append((cat, qa["question"], gold, str(pred), cat == "4"))

    print(f"选择题已判: {sum(v['correct']+v['wrong'] for v in stats.values())} 题；"
          f"待裁判: {len(judge_tasks)} 题 (judge={JUDGE_MODEL})", flush=True)

    client = AsyncOpenAI(api_key=os.environ["DASHSCOPE_API_KEY"], base_url=DASHSCOPE_BASE, timeout=120)
    sem = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(*[
        judge_one(client, sem, q, g, p, ua) for _, q, g, p, ua in judge_tasks
    ])

    for (cat, q, *_), ok in zip(judge_tasks, results):
        if ok is None:
            stats[cat]["fail"] += 1
        else:
            stats[cat]["correct" if ok else "wrong"] += 1
            detail[q] = ok

    print(f"\n{'='*60}\n  LifeBench Judge Results (judge={JUDGE_MODEL})\n{'='*60}")
    total_c = total_w = 0
    out = {"file": args.file, "judge": JUDGE_MODEL, "prediction_key": pk, "by_category": {}}
    for cat in ["0", "1", "2", "3", "4"]:
        v = stats.get(cat)
        if not v:
            continue
        n = v["correct"] + v["wrong"]
        acc = v["correct"] / n if n else 0
        out["by_category"][CAT_NAMES[cat]] = {
            "acc": round(acc, 4), "n": n, "judge_fail": v["fail"]
        }
        print(f"  {CAT_NAMES[cat]:4s} (cat{cat}): acc = {acc:.3f}  (n={n}, fail={v['fail']})")
        total_c += v["correct"]
        total_w += v["wrong"]
    n_all = total_c + total_w
    out["overall"] = {"acc": round(total_c / n_all, 4) if n_all else 0, "n": n_all}
    print(f"\n  Overall: acc = {out['overall']['acc']:.3f}  (n={n_all})")

    out_path = Path(args.file).with_name(Path(args.file).stem + "_judge.json")
    json.dump(out, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"\n结果保存: {out_path}")

    detail_path = Path(args.file).with_name(Path(args.file).stem + "_judge_detail.json")
    json.dump(detail, open(detail_path, "w"), indent=2, ensure_ascii=False)
    print(f"逐题判定: {detail_path}")


if __name__ == "__main__":
    asyncio.run(amain())
