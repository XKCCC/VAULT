#!/usr/bin/env python3
"""LongMemEval LLM-judge：官方 anscheck prompt + qwen3.7-max 裁判

prompt 逐字移植自 LongMemEval 官方 evaluate_qa.py（get_anscheck_prompt），
覆盖全部题型分支：
  - single-session-user / single-session-assistant / multi-session：标准版
  - temporal-reasoning：天数 off-by-one 宽容版
  - knowledge-update：允许带旧信息、只要更新答案正确
  - single-session-preference：rubric 版（answer 字段是期望回复 rubric）
  - abstention（question_id 含 _abs）：判"是否正确识别不可答"

⚠️ 官方裁判是 gpt-4o；我们用 qwen3.7-max（推理模型须 enable_thinking=False），
跨表对位只作水位参考，内部排序可信。

用法:
  python emo/scripts/judge_longmemeval.py --file outputs/longmemeval/emo_lme_s_results.json
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
from openai import AsyncOpenAI

DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
JUDGE_MODEL = "qwen3.7-max"


# ── 官方 prompt（逐字移植，勿改措辞）──
def get_anscheck_prompt(task, question, answer, response, abstention=False):
    if not abstention:
        if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'temporal-reasoning':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'knowledge-update':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'single-session-preference':
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        else:
            raise NotImplementedError(f"unknown task type: {task}")
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        prompt = template.format(question, answer, response)
    return prompt


# ── MemPro 版裁判 prompt（移植自 MemPro/eval/longmemeval_test.py，
#    规则逐字保留；判定对象措辞 "research summary"→"model response" 以适配我们的短答案，
#    宽松度规则不变：实体出现即算对/语义等价优先/缺上下文不罚）──
MEMPRO_JUDGE_PROMPT = """\
You are judging whether a model response contains the correct answer to a question.

Question:
{question}

Gold answer:
{answer}

Model response:
{response}

Return a JSON object with exactly these keys:
- "contains_answer": boolean
- "reason": short string

Rules:
- Mark true if the response contains the gold answer, a close paraphrase, or the main answer entity/phrase.
- For entity answers, be lenient: if the gold entity is present anywhere in the response, count it as correct even if extra context is missing.
- For short answers, do not require exact formatting if the meaning is clearly the same.
- Prefer semantic equivalence over exact wording.
- If the response clearly supports the intended answer, even with extra context or minor wording differences, mark true.
- Mark false if the response does not contain enough information to recover the gold answer.
- Do not output anything except JSON.
"""


async def judge_one(client, sem, record, judge_model, prompt_style, extra_body):
    qtype = record["question_type"]
    if prompt_style == "mempro":
        prompt = MEMPRO_JUDGE_PROMPT.format(
            question=record["question"], answer=str(record["answer"]),
            response=record["hypothesis"])
        max_tokens = 128
    else:
        prompt = get_anscheck_prompt(
            qtype,
            record["question"],
            str(record["answer"]),
            record["hypothesis"],
            abstention="_abs" in record["question_id"],
        )
        max_tokens = 16
    # O 系列/gpt-5 推理模型不支持 temperature，用 max_completion_tokens；
    # 且推理 token 会占用输出预算——给足空间否则 verdict 被截断为空
    reasoning = bool(re.match(r"^(o\d|gpt-5)", judge_model))
    async with sem:
        for attempt in range(8):
            try:
                kw = {"extra_body": extra_body} if extra_body else {}
                if reasoning:
                    kw["max_completion_tokens"] = max(max_tokens, 1024)
                else:
                    kw.update(temperature=0, max_tokens=max_tokens)
                t0 = time.time()
                r = await client.chat.completions.create(
                    model=judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    **kw,
                )
                log_cost("judge_lme", judge_model, r, time.time() - t0)
                txt = r.choices[0].message.content.strip()
                if prompt_style == "mempro":
                    m = re.search(r"\{.*\}", txt, re.DOTALL)
                    if m:
                        return bool(json.loads(m.group(0)).get("contains_answer"))
                    return None
                return "yes" in txt.lower()
            except Exception as e:
                err = str(e)
                transient = any(k in err for k in ("402", "429", "500", "502", "503", "504", "timeout", "Connection"))
                if attempt < 7:
                    await asyncio.sleep(25 if transient else 2 ** attempt)
                else:
                    print(f"  judge failed ({record['question_id']}): {err[:100]}", flush=True)
    return None


async def amain():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="eval_longmemeval.py 产出的结果文件")
    ap.add_argument("--judge-model", default="qwen3.7-max",
                    help="裁判模型名（litellm 代理按清单 key 写，如 openai/gpt-4o-mini）")
    ap.add_argument("--judge-base-url", default=None,
                    help="指定裁判 API 端点（如 litellm 代理 https://www.litellm.org/），key 读 OPENAI_API_KEY")
    ap.add_argument("--prompt-style", choices=["anscheck", "mempro"], default="anscheck",
                    help="anscheck=官方严格口径；mempro=MemPro 宽松口径（社区对位用）")
    ap.add_argument("--concurrency", type=int, default=10)
    args = ap.parse_args()

    global JUDGE_MODEL
    JUDGE_MODEL = args.judge_model

    records = json.load(open(args.file))
    todo = [r for r in records if r.get("hypothesis")]
    print(f"{len(todo)}/{len(records)} 题待裁判 (judge={JUDGE_MODEL}, style={args.prompt_style})", flush=True)

    if args.judge_base_url or JUDGE_MODEL.startswith("gpt"):
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            sys.exit("错误：未检测到 OPENAI_API_KEY 环境变量。请先执行：\n"
                     "  export OPENAI_API_KEY=\"sk-...\"   # litellm 代理 key")
        client = AsyncOpenAI(
            api_key=key,
            base_url=args.judge_base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            timeout=120)
        extra_body = None
    else:
        key = os.environ.get("DASHSCOPE_API_KEY")
        if not key:
            sys.exit("错误：未检测到 DASHSCOPE_API_KEY 环境变量。")
        client = AsyncOpenAI(api_key=key, base_url=DASHSCOPE_BASE, timeout=120)
        extra_body = {"enable_thinking": False} if "qwen3.7" in JUDGE_MODEL else None
    sem = asyncio.Semaphore(args.concurrency)

    results = await asyncio.gather(*[
        judge_one(client, sem, r, JUDGE_MODEL, args.prompt_style, extra_body) for r in todo
    ])

    stats = defaultdict(lambda: {"correct": 0, "wrong": 0, "fail": 0})
    for r, ok in zip(todo, results):
        qt = r["question_type"] + ("_abs" if "_abs" in r["question_id"] else "")
        if ok is None:
            stats[qt]["fail"] += 1
        elif ok:
            stats[qt]["correct"] += 1
        else:
            stats[qt]["wrong"] += 1

    # 汇总（abs 单列，也并入总准确率——官方口径 abstention 是 500 题的一部分）
    print(f"\n{'='*60}\n  LongMemEval Judge Results (judge={JUDGE_MODEL}, style={args.prompt_style})\n{'='*60}")
    total_c = total_w = 0
    out = {"file": args.file, "judge": JUDGE_MODEL, "prompt_style": args.prompt_style, "by_type": {}}
    for qt in sorted(stats):
        v = stats[qt]
        n = v["correct"] + v["wrong"]
        acc = v["correct"] / n if n else 0
        out["by_type"][qt] = {"acc": round(acc, 4), "n": n, "judge_fail": v["fail"]}
        print(f"  {qt:32s}: acc = {acc:.3f}  (n={n}, fail={v['fail']})")
        total_c += v["correct"]
        total_w += v["wrong"]
    n_all = total_c + total_w
    out["overall"] = {"acc": round(total_c / n_all, 4) if n_all else 0, "n": n_all}
    print(f"\n  {'Overall':32s}: acc = {out['overall']['acc']:.3f}  (n={n_all})")

    # 默认口径（qwen3.7-max + anscheck）保持历史文件名不串档；其他组合带口径后缀
    legacy = (JUDGE_MODEL == "qwen3.7-max" and args.prompt_style == "anscheck")
    suffix = "" if legacy else f"_{JUDGE_MODEL.replace('/', '_')}_{args.prompt_style}"
    out_path = Path(args.file).with_name(Path(args.file).stem + f"_judge{suffix}.json")
    json.dump(out, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"\n结果保存: {out_path}")

    # 逐题判定落盘：消融/配置间的配对差分分析依赖它
    detail = {r["question_id"]: ok for r, ok in zip(todo, results)}
    detail_path = Path(args.file).with_name(Path(args.file).stem + f"_judge{suffix}_detail.json")
    json.dump(detail, open(detail_path, "w"), indent=2, ensure_ascii=False)
    print(f"逐题判定: {detail_path}")


if __name__ == "__main__":
    asyncio.run(amain())
