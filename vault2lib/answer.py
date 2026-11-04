"""读侧答题主链路：检索 → session 合并 → 判缺补充 → 证据条件检测 → 统一答题。"""
import argparse
import asyncio
import json
import os
import re
from pathlib import Path

from reanswer_locomo_fulltext import MEMORY_CONTEXT_HEADER, fetch_full_lines

from vault2lib import constants as C
from vault2lib.llm import call_llm, make_client, parse_json_obj
from vault2lib.prompts import INFER_DETECT_PROMPT, REASON_ADDON, UNIFIED_ANSWER_PROMPT
from vault2lib.retrieval import csr_retrieve, gap_supplements, session_merge, timeline_block
from vault2lib.temporal import duration_ref


async def answer_one(client, sem, db_dir, sid, question, answer_prompt=None):
    top_ids, top_c = csr_retrieve(db_dir, sid, question)
    lines, miss = fetch_full_lines(str(Path(db_dir) / f"{sid}.db"), top_ids)
    valid_ids = [m for m in top_ids if m not in set(miss)]
    lines = session_merge(db_dir, sid, valid_ids, lines)
    supplements = await gap_supplements(client, sem, db_dir, sid, question, lines)
    numbered = "\n".join(f"{i+1}. {l[:C.TRUNC_DETECT]}" for i, l in enumerate(lines))
    det = parse_json_obj(await call_llm(client, sem, C.AUX_MODEL,
                                        INFER_DETECT_PROMPT.format(question=question, lines=numbered),
                                        "infer_detect", max_tokens=30))
    infer, temporal = bool(det.get("infer")), bool(det.get("temporal"))
    tl = "" if temporal else timeline_block(db_dir, sid, top_c)
    header = MEMORY_CONTEXT_HEADER.format(speaker_a="the user", speaker_b="the assistant")
    ctx = header + (tl + "\n" if tl else "") + "\n".join(f"- {l}" for l in lines) \
        + ("\n" + "\n".join(f"- {s}" for s in supplements) if supplements else "")
    ref = await duration_ref(client, sem, question, lines) if C.DUR_Q_RE.search(question) else ""
    prompt = (answer_prompt or UNIFIED_ANSWER_PROMPT).format(question=question, ref=ref)
    if infer:
        prompt += REASON_ADDON
    out_text = await call_llm(client, sem, C.READER_MODEL, ctx + "\n\n" + prompt,
                              "answer", max_tokens=400)
    if infer:
        m = re.search(r"Answer:\s*(.+?)\s*$", out_text, re.MULTILINE)
        return (m.group(1) if m else out_text.split("\n")[-1]).strip()
    return out_text.strip()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    qs = [json.loads(l) for l in open(args.questions)]
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["qa_id"] for l in open(args.out)}
    todo = [q for q in qs if q["qa_id"] not in done]
    print(f"vault2: {len(qs)} 题，待跑 {len(todo)}", flush=True)

    client = make_client()
    sem = asyncio.Semaphore(args.concurrency)

    async def one(q):
        try:
            pred = await answer_one(client, sem, args.db, q["sample_id"], q["question"])
        except Exception as e:  # noqa: BLE001  与 csr_ab 同口径：失败题不提交（分母 1380）
            print(f"  [skip] {q['qa_id']}: {type(e).__name__}: {e}", flush=True)
            return None
        return {"qa_id": q["qa_id"], "predicted_answer": pred}

    with open(args.out, "a") as f:
        for i in range(0, len(todo), 50):
            batch = await asyncio.gather(*[one(q) for q in todo[i:i + 50]])
            for r in batch:
                if r:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  [{min(i + 50, len(todo))}/{len(todo)}]", flush=True)
    print("vault2 完成 →", args.out, flush=True)
