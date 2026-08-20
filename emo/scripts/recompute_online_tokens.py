"""离线复算在线答题输入 token 分布（零 API）——论文 Table 3 / fig5 的数据出处。

复用 reanswer_lme_fulltext.py 的 fetch_full_texts + QA_PROMPT（同一函数，保证口径一致），
tokenizer 用本地 Qwen2.5-7B-Instruct（与 qwen-plus 同族 BPE；近似口径，脚注声明）。

用法：
  python emo/scripts/recompute_online_tokens.py \
      --results-file outputs/longmemeval/emo_lme_s_dreamfull_fulltext_results.json \
      --sqlite-dir emo/memory/lme_sqlite_dream \
      --out-file outputs/cost_stats/lme_dreamfull_online_tokens.json
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reanswer_lme_fulltext import QA_PROMPT, fetch_full_texts  # noqa: E402

from transformers import AutoTokenizer  # noqa: E402

TOK = None


def count(text: str) -> int:
    return len(TOK.encode(text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-file", required=True)
    ap.add_argument("--sqlite-dir", required=True)
    ap.add_argument("--out-file", required=True)
    ap.add_argument("--tokenizer", default="Qwen2.5-7B-Instruct")
    args = ap.parse_args()

    global TOK
    TOK = AutoTokenizer.from_pretrained(args.tokenizer)

    records = json.load(open(args.results_file))
    per_q = []
    for rec in records:
        qid = rec["question_id"]
        db = str(Path(args.sqlite_dir) / "longmemeval_s_cleaned" / f"{qid}.db")
        texts = fetch_full_texts(db, rec.get("retrieved_ids", []))
        memories = (
            "\n\n".join(f"- {t}" for t in texts)
            if texts
            else "(No relevant memories were retrieved.)\n"
        )
        prompt = QA_PROMPT.format(
            memories=memories,
            question_date=rec.get("question_date", ""),
            question=rec["question"],
        )
        per_q.append(
            {
                "question_id": qid,
                "n_items": len(texts),
                "input_tokens": count(prompt),
                "answer_chars": len(rec.get("hypothesis") or ""),
            }
        )

    toks = [r["input_tokens"] for r in per_q]
    stats = {
        "source": args.results_file,
        "tokenizer": args.tokenizer,
        "n_questions": len(per_q),
        "input_tokens": {
            "median": statistics.median(toks),
            "mean": round(statistics.mean(toks), 1),
            "p90": sorted(toks)[int(len(toks) * 0.9)],
            "max": max(toks),
            "total": sum(toks),
        },
        "items_per_q_median": statistics.median(r["n_items"] for r in per_q),
    }
    out = Path(args.out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"stats": stats, "per_question": per_q}, open(out, "w"), ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
