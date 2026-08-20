"""真 gpt-4o-mini 重判后的全量配对分析（2026-08-18 误标事故修复）。

输入：emo_lme_s_{fulltext,dreamfull_fulltext}_results_judge_{gpt-4o-mini_anscheck,
gpt-4o-mini_mempro,(qwen 默认)}_detail.json + 结果文件（question_type）
输出：McNemar / excl-abs / 分品类归因（图4数据源）/ 跨裁判翻转重合 / 成本汇总
"""
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

OUT = Path("outputs/longmemeval")


def load(name):
    return json.load(open(OUT / name))


def mcnemar(base, treat):
    """返回 b(仅基线对), c(仅处理对), 精确双侧 p"""
    b = sum(1 for q in base if base[q] and not treat.get(q, False))
    c = sum(1 for q in base if not base[q] and treat.get(q, False))
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)
    return b, c, p


def acc(d, qids=None):
    qs = [q for q, v in d.items() if qids is None or q in qids]
    return sum(1 for q in qs if d[q]) / len(qs), len(qs)


def main():
    res = {r["question_id"]: r for r in load("emo_lme_s_fulltext_results.json")}
    qtype = {q: r["question_type"] for q, r in res.items()}
    abs_ids = {q for q in res if q.endswith("_abs")}

    d = {
        "4omini_ans": {
            "base": load("emo_lme_s_fulltext_results_judge_gpt-4o-mini_anscheck_detail.json"),
            "dream": load("emo_lme_s_dreamfull_fulltext_results_judge_gpt-4o-mini_anscheck_detail.json"),
        },
        "4omini_mem": {
            "base": load("emo_lme_s_fulltext_results_judge_gpt-4o-mini_mempro_detail.json"),
            "dream": load("emo_lme_s_dreamfull_fulltext_results_judge_gpt-4o-mini_mempro_detail.json"),
        },
        "qwen_ans": {
            "base": load("emo_lme_s_fulltext_results_judge_detail.json"),
            "dream": load("emo_lme_s_dreamfull_fulltext_results_judge_detail.json"),
        },
    }

    print("=" * 70)
    print("① 总体 + McNemar（base vs dream，同题配对）")
    for tag, pair in d.items():
        a_b, n_b = acc(pair["base"])
        a_d, _ = acc(pair["dream"])
        b, c, p = mcnemar(pair["base"], pair["dream"])
        print(f"  {tag}: {a_b:.3f} -> {a_d:.3f}  (Δ{(a_d-a_b)*100:+.1f})  "
              f"McNemar {b}↔{c} 净{c-b:+d}  p={p:.2e}")

    print("=" * 70)
    print("② excl-abs 口径（剔除 _abs 题）")
    for tag, pair in d.items():
        if tag == "4omini_mem":
            continue
        non_abs = {q for q in res} - abs_ids
        a_b, _ = acc(pair["base"], non_abs)
        a_d, _ = acc(pair["dream"], non_abs)
        print(f"  {tag}: base {a_b*100:.1f} / dream {a_d*100:.1f}  (n={len(non_abs)})")

    print("=" * 70)
    print("③ 分品类归因（4o-mini anscheck；净翻转 = dream 仅对 - base 仅对）")
    pair = d["4omini_ans"]
    by_type = defaultdict(lambda: [0, 0, 0, 0])  # b_only, c_only, n, ...
    for q, t in qtype.items():
        vb, vd = pair["base"].get(q, False), pair["dream"].get(q, False)
        by_type[t][2] += 1
        if vb and not vd:
            by_type[t][0] += 1
        elif vd and not vb:
            by_type[t][1] += 1
    for t, (bo, co, n, _) in sorted(by_type.items()):
        a_b = sum(1 for q in qtype if qtype[q] == t and pair["base"].get(q)) / n
        a_d = sum(1 for q in qtype if qtype[q] == t and pair["dream"].get(q)) / n
        print(f"  {t:28s} n={n:3d}  {a_b:.3f}->{a_d:.3f}  净翻转 {co-bo:+d} ({bo}↔{co})")

    print("=" * 70)
    print("④ 跨裁判翻转清单重合（qwen anscheck vs 4o-mini anscheck）")
    def flips(pair):
        up = {q for q in pair["base"] if not pair["base"][q] and pair["dream"].get(q)}
        down = {q for q in pair["base"] if pair["base"][q] and not pair["dream"].get(q)}
        return up, down
    up_q, down_q = flips(d["qwen_ans"])
    up_g, down_g = flips(d["4omini_ans"])
    print(f"  上行翻转: qwen {len(up_q)} / 4omini {len(up_g)} / 交集 {len(up_q & up_g)}"
          f"  (覆盖率 {len(up_q & up_g)/max(len(up_q | up_g),1):.2f} Jaccard)")
    print(f"  下行翻转: qwen {len(down_q)} / 4omini {len(down_g)} / 交集 {len(down_q & down_g)}")

    print("=" * 70)
    print("⑤ 本轮实烧成本（outputs/cost_log.jsonl）")
    by_src = defaultdict(lambda: {"n": 0, "pt": 0, "ct": 0, "sec": []})
    for line in open("outputs/cost_log.jsonl"):
        r = json.loads(line)
        s = by_src[(r["src"], r["model"])]
        s["n"] += 1
        s["pt"] += r.get("pt") or 0
        s["ct"] += r.get("ct") or 0
        s["sec"].append(r.get("sec") or 0)
    tot = 0.0
    for (src, model), s in sorted(by_src.items()):
        cost = s["pt"] / 1e6 * 0.07 + s["ct"] / 1e6 * 0.30
        tot += cost
        print(f"  {src:14s} {model:24s} {s['n']:5d} 调用  "
              f"in {s['pt']/1e6:.2f}M  out {s['ct']/1e6:.2f}M  "
              f"时延中位 {statistics.median(s['sec']):.1f}s  ${cost:.3f}")
    print(f"  合计 ${tot:.3f}（naga 价 $0.07/$0.30）")


if __name__ == "__main__":
    main()
