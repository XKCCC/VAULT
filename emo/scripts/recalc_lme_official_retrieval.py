#!/usr/bin/env python3
"""LongMemEval 检索指标官方口径重算（离线，零重跑）

口径来源：LongMemEval/src/retrieval/eval_utils.py（官方实现逐行移植）
  - recall_any@k  : top-k 命中任一金标 turn 即 1
  - recall_all@k  : top-k 命中全部金标 turn 才算 1（论文表 3 报的 Recall）
  - ndcg@k        : 二元相关性，ideal DCG = 全集相关性降序截断 k（= ndcg_any@k）
  - session 级    : turn2session effective_k 扩展（排名里集齐 k 个去重 session 为止）
                    ⚠️ 近似：我们的检索只返回 top-k+时间并集余量，扩展深不过已返回列表，
                    因此 session 级偏保守（低估）；turn 级是精确值。

输入：emo_lme_s_results.json 的 retrieved_ids（主结果已落盘，无需重跑检索）
金标：longmemeval_s_cleaned.json 的 has_answer turn → doc_id 与 loader mem_id 同 scheme
过滤：与官方 print_retrieval_metrics.py 一致，剔除 _abs 拒答题
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
K = 10


def dcg(rels, k):
    rels = rels[:k]
    if not rels:
        return 0.0
    return rels[0] + sum(r / math.log2(i + 1) for i, r in enumerate(rels) if i >= 1)


def ndcg(ranked_ids, correct, k):
    rels = [1 if d in correct else 0 for d in ranked_ids[:k]]
    # ideal：金标全在最前（官方用全 corpus 相关性降序，等价于 min(len(correct),k) 个 1 打头）
    ideal = [1] * len(correct)
    idcg = dcg(ideal, k)
    return (dcg(rels, k) / idcg) if idcg else None


def eval_turn(ranked_ids, correct, k=K):
    top = ranked_ids[:k]
    recall_any = float(any(d in top for d in correct))
    recall_all = float(all(d in top for d in correct))
    return recall_any, recall_all, ndcg(ranked_ids, correct, k)


def eval_session(ranked_ids, correct_sessions, n_correct_corpus_turns, k=K):
    """turn2session：doc_id 去尾段得 session id；effective_k 扩到集齐 k 个去重 session

    n_correct_corpus_turns：全 corpus 中属于金标 session 的 turn 总数——官方 ideal
    DCG 按 corpus 相关性降序，同一金标 session 的每条 turn 都计 1。
    """
    def strip(d):
        return d.rsplit("_t", 1)[0]

    ranked_sessions = [strip(d) for d in ranked_ids]
    effective_k = min(k, len(ranked_sessions))
    uniq = set(ranked_sessions[:effective_k])
    while effective_k < len(ranked_sessions) and len(uniq) < k:
        effective_k += 1
        uniq = set(ranked_sessions[:effective_k])
    top = set(ranked_sessions[:effective_k])
    recall_any = float(any(s in top for s in correct_sessions))
    recall_all = float(all(s in top for s in correct_sessions))
    # 官方 ndcg(k=effective_k)：actual 与 ideal 同按 effective_k 截断；
    # ideal = corpus 相关性降序（金标 session 的全部 turn 都为 1）
    rels = [1 if s in correct_sessions else 0 for s in ranked_sessions[:effective_k]]
    ideal = [1] * n_correct_corpus_turns
    idcg = dcg(ideal, effective_k)
    ndcg_s = (dcg(rels, effective_k) / idcg) if idcg else None
    return recall_any, recall_all, ndcg_s


def main():
    results_file = sys.argv[1] if len(sys.argv) > 1 else str(
        PROJECT_ROOT / "outputs/longmemeval/emo_lme_s_results.json")
    data_file = sys.argv[2] if len(sys.argv) > 2 else str(
        PROJECT_ROOT / "LongMemEval/data/longmemeval_s_cleaned.json")
    out_file = sys.argv[3] if len(sys.argv) > 3 else str(
        PROJECT_ROOT / "outputs/longmemeval/retrieval_official_metrics.json")

    results = json.load(open(results_file))
    data = {inst["question_id"]: inst for inst in json.load(open(data_file))}

    per_q = []
    missing_gold, missing_ret = 0, 0
    for r in results:
        qid = r["question_id"]
        if "_abs" in qid:  # 官方过滤拒答题
            continue
        inst = data.get(qid)
        if inst is None:
            missing_gold += 1
            continue
        # 金标 turn doc_id（与 loader mem_id 同 scheme：qid_s{si}_t{ti}）
        correct = set()
        for si, sess in enumerate(inst["haystack_sessions"]):
            for ti, turn in enumerate(sess):
                if turn.get("has_answer"):
                    correct.add(f"{qid}_s{si}_t{ti}")
        if not correct:
            missing_gold += 1
            continue
        retrieved = r.get("retrieved_ids") or r.get("retrieved") or []
        if not retrieved:
            missing_ret += 1
            continue

        t_any, t_all, t_ndcg = eval_turn(retrieved, correct)
        correct_sess = {d.rsplit("_t", 1)[0] for d in correct}
        # corpus 中属于金标 session 的 turn 总数（session 级 ideal DCG 用）
        gold_si = {int(d.rsplit("_s", 1)[1].rsplit("_t", 1)[0]) for d in correct}
        n_correct_corpus_turns = sum(
            len(inst["haystack_sessions"][si]) for si in gold_si
        )
        s_any, s_all, s_ndcg = eval_session(retrieved, correct_sess, n_correct_corpus_turns)
        per_q.append({
            "question_id": qid,
            "question_type": inst["question_type"],
            "n_gold_turns": len(correct),
            "n_returned": len(retrieved),
            "turn_recall_any@10": t_any, "turn_recall_all@10": t_all, "turn_ndcg@10": t_ndcg,
            "sess_recall_any@10": s_any, "sess_recall_all@10": s_all, "sess_ndcg@10": s_ndcg,
        })

    keys = ["turn_recall_any@10", "turn_recall_all@10", "turn_ndcg@10",
            "sess_recall_any@10", "sess_recall_all@10", "sess_ndcg@10"]
    by_type = defaultdict(lambda: defaultdict(list))
    overall = defaultdict(list)
    gold_dist = defaultdict(int)
    for q in per_q:
        gold_dist[q["n_gold_turns"]] += 1
        for k_ in keys:
            if q[k_] is not None:
                by_type[q["question_type"]][k_].append(q[k_])
                overall[k_].append(q[k_])

    print(f"\n{'='*95}")
    print(f"  LongMemEval 检索【官方口径】重算（{Path(results_file).name}，turn 级精确 / session 级近似偏低）")
    print(f"{'='*95}")
    print(f"  {'type':28s} {'t_r_any':>8s} {'t_r_all':>8s} {'t_ndcg':>8s} {'s_r_any':>8s} {'s_r_all':>8s} {'s_ndcg':>8s} {'n':>5s}")
    for qt in sorted(by_type):
        a = by_type[qt]
        row = f"  {qt:28s}"
        for k_ in keys:
            row += f" {sum(a[k_])/len(a[k_]):>8.3f}"
        row += f" {len(a['turn_ndcg@10']):>5d}"
        print(row)
    row = f"  {'Overall':28s}"
    for k_ in keys:
        row += f" {sum(overall[k_])/len(overall[k_]):>8.3f}"
    row += f" {len(overall['turn_ndcg@10']):>5d}"
    print(row)
    print(f"\n  金标 turn 数分布: {dict(sorted(gold_dist.items()))}")
    print(f"  跳过: 无金标 {missing_gold}，无 retrieved_ids {missing_ret}")

    out = {"config": {"results_file": results_file, "k": K,
                      "note": "turn 级=官方精确口径；session 级=turn2session 近似（扩展深不过已返回列表，偏保守）"},
           "overall": {k_: round(sum(v)/len(v), 4) for k_, v in overall.items()},
           "by_type": {qt: {k_: round(sum(v)/len(v), 4) for k_, v in a.items()} for qt, a in by_type.items()},
           "per_question": per_q}
    json.dump(out, open(out_file, "w"), indent=2, ensure_ascii=False)
    print(f"\n明细保存: {out_file}")


if __name__ == "__main__":
    main()
