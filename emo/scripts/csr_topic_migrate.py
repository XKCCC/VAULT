#!/usr/bin/env python3
"""话题簇迁移（做梦即话题组织，2026-09-29）：conv-26 冒烟。

三段式（全 LLM，token 不省在该用的地方）：
  Pass1 主题发现：全库一句话摘要 → LLM 提出 10-25 个话题（一件事=一簇）
  Pass2 逐条归派：批量 30 条/次 → 每条分 1-2 个话题
  Pass3 主题文档：每簇一次 LLM → 叙事概述 + 时间线（绝对日期）
产出落 conv-26 库表 topic_cluster/topic_member，打印样例文档，并测 round-1
主题命中率（dense 查主题文档 top-3，看证据 turn 所在簇是否命中）。
"""
import json
import re
import sqlite3
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
import sys as _sys

_SQLITE_DIR = PROJECT_ROOT / "emo/memory/locomo_sqlite_dream_bge"
if "--dir" in _sys.argv:
    _SQLITE_DIR = Path(_sys.argv[_sys.argv.index("--dir") + 1])
    _sys.argv = [a for a in _sys.argv if a != "--dir" and a != str(_SQLITE_DIR)]
_MODEL = "qwen-plus"
if "--model" in _sys.argv:
    _MODEL = _sys.argv[_sys.argv.index("--model") + 1]
    _sys.argv = [a for a in _sys.argv if a != "--model" and a != _MODEL]
DB = _SQLITE_DIR / (_sys.argv[1] + ".db") if len(_sys.argv) > 1 else _SQLITE_DIR / "conv-26.db"
OUT = PROJECT_ROOT / "outputs/csr"
DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
import os
KEY = os.environ.get("DASHSCOPE_API_KEY", "")
COST = []

P1_PROMPT = """下面是一位用户记忆库中全部记忆的一句话摘要（{n} 条，按时间排）。

任务：把这些记忆归成 10-25 个**话题簇**——每个话题是"一件持续的事/一条事件线"（如"搬家到巴黎始末"、"学习陶艺"、"篮球生涯与退役"、"筹备领养孩子"），不要泛泛的主题词（如"工作"、"家庭"）。
要求：话题之间尽量不重叠；每个话题给话题名（具体、叙事性）+ 一句描述（谁、什么事）。

输出 JSON（只输出 JSON）：
{{"topics": [{{"id": "t1", "name": "...", "desc": "..."}}]}}

摘要：
{lines}"""

P2_PROMPT = """话题表：
{taxonomy}

下面是记忆条目（编号. (日期) 摘要）。给每条分配 1-2 个最相关的话题 id（实在没有合适的给 "new:话题名"）。

输出 JSON（只输出 JSON）：
{{"assign": {{"1": ["t3"], "2": ["t1", "t5"]}}}}

条目：
{lines}"""

P3_PROMPT = """话题「{name}」：{desc}

下面是该话题成员的记忆摘要，按时间先后排列（编号. (日期) 摘要）。

任务（只输出 JSON）：
1. narrative：≤120 字的叙事概述——谁、什么事、起因、经过、结果/现状（与原语言一致；要写成"任何问法都能对上"的完整小故事，关键人名/地名/物名必须出现）。
2. timeline：提取带日期的事件/槽位值变化，每条 {{"date": "YYYY-MM-DD 或原文日期", "slot": "方面", "value": "该时点的值或事件", "status": "current|history|event"}}；同一 slot 后值不同（搬家/改口/换偏好）：旧值 history、最新 current；演进全过程保留，history 不省略。

**日期权威约束（最高优先级）**：timeline 每条的 date **必须直接采用对应输入条目括号里的日期**，原样照抄；禁止根据"last Friday/yesterday/上周"等相对时间表达自行重算或另选日期。若某事件无法对应到输入条目的日期，则不要收录该事件。

条目：
{lines}"""


def llm(prompt, max_tokens=2500, model=None):
    model = model or _MODEL
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "temperature": 0, "max_tokens": max_tokens}
    for attempt in range(3):
        try:
            r = httpx.post(DASHSCOPE_BASE + "/chat/completions",
                           headers={"Authorization": f"Bearer {KEY}"}, json=payload, timeout=180)
            r.raise_for_status()
            d = r.json()
            COST.append({"pt": d["usage"]["prompt_tokens"], "ct": d["usage"]["completion_tokens"]})
            text = d["choices"][0]["message"]["content"].strip()
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                return json.loads(m.group(0))
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ llm: {type(e).__name__} {str(e)[:120]}", flush=True)
            time.sleep(4 * (attempt + 1))
    return {}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB))
    con.row_factory = sqlite3.Row
    mems = con.execute(
        "SELECT mem_id, session_id, turn_index, summary, raw_content FROM memory_files"
        " WHERE mem_id NOT LIKE 'l3_%' AND mem_id NOT LIKE '%session%' ORDER BY turn_index").fetchall()
    dates = {}
    for m in mems:
        dm = re.search(r"DATE: (.*?)\n", m["raw_content"] or "")
        dates[m["mem_id"]] = dm.group(1) if dm else ""
    print(f"记忆 {len(mems)} 条", flush=True)

    # ── Pass1：主题发现（一次喂全量单行摘要）──
    lines = "\n".join(f"{i+1}. ({dates[m['mem_id']]}) {(m['summary'] or '')[:80]}"
                      for i, m in enumerate(mems))
    p1 = llm(P1_PROMPT.format(n=len(mems), lines=lines), max_tokens=3000)
    topics = {t["id"]: t for t in p1.get("topics", [])}
    print(f"Pass1 话题 {len(topics)} 个", flush=True)

    # ── Pass2：逐条归派（30 条/批）──
    taxonomy = "\n".join(f"{t['id']}. {t['name']}：{t['desc']}" for t in topics.values())
    assign = {}
    B = 30
    batches = []
    for i in range(0, len(mems), B):
        batch = mems[i:i + B]
        blines = "\n".join(f"{i+j+1}. ({dates[m['mem_id']]}) {(m['summary'] or '')[:100]}"
                           for j, m in enumerate(batch))
        batches.append((i, blines))
    par = max(1, int(os.environ.get("MIGRATE_INTRA_PAR", "1")))

    def _p2_one(item):
        i, blines = item
        return i, llm(P2_PROMPT.format(taxonomy=taxonomy, lines=blines), max_tokens=2000)

    if par > 1 and len(batches) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(par, len(batches))) as ex:
            p2_results = list(ex.map(_p2_one, batches))
    else:
        p2_results = [_p2_one(b) for b in batches]
    for i, p2 in p2_results:
        for k, tids in (p2.get("assign") or {}).items():
            try:
                mid = mems[int(k) - 1]["mem_id"]
            except Exception:  # noqa: BLE001
                continue
            for tid in (tids if isinstance(tids, list) else [tids])[:2]:
                assign.setdefault(str(tid), []).append(mid)
        print(f"  归派 [{min(i+B, len(mems))}/{len(mems)}]", flush=True)
    # new: 话题并入最近邻（冒烟简化：落单独'杂项'）
    new_ids = [t for t in assign if t.startswith("new") or t not in topics]
    for t in new_ids:
        assign.setdefault("_misc", []).extend(assign.pop(t))

    # ── Pass3：主题文档 ──
    mem_by_id = {m["mem_id"]: m for m in mems}
    docs = {}
    p3_jobs = []
    for tid, name_desc in list(topics.items()) + [("_misc", {"name": "杂项", "desc": "未归派记忆"})]:
        mids = assign.get(tid, [])
        if not mids:
            continue
        def _sess_num(m):
            sm = re.search(r"(\d+)", mem_by_id[m]["session_id"] or "")
            return int(sm.group(1)) if sm else 0
        ordered = sorted(mids, key=lambda m: (_sess_num(m), mem_by_id[m]["turn_index"]))
        dlines = "\n".join(f"{j+1}. ({dates[m]}) {(mem_by_id[m]['summary'] or '')[:120]}"
                           for j, m in enumerate(ordered[:35]))
        p3_jobs.append((tid, name_desc, ordered, dlines))

    def _p3_one(job):
        tid, name_desc, ordered, dlines = job
        return tid, name_desc, ordered, llm(
            P3_PROMPT.format(name=name_desc["name"], desc=name_desc["desc"], lines=dlines),
            max_tokens=2500)

    if par > 1 and len(p3_jobs) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(par, len(p3_jobs))) as ex:
            p3_results = list(ex.map(_p3_one, p3_jobs))
    else:
        p3_results = [_p3_one(j) for j in p3_jobs]
    for tid, name_desc, ordered, p3 in p3_results:
        docs[tid] = {"name": name_desc["name"], "desc": name_desc["desc"],
                     "narrative": p3.get("narrative", ""), "timeline": p3.get("timeline", []),
                     "members": ordered}
        print(f"  文档 [{len(docs)}] {name_desc['name']}: 成员 {len(ordered)}", flush=True)

    # ── 落库 + 样例打印 ──
    con.execute("""CREATE TABLE IF NOT EXISTS topic_cluster (
        topic_id TEXT PRIMARY KEY, name TEXT, narrative TEXT,
        timeline_json TEXT DEFAULT '[]', n_members INT DEFAULT 0)""")
    con.execute("""CREATE TABLE IF NOT EXISTS topic_member (
        topic_id TEXT NOT NULL, mem_id TEXT NOT NULL,
        PRIMARY KEY (topic_id, mem_id))""")
    con.execute("DELETE FROM topic_cluster")
    con.execute("DELETE FROM topic_member")
    for tid, d in docs.items():
        con.execute("INSERT OR REPLACE INTO topic_cluster VALUES (?,?,?,?,?)",
                    (tid, d["name"], d["narrative"],
                     json.dumps(d["timeline"], ensure_ascii=False), len(d["members"])))
        con.executemany("INSERT OR IGNORE INTO topic_member VALUES (?,?)",
                        [(tid, m) for m in d["members"]])
    con.commit()

    show = sorted(docs.values(), key=lambda d: -len(d["members"]))[:3]
    for d in show:
        print(f"\n══ 话题簇样例: {d['name']}（成员 {len(d['members'])}）")
        print(f"  narrative: {d['narrative'][:250]}")
        for t in d["timeline"][:6]:
            print(f"    [{t.get('status','?'):8s}] {t.get('date','')} {t.get('slot','')}: {t.get('value','')[:60]}")

    (OUT / "topic_smoke.json").write_text(json.dumps({
        "topics": [{"id": tid, **{k: v for k, v in d.items() if k != "members"}}
                   for tid, d in docs.items()],
        "cost": {"llm_calls": len(COST), "tokens": sum(c["pt"] + c["ct"] for c in COST)},
    }, ensure_ascii=False, indent=2))
    print(f"\n→ {OUT / 'topic_smoke.json'}（成本 {len(COST)} 调用 {sum(c['pt']+c['ct'] for c in COST):,} tokens）")
    con.close()


if __name__ == "__main__":
    main()
