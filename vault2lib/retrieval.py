"""读侧检索与证据组装：CSR 两级检索 / session 合并 / 判缺补充 / 时间线块。"""
import json
import math
import re
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
from bench_utils import get_embedding_fn
from reanswer_locomo_fulltext import fetch_full_lines

from vault2lib import constants as C
from vault2lib.llm import call_llm, parse_json_obj
from vault2lib.prompts import CHAIN_PROMPT, DIAG_PROMPT

_ENV = {}
_RAW = {}
_BM25 = {}
_XE = None


def get_xe():
    global _XE
    if _XE is None:
        from sentence_transformers import CrossEncoder

        _XE = CrossEncoder(C.RERANKER)
    return _XE


def get_env(db_dir, sid):
    key = (str(db_dir), sid)
    if key in _ENV:
        return _ENV[key]
    con = sqlite3.connect(str(Path(db_dir) / f"{sid}.db"))
    mems = {r[0]: (r[1], r[2] or "") for r in con.execute(
        "SELECT mem_id, session_id, summary FROM memory_files WHERE mem_id NOT LIKE 'l3_%'")}
    rows = con.execute(
        "SELECT topic_id, name, narrative, timeline_json FROM topic_cluster").fetchall()
    mrows = con.execute("SELECT topic_id, mem_id FROM topic_member").fetchall()
    ets = {r[0]: (r[1] or "") for r in con.execute(
        "SELECT mem_id, event_timestamp FROM memory_files")}
    turns = {r[0]: r[1] for r in con.execute("SELECT mem_id, turn_index FROM memory_files")}
    con.close()
    mem_of = {}
    for cid, mid in mrows:
        mem_of.setdefault(cid, []).append(mid)
    clusters = [{"cid": cid, "name": name, "overview": ov or "",
                 "timeline": json.loads(tl or "[]"), "members": mem_of.get(cid, [])}
                for cid, name, ov, tl in rows]
    embed = get_embedding_fn(C.EMBED)
    texts = [f"{c['name']}: {c['overview']} " + " ".join(
        f"{t.get('date', '')} {t.get('slot', '')}={t.get('value', '')}"
        for t in c["timeline"][:6]) for c in clusters]
    cl_v = None
    if clusters:
        v = np.array(embed(texts))
        cl_v = v / np.linalg.norm(v, axis=1, keepdims=True)
    mem_ids = sorted(mems)
    mv = np.array(embed([mems[m][1] for m in mem_ids]))
    mv = mv / np.linalg.norm(mv, axis=1, keepdims=True)
    _ENV[key] = {"con_path": str(Path(db_dir) / f"{sid}.db"), "mems": mems,
                 "clusters": clusters, "cl_v": cl_v, "embed": embed,
                 "mem_ids": mem_ids, "mv": mv, "midx": {m: i for i, m in enumerate(mem_ids)},
                 "ets": ets, "turns": turns}
    return _ENV[key]


def raw_map(db_dir, sid):
    key = (str(db_dir), sid)
    if key not in _RAW:
        con = sqlite3.connect(str(Path(db_dir) / f"{sid}.db"))
        _RAW[key] = {r[0]: (r[1] or "", bool(r[2]), r[3] or "")
                     for r in con.execute(
                         "SELECT mem_id, raw_content, superseded_by, session_id FROM memory_files"
                         " WHERE mem_id NOT LIKE 'l3_%'")}
        con.close()
    return _RAW[key]


def bm25_index(db_dir, sid):
    key = (str(db_dir), sid)
    if key not in _BM25:
        raws = raw_map(db_dir, sid)
        docs, df = {}, Counter()
        for mid, (raw, sup, sess) in raws.items():
            tf = Counter(re.findall(r"[a-z0-9]+", raw[:1500].lower()))
            docs[mid] = (tf, sup, sess)
            for t in tf:
                df[t] += 1
        n = len(docs)
        idf = {t: max(0.0, math.log((n - d + 0.5) / (d + 0.5) + 1)) for t, d in df.items()}
        _BM25[key] = (docs, idf, n)
    return _BM25[key]


def hybrid_recall(db_dir, sid, entity, facet, aka):
    env = get_env(db_dir, sid)
    aka = [a for a in (aka or []) if a and a.strip()][:3]
    q = f"{entity} {facet}".strip()
    qv = np.array(env["embed"]([q]))
    qv = qv / np.linalg.norm(qv, axis=1, keepdims=True)
    sem_rank = [env["mem_ids"][i] for i in (env["mv"] @ qv.T).flatten().argsort()[::-1][:15]]
    docs, idf, _ = bm25_index(db_dir, sid)
    q_tokens = set(re.findall(r"[a-z0-9]+", (entity + " " + facet + " " + " ".join(aka)).lower()))
    lex = [(mid, sum(idf.get(t, 0.0) for t in q_tokens if t in tf))
           for mid, (tf, sup, sess) in docs.items()]
    lex = [x for x in lex if x[1] > 0]
    lex_rank = [m for m, _ in sorted(lex, key=lambda x: x[1], reverse=True)[:15]]
    rrf = {}
    for rk, mid in enumerate(sem_rank):
        rrf[mid] = rrf.get(mid, 0.0) + 1.0 / (C.RRF_K + rk + 1)
    for rk, mid in enumerate(lex_rank):
        rrf[mid] = rrf.get(mid, 0.0) + 1.0 / (C.RRF_K + rk + 1)
    top = sorted(rrf, key=lambda m: rrf[m], reverse=True)[:8]
    raws = raw_map(db_dir, sid)
    out = [{"mem_id": m, "sess": raws[m][2], "superseded": raws[m][1],
            "text": raws[m][0][:260]} for m in top if m in raws]
    out.sort(key=lambda x: int(re.search(r"(\d+)", x["sess"] or "0").group(1))
             if re.search(r"(\d+)", x["sess"] or "0") else 0)
    return out


def csr_retrieve(db_dir, sid, question):
    env = get_env(db_dir, sid)
    qv = np.array(env["embed"]([question]))
    qv = qv / np.linalg.norm(qv, axis=1, keepdims=True)
    top_c = set()
    if env["cl_v"] is not None:
        sims = (env["cl_v"] @ qv.T).flatten()
        top_c = set(sims.argsort()[::-1][:C.N_CLUSTERS].tolist())
    cand = list({m for ci in top_c for m in env["clusters"][ci]["members"] if m in env["mems"]})
    cand_set = set(cand)
    rest_idx = [env["midx"][m] for m in env["mem_ids"] if m not in cand_set]
    if rest_idx:
        s3 = (env["mv"][rest_idx] @ qv.T).flatten()
        order = sorted(range(len(rest_idx)), key=lambda i: float(s3[i]), reverse=True)[:C.N_FALLBACK]
        cand += [env["mem_ids"][rest_idx[i]] for i in order]
    if not cand:
        return [], []
    scores = get_xe().predict([(question, env["mems"][m][1]) for m in cand])
    s_rank = sorted(range(len(cand)), key=lambda i: float(scores[i]), reverse=True)
    cand30 = [cand[i] for i in s_rank[:C.RRF_CAND]]
    raw_lines, miss = fetch_full_lines(env["con_path"], cand30)
    valid30 = [m for m in cand30 if m not in set(miss)]
    if len(raw_lines) == len(valid30) and valid30:
        r_scores = get_xe().predict([(question, l) for l in raw_lines])
        r_rank = sorted(range(len(valid30)), key=lambda i: float(r_scores[i]), reverse=True)
        s_pos = {mid: rk for rk, mid in enumerate(valid30)}
        r_pos = {valid30[i]: rk for rk, i in enumerate(r_rank)}
        fused = sorted(valid30,
                       key=lambda m: 1.0 / (C.RRF_K + s_pos[m] + 1) + 1.0 / (C.RRF_K + r_pos[m] + 1),
                       reverse=True)
        top = fused[:C.TOP_N]
    else:
        top = valid30[:C.TOP_N]
    return top, sorted(top_c)


def session_merge(db_dir, sid, top_ids, lines):
    env = get_env(db_dir, sid)
    mems, turns = env["mems"], env["turns"]
    segs, order = {}, []
    for mid, line in zip(top_ids, lines):
        sess = (mems.get(mid) or ("", ""))[0]
        if sess not in segs:
            segs[sess] = []
            order.append(sess)
        segs[sess].append({"turn": turns.get(mid, 0), "text": line})
    out = []
    for sess in order:
        grp = sorted(segs[sess], key=lambda x: x["turn"])
        out.append(grp[0]["text"] if len(grp) == 1 else "\n".join(g["text"] for g in grp))
    return out


def timeline_block(db_dir, sid, top_c):
    env = get_env(db_dir, sid)
    parts = []
    for ci in top_c[:3]:
        c = env["clusters"][ci]
        if not c["timeline"]:
            continue
        lines = [f"  - ({t.get('date', '')}) {t.get('slot', '')}: {t.get('value', '')}"
                 f"{'【当前】' if t.get('status') == 'current' else ''}"
                 for t in c["timeline"][:8]]
        parts.append(f"[时间线] {c['name']}:\n" + "\n".join(lines))
    return "\n".join(parts)


async def gap_supplements(client, sem, db_dir, sid, question, lines):
    numbered = "\n".join(f"{i+1}. {l[:C.TRUNC_DIAG]}" for i, l in enumerate(lines))
    diag = await call_llm(client, sem, C.AUX_MODEL,
                          DIAG_PROMPT.format(question=question, lines=numbered),
                          "diagnose", max_tokens=500)
    gaps = parse_json_obj(diag).get("gaps", []) or []
    supplements = []
    for g in gaps[:C.GAPS_MAX]:
        ent = (g.get("entity") or "").strip()
        if not ent:
            continue
        cand = hybrid_recall(db_dir, sid, ent, g.get("facet", ""), g.get("aka"))
        if not cand:
            continue
        chain_lines = "\n".join(
            f"{i+1}. {'[已被后续更新] ' if c['superseded'] else ''}{c['text']}"
            for i, c in enumerate(cand))
        sel = await call_llm(client, sem, C.AUX_MODEL,
                             CHAIN_PROMPT.format(entity=ent, facet=g.get("facet", ""),
                                                 time=g.get("time", "any"), question=question,
                                                 chain_lines=chain_lines),
                             "chain_select", max_tokens=200)
        picked = parse_json_obj(sel).get("selected", []) or []
        sel_ids = [cand[int(p) - 1]["mem_id"] for p in picked[:C.SUPP_MAX]
                   if isinstance(p, (int, float)) and 1 <= int(p) <= len(cand)]
        new_lines, _ = fetch_full_lines(str(Path(db_dir) / f"{sid}.db"), sel_ids)
        pool_set = set(lines)
        for l in new_lines:
            if l not in pool_set:
                supplements.append(f"[补充证据] {l}")
    return supplements
