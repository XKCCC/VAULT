# Reproducing VAULT's headline numbers

Every number below is backed by evidence files shipped in this repository (`outputs/`), so you can
also verify aggregates without re-running anything (see §7).

## 0. Environment

```bash
# Python 3.10+; a CUDA GPU is strongly recommended (bge-m3 embedding + CrossEncoder rerank).
# CPU works but is 30-60x slower on import/indexing.
pip install -r requirements.txt

# Local models (BAAI official sources; HuggingFace or ModelScope mirrors work):
mkdir -p emo/models
huggingface-cli download BAAI/bge-m3 --local-dir emo/models/bge-m3
huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir emo/models/bge-reranker-v2-m3

# API keys (OpenAI-compatible endpoints):
export DASHSCOPE_API_KEY=sk-...     # qwen-plus / qwen3.7-max (answering, dreaming, judging)
export OPENAI_API_KEY=sk-...        # gpt-4o-mini judge (official OpenAI or compatible proxy)
```

## 1. Data (not redistributed — fetch from upstream, respect upstream licenses)

| Dataset | Upstream | Expected location |
|---|---|---|
| LoCoMo (`locomo10.json`) | https://github.com/snap-research/locomo | `locomo/data/locomo10.json` |
| LongMemEval (`longmemeval_s_cleaned.json`) | https://github.com/xiaowu0162/LongMemEval | `LongMemEval/data/longmemeval_s_cleaned.json` |
| LifeBench | https://github.com/zhiyuan5986/LifeBench | `LifeBench-memory/life_bench_data/locomo_format/our_en.json` |
| mem0 memory-benchmarks (judge prompts for LoCoMo) | https://github.com/mem0ai/memory-benchmarks | `memory-benchmarks/` (repo root) |

All eval scripts accept `--data-file` to override these defaults.

## 2. LoCoMo headline (J 83.19 / 86.97 top-10; 86.11 / 89.55 top-50)

```bash
# First pass: import raw turns with bge-m3, dream (steps 1,2,4), answer top-10 (~3-5 h on one GPU).
# Resumable: re-running the same command skips already-dreamed/imported items.
EMO_COST_LOG=$PWD/outputs/cost_log_locomo_dream_bge.jsonl \
python emo/scripts/eval_locomo.py \
  --model dashscope/qwen-plus --prompt-version v3 --rerank --graph-expand \
  --dream --dream-steps 1,2,4 --dream-batch 32 --raw-turns \
  --embed-model emo/models/bge-m3 \
  --chroma-dir emo/memory/locomo_chroma_dream_bge --sqlite-dir emo/memory/locomo_sqlite_dream_bge \
  --out-file outputs/locomo/locomo_dream_bge_full.json

# Second pass (same library, retrieval+answering only, top-50 SOTA row, ~1 h):
python emo/scripts/eval_locomo.py \
  --model dashscope/qwen-plus --prompt-version v3 --rerank --graph-expand --top-k 50 \
  --embed-model emo/models/bge-m3 \
  --chroma-dir emo/memory/locomo_chroma_dream_bge --sqlite-dir emo/memory/locomo_sqlite_dream_bge \
  --out-file outputs/locomo/locomo_dream_bge_top50_v3.json

# Judges (qwen3.7-max via DashScope; gpt-4o-mini via OPENAI_API_KEY + --judge-base-url):
python emo/scripts/judge_locomo.py \
  --files outputs/locomo/locomo_dream_bge_full.json outputs/locomo/locomo_dream_bge_top50_v3.json
python emo/scripts/judge_locomo.py \
  --files outputs/locomo/locomo_dream_bge_full.json outputs/locomo/locomo_dream_bge_top50_v3.json \
  --judge-model gpt-4o-mini --judge-base-url https://api.openai.com/v1 --concurrency 16

# No-dream same-embedder control (the +4.4/+4.7 J comparison):
python emo/scripts/eval_locomo.py \
  --model dashscope/qwen-plus --prompt-version v3 --rerank --graph-expand \
  --embed-model emo/models/bge-m3 \
  --chroma-dir emo/memory/locomo_chroma_bge --sqlite-dir emo/memory/locomo_sqlite_bge \
  --out-file outputs/locomo/locomo_nodream_bge_v3.json
python emo/scripts/judge_locomo.py --files outputs/locomo/locomo_nodream_bge_v3.json
```

## 3. LongMemEval headline (0.716 → 0.816, n=500)

500 questions, each with an independent memory library (~247k turns total). GPU required.
All runs are resumable (per-question checkpointing); shard with `--shard i/N` for parallelism
(~8 GB VRAM per process; multiple processes may share one GPU).

```bash
# --- NoOrg baseline (retrieval only, main library) ---
python emo/scripts/eval_longmemeval.py --model dashscope/qwen-plus --rerank \
  --out-file outputs/longmemeval/emo_lme_s_results.json
# (eval_locomo/eval_longmemeval now answer from full raw_content; no separate re-answer step needed)

# --- VAULT full (dream organization; SEPARATE library dirs — dreaming rewrites the library in place) ---
python emo/scripts/eval_longmemeval.py --model dashscope/qwen-plus --rerank \
  --dream --dream-batch 8 \
  --chroma-dir emo/memory/lme_chroma_dream --sqlite-dir emo/memory/lme_sqlite_dream \
  --out-file outputs/longmemeval/emo_lme_s_dreamfull_results.json

# --- Judges: two judges × two prompt conventions ---
python emo/scripts/judge_longmemeval.py --file outputs/longmemeval/emo_lme_s_results.json          # qwen3.7-max, anscheck
python emo/scripts/judge_longmemeval.py --file outputs/longmemeval/emo_lme_s_dreamfull_results.json
python emo/scripts/judge_longmemeval.py --file outputs/longmemeval/emo_lme_s_results.json \
  --judge-model gpt-4o-mini --judge-base-url https://api.openai.com/v1 --prompt-style anscheck --concurrency 16
python emo/scripts/judge_longmemeval.py --file outputs/longmemeval/emo_lme_s_dreamfull_results.json \
  --judge-model gpt-4o-mini --judge-base-url https://api.openai.com/v1 --prompt-style anscheck --concurrency 16
# --prompt-style mempro gives the MemPro-convention numbers (0.758 / 0.842).

# --- Official retrieval metrics (recall_all / recall_any / NDCG@10, offline, seconds) ---
python emo/scripts/recalc_lme_official_retrieval.py \
  outputs/longmemeval/emo_lme_s_results.json \
  LongMemEval/data/longmemeval_s_cleaned.json \
  outputs/longmemeval/retrieval_official_metrics.json
```

## 4. Dream ablation (organization vs no-organization, same embedder)

Controlled pairs already shipped under `outputs/`:

| Benchmark | NoOrg | VAULT | Evidence files |
|---|---|---|---|
| LongMemEval | 0.716 | 0.816 | `emo_lme_s_fulltext_results_judge_detail.json` vs `emo_lme_s_dreamfull_fulltext_results_judge_detail.json` |
| LoCoMo (J, 4o-mini) | 82.30 | 86.97 | `outputs/locomo/judge_results.json` (`locomo_nodream_bge_v3` vs `locomo_dream_bge_full`) |

To re-run a pair, use §2/§3 commands with the two library directories shown there
(main library = NoOrg; `*_dream` library = VAULT). Component ablations (`--no-l3`,
`--no-supersede`, `--multi-channel`, budget curves) are documented in
`outputs/locomo/BENCHMARK_REPORT.md` and the per-file outputs under `outputs/longmemeval/`.

## 5. Cost measurement (paper efficiency table)

```bash
# All LLM calls (dreaming / answering / judging) log token usage + latency when EMO_COST_LOG is set:
EMO_COST_LOG=$PWD/outputs/cost_log_gpu.jsonl \
python emo/scripts/eval_longmemeval.py --model dashscope/qwen-plus --rerank \
  --dream --dream-batch 8 --stratified 4 \
  --chroma-dir emo/memory/lme_chroma_costmeasure --sqlite-dir emo/memory/lme_sqlite_costmeasure \
  --out-file outputs/longmemeval/emo_lme_s_costmeasure_s24.json

# Offline re-computation of online tokens from result files:
python emo/scripts/recompute_online_tokens.py
```

Raw logs backing the paper's numbers are shipped: `outputs/cost_log*.jsonl`, `outputs/cost_stats/*.json`.

## 6. LifeBench (secondary benchmark)

```bash
python emo/scripts/eval_lifebench.py --model dashscope/qwen-plus --rerank \
  --out-file outputs/lifebench/emo_lifebench_en_results.json
python emo/scripts/judge_lifebench.py \
  --file outputs/lifebench/emo_lifebench_en_results.json \
  --prediction-key emo_dashscope_qwen-plus_top10_rr_prediction
```

## 7. Verifying aggregates from shipped evidence (no re-run)

```python
import json, glob
for f in sorted(glob.glob('outputs/longmemeval/*_judge_*detail.json')):
    d = json.load(open(f))
    print(f"{f.split('/')[-1]:80s} acc = {sum(map(bool, d.values()))/len(d):.4f}  (n={len(d)})")
```

## 8. AML (Agent Memory Leaderboard) adapter

`emo/aml/` implements the AML Add/Search protocol on top of VAULT
(raw-index-immediately + background dream upgrade). See `emo/aml/README.md`
and `emo/aml/smoke_local.py` for the local smoke test.
