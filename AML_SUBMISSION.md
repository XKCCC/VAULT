# VAULT — AML Submission

Submission of the **VAULT** agent-memory system to the Agent Memory Leaderboard (AML).
This document covers: how to run the repository, the Docker command, where the
Add/Search protocol is implemented, method authorship & technical report, and the
complete list of methods used.

- Memory system: **VAULT** — *Versioned, Associative, and Unabridged Long-term Traces for Agent Memory*
- Repository: https://github.com/XKCCC/VAULT
- Contact: via the repository (issues) or the AML registration channel

---

## 1. How to run the repository

### 1.1 Requirements

- Python 3.10+, CUDA GPU recommended (bge-m3 embedding + CrossEncoder rerank; CPU works but is much slower)
- `pip install -r requirements.txt`
- Local models (BAAI official sources, HuggingFace or ModelScope):
  ```bash
  mkdir -p emo/models
  huggingface-cli download BAAI/bge-m3 --local-dir emo/models/bge-m3
  huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir emo/models/bge-reranker-v2-m3
  ```

### 1.2 Configuration (all via environment variables — no keys on disk)

| Variable | Default | Meaning |
|---|---|---|
| `AML_LLM_BASE` | dashscope compatible-mode | OpenAI-compatible endpoint of the **internal** LLM (used only by our offline dream pipeline) |
| `AML_LLM_MODEL` | `qwen-plus` | internal LLM model name — **set to `gpt-4o-mini` for official AML runs** |
| `AML_LLM_KEY_ENV` | `DASHSCOPE_API_KEY` | name of the env var holding the API key |
| `AML_API_KEY` | empty | Memory System Key we issue to the platform; empty = open smoke mode (no auth) |
| `AML_ROOT` | `emo/memory/aml_libs` | root of per-user memory libraries |
| `AML_DREAM_BATCH` | `16` | dream concurrency (32 on GPU) |
| `AML_ADD_PATH` / `AML_SEARCH_PATH` | `/add` / `/search` | endpoint paths (contract allows customization) |

### 1.3 Local self-test (does not consume platform quota)

```bash
export OPENAI_API_KEY=...            # or DASHSCOPE_API_KEY for dev
python emo/aml/smoke_local.py        # in-process TestClient: /health, /add, /search
```

### 1.4 Production server

```bash
export AML_LLM_BASE=https://api.naga.ac/v1   # gpt-4o-mini chain for official runs
export AML_LLM_MODEL=gpt-4o-mini
export AML_LLM_KEY_ENV=OPENAI_API_KEY
export OPENAI_API_KEY=...
export AML_API_KEY=<memory-system-key-issued-to-platform>
python emo/aml/server.py --host 0.0.0.0 --port 8000
```

Auth: `Authorization: Token|Bearer <key>` or `X-Api-Key: <key>`; `GET /health` is always open.

## 2. Docker

`Dockerfile` and `.dockerignore` are at the repository root. The image contains
Python deps + both local models (downloaded at build time); LLM access is provided
at run time via env vars (outbound HTTPS to the LLM endpoint is required).

```bash
# build (~5 GB image: torch CUDA base + bge-m3 + bge-reranker-v2-m3)
docker build -t vault-aml .

# run (GPU)
docker run --gpus all -p 8000:8000 \
  -e AML_LLM_BASE=https://api.naga.ac/v1 \
  -e AML_LLM_MODEL=gpt-4o-mini \
  -e AML_LLM_KEY_ENV=OPENAI_API_KEY \
  -e OPENAI_API_KEY=<key> \
  -e AML_API_KEY=<memory-system-key> \
  -v vault_aml_libs:/app/emo/memory/aml_libs \
  vault-aml

# smoke against the container
curl localhost:8000/health
curl -X POST localhost:8000/add -H 'Content-Type: application/json' \
  -d '{"request_id":"r1","user_id":"u1","session_id":"s1","messages":[{"role":"user","content":"I started learning piano on 2023-05-08."}]}'
curl -X POST localhost:8000/search -H 'Content-Type: application/json' \
  -d '{"query":"when did I start learning piano","user_id":"u1","top_k":5}'
```

Notes: the volume keeps per-user libraries persistent across restarts (required for
the 30-day availability window). For CPU-only hosts, drop `--gpus all` and set
`AML_DREAM_BATCH=8`. To skip the build-time model download, mount pre-downloaded
models with `-v /host/models:/app/emo/models`.

## 3. Where Add/Search is implemented

| Layer | File | Role |
|---|---|---|
| HTTP endpoints | `emo/aml/server.py` | FastAPI app: `POST /add`, `POST /search`, `GET /health`; Token/Bearer/X-Api-Key auth |
| Protocol → VAULT adapter | `emo/aml/adapter.py` | per-`user_id` isolated libraries; Add = persist raw + index synchronously (contract), dream upgrade in background; Search = retrieval + full-content return |
| Configuration | `emo/aml/config.py` | every knob via env vars |
| Memory engine | `emo/memory/` | `dreamer.py` (organization), `retriever.py` (recall), `index_store.py` (ChromaDB hot index), `persistent_store.py` (SQLite), `buffer.py`, `temporal.py`, `cache.py`, `assembler.py`, `schema.py` |
| Local contract test | `emo/aml/smoke_local.py` | end-to-end /add → /search without platform quota |

Behavioral contract notes: Add returns only after the content is searchable
(synchronous, as required). Search returns the **full raw content** of each hit
(unabridged — a core claim of VAULT), sorted by relevance, with `id`, `content`,
`score`, and `created_at` when known. `top_k` is honored up to the contract maximum of 100.

## 4. Original method authors & technical report

- **Method**: VAULT — *Versioned, Associative, and Unabridged Long-term Traces for Agent Memory*.
  Authors: the VAULT authors (anonymous, ICLR 2027 submission under double-blind review;
  author identities will be added upon publication).
- **Technical report**: the VAULT paper (ICLR 2027 submission; public link will be added
  here upon publication). Reproduction chains for every headline number: `REPRODUCING.md`.
- **Third-party components** (used, not claimed): bge-m3 and bge-reranker-v2-m3
  (BAAI, FlagEmbedding), ChromaDB, SQLite, Faiss, FastAPI, sentence-transformers.
- **Evaluation-only references** (not part of the memory system): mem0
  `memory-benchmarks` judge prompt for LoCoMo; MemPro judge prompt convention for
  LongMemEval. The dream pipeline's insight-extraction step is inspired by G-Memory.

## 5. Complete list of methods used

| # | Method | Where |
|---|---|---|
| 1 | **Immediate raw indexing on Add** — raw turns persisted (`status=raw`) and embedded synchronously so they are searchable before organization | `adapter.py`, `index_store.py`, `persistent_store.py` |
| 2 | **Background dream upgrade** — after a write burst settles (~15 s), raw entries are restructured into dated, normalized long-term memories | `adapter.py` (`_dream_loop`), `dreamer.py` |
| 3 | **Association graph + 1-hop graph expansion** — dream-built links between memories; query-time neighbor expansion | `dreamer.py`, `retriever.py` (`expand_graph`) |
| 4 | **Supersede versioning** — contradicted facts marked `superseded_by`, kept for provenance, excluded from retrieval | `dreamer.py`, `index_store.py` |
| 5 | **Absolute-date temporal anchoring + timeline union** — temporal expressions resolved to dates; temporal questions union a date-range channel | `temporal.py`, `persistent_store.py`, `retriever.py` |
| 6 | **L3 topic fusion / hierarchical channel** — dream clusters related memories into topic-level insights recalled alongside raw channels | `dreamer.py`, `retriever.py` |
| 7 | **Multi-channel recall + RRF fusion** — session buffer (Faiss + BM25) merged with long-term channels | `buffer.py`, `retriever.py` |
| 8 | **Dense retrieval with bge-m3** (GPU embeddings, ChromaDB) | `index_store.py`, `bench_utils.py` |
| 9 | **CrossEncoder rerank** with bge-reranker-v2-m3 | `bench_utils.py`, `adapter.py` |
| 10 | **MMR diversification** of the final list | `retriever.py` |
| 11 | **L1/L2 activation cache** for hot memories | `cache.py` |
| 12 | **Unabridged full-content injection** — Search returns complete `raw_content`, never summaries | `adapter.py` (`search`) |
| 13 | **Strict per-`user_id` isolation** — one independent library (Chroma + SQLite) per user, no cross-user sharing | `adapter.py` |
| 14 | **gpt-4o-mini as the only internal LLM** (dream pipeline) for official AML runs | `config.py` |

Answering and judging on the AML side are fixed by the platform and are outside this repository.
