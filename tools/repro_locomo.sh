#!/bin/bash
# LoCoMo refined 一键复现：建库（断点可续）→ 答 1382 题 → 官方可比尺判分。
# 用法: bash tools/repro_locomo.sh [lib_dir]
# 依赖: DASHSCOPE_API_KEY；权重见 emo/models/README.md（或 VAULT2_MODELS_DIR 改址）；
#       判分需 refined 仓库（REFINED_REPO 环境变量，默认 /tmp/LoCoMo_refined）
#       与本地裁判服务（tools/serve_judge_qwen3_14b.sh，关思考）。
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LIB="${1:-$ROOT/libs/locomo_repro}"
cd "$ROOT"
python run_build.py --out "$LIB"
python run_answer.py --db "$LIB" --questions locomo/data/refined_questions.jsonl \
    --out "${LIB}_preds.jsonl" --concurrency 8
bash tools/score_refined.sh "${LIB}_preds.jsonl" "${LIB}_preds_scored.jsonl"
