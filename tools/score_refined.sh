#!/bin/bash
# LoCoMo refined 官方可比尺判分：Qwen3-14B 权重 + 官方 judge prompt。
# 默认指向本地 vLLM（tools/serve_judge_qwen3_14b.sh，:8001，关思考）；
# 关思考是尺等价的前提（vLLM 默认开思考会收紧约 5 分）。
#
# 用法: bash tools/score_refined.sh <predictions.jsonl> [output_scored.jsonl]
# 覆盖: REFINED_REPO / JUDGE_BASE_URL / JUDGE_API_KEY / JUDGE_MODEL
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PRED="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
OUTP="${2:-${PRED%.jsonl}_scored.jsonl}"
mkdir -p "$(dirname "$OUTP")"
OUTP="$(cd "$(dirname "$OUTP")" && pwd)/$(basename "$OUTP")"
REFINED="${REFINED_REPO:-/tmp/LoCoMo_refined}"
cd "$REFINED" || exit 1
EVALUATOR_MODEL="${JUDGE_MODEL:-qwen3-14b}" \
EVALUATOR_BASE_URL="${JUDGE_BASE_URL:-http://localhost:8001/v1}" \
EVALUATOR_API_KEY="${JUDGE_API_KEY:-token-abc}" \
conda run -n emo --no-capture-output python src/evaluate.py \
  --questions-path data/public/questions.jsonl \
  --predictions-path "$PRED" \
  --output-path "$OUTP" \
  --concurrency 8
