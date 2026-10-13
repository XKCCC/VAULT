#!/bin/bash
# 本地 Qwen3-14B 裁判服务（LoCoMo refined 官方可比尺）。
# 关键：--default-chat-template-kwargs 关思考——vLLM 默认开思考会把尺收紧约 5 分，
# 与云端/dashscope 默认行为不等价（2026-10-12 实测根因）。
#
# 用法（按需覆盖环境变量）：
#   VAULT2_MODELS_DIR=/path/to/models JUDGE_CONDA_ENV=vllm2 JUDGE_TP=2 JUDGE_UTIL=0.7 \
#   CU13COMPAT_LIB=/path/to/cu13/libs bash tools/serve_judge_qwen3_14b.sh
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="${VAULT2_MODELS_DIR:-$ROOT/emo/models}"
# vllm2 环境的 cu13 运行时库（libcudart/libnvrtc 的 .so.13，vLLM 0.27 启动必需）；
# CU13COMPAT_LIB 可额外前置自定义目录
CONDA_BASE="$(conda info --base 2>/dev/null)"
export LD_LIBRARY_PATH="${CU13COMPAT_LIB:+$CU13COMPAT_LIB:}$CONDA_BASE/envs/${JUDGE_CONDA_ENV:-vllm2}/lib/python3.14/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH"
exec conda run -n "${JUDGE_CONDA_ENV:-vllm2}" --no-capture-output python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODELS_DIR/Qwen3-14B" \
  --served-model-name qwen3-14b \
  --tensor-parallel-size "${JUDGE_TP:-2}" \
  --port "${JUDGE_PORT:-8001}" \
  --max-model-len 8192 \
  --gpu-memory-utilization "${JUDGE_UTIL:-0.7}" \
  --enforce-eager \
  --default-chat-template-kwargs '{"enable_thinking": false}'
