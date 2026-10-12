# 权重目录（不进 git）

默认权重根目录；可用环境变量 `VAULT2_MODELS_DIR` 整体改址。

| 目录 | 用途 | HF id |
|---|---|---|
| `bge-m3/` | 写侧建索引 + 读侧稠密检索嵌入 | `BAAI/bge-m3` |
| `bge-reranker-v2-m3/` | 读侧精排（CrossEncoder） | `BAAI/bge-reranker-v2-m3` |
| `Qwen3-14B/` | 本地裁判权重（可选，LoCoMo refined 官方可比尺） | `Qwen/Qwen3-14B` |

拉取（国内走 hf-mirror）：

```bash
HF_ENDPOINT=https://hf-mirror.com python -c \
  "from huggingface_hub import snapshot_download as d; d('BAAI/bge-m3', local_dir='emo/models/bge-m3')"
HF_ENDPOINT=https://hf-mirror.com python -c \
  "from huggingface_hub import snapshot_download as d; d('BAAI/bge-reranker-v2-m3', local_dir='emo/models/bge-reranker-v2-m3')"
```
