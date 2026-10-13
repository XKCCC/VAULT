"""AML 适配层配置（全环境变量；key 不落盘、不进 git）。

两档切换：
  开发自测：AML_LLM_BASE=dashscope 兼容端点 / AML_LLM_MODEL=qwen-plus / AML_LLM_KEY_ENV=DASHSCOPE_API_KEY
  正式评测：AML_LLM_BASE=https://api.naga.ac/v1 / AML_LLM_MODEL=gpt-4o-mini / AML_LLM_KEY_ENV=OPENAI_API_KEY
            （AML full 前置清单要求内部模型 gpt-4o-mini）
"""
import os
from dataclasses import dataclass

from vault2lib.constants import EMBED, RERANKER


@dataclass(frozen=True)
class AMLConfig:
    llm_base: str = os.environ.get(
        "AML_LLM_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    llm_model: str = os.environ.get("AML_LLM_MODEL", "qwen-plus")
    llm_key_env: str = os.environ.get("AML_LLM_KEY_ENV", "DASHSCOPE_API_KEY")

    embed_model: str = os.environ.get("AML_EMBED_MODEL", EMBED)
    rerank_model: str = os.environ.get("AML_RERANK_MODEL", RERANKER)

    # 每 user_id 一座独立库（AML 硬性要求，严禁跨 user 共享）
    root: str = os.environ.get("AML_ROOT", str(__import__("pathlib").Path(__file__).resolve().parents[1] / "aml_libs"))

    # 参赛方签发的 Memory System Key；留空=不校验（仅限 smoke）
    api_key: str = os.environ.get("AML_API_KEY", "")

    add_path: str = os.environ.get("AML_ADD_PATH", "/add")
    search_path: str = os.environ.get("AML_SEARCH_PATH", "/search")

    dream_batch: int = int(os.environ.get("AML_DREAM_BATCH", "16"))
    quiet_s: float = float(os.environ.get("AML_QUIET_S", "15"))

    # AML 正式评测 top_k 固定 100；按平台传值原样执行（上限 100）
    max_top_k: int = 100
