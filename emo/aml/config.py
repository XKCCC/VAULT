"""AML 适配层配置（全部走环境变量，key 不落盘）。

两档切换：
  开发自测（dashscope/qwen-plus）：
    export AML_LLM_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
    export AML_LLM_MODEL=qwen-plus
    export AML_LLM_KEY_ENV=DASHSCOPE_API_KEY
  正式评测（内部模型必须 gpt-4o-mini，AML full 前置清单要求）：
    export AML_LLM_BASE=https://api.naga.ac/v1
    export AML_LLM_MODEL=gpt-4o-mini
    export AML_LLM_KEY_ENV=OPENAI_API_KEY   # 配合 .env.openai_official
"""
import os
from dataclasses import dataclass
from pathlib import Path

_EMO_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class AMLConfig:
    # LLM（做梦用；评测链路的答题/裁判由 AML 平台固定，与我们无关）
    llm_base: str = os.environ.get(
        "AML_LLM_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    llm_model: str = os.environ.get("AML_LLM_MODEL", "qwen-plus")
    llm_key_env: str = os.environ.get("AML_LLM_KEY_ENV", "DASHSCOPE_API_KEY")

    # 本地模型
    embed_model: str = os.environ.get(
        "AML_EMBED_MODEL", str(_EMO_DIR / "models" / "bge-m3")
    )
    rerank_model: str = os.environ.get(
        "AML_RERANK_MODEL", str(_EMO_DIR / "models" / "bge-reranker-v2-m3")
    )

    # 库存放根目录（每 user_id 一个独立库，严禁跨 user_id 共享——AML 硬性要求）
    root: str = os.environ.get("AML_ROOT", str(_EMO_DIR / "memory" / "aml_libs"))

    # 参赛方签发的 Memory System Key（平台调我们 Add/Search 时校验）；
    # 留空 = 不校验（仅限公开 smoke 阶段）
    api_key: str = os.environ.get("AML_API_KEY", "")

    # 接口路径（契约允许参赛方自定）
    add_path: str = os.environ.get("AML_ADD_PATH", "/add")
    search_path: str = os.environ.get("AML_SEARCH_PATH", "/search")

    # 做梦并发（dream-batch；GPU 机建议 32，CPU 机器 8）
    dream_batch: int = int(os.environ.get("AML_DREAM_BATCH", "16"))

    # 检索参数（与论文最佳组合一致：rerank + 图扩展）
    graph_extra: int = int(os.environ.get("AML_GRAPH_EXTRA", "5"))
    # AML 正式评测 top_k 固定 100；我们按平台传的值原样执行（上限 100）
    max_top_k: int = 100
