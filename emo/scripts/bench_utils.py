"""Benchmark 评测共享工具 — LongMemEval / LifeBench harness 共用

从 eval_locomo.py 抽取的可复用件（eval_locomo.py 本身不动）：
  - create_client / generate_answer：LLM 客户端（dashscope / OpenAI 兼容 / 本地 HF）
  - make_retrieve_fn：检索开关装配（rerank/mmr/hier/iter2/graph/temporal），
    temporal_now 改为逐题传参（LongMemEval 每题有自己的 question_date）

约定：
  - 读路径不写库（访问计数只在内存，评测不污染）
  - dashscope 推理模型（qwen3.7-*）调用方需自行 extra_body enable_thinking=False
"""

import logging
import os
import sys
import time
from pathlib import Path

from cost_log import log_cost

logger = logging.getLogger("bench_utils")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EMO_DIR = PROJECT_ROOT / "emo"

# 默认 embedding / 精排模型（LoCoMo 最佳组合同款）
DEFAULT_EMBED_MODEL = str(EMO_DIR / "models" / "bge-m3")
DEFAULT_RERANK_MODEL = str(EMO_DIR / "models" / "bge-reranker-v2-m3")
DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"


# ════════════════════════════════════════════════════════════════
#  共享 embedding function（批量建库时避免每个库重复加载模型）
# ════════════════════════════════════════════════════════════════

_EMBED_FN_CACHE = {}


def get_embedding_fn(model_path: str):
    """按模型路径缓存 embedding function 实例

    LongMemEval 要建 500 个独立库：若每个 IndexStore 各自加载 bge-m3，
    CPU 上每次约 2 分钟，光加载就要十几小时。共享一个实例即可。

    实现要点（两个踩坑，2026-08-10）：
    1. chroma 的 SentenceTransformerEmbeddingFunction 默认 device="cpu"——
       不传就在 GPU 机器上用 CPU 编码（35+ min/500 条 vs GPU <1 min）
    2. encode 默认 batch_size=32，bge-m3 处理 8k token 长文档时单批 4D
       attention mask ~8GB，共享卡 OOM → cuda 侧收窄到 8
    3. 必须子类化而非自写新类：chroma 重开已建库时按 name() 校验 EF 身份
       （Embedding function conflict: new: X vs persisted: sentence_transformer），
       子类化保证 name()/get_config() 与建库时完全一致
    """
    if model_path not in _EMBED_FN_CACHE:
        import numpy as np
        from chromadb.utils.embedding_functions import (
            SentenceTransformerEmbeddingFunction as _STEF,
        )
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
        batch_size = 8 if device == "cuda" else 32

        class _BatchSafeSTEF(_STEF):
            def __call__(self, input):
                embeddings = self._model.encode(
                    list(input),
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=self.normalize_embeddings,
                )
                return [np.array(e, dtype=np.float32) for e in embeddings]

        logger.info(f"加载共享 embedding 模型: {model_path} (device={device}, batch={batch_size})")
        _EMBED_FN_CACHE[model_path] = _BatchSafeSTEF(model_name=model_path, device=device)
    return _EMBED_FN_CACHE[model_path]


_RERANK_CACHE = {}


def get_reranker(model_path: str):
    """按模型路径缓存 CrossEncoder 精排器（防逐实例重建的显存爬升）"""
    if model_path not in _RERANK_CACHE:
        from sentence_transformers import CrossEncoder
        logger.info(f"加载共享精排模型: {model_path}")
        _RERANK_CACHE[model_path] = CrossEncoder(model_path)
    return _RERANK_CACHE[model_path]


# ════════════════════════════════════════════════════════════════
#  LLM 客户端
# ════════════════════════════════════════════════════════════════

def create_client(args):
    """创建 LLM 客户端，返回 (client, model_name, mode)"""
    if getattr(args, "local", False):
        return _create_local_model(args)

    try:
        from openai import OpenAI
    except ImportError:
        logger.error("需要 openai 库: pip install openai")
        sys.exit(1)

    if args.model.startswith("dashscope/"):
        model_name = args.model.split("/", 1)[1]
        client = OpenAI(
            api_key=args.api_key or os.environ.get("DASHSCOPE_API_KEY", ""),
            base_url=args.api_base,
        )
        return client, model_name, "api"

    client = OpenAI(
        api_key=args.api_key or os.environ.get("OPENAI_API_KEY", ""),
        base_url=args.api_base,
    )
    return client, args.model, "api"


def _create_local_model(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"加载模型: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    )
    if args.adapter:
        from peft import PeftModel
        logger.info(f"加载 adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)

    model.eval()
    return (model, tokenizer), args.base_model, "local"


def generate_answer(client_info, prompt: str, max_tokens: int = 256) -> str:
    client, model_name, mode = client_info
    if mode == "local":
        return _generate_local(client, prompt, max_tokens)
    return _generate_api(client, model_name, prompt, max_tokens)


def _generate_api(client, model_name: str, prompt: str, max_tokens: int) -> str:
    # dashscope 推理模型（qwen3.7-*）必须显式关思考链，否则 thinking 吃光
    # max_tokens 导致空回答，且延迟/成本翻倍（与裁判同一约定，TODO 坑#3）
    extra = {"enable_thinking": False} if "qwen3.7" in model_name else None
    for attempt in range(3):
        try:
            t0 = time.time()
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0,
                extra_body=extra,
            )
            log_cost("bench_utils", model_name, response, time.time() - t0)
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"API 调用失败 (尝试 {attempt + 1}/3): {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
    return ""


def _generate_local(model_and_tokenizer, prompt: str, max_tokens: int) -> str:
    import torch
    model, tokenizer = model_and_tokenizer

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=6000)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ════════════════════════════════════════════════════════════════
#  检索开关装配
# ════════════════════════════════════════════════════════════════

def make_retrieve_fn(args, retriever, client_info):
    """把 eval_locomo 的 _retrieve 闭包泛化：temporal_now 逐题传参

    Returns:
        retrieve(question, temporal_now=None) -> [(IndexEntry, score)]
    """
    client, model_name, mode = client_info

    # HyDE query 改写（可选；原 query 仍用于时间解析）
    # dashscope 推理模型须关思考链（与 _generate_api 同一约定）
    _extra = {"enable_thinking": False} if "qwen3.7" in str(model_name) else None

    query_rewriter = None
    if getattr(args, "query_rewrite", False):
        if mode == "api":
            def query_rewriter(q: str) -> str:
                r = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": (
                        "Write a short hypothetical answer to the question as one factual "
                        "declarative sentence (with names and specific dates when relevant). "
                        "Output only the sentence.\n\n"
                        f"Question: {q}\nSentence:"
                    )}],
                    temperature=0,
                    max_tokens=64,
                    extra_body=_extra,
                )
                return r.choices[0].message.content.strip() or q

    # CrossEncoder 精排（可选；全局缓存——逐实例新建会让显存越爬越高，
    # 每个 ~2.3GB，GC 不及时就是泄漏，2026-08-11 实锤）
    reranker = None
    if getattr(args, "rerank", False):
        _xe = get_reranker(args.rerank_model)

        def reranker(q, pairs):
            if not pairs:
                return pairs
            scores = _xe.predict([(q, (e.summary or "")) for e, _ in pairs])
            order = sorted(range(len(pairs)), key=lambda i: float(scores[i]), reverse=True)
            return [(pairs[i][0], float(scores[i])) for i in order]

    def _once(question: str, temporal_now):
        if getattr(args, "multi_channel", False):
            # 多路召回 + RRF（dense/graph/l3/temporal 四路）；graph_expand/hier 被涵括
            return retriever.retrieve_multi(
                question,
                rrf_k=args.rrf_k,
                l3_cards=args.l3_cards,
                l3_member_top=getattr(args, "l3_member_top", 3),
                channel_weights={"l3": getattr(args, "w_l3", 1.0)},
                reranker=reranker,
                temporal_now=temporal_now,
                l3_approx=getattr(args, "l3_approx", False),
            )
        return retriever.retrieve(
            question,
            expand_graph=args.graph_expand,
            graph_decay=args.graph_decay,
            graph_extra=args.graph_extra,
            query_rewriter=query_rewriter,
            reranker=reranker,
            hierarchical=args.hier,
            mmr=args.mmr,
            temporal_now=temporal_now,
            include_superseded=getattr(args, "no_supersede", False),
        )

    def retrieve(question: str, temporal_now=None):
        r1 = _once(question, temporal_now)
        if args.no_l3:
            r1 = [(e, s) for e, s in r1 if e.category != "semantic"]
        if not args.iter2 or not r1 or mode != "api":
            return r1

        # iter2 多跳补查：第一轮判缺 → 生成补查 query → 合并
        mem_lines = "\n".join(f"- {e.summary}" for e, _ in r1[:5])
        gap = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": (
                "Given a question and the top retrieved memories, decide if they contain "
                "enough information to answer it. If yes, output exactly: NONE. "
                "If no, write ONE short search query that would find the missing piece "
                "(include names/dates when relevant). Output only the query.\n\n"
                f"Question: {question}\n\nMemories:\n{mem_lines}\n\nOutput:"
            )}],
            temperature=0,
            max_tokens=48,
            extra_body=_extra,
        )
        gap_q = gap.choices[0].message.content.strip()
        if not gap_q or gap_q.upper().startswith("NONE"):
            return r1
        r2 = _once(gap_q, temporal_now)
        if args.no_l3:
            r2 = [(e, s) for e, s in r2 if e.category != "semantic"]
        seen = {e.mem_id for e, _ in r1}
        merged = list(r1)
        for e, s in r2:
            if e.mem_id not in seen:
                merged.append((e, s * 0.9))
                seen.add(e.mem_id)
        merged.sort(key=lambda x: x[1], reverse=True)
        return merged[: args.top_k + args.graph_extra + 3]

    return retrieve


# ════════════════════════════════════════════════════════════════
#  检索开关 CLI 参数（两个 harness 共用一组定义，保证命令行一致）
# ════════════════════════════════════════════════════════════════

def add_retrieval_args(parser):
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--graph-expand", action="store_true",
                        help="沿 related_ids 图扩展一跳（需要先做梦想 Step2 建关联）")
    parser.add_argument("--graph-extra", type=int, default=5)
    parser.add_argument("--graph-decay", type=float, default=0.85)
    parser.add_argument("--query-rewrite", action="store_true",
                        help="HyDE query 改写（LoCoMo 上为负效果，默认关）")
    parser.add_argument("--rerank", action="store_true",
                        help="语义 top-2k 种子经 CrossEncoder 精排后保留 top-k（最佳组合组件）")
    parser.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL)
    parser.add_argument("--iter2", action="store_true",
                        help="多跳迭代检索：第一轮判缺后生成补查 query 再检索一轮")
    parser.add_argument("--hier", action="store_true",
                        help="L3 层级检索（需要做梦 Step4 出语义簇）")
    parser.add_argument("--mmr", action="store_true", help="MMR 多样性去重")
    parser.add_argument("--no-l3", action="store_true", help="消融用：剔除 L3 语义簇")
    parser.add_argument("--no-supersede", action="store_true",
                        help="消融用：检索侧不滤除被 supersede 的条目")
    parser.add_argument("--multi-channel", action="store_true",
                        help="多路召回 + RRF 融合（dense/graph/l3/temporal 四路，涵括 graph-expand/hier）")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF 融合常数")
    parser.add_argument("--l3-cards", type=int, default=2,
                        help="多路召回中 L3 主题背景卡最多追加几条（不占 top-k）")
    parser.add_argument("--l3-approx", action="store_true",
                        help="消融用：L3 通道强制走摘要二次查询近似带块（不用持久化的簇成员连接）")
    parser.add_argument("--l3-member-top", type=int, default=3,
                        help="多路召回 L3 通道每簇展开成员数（默认 3；8 会造成簇成员洪泛挤占证据位）")
    parser.add_argument("--w-l3", type=float, default=1.0,
                        help="多路召回 RRF 中 L3 通道权重（默认 1.0；可调低防洪泛）")
    return parser
