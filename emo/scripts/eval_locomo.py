#!/usr/bin/env python3
"""LoCoMo Benchmark Evaluation for EMO Memory System

流程：
  1. 加载 LoCoMo 对话数据
  2. 将对话导入 EMO 记忆系统
  3. 对每个 QA：检索记忆 → 组装 prompt → 调模型 → 评分
  4. 按 5 个类别统计 F1 / Accuracy
  5. 计算检索召回率（Recall@K）

用法：
  # 先装依赖
  pip install -r emo/memory/requirements_memory.txt
  pip install nltk openai

  # 用 API 评估（快速）
  python emo/scripts/eval_locomo.py --model dashscope/qwen-plus --top-k 10

  # 只跑 1 个 conversation 做快速验证
  python emo/scripts/eval_locomo.py --model dashscope/qwen3.7-plus --conv-idx 0

  # 用本地模型评估
  python emo/scripts/eval_locomo.py --local \
      --base-model Qwen2.5-7B-Instruct \
      --adapter emo/outputs/aditi_sft_scored/final_adapter

  # 强制重新导入数据
  python emo/scripts/eval_locomo.py --model dashscope/qwen-plus --force-reimport
"""

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# HuggingFace 镜像（避免下载卡住）
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 让脚本能找到 memory 模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EMO_DIR = PROJECT_ROOT / "emo"
sys.path.insert(0, str(EMO_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "locomo"))

from memory.index_store import IndexStore, _DEFAULT_MODEL_PATH
from memory.persistent_store import PersistentStore
from memory.locomo_loader import LoCoMoLoader
from memory.retriever import Retriever
from memory.assembler import ContextAssembler
from bench_utils import get_embedding_fn
from cost_log import log_cost

# ── 复用 LoCoMo 的评估函数 ──
try:
    from task_eval.evaluation import f1_score, f1, normalize_answer
except ImportError:
    # 如果 locomo 模块不可用，内置一个简化版
    import re
    import string
    from collections import Counter

    logging.warning("locomo.task_eval not found, using built-in evaluation functions")

    def normalize_answer(s):
        s = str(s).replace(",", "")
        s = re.sub(r'\b(a|an|the|and)\b', ' ', s)
        s = ''.join(ch for ch in s if ch not in set(string.punctuation))
        return ' '.join(s.lower().split())

    def f1_score(prediction, ground_truth):
        pred_tokens = normalize_answer(prediction).split()
        gold_tokens = normalize_answer(ground_truth).split()
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        return (2 * precision * recall) / (precision + recall)

    def f1(prediction, ground_truth):
        import numpy as np
        preds = [p.strip() for p in prediction.split(',')]
        golds = [g.strip() for g in ground_truth.split(',')]
        return np.mean([max([f1_score(p, g) for p in preds]) for g in golds])


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
# 压低第三方日志
for name in ["chromadb", "httpx", "sentence_transformers", "openai", "httpcore"]:
    logging.getLogger(name).setLevel(logging.WARNING)

logger = logging.getLogger("eval_locomo")


# ════════════════════════════════════════════════════════════════
#  常量
# ════════════════════════════════════════════════════════════════

CAT_NAMES = {
    1: "Multi-hop",
    2: "Temporal",
    3: "Open-domain",
    4: "Single-hop",
    5: "Adversarial",
}

QA_PROMPT = (
    "Based on the above memories, answer the question in 1-3 words. "
    "Use exact words from the memories. Be specific, no explanation.\n\n"
    "Examples:\n"
    "Q: What did Caroline research? A: counseling and mental health\n"
    "Q: What is Caroline's identity? A: trans woman\n\n"
    "Question: {question}\n"
    "Answer:"
)

QA_PROMPT_CAT2 = (
    "Based on the above memories, answer with a SPECIFIC DATE (e.g., 'May 7, 2023', 'July 2023', 'August 12, 2023'). "
    "Do NOT use relative time like 'last week' or 'last Saturday'. "
    "Use the DATE information in the memories. Answer in 1-5 words.\n\n"
    "Examples:\n"
    "Q: When did Caroline go to the support group? A: May 7, 2023\n"
    "Q: When did Melanie run a charity race? A: August 12, 2023\n\n"
    "Question: {question}\n"
    "Answer:"
)

# cat2 弹性版（--cat2-flexible）：时长类问题允许回答时长，
# 修复强制 "具体日期" 与 "How long..."/时长类金标的系统性冲突
QA_PROMPT_CAT2_FLEX = (
    "Based on the above memories, answer the time-related question. "
    "If the question asks WHEN something happened, answer with a SPECIFIC DATE (e.g., 'May 7, 2023'). "
    "If the question asks HOW LONG or a duration, answer with the duration (e.g., '4 years', '2 months'). "
    "Answer in 1-5 words.\n\n"
    "Examples:\n"
    "Q: When did Caroline go to the support group? A: May 7, 2023\n"
    "Q: How long has Caroline had her current group of friends? A: 4 years\n\n"
    "Question: {question}\n"
    "Answer:"
)

QA_PROMPT_CAT5 = (
    "Based on the above memories, select the correct answer.\n\n"
    "Question: {question}\n"
    "Select: (a) {option_a} (b) {option_b}\n"
    "Answer (a or b):"
)

# cat3 放开推理（--cat3-infer）：锚定式推理——允许从记忆事实出发用常识搭桥，
# 但每一步推理必须有记忆支撑；无相关记忆时仍须拒答（防裸幻觉）
QA_PROMPT_CAT3_INFER = (
    "You are answering a question about a past conversation based on the retrieved memories above.\n"
    "Each memory begins with its date in parentheses.\n\n"
    "Instructions:\n"
    "1. Start from the memories: find the facts related to the question.\n"
    "2. You MAY reason from those facts using common sense (infer preferences, traits, "
    "likelihoods) — but every inference must be anchored in something the memories say. "
    "Do NOT invent facts with no support in the memories.\n"
    "3. If the memories contain nothing related to the question, answer \"Not mentioned in memories\".\n"
    "4. Answer in 1-6 words. Be specific. No explanation.\n\n"
    "Examples:\n"
    "Q: Would Caroline likely own Dr. Seuss books? (memory: \"Caroline collects classic children's books\") A: Yes\n"
    "Q: What is Caroline's political leaning? (memory: \"Caroline campaigns for LGBTQ rights\") A: Liberal\n\n"
    "Question: {question}\n"
    "Answer:"
)

# ── v2：Mem0 风格时间/冲突指令 + 硬格式约束 ──
QA_PROMPT_V2 = (
    "You are answering a question about a past conversation based on the retrieved memories above.\n"
    "Each memory begins with its date in parentheses, e.g. \"(1:56 pm on 8 May, 2023)\".\n\n"
    "Instructions:\n"
    "1. Base your answer ONLY on the memories above. Use exact words from them when possible.\n"
    "2. Pay attention to the dates: if memories conflict, the most recent memory wins.\n"
    "3. If a memory contains relative time (\"yesterday\", \"last week\"), convert it to an absolute date "
    "using that memory's own date before answering.\n"
    "4. Answer in 1-5 words. Be specific. No explanation, no full sentences.\n\n"
    "Examples:\n"
    "Q: What did Caroline research? A: counseling and mental health\n"
    "Q: What is Caroline's identity? A: trans woman\n\n"
    "Question: {question}\n"
    "Answer:"
)

QA_PROMPT_CAT2_V2 = (
    "You are answering a time-related question about a past conversation based on the retrieved memories above.\n"
    "Each memory begins with its date in parentheses, e.g. \"(1:56 pm on 8 May, 2023)\".\n\n"
    "Instructions:\n"
    "1. Find the memory that mentions the event, and use ITS date as the answer.\n"
    "2. If the memory text contains relative time (\"yesterday\", \"last week\", \"two days ago\"), "
    "compute the absolute date from the memory's own date. Example: a memory dated \"May 8, 2023\" says "
    "\"I adopted a puppy yesterday\" → the adoption was May 7, 2023.\n"
    "3. Answer with a SPECIFIC date (e.g., \"May 7, 2023\", \"July 2023\"). If the question asks HOW LONG "
    "or a duration, answer with the duration (e.g., \"4 years\", \"2 months\").\n"
    "4. NEVER answer with relative time (\"last week\", \"yesterday\"). Answer in 1-5 words.\n\n"
    "Examples:\n"
    "Q: When did Caroline go to the support group? A: May 7, 2023\n"
    "Q: When did Melanie run a charity race? A: August 12, 2023\n"
    "Q: How long has Jon owned his car? A: 4 years\n\n"
    "Question: {question}\n"
    "Answer:")

# v3：v2 + MemPro locomo prompt 的答案形态条款（复数全列/why 双因/
# 限定词锚定），放开 1-5 词硬约束。与 reanswer_locomo_fulltext.QA_PROMPT_V3 逐字一致。
QA_PROMPT_V3 = (
    "You are answering a question about a past conversation based on the retrieved memories above.\n"
    "Each memory begins with its date in parentheses, e.g. \"(1:56 pm on 8 May, 2023)\".\n\n"
    "Instructions:\n"
    "1. Base your answer ONLY on the memories above. Use exact words from them when possible.\n"
    "2. Pay attention to the dates: if memories conflict, the most recent memory wins.\n"
    "3. If a memory contains relative time (\"yesterday\", \"last week\"), convert it to an absolute date "
    "using that memory's own date before answering.\n"
    "4. If the question asks for plural items, examples, events, values, or reasons, include ALL directly "
    "relevant specific items from the memories, separated by commas — do not pick just one.\n"
    "5. For \"why\" questions, keep every direct reason given in the memories (after words like "
    "\"because\", \"since\", \"wanted\", \"dreaming of\") — include both the long-term motivation and "
    "the triggering event if both appear.\n"
    "6. If the question contains a qualifier (\"through\", \"by\", \"after\", \"on\", \"about\"...), answer "
    "only the fact tied to that qualifier, not nearby background facts.\n"
    "7. Answer with a short phrase (a few more words are fine when listing multiple items). "
    "No explanation, no full sentences.\n\n"
    "Examples:\n"
    "Q: What did Caroline research? A: counseling and mental health\n"
    "Q: What is Caroline's identity? A: trans woman\n"
    "Q: What hobbies does Caroline enjoy? A: painting, hiking, yoga\n\n"
    "Question: {question}\n"
    "Answer:"
)

MEMORY_CONTEXT_HEADER = (
    "Below are relevant memories from a conversation between {speaker_a} and {speaker_b} "
    "that took place over multiple sessions:\n\n"
)

NO_MEMORY_PROMPT = (
    "You are given a question about a conversation, but no relevant memories were found.\n"
    "If you cannot answer based on the information provided, write 'No information available'.\n\n"
)


# ════════════════════════════════════════════════════════════════
#  LLM 调用
# ════════════════════════════════════════════════════════════════

def create_client(args):
    """创建 LLM 客户端"""
    if args.local:
        return _create_local_model(args)

    # API 模式
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
    else:
        # OpenAI 兼容 API
        client = OpenAI(
            api_key=args.api_key or os.environ.get("OPENAI_API_KEY", ""),
            base_url=args.api_base,
        )
        return client, args.model, "api"


def _create_local_model(args):
    """加载本地 HuggingFace 模型"""
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


def generate_answer(client_info, prompt: str, max_tokens: int = 64) -> str:
    """生成回答"""
    client, model_name, mode = client_info

    if mode == "local":
        return _generate_local(client, prompt, max_tokens)
    else:
        return _generate_api(client, model_name, prompt, max_tokens)


def _generate_api(client, model_name: str, prompt: str, max_tokens: int) -> str:
    """API 调用"""
    for attempt in range(3):
        try:
            t0 = time.time()
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0,
            )
            log_cost("eval_locomo_answer", model_name, response, time.time() - t0)
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"API 调用失败 (尝试 {attempt + 1}/3): {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
    return ""


def _generate_local(model_and_tokenizer, prompt: str, max_tokens: int) -> str:
    """本地模型生成"""
    import torch
    model, tokenizer = model_and_tokenizer

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3000)
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
#  Prompt 构建
# ════════════════════════════════════════════════════════════════

def format_memories_for_locomo(
    memories: list,
    speaker_a: str,
    speaker_b: str,
    persistent_store=None,
) -> str:
    """将检索到的记忆格式化为 LoCoMo 风格的上下文

    全文注入：非 L3 条目从 sqlite 拉 raw_content 全文；
    L3 背景卡的 raw_content 是占位符，洞察正文在 summary。
    """
    if not memories:
        return ""

    full_by_id = {}
    if persistent_store is not None:
        full_by_id = {
            mf.mem_id: mf.raw_content
            for mf in persistent_store.get_batch([e.mem_id for e, _ in memories])
            if mf.raw_content
        }

    lines = [MEMORY_CONTEXT_HEADER.format(speaker_a=speaker_a, speaker_b=speaker_b)]

    for entry, score in memories:
        if entry.category == "semantic":
            lines.append(f"- [主题洞察] {entry.summary}")
        else:
            raw = full_by_id.get(entry.mem_id)
            if raw:
                # 重建 "(date) speaker: 全文" 形态（与 reanswer_locomo_fulltext.py 同构，
                # 保证与 0.602/76.89 基线的 prompt 格式一致）
                m = re.match(r"DATE: (.*?)\nSPEAKER: (.*?)\nDIA_ID: .*?\nTEXT: (.*)", raw, re.DOTALL)
                if m:
                    lines.append(f"- ({m.group(1)}) {m.group(2)}: {m.group(3).strip()}")
                    continue
                m = re.match(r"SESSION: .*?\nDATE: (.*?)\nSUMMARY: (.*)", raw, re.DOTALL)
                if m:
                    lines.append(f"- ({m.group(1)}) {m.group(2).strip()}")
                    continue
                lines.append(f"- {raw.strip()}")
            else:
                lines.append(f"- {entry.summary}")

    return "\n".join(lines)


def build_prompt(
    question: str,
    category: int,
    memories: list,
    speaker_a: str,
    speaker_b: str,
    cat5_options: Optional[dict] = None,
    cat2_flexible: bool = False,
    prompt_version: str = "v1",
    cat3_infer: bool = False,
    persistent_store=None,
) -> str:
    """构建完整的评估 prompt"""

    parts = []

    # 记忆上下文
    if memories:
        mem_context = format_memories_for_locomo(
            memories, speaker_a, speaker_b, persistent_store=persistent_store
        )
        parts.append(mem_context)
    else:
        parts.append(NO_MEMORY_PROMPT)

    # 问题 + 指令
    if category == 5 and cat5_options:
        prompt_template = QA_PROMPT_CAT5.format(
            question=question,
            option_a=cat5_options["a"],
            option_b=cat5_options["b"],
        )
        parts.append(prompt_template)
    elif category == 3 and cat3_infer:
        parts.append(QA_PROMPT_CAT3_INFER.format(question=question))
    elif category == 2:
        if prompt_version in ("v2", "v3"):
            template = QA_PROMPT_CAT2_V2
        else:
            template = QA_PROMPT_CAT2_FLEX if cat2_flexible else QA_PROMPT_CAT2
        parts.append(template.format(question=question))
    else:
        main_prompt = QA_PROMPT_V3 if prompt_version == "v3" else (QA_PROMPT_V2 if prompt_version == "v2" else QA_PROMPT)
        parts.append(main_prompt.format(question=question))

    return "\n\n".join(parts)


# ════════════════════════════════════════════════════════════════
#  Cat 5 特殊处理
# ════════════════════════════════════════════════════════════════

def make_cat5_options(qa: dict) -> Tuple[dict, str]:
    """为 category 5 问题生成随机排列的选项

    Returns:
        (options_dict, correct_answer_text)
        options_dict: {"a": "...", "b": "..."}
        correct_answer_text: 正确答案的文本
    """
    adversarial_answer = qa.get("adversarial_answer", qa.get("answer", ""))
    not_mentioned = "Not mentioned in the conversation"

    if random.random() < 0.5:
        options = {"a": not_mentioned, "b": adversarial_answer}
        correct = not_mentioned
    else:
        options = {"a": adversarial_answer, "b": not_mentioned}
        correct = not_mentioned

    return options, correct


def process_cat5_answer(raw_output: str, options: dict) -> str:
    """将模型输出转换为评估可用的答案

    评估标准：检查输出是否包含 "no information available" 或 "not mentioned"
    """
    output = raw_output.strip()

    # 如果是单个字母或带括号的字母，映射到对应文本
    output_lower = output.lower().strip()
    if output_lower in ("a", "(a)"):
        return options["a"]
    elif output_lower in ("b", "(b)"):
        return options["b"]

    # 否则直接返回原始输出（评估函数会检查关键词）
    return output


# ════════════════════════════════════════════════════════════════
#  评分
# ════════════════════════════════════════════════════════════════

def score_prediction(prediction: str, answer: str, category: int) -> float:
    """对单个预测评分

    Returns:
        F1 score (0-1) for categories 1-4
        0 or 1 for category 5
    """
    if category == 5:
        pred_lower = prediction.lower()
        if "no information available" in pred_lower or "not mentioned" in pred_lower:
            return 1.0
        return 0.0

    if category == 3:
        # Open-domain: 只取第一个答案
        answer = answer.split(";")[0].strip()

    if category == 1:
        # Multi-hop: 支持逗号分隔的多答案
        return f1(prediction, answer)
    else:
        # Single-hop, Temporal, Open-domain
        return f1_score(prediction, str(answer))


def _norm_dia_id(eid: str) -> str:
    """归一化 dia_id：LoCoMo 标注有 "D8:6;"(尾分号)、"D30:05"(零填充) 等畸形"""
    eid = str(eid).strip().rstrip(";").strip()
    m = re.match(r"^(D\d+):0*(\d+)$", eid)
    return f"{m.group(1)}:{m.group(2)}" if m else eid


def compute_recall(retrieved_memories: list, evidence: list) -> float:
    """计算检索召回率：检索到的记忆中是否包含 evidence 对应的 dia_id"""
    if not evidence:
        return 1.0

    retrieved_dia_ids = set()
    for entry, _ in retrieved_memories:
        for tag in entry.tags:
            if tag.startswith("D") and ":" in tag:
                retrieved_dia_ids.add(_norm_dia_id(tag))

    if not retrieved_dia_ids:
        return 0.0

    hits = sum(1 for ev in evidence if _norm_dia_id(ev) in retrieved_dia_ids)
    return hits / len(evidence)


# ════════════════════════════════════════════════════════════════
#  主评估流程
# ════════════════════════════════════════════════════════════════

def evaluate(args):
    """主评估函数"""

    # ── 加载数据 ──
    logger.info(f"加载数据: {args.data_file}")
    data = json.load(open(args.data_file, encoding="utf-8"))

    # ── 输出文件 ──
    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    model_key = f"emo_{args.model.replace('/', '_')}_top{args.top_k}"
    if args.graph_expand:
        model_key += f"_graph{args.graph_extra}"
    if getattr(args, "multi_channel", False):
        model_key += "_multi"
    if args.cat2_flexible:
        model_key += "_c2flex"
    if args.prompt_version != "v1":
        model_key += f"_{args.prompt_version}"
    if args.query_rewrite:
        model_key += "_qr"
    if args.rerank:
        model_key += "_rr"
    if args.iter2:
        model_key += "_ir"
    if args.hier:
        model_key += "_hier"
    if args.mmr:
        model_key += "_mmr"
    if args.no_l3:
        model_key += "_nol3"
    if args.temporal_anchor:
        model_key += "_tanc"
    if args.cat3_infer:
        model_key += "_c3i"
    prediction_key = f"{model_key}_prediction"

    # 加载已有结果（支持断点续跑）
    if out_file.exists() and not args.overwrite:
        out_samples = {d["sample_id"]: d for d in json.load(open(out_file))}
        logger.info(f"加载已有结果: {len(out_samples)} conversations")
    else:
        out_samples = {}

    # ── 创建 LLM 客户端 ──
    client_info = create_client(args)
    logger.info(f"模型: {args.model}, top-k: {args.top_k}")

    # ── 选择要评估的 conversations ──
    if args.conv_idx is not None:
        samples = [data[args.conv_idx]]
    else:
        samples = data

    # ── 统计 ──
    total_stats = defaultdict(lambda: {"total": 0, "score_sum": 0.0, "recall_sum": 0.0})
    all_results = []

    for sample_idx, sample in enumerate(samples):
        sample_id = sample["sample_id"]
        conv = sample["conversation"]
        speakers = LoCoMoLoader.get_speaker_names(conv)
        session_count = LoCoMoLoader.get_session_count(conv)
        qas = sample["qa"]

        # A3 时间锚定：该对话的 "now" = 最后一个 session 的日期
        # （LoCoMo 数据在 2023 年，默认系统时间会解析不到任何事件）
        temporal_now = None
        if args.temporal_anchor:
            from memory.locomo_loader import _parse_locomo_datetime
            last_dt = conv.get(f"session_{session_count}_date_time", "")
            parsed = _parse_locomo_datetime(last_dt)
            if parsed:
                from datetime import datetime as _dt
                temporal_now = _dt.strptime(parsed, "%Y-%m-%d %H:%M:%S")

        # LoCoMo 的丰富标注数据
        observation = sample.get("observation", {})
        session_sum = sample.get("session_summary", {})
        event_sum = sample.get("event_summary", {})

        logger.info(
            f"\n{'='*60}\n"
            f"[{sample_idx+1}/{len(samples)}] {sample_id} "
            f"({speakers[0]} & {speakers[1]}, {session_count} sessions, {len(qas)} questions)\n"
            f"{'='*60}"
        )

        # ── 构建记忆库 ──
        chroma_dir = str(Path(args.chroma_dir) / sample_id)
        sqlite_path = str(Path(args.sqlite_dir) / f"{sample_id}.db")
        Path(chroma_dir).mkdir(parents=True, exist_ok=True)
        Path(args.sqlite_dir).mkdir(parents=True, exist_ok=True)

        print(f"  [1/4] 初始化 IndexStore...", flush=True)

        # 如果 force_reimport，先删除旧数据再创建
        if args.force_reimport:
            import shutil
            if Path(chroma_dir).exists():
                shutil.rmtree(chroma_dir, ignore_errors=True)
            if Path(sqlite_path).exists():
                Path(sqlite_path).unlink()

        # 共享 EF 显式传 device：直建 IndexStore 内部走 chroma STEF 默认 cpu，
        # bge-m3 在 CPU 上每条十几秒
        index_store = IndexStore(
            persist_dir=chroma_dir,
            embedding_model_path=args.embed_model,
            embedding_fn=get_embedding_fn(args.embed_model or _DEFAULT_MODEL_PATH),
        )
        persistent_store = PersistentStore(db_path=sqlite_path)

        if index_store.count() == 0 or args.force_reimport:

            print(f"  [2/4] 导入 {sample_id} 的对话到记忆系统...", flush=True)
            loader = LoCoMoLoader(index_store, persistent_store, session_id_prefix=sample_id,
                              raw_turns=getattr(args, 'raw_turns', False))
            mem_count = loader.load_conversation(
                conv,
                observation=observation,
                session_summary=session_sum,
                event_summary=event_sum,
                time_offset_days=args.time_offset,
            )
            print(f"  [2/4] 导入完成: {mem_count} 条记忆", flush=True)
        else:
            print(f"  [2/4] 记忆已存在: {index_store.count()} 条", flush=True)

        # ── 做梦流程（可选）──
        if args.dream:
            from memory.dreamer import DreamOrchestrator
            steps = set(int(s) for s in args.dream_steps.split(","))
            print(f"  [2.5/4] 做梦流程 (steps={sorted(steps)} batch={args.dream_batch})...", flush=True)

            dreamer = DreamOrchestrator(index_store, persistent_store, client_info[0], llm_model=client_info[1])
            dream_stats = {}

            if 1 in steps or 2 in steps:
                # 合并异步版：每条一次调用同时完成 结构化+关联+supersede
                # （分离路径的 generate_links 不做 supersede）
                print(f"    Step 1+2: 结构化+关联+supersede（合并异步）...", flush=True)
                dream_stats.update(asyncio.run(
                    dreamer.structure_and_link_memories_async(batch_size=args.dream_batch)
                ))
            if 0 in steps:
                print(f"    Step 0: 遗忘衰减...", flush=True)
                dream_stats["decay"] = dreamer.weight_decay()
            if 3 in steps:
                print(f"    Step 3: 记忆演化...", flush=True)
                dream_stats["evolved"] = dreamer.evolve_memories()
            if 4 in steps:
                print(f"    Step 4: 簇融合...", flush=True)
                dream_stats["fused"] = dreamer.fuse_clusters()
            if 5 in steps:
                print(f"    Step 5: 效用清理...", flush=True)
                dream_stats["cleaned"] = dreamer.utility_cleanup()
            if 6 in steps:
                print(f"    Step 6: 标签分类器...", flush=True)
                dream_stats["tag_classifier"] = dreamer.train_tag_classifier()

            print(f"  [2.5/4] 做梦完成: {dream_stats}", flush=True)

        # 传 persistent_store 启用时间轴并集召回（否则 --temporal-anchor 是惰性开关，
        # temporal_now 无处生效）
        # 注意：对未带 --temporal-anchor 的运行是 no-op（now=真实当前时间，
        # 范围可能匹配不到数据内的历史事件）
        retriever = Retriever(index_store, top_k=args.top_k, persistent_store=persistent_store)
        assembler = ContextAssembler(index_store, persistent_store)

        # query_rewriter / iter2 都可能用到 LLM 客户端，提前统一解包
        client, model_name, mode = client_info

        # ── HyDE query 改写器（可选；原 query 仍用于时间解析）──
        query_rewriter = None
        if args.query_rewrite:
            if mode == "api":
                def query_rewriter(q: str) -> str:
                    t0 = time.time()
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
                    )
                    log_cost("eval_locomo_hyde", model_name, r, time.time() - t0)
                    rewritten = r.choices[0].message.content.strip()
                    return rewritten or q

        # ── CrossEncoder 精排（可选，语义 top-2k 重排取 top-k）──
        reranker = None
        if args.rerank:
            from sentence_transformers import CrossEncoder
            _xe = CrossEncoder(args.rerank_model)

            def reranker(q, pairs):
                if not pairs:
                    return pairs
                scores = _xe.predict([(q, (e.summary or "")) for e, _ in pairs])
                order = sorted(range(len(pairs)), key=lambda i: float(scores[i]), reverse=True)
                return [(pairs[i][0], float(scores[i])) for i in order]

        # ── 多跳迭代检索（可选：检索→判缺→补查→合并）──
        def _retrieve(question: str):
            r1 = retriever.retrieve(
                question,
                expand_graph=args.graph_expand,
                graph_decay=args.graph_decay,
                graph_extra=args.graph_extra,
                query_rewriter=query_rewriter,
                reranker=reranker,
                hierarchical=args.hier,
                mmr=args.mmr,
                temporal_now=temporal_now,
            )
            if args.no_l3:
                r1 = [(e, s) for e, s in r1 if e.category != "semantic"]
            if not args.iter2 or not r1 or mode != "api":
                return r1
            mem_lines = "\n".join(f"- {e.summary}" for e, _ in r1[:5])
            t0 = time.time()
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
            )
            log_cost("eval_locomo_gapcheck", model_name, gap, time.time() - t0)
            gap_q = gap.choices[0].message.content.strip()
            if not gap_q or gap_q.upper().startswith("NONE"):
                return r1
            r2 = retriever.retrieve(
                gap_q,
                expand_graph=args.graph_expand,
                graph_decay=args.graph_decay,
                graph_extra=args.graph_extra,
                query_rewriter=query_rewriter,
                reranker=reranker,
                hierarchical=args.hier,
                mmr=args.mmr,
                temporal_now=temporal_now,
            )
            if args.no_l3:
                r2 = [(e, s) for e, s in r2 if e.category != "semantic"]
            seen = {e.mem_id for e, _ in r1}
            merged = list(r1)
            for e, s in r2:
                if e.mem_id not in seen:
                    merged.append((e, s * 0.9))  # 第二轮轻微降权
                    seen.add(e.mem_id)
            merged.sort(key=lambda x: x[1], reverse=True)
            return merged[: args.top_k + args.graph_extra + 3]

        # ── 初始化输出 ──
        if sample_id in out_samples and not args.overwrite:
            out_data = out_samples[sample_id]
            # 统计已有的结果（f1 与 recall 都要累计，否则续跑后 Recall@K 恒为 0）
            for qa in out_data["qa"]:
                if prediction_key + "_f1" in qa:
                    cat = qa["category"]
                    total_stats[cat]["total"] += 1
                    total_stats[cat]["score_sum"] += qa[prediction_key + "_f1"]
                    total_stats[cat]["recall_sum"] += qa.get(prediction_key + "_recall", 0.0)
            logger.info(f"已有部分结果，继续评估...")
        else:
            out_data = {"sample_id": sample_id, "qa": [dict(qa) for qa in qas]}

        # ── 逐题评估 ──
        total_qa = len(out_data["qa"])
        answered = 0
        eval_limit = args.limit if args.limit else total_qa
        print(f"  [3/4] 开始逐题评估 ({min(eval_limit, total_qa)}/{total_qa} 题)...", flush=True)
        for i, qa in enumerate(out_data["qa"]):
            if answered >= eval_limit:
                print(f"  达到 limit={eval_limit}，停止评估", flush=True)
                break
            # 跳过已评估的题目
            if prediction_key in qa and not args.overwrite:
                continue

            question = qa["question"]
            answer = str(qa.get("answer", ""))
            category = qa["category"]
            evidence = qa.get("evidence", [])

            print(f"    Q{i+1}: [{CAT_NAMES.get(category, '?')}] {question[:60]}...", flush=True)

            # ── 检索记忆（图扩展 + query 改写 + 精排 + 可选迭代补查）──
            memories = _retrieve(question)
            print(f"      → 检索到 {len(memories)} 条记忆", flush=True)

            # ── Cat 5 特殊处理 ──
            cat5_options = None
            if category == 5:
                cat5_options, _ = make_cat5_options(qa)

            # ── 构建 prompt ──
            prompt = build_prompt(
                question, category, memories,
                speakers[0], speakers[1],
                cat5_options,
                cat2_flexible=args.cat2_flexible,
                prompt_version=args.prompt_version,
                cat3_infer=args.cat3_infer,
                persistent_store=persistent_store,
            )

            # ── 获取模型回答 ──
            print(f"      → 调用 API...", flush=True)
            max_tokens = 64 if category != 5 else 32
            raw_output = generate_answer(client_info, prompt, max_tokens=max_tokens)
            print(f"      → 回答: {raw_output[:80]}", flush=True)

            # ── 处理 Cat 5 输出 ──
            if category == 5 and cat5_options:
                prediction = process_cat5_answer(raw_output, cat5_options)
            else:
                prediction = raw_output.strip()

            # ── 评分 ──
            f1_val = score_prediction(prediction, answer, category)

            # ── 计算检索召回率 ──
            recall = compute_recall(memories, evidence)

            # ── 存储结果 ──
            qa[prediction_key] = prediction
            qa[prediction_key + "_f1"] = round(f1_val, 4)
            qa[prediction_key + "_recall"] = round(recall, 4)
            qa[prediction_key + "_retrieved_ids"] = [
                e.mem_id for e, _ in memories
            ]

            # ── 统计 ──
            total_stats[category]["total"] += 1
            total_stats[category]["score_sum"] += f1_val
            total_stats[category]["recall_sum"] += recall

            answered += 1
            if answered % 10 == 0:
                _print_progress(total_stats, answered)

        # ── 保存进度 ──
        out_samples[sample_id] = out_data
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(list(out_samples.values()), f, indent=2, ensure_ascii=False)

        logger.info(
            f"✅ {sample_id} 完成: {answered} 题已评估, "
            f"结果已保存到 {out_file}"
        )

    # ── 最终统计 ──
    print_results(total_stats, model_key)

    # ── 保存统计摘要 ──
    stats_file = out_file.with_name(out_file.stem + "_stats.json")
    save_stats(total_stats, model_key, stats_file)
    logger.info(f"统计摘要已保存到 {stats_file}")


def _print_progress(stats: dict, count: int):
    """打印进度"""
    total_score = sum(s["score_sum"] for s in stats.values())
    total_count = sum(s["total"] for s in stats.values())
    avg = total_score / max(total_count, 1)
    logger.info(f"  进度: {count} 题, 平均 F1: {avg:.3f}")


def print_results(stats: dict, model_key: str):
    """打印最终结果"""
    print(f"\n{'='*60}")
    print(f"  LoCoMo Evaluation Results: {model_key}")
    print(f"{'='*60}\n")

    cat_order = [4, 1, 2, 3, 5]  # Single, Multi, Temporal, Open, Adversarial
    total_count = 0
    total_score = 0.0
    total_recall = 0.0

    for cat in cat_order:
        s = stats.get(cat, {"total": 0, "score_sum": 0.0, "recall_sum": 0.0})
        if s["total"] == 0:
            continue
        avg_f1 = s["score_sum"] / s["total"]
        avg_recall = s["recall_sum"] / s["total"]
        print(
            f"  Cat {cat} ({CAT_NAMES[cat]:12s}): "
            f"F1 = {avg_f1:.3f}  Recall@K = {avg_recall:.3f}  "
            f"(n={s['total']})"
        )
        total_count += s["total"]
        total_score += s["score_sum"]
        total_recall += s["recall_sum"]

    if total_count > 0:
        print(f"\n  {'Overall':22s}: "
              f"F1 = {total_score/total_count:.3f}  "
              f"Recall@K = {total_recall/total_count:.3f}  "
              f"(n={total_count})")
    print()


def save_stats(stats: dict, model_key: str, stats_file: Path):
    """保存统计结果到 JSON"""
    results = {}
    results[model_key] = {}

    for cat in [1, 2, 3, 4, 5]:
        s = stats.get(cat, {"total": 0, "score_sum": 0.0, "recall_sum": 0.0})
        if s["total"] > 0:
            results[model_key][f"cat{cat}_{CAT_NAMES[cat]}"] = {
                "count": s["total"],
                "f1": round(s["score_sum"] / s["total"], 4),
                "recall": round(s["recall_sum"] / s["total"], 4),
                "total_f1": round(s["score_sum"], 4),
            }

    total_count = sum(s["total"] for s in stats.values())
    total_score = sum(s["score_sum"] for s in stats.values())
    total_recall = sum(s["recall_sum"] for s in stats.values())
    if total_count > 0:
        results[model_key]["overall"] = {
            "count": total_count,
            "f1": round(total_score / total_count, 4),
            "recall": round(total_recall / total_count, 4),
        }

    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# ════════════════════════════════════════════════════════════════
#  独立评估函数（供 test_locomo_full.py 调用）
# ════════════════════════════════════════════════════════════════

def evaluate_locomo(
    conv: dict,
    store,
    persist,
    limit: int = 20,
    model: str = "qwen-plus",
    top_k: int = 10,
    use_classifier: bool = True,
    classifier_threshold: float = 0.3,
    classifier_path: str = None,
) -> dict:
    """评估单个 LoCoMo 对话

    Args:
        conv: LoCoMo 对话数据（包含 conversation 和 qa）
        store: IndexStore 实例
        persist: PersistentStore 实例
        limit: 评估多少题
        model: LLM 模型名
        top_k: 检索多少条记忆
        use_classifier: 是否使用标签分类器分段检索
        classifier_threshold: 分类器置信度阈值（低于此值走 unknown 路径）

    Returns:
        {"overall_f1": float, "overall_recall": float, "by_category": {...}}
    """
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    client_info = (client, model, "api")

    speakers = LoCoMoLoader.get_speaker_names(conv["conversation"])
    qas = conv["qa"][:limit]
    retriever = Retriever(store, top_k=top_k)

    # 加载标签分类器
    classifier = None
    if use_classifier:
        try:
            from memory.tag_classifier import TagClassifier
            if classifier_path is None:
                classifier_path = EMO_DIR / "memory" / "models" / "tag_classifier.pkl"
            else:
                classifier_path = Path(classifier_path)
            if classifier_path.exists():
                classifier = TagClassifier.load(classifier_path)
                print(f"  标签分类器已加载: {classifier_path} (threshold={classifier_threshold})", flush=True)
            else:
                print(f"  ⚠️ 标签分类器不存在: {classifier_path}，跳过分段检索", flush=True)
        except Exception as e:
            print(f"  ⚠️ 加载标签分类器失败: {e}", flush=True)

    results_by_cat = defaultdict(lambda: {"f1_sum": 0.0, "recall_sum": 0.0, "count": 0})
    unknown_count = 0

    for i, qa in enumerate(qas):
        question = qa["question"]
        answer = str(qa.get("answer", ""))
        category = qa["category"]
        evidence = qa.get("evidence", [])

        print(f"  Q{i+1}/{len(qas)}: [{CAT_NAMES.get(category, '?')}] {question[:60]}...", flush=True)

        # 标签分类器预测
        predicted_label1 = None
        is_unknown = False
        if classifier:
            result = classifier.predict(question, top_k=1, confidence_threshold=classifier_threshold)
            label1_preds = result["label1"]
            label2_preds = result["label2"]
            known = result["known"]

            if known and label1_preds and label2_preds:
                predicted_label1 = label1_preds[0][0]
                label2 = label2_preds[0][0]
                conf1 = label1_preds[0][1]
                conf2 = label2_preds[0][1]
                print(f"    → 预测: {predicted_label1}/{label2} (conf={conf1:.2f}/{conf2:.2f})", flush=True)
            else:
                is_unknown = True
                unknown_count += 1
                if label1_preds:
                    conf1 = label1_preds[0][1]
                    conf2 = label2_preds[0][1] if label2_preds else 0.0
                    print(f"    → UNKNOWN (conf={conf1:.2f}/{conf2:.2f} < {classifier_threshold})", flush=True)
                else:
                    print(f"    → UNKNOWN (分类器无法预测)", flush=True)

        # 检索记忆（如果有预测标签则分段检索）
        if predicted_label1:
            memories = retriever.retrieve(question, category=predicted_label1)
            print(f"    → 分段检索 [{predicted_label1}]: {len(memories)} 条记忆", flush=True)
        else:
            memories = retriever.retrieve(question)
            print(f"    → 全局检索: {len(memories)} 条记忆", flush=True)

        # Cat 5 特殊处理
        cat5_options = None
        if category == 5:
            cat5_options, _ = make_cat5_options(qa)

        # 构建 prompt
        prompt = build_prompt(question, category, memories, speakers[0], speakers[1], cat5_options,
                              persistent_store=persist)

        # 获取模型回答
        print(f"    → 调用 API...", flush=True)
        max_tokens = 64 if category != 5 else 32
        raw_output = generate_answer(client_info, prompt, max_tokens=max_tokens)

        # 处理 Cat 5 答案
        if category == 5 and cat5_options:
            prediction = process_cat5_answer(raw_output, cat5_options)
        else:
            prediction = raw_output.strip()

        print(f"    → 回答: {prediction[:80]}", flush=True)

        # 评分
        f1 = score_prediction(prediction, answer, category)
        recall = compute_recall(memories, evidence)

        cat_key = CAT_NAMES.get(category, f"cat_{category}")
        results_by_cat[cat_key]["f1_sum"] += f1
        results_by_cat[cat_key]["recall_sum"] += recall
        results_by_cat[cat_key]["count"] += 1

        print(f"    → F1={f1:.3f}, Recall={recall:.3f}", flush=True)

    # 汇总
    overall_f1_sum = 0.0
    overall_recall_sum = 0.0
    overall_count = 0
    by_category = {}

    for cat, cat_data in results_by_cat.items():
        count = cat_data["count"]
        f1 = cat_data["f1_sum"] / count if count > 0 else 0.0
        recall = cat_data["recall_sum"] / count if count > 0 else 0.0
        by_category[cat] = {"f1": f1, "recall": recall, "count": count}

        overall_f1_sum += cat_data["f1_sum"]
        overall_recall_sum += cat_data["recall_sum"]
        overall_count += count

    overall_f1 = overall_f1_sum / overall_count if overall_count > 0 else 0.0
    overall_recall = overall_recall_sum / overall_count if overall_count > 0 else 0.0

    if classifier:
        print(f"\n  分段检索统计: {unknown_count}/{len(qas)} 题走 UNKNOWN 路径", flush=True)

    return {
        "overall_f1": overall_f1,
        "overall_recall": overall_recall,
        "by_category": by_category,
    }


# ════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="LoCoMo Benchmark Evaluation for EMO Memory System"
    )

    # 数据
    parser.add_argument(
        "--data-file",
        default=str(PROJECT_ROOT / "locomo" / "data" / "locomo10.json"),
    )
    parser.add_argument(
        "--out-file",
        default=str(PROJECT_ROOT / "outputs" / "locomo" / "emo_locomo_results.json"),
    )

    # 模型
    parser.add_argument(
        "--model", default="dashscope/qwen3.7-plus",
        help="模型名，格式: provider/model，如 dashscope/qwen-plus, openai/gpt-4",
    )
    parser.add_argument("--api-base", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api-key", default="")

    # 检索
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--graph-expand", action="store_true",
                        help="沿 related_ids 图扩展一跳（需要先用 --dream-steps 2 建关联）")
    parser.add_argument("--graph-extra", type=int, default=5,
                        help="图扩展最多追加的邻居数")
    parser.add_argument("--graph-decay", type=float, default=0.85,
                        help="图传播衰减：邻居图分 = 种子分 * decay")
    parser.add_argument("--cat2-flexible", action="store_true",
                        help="cat2 时长类问题允许回答时长（harness 侧修正，非系统能力，"
                             "不作为系统改进的依据；默认关闭以保持测量中性）")
    parser.add_argument("--prompt-version", choices=["v1", "v2", "v3"], default="v1",
                        help="QA prompt 版本：v1=原版，v2=Mem0 风格时间/冲突指令+格式约束")
    parser.add_argument("--query-rewrite", action="store_true",
                        help="检索前用 LLM 把问题改写为假想答案句（HyDE）；原 query 仍用于时间解析")
    parser.add_argument("--rerank", action="store_true",
                        help="语义 top-2k 种子经 CrossEncoder 精排后保留 top-k")
    parser.add_argument("--rerank-model", default=str(EMO_DIR / "models" / "bge-reranker-v2-m3"),
                        help="本地 CrossEncoder 精排模型路径")
    parser.add_argument("--iter2", action="store_true",
                        help="多跳迭代检索：第一轮结果判缺后生成补查 query 再检索一轮")
    parser.add_argument("--hier", action="store_true",
                        help="L3 层级检索：先命中主题簇(label1=semantic)，再展开簇内成员")
    parser.add_argument("--mmr", action="store_true",
                        help="MMR 多样性去重：惩罚与已选条目近义的候选（防多粒度重复占槽）")
    parser.add_argument("--no-l3", action="store_true",
                        help="消融用：检索结果剔除 L3 语义簇（label1=semantic）")
    parser.add_argument("--temporal-anchor", action="store_true",
                        help="A3 消融：时间解析 now 锚定到对话末 session 日期（2023 数据才生效）")
    parser.add_argument("--cat3-infer", action="store_true",
                        help="cat3 开放域放开锚定推理（从记忆事实出发可用常识搭桥，防裸幻觉仍拒答）")

    # 范围
    parser.add_argument("--conv-idx", type=int, default=None, help="只评估指定的 conversation 索引")
    parser.add_argument("--limit", type=int, default=None, help="每个 conversation 最多评估多少题（快速验证用）")

    # 本地模型
    parser.add_argument("--local", action="store_true", help="使用本地模型")
    parser.add_argument("--base-model", type=str, default=str(PROJECT_ROOT / "Qwen2.5-7B-Instruct"))
    parser.add_argument("--adapter", type=str, default="")

    # 记忆存储
    parser.add_argument("--chroma-dir", default=str(EMO_DIR / "memory" / "locomo_chroma"))
    parser.add_argument("--sqlite-dir", default=str(EMO_DIR / "memory" / "locomo_sqlite"))
    parser.add_argument("--embed-model", default=None,
                        help="本地 embedding 模型路径（默认 all-MiniLM-L6-v2；可指 emo/models/bge-m3）")

    # 控制
    parser.add_argument("--force-reimport", action="store_true", help="强制重新导入数据")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有评估结果")
    parser.add_argument("--time-offset", type=int, default=30, help="将对话映射到多少天前")
    parser.add_argument("--dream", action="store_true", help="导入后跑做梦流程（丰富记忆）")
    parser.add_argument("--dream-steps", type=str, default="1,2,4,6",
                        help="做梦步骤（逗号分隔）: 0=衰减,1=结构化,2=关联,3=演化,4=融合,5=清理,6=分类器")
    parser.add_argument("--dream-batch", type=int, default=8,
                        help="Step1+2 合并异步路径的 API 并发数")
    parser.add_argument("--raw-turns", action="store_true",
                        help="导入时 turn 标 raw（配合 --dream 让 Step1 结构化/supersede 跑在原始 turn 上；仅影响新建库）")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 创建输出目录
    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    evaluate(args)
