"""D10: DreamOrchestrator — 做梦流程编排

做梦流程（离线/后台执行）:
  Step 1: 结构化（raw → dreamed）
    对每条 status="raw" 的记忆用 LLM 提取 summary/facts/relations
  Step 2: 关联建立（A-MEM 式）
    检索语义近邻 → LLM 判断关系类型 → 写入 related_ids
  Step 3: 记忆演化
    新记忆触发已有记忆的更新（summary/tags 演化）
  Step 4: 图分析（GFM-RAG 启发）
    构建记忆关系图 → 发现簇/孤立节点 → 融合/遗忘
  Step 5: 效用清理
    长期未被检索的记忆降权/压缩

依赖:
  - IndexStore（ChromaDB）
  - PersistentStore（SQLite）
  - LLM API（DashScope / OpenAI 兼容）
"""

from __future__ import annotations

import json
import logging
import os
import re
import time

try:
    from cost_log import log_cost
except ImportError:  # dreamer 作为库被引用时 emo/scripts 可能不在 sys.path
    def log_cost(*a, **k):
        pass
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .index_store import IndexStore
from .persistent_store import PersistentStore
from .schema import IndexEntry, MemoryFile
from .temporal import weekday_name

logger = logging.getLogger(__name__)

# ── LLM 提示词模板 ──

STRUCTURE_PROMPT = """You are a memory structuring assistant. Given a raw conversation memory, extract structured information.

**CRITICAL LANGUAGE RULE: All output fields MUST be in the SAME language as the original memory. If the original memory is in English, ALL output must be in English. If in Chinese, ALL output must be in Chinese. Never mix languages.**

**Current Label1 set (primary category)**: {label1_set}
**Current Label2 set (secondary category)**: {label2_set}

{time_anchor}

Original memory:
{raw_content}

Instructions:
1. Pick ONE label1 from the Label1 set. If none fits, suggest a NEW label1 (keep it short, 1-3 words).
2. Pick ONE label2 from the Label2 set. If none fits, suggest a NEW label2 (keep it short, 1-3 words).
3. Extract facts, relations, summary, tags, keywords, and context description.
4. Extract temporal information (when did this happen?).

**Important definitions:**
- summary: A concise one-sentence description of what happened (under 50 words)
- keywords: Core concepts/entities mentioned (3-5 short terms, e.g. "gym", "dosa", "Bangalore")
- context_description: A brief semantic description of the conversational context and emotional tone (1-2 sentences). This captures the "why" and "how" of the conversation, not just "what".
- temporal_context: When did this event happen? Extract explicit dates OR infer approximate time from context (e.g., "last summer" → "2023 summer, approx June-August", "college days" → "approx 2019-2023"). If no temporal clues, use "unknown time".
- event_sequence: Where does this event fit in the timeline relative to other life events? (e.g., "after changing jobs", "before moving to Bangalore", "during the first year of knowing the user")
- event_timestamp: Best estimate of when this happened in YYYY-MM-DD HH:MM:SS format.
  * Explicit dates in the text → use them directly.
  * Relative expressions ("yesterday", "last weekend", "5 minutes ago") → resolve against the conversation time anchor; keep hour-minute precision for sub-day expressions ("5 minutes ago" → e.g. 14:25:00, not 00:00:00).
  * Approximate periods ("summer 2023") → use the middle of the range (e.g. "2023-07-15 12:00:00").
  * Timeless facts or preferences with no event → use the conversation date (i.e., when it was told).
  * Events at a genuinely indeterminate time ("when I was a kid", "years ago") → leave event_timestamp as "" (empty string); do NOT fabricate a date.

Output JSON format:
{{
  "summary": "One-sentence summary (same language as input)",
  "facts": ["fact entry 1", "fact entry 2"],
  "relations": ["relation entry 1", "relation entry 2"],
  "label1": "chosen or new label1",
  "label2": "chosen or new label2",
  "label1_is_new": true/false,
  "label2_is_new": true/false,
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "context_description": "Brief description of the conversational context and emotional tone",
  "temporal_context": "When this happened (extracted or inferred)",
  "event_sequence": "Timeline position relative to other events",
  "event_timestamp": "YYYY-MM-DD HH:MM:SS (best estimate)",
  "tags": ["tag1", "tag2"]
}}

Output JSON only, no other text."""

# ── 优化版：合并 Step 1 + Step 2 的提示词 ──

STRUCTURE_AND_LINK_PROMPT = """You are a memory structuring and linking assistant. Given a raw conversation memory and its semantic neighbors, perform TWO tasks in ONE response.

**CRITICAL LANGUAGE RULE: All output fields MUST be in the SAME language as the original memory. If the original memory is in English, ALL output must be in English. If in Chinese, ALL output must be in Chinese. Never mix languages.**

**Current Label1 set (primary category)**: {label1_set}
**Current Label2 set (secondary category)**: {label2_set}

{time_anchor}

=== TASK 1: Structure the raw memory ===

Original memory:
{raw_content}

Extract:
1. Pick ONE label1 from the Label1 set. If none fits, suggest a NEW label1 (keep it short, 1-3 words).
2. Pick ONE label2 from the Label2 set. If none fits, suggest a NEW label2 (keep it short, 1-3 words).
3. Extract facts, relations, summary, tags, keywords, context description, and temporal information.

**Important definitions:**
- summary: A concise one-sentence description of what happened (under 50 words)
- keywords: Core concepts/entities mentioned (3-5 short terms, e.g. "gym", "dosa", "Bangalore")
- context_description: A brief semantic description of the conversational context and emotional tone (1-2 sentences)
- temporal_context: When did this event happen? Extract explicit dates OR infer approximate time from context
- event_sequence: Where does this event fit in the timeline relative to other life events?
- event_timestamp: Best estimate of when this happened in YYYY-MM-DD HH:MM:SS format.
  * Explicit dates in the text → use them directly.
  * Relative expressions ("yesterday", "last weekend", "5 minutes ago") → resolve against the conversation time anchor; keep hour-minute precision for sub-day expressions ("5 minutes ago" → e.g. 14:25:00, not 00:00:00).
  * Approximate periods ("summer 2023") → use the middle of the range (e.g. "2023-07-15 12:00:00").
  * Timeless facts or preferences with no event → use the conversation date (i.e., when it was told).
  * Events at a genuinely indeterminate time ("when I was a kid", "years ago") → leave event_timestamp as "" (empty string); do NOT fabricate a date.

=== TASK 2: Judge which neighbors are related ===

Semantic neighbors (retrieved by similarity):
{neighbors}

For each neighbor, judge if it is genuinely related to the original memory. Only mark as related if there is a meaningful connection (same topic, same person, causal, temporal, or supplementary information).

Additionally, check for CONTRADICTIONS: if a neighbor is an OLDER statement about the same subject that the new fact makes outdated (e.g. neighbor says "user likes coffee" but the new fact says "user no longer drinks coffee", or neighbor says "user works at A" but the new fact says "user moved to company B"), include that neighbor's id in "supersede_ids". Only mark genuine contradictions — supplementary or evolving detail is NOT a contradiction.

Output JSON format:
{{
  "summary": "One-sentence summary (same language as input)",
  "facts": ["fact entry 1", "fact entry 2"],
  "relations": ["relation entry 1", "relation entry 2"],
  "label1": "chosen or new label1",
  "label2": "chosen or new label2",
  "label1_is_new": true/false,
  "label2_is_new": true/false,
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "context_description": "Brief description of the conversational context and emotional tone",
  "temporal_context": "When this happened (extracted or inferred)",
  "event_sequence": "Timeline position relative to other events",
  "event_timestamp": "YYYY-MM-DD HH:MM:SS (best estimate)",
  "tags": ["tag1", "tag2"],
  "related_ids": [
    {{"id": "neighbor_id_1", "related": true, "relation_type": "同主题/同人物/因果/时序/补充"}},
    {{"id": "neighbor_id_2", "related": false}},
    ...
  ],
  "supersede_ids": ["neighbor_id_x"]
}}

Output JSON only, no other text."""

LINK_PROMPT = """你是一个记忆关联分析助手。给定一条新记忆和若干候选记忆，判断它们之间是否存在关联。

新记忆:
{new_memory}

候选记忆:
{candidates}

请输出 JSON 格式（只列出有关系的候选项）:
{{
  "links": [
    {{"id": "候选ID", "relation": "同主题/同人物/因果/时序/补充"}}
  ]
}}

如果没有关系，输出 {{"links": []}}。只输出 JSON。"""


def _time_anchor_text(mf: MemoryFile) -> str:
    """对话时间锚点文本（做梦时相对时间 → 绝对日期的解析基准）

    来源优先级：raw_content 的 DATE: 头 > original_date。
    交互 turn 落盘时必带 DATE 头（serve_persona._persist_turns），
    因此 "yesterday/last weekend/5 minutes ago" 可以确定性换算为完整年月日时分。
    注意不能用 timestamp 字段兜底：对导入数据它是合成铺开时间（非对话时间），
    会把无日期线索的记忆钉到假时间上。
    """
    date_str = ""
    m = re.search(r"^DATE:\s*(.+)$", mf.raw_content or "", re.MULTILINE)
    if m:
        date_str = m.group(1).strip()
    elif mf.original_date:
        date_str = mf.original_date.strip()

    if not date_str:
        return (
            "**Conversation time anchor**: unknown. "
            "Make your best estimate of event_timestamp with full year, month and day. "
            "Do NOT use any import/processing time as the event time."
        )
    wd = weekday_name(date_str)
    wd_part = f" ({wd})" if wd else ""
    return (
        f"**Conversation time anchor**: this memory comes from a conversation at "
        f"{date_str}{wd_part}. Use this as the reference \"now\" to resolve ALL relative "
        "time expressions (\"yesterday\", \"last weekend\", \"next week\", \"5 minutes ago\") "
        "into absolute dates with full year, month and day — and keep hour-minute precision "
        "for sub-day expressions (\"5 minutes ago\", \"3 hours ago\"). "
        "Future plans should be anchored after this date."
    )


EVOLVE_PROMPT = """You are a memory evolution assistant. Given a target memory and its related memories, determine if the target memory should be enriched with genuinely new information from its related memories.

**CRITICAL LANGUAGE RULE**: All output MUST be in the SAME language as the input memories. Never mix languages.

Target memory:
  Summary: {summary}
  Facts: {facts}
  Tags: {tags}

Related memories:
{related_summaries}

Rules:
- Only update if related memories provide GENUINELY NEW information not already in the target
- Do NOT just rephrase — add new facts, details, people, places, or context
- Keep the updated summary concise (under 50 words)
- Add at most 2 new facts and 3 new tags

Output JSON:
{{"update": true, "summary": "enriched summary", "new_facts": ["new fact 1"], "new_tags": ["new_tag"]}}
or
{{"update": false}}

Output JSON only."""

FUSE_PROMPT = """You are a memory insight extraction assistant. Given a group of related memories about the same topic, analyze their common patterns and differences to extract a high-level insight.

**CRITICAL LANGUAGE RULE**: Output MUST be in the SAME language as the input memories.

Related memories:
{memories}

Instructions:
1. Identify the COMMON PATTERN across these memories (what keeps recurring?)
2. Identify NOTABLE DIFFERENCES (what varies?)
3. Extract ONE concise insight (under 30 words) that captures the overarching behavioral or thematic pattern

This insight should be actionable — it should help understand HOW the person behaves in this type of situation, not just WHAT happened.

Output the insight text only, no JSON, no extra text."""


class DreamOrchestrator:
    """做梦流程编排器"""

    def __init__(
        self,
        index_store: IndexStore,
        persistent_store: PersistentStore,
        llm_client,
        llm_model: str = "qwen-plus",
        label_config_path: str = "emo/memory/models/label_sets.json",
    ):
        self._index = index_store
        self._persist = persistent_store
        self._llm = llm_client
        self._model = llm_model
        self._label_config_path = Path(label_config_path)
        self._embed_model = None  # 延迟加载（Box 聚类时需要）
        self._db_lock = threading.Lock()  # 保护 SQLite 并发写入

        # 初始标签集（可扩展）
        self._label1_set: List[str] = []
        self._label2_set: List[str] = []
        self._load_label_sets()

    # ── 标签集管理 ──

    # 默认初始标签集
    DEFAULT_LABEL1 = ["conversation", "knowledge", "opinion", "fact", "capability"]
    DEFAULT_LABEL2 = [
        "interest", "astronomy", "geography", "history", "food",
        "news", "politics", "sports", "technology", "art",
        "music", "travel", "health", "work", "family",
        "relationship", "daily_life", "emotion", "education", "entertainment",
    ]

    def _load_label_sets(self) -> None:
        """加载标签集（从文件或默认值）"""
        if self._label_config_path.exists():
            with open(self._label_config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            self._label1_set = config.get("label1", self.DEFAULT_LABEL1)
            self._label2_set = config.get("label2", self.DEFAULT_LABEL2)
            logger.info(
                f"Loaded label sets: {len(self._label1_set)} L1, "
                f"{len(self._label2_set)} L2"
            )
        else:
            self._label1_set = list(self.DEFAULT_LABEL1)
            self._label2_set = list(self.DEFAULT_LABEL2)
            self._save_label_sets()

    def _save_label_sets(self) -> None:
        """保存标签集到文件"""
        self._label_config_path.parent.mkdir(parents=True, exist_ok=True)
        config = {
            "label1": self._label1_set,
            "label2": self._label2_set,
        }
        with open(self._label_config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def _call_llm(self, prompt: str, max_tokens: int = 300) -> str:
        """调用 LLM"""
        try:
            t0 = time.time()
            resp = self._llm.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            log_cost("dreamer", self._model, resp, time.time() - t0)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ""

    def _parse_json(self, text: str) -> Optional[dict]:
        """解析 LLM 输出的 JSON（带容错）"""
        text = text.strip()
        # 去掉 markdown 代码块
        if "```" in text:
            parts = text.split("```")
            for p in parts:
                p = p.strip()
                if p.startswith("json"):
                    p = p[4:].strip()
                if p.startswith("{"):
                    text = p
                    break

        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 容错：尝试修复截断的 JSON
        # 找到最后一个完整的 key-value 对，截断到那里并闭合
        if text.startswith("{"):
            # 尝试逐步移除尾部直到 JSON 有效
            for end_marker in ['"', "]", "}", ",", " ", "\n"]:
                idx = text.rfind(end_marker)
                while idx > 10:
                    candidate = text[:idx + 1]
                    # 闭合未关闭的括号
                    open_braces = candidate.count("{") - candidate.count("}")
                    open_brackets = candidate.count("[") - candidate.count("]")
                    candidate += "]" * max(0, open_brackets)
                    candidate += "}" * max(0, open_braces)
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        idx = text.rfind(end_marker, 0, idx)
                        if idx <= 10:
                            break

        logger.warning(f"Failed to parse JSON: {text[:120]}...")
        return None

    # ── Step 0: 遗忘衰减（Ebbinghaus）──

    def weight_decay(
        self,
        threshold_retrieve: float = 0.1,
        threshold_archive: float = 0.05,
    ) -> dict:
        """Ebbinghaus 式遗忘衰减

        扫描所有记忆，计算当前权重：
          weight = base_weight * decay(days_idle) * access_boost

        衰减参数（FadeMem 式）:
          重要记忆 (base_weight >= 0.7): beta=0.8, 半衰期 ~11天
          一般记忆 (base_weight < 0.7):  beta=1.2, 半衰期 ~5天

        根据权重执行:
          weight >= threshold_retrieve → 保留在 ChromaDB（可检索）
          weight < threshold_retrieve  → 从 ChromaDB 移除（不可检索，SQLite 保留）
          weight < threshold_archive   → 标记为 "archived"

        Args:
            threshold_retrieve: 最低可检索权重
            threshold_archive: 归档权重阈值

        Returns:
            {"demoted": N, "archived": N, "kept": N}
        """
        import math

        all_entries = self._index.get_all_entries()
        if not all_entries:
            return {"demoted": 0, "archived": 0, "kept": 0}

        now = datetime.now()
        demoted = 0
        archived = 0
        kept = 0
        to_remove = []

        for entry in all_entries:
            # 跳过 L3 语义记忆和常识种子（不衰减）
            if entry.category == "semantic" or entry.source == "commonsense":
                kept += 1
                continue

            # 获取事件时间：优先用 event_timestamp，fallback 到 created_at
            mf = self._persist.get(entry.mem_id)
            event_time_str = ""
            if mf and mf.event_timestamp:
                event_time_str = mf.event_timestamp
            if not event_time_str:
                event_time_str = entry.created_at

            # 计算 days_idle
            try:
                event_time = datetime.strptime(event_time_str, "%Y-%m-%d %H:%M:%S")
                days_idle = max((now - event_time).days, 0)
            except (ValueError, TypeError):
                days_idle = 0

            # Ebbinghaus 衰减
            beta = 0.8 if entry.base_weight >= 0.7 else 1.2
            lambda_i = 0.1 * math.exp(-0.5 * entry.base_weight)
            decay = math.exp(-lambda_i * (days_idle ** beta))

            # 访问增强
            access_boost = entry.access_count / (1 + entry.access_count)

            # 最终权重
            weight = entry.base_weight * decay * (0.5 + 0.5 * access_boost)
            weight = max(weight, 0.01)

            if weight < threshold_archive:
                # 归档
                to_remove.append(entry.mem_id)
                mf = self._persist.get(entry.mem_id)
                if mf:
                    mf.status = "archived"
                    self._persist.save([mf])
                archived += 1
            elif weight < threshold_retrieve:
                # 从检索索引移除
                to_remove.append(entry.mem_id)
                demoted += 1
            else:
                kept += 1

        # 批量从 ChromaDB 移除
        if to_remove:
            self._index.delete(to_remove)

        stats = {"demoted": demoted, "archived": archived, "kept": kept}
        logger.info(f"Weight decay: kept={kept}, demoted={demoted}, archived={archived}")
        print(f"  衰减结果: 保留={kept}, 降权={demoted}, 归档={archived}", flush=True)
        return stats

    # ── Step 1: 结构化 ──

    def structure_raw_memories(self, limit: int = None) -> int:
        """处理所有 status="raw" 的记忆，提取结构化信息

        Args:
            limit: 最多处理 N 条（None 表示全部）

        Returns:
            处理的记忆数量
        """
        # 查找所有 raw 记忆
        raw_memories = self._persist.get_by_status("raw")
        if limit:
            raw_memories = raw_memories[:limit]
        if not raw_memories:
            logger.info("No raw memories to process")
            return 0

        logger.info(f"Processing {len(raw_memories)} raw memories...")
        logger.info(f"Label1 set ({len(self._label1_set)}): {self._label1_set}")
        logger.info(f"Label2 set ({len(self._label2_set)}): {self._label2_set[:10]}...")
        count = 0
        new_labels = {"label1": set(), "label2": set()}

        for i, mf in enumerate(raw_memories):
            result = self._structure_one(mf, new_labels)
            if result:
                count += 1
            if (i + 1) % 10 == 0 or (i + 1) == len(raw_memories):
                print(f"  进度: {i+1}/{len(raw_memories)} (成功 {count})", flush=True)

        # 扩展标签集
        for new_l1 in new_labels["label1"]:
            if new_l1 not in self._label1_set:
                self._label1_set.append(new_l1)
                logger.info(f"  🆕 New label1: {new_l1}")
        for new_l2 in new_labels["label2"]:
            if new_l2 not in self._label2_set:
                self._label2_set.append(new_l2)
                logger.info(f"  🆕 New label2: {new_l2}")

        if new_labels["label1"] or new_labels["label2"]:
            self._save_label_sets()

        logger.info(f"Structured {count}/{len(raw_memories)} memories")
        return count

    def _structure_one(self, mf: MemoryFile, new_labels: dict) -> bool:
        """结构化单条记忆"""
        prompt = STRUCTURE_PROMPT.format(
            raw_content=mf.raw_content,
            label1_set=", ".join(self._label1_set),
            label2_set=", ".join(self._label2_set),
            time_anchor=_time_anchor_text(mf),
        )
        output = self._call_llm(prompt, max_tokens=500)
        data = self._parse_json(output)

        if not data:
            return False

        # 提取 label1 / label2
        label1 = data.get("label1", "")
        label2 = data.get("label2", "")

        # 追踪新标签
        if data.get("label1_is_new") and label1:
            new_labels["label1"].add(label1)
        if data.get("label2_is_new") and label2:
            new_labels["label2"].add(label2)

        # 更新 MemoryFile
        mf.summary = data.get("summary", mf.summary)
        mf.fact_entries = json.dumps(data.get("facts", []), ensure_ascii=False)
        mf.rel_entries = json.dumps(data.get("relations", []), ensure_ascii=False)
        mf.label1 = label1
        mf.label2 = label2
        mf.category = label1        # 兼容旧字段
        mf.sub_category = label2    # 兼容旧字段
        mf.tags = ",".join(data.get("tags", []))
        mf.keywords = json.dumps(data.get("keywords", []), ensure_ascii=False)
        mf.context_description = data.get("context_description", "")
        mf.temporal_context = data.get("temporal_context", "")
        mf.event_sequence = data.get("event_sequence", "")
        # LLM 返回空（不可考事件）时保留已有值——交互 turn 落盘时已填
        # 提及时间，清空会让它掉出时间轴
        mf.event_timestamp = data.get("event_timestamp", "") or mf.event_timestamp
        mf.status = "dreamed"

        # 保存回 SQLite
        self._persist.save([mf])

        # 重建 ChromaDB 索引（先删后写避免重复，保留统计与关联）
        self._rebuild_index_entry(mf)

        return True

    def _build_embedding_text(self, mf: MemoryFile) -> str:
        """构建 embedding 文本（A-MEM 式：联合编码内容+关键词+标签+上下文描述）

        A-MEM: e_i = f_enc[concat(c_i, K_i, G_i, X_i)]
        """
        parts = []

        # c_i: 内容（用 summary 而非 raw_content，更精炼）
        if mf.summary:
            parts.append(mf.summary)

        # K_i: 关键词
        keywords = mf.get_keywords() if hasattr(mf, 'get_keywords') else []
        if not keywords and mf.keywords:
            try:
                keywords = json.loads(mf.keywords)
            except (json.JSONDecodeError, TypeError):
                keywords = []
        if keywords:
            parts.append(" ".join(keywords[:5]))

        # G_i: 标签
        if mf.tags:
            parts.append(mf.tags.replace(",", " "))

        # X_i: 上下文描述
        if mf.context_description:
            parts.append(mf.context_description)

        return ". ".join(parts)

    def _rebuild_index_entry(self, mf: MemoryFile) -> None:
        """重建索引条目（先删后写避免重复），保留统计与关联

        结构化/演化会改写 summary 等语义字段、需要重建 embedding，
        但 access_count / utility_count / base_weight / related_ids
        是在线运行与历史做梦积累的资产，必须随重建保留——
        否则每次做梦都会把统计数据清零、双存储的图发散。
        """
        old = self._index.get_by_id(mf.mem_id)
        self._index.delete([mf.mem_id])
        embedding_text = self._build_embedding_text(mf)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # tags 合并而非覆盖：结构化把 mf.tags 换成了 LLM 语义标签，但旧条目
        # 里的溯源标签（session_id/dia_id/has_answer）是召回统计与审计的
        # 唯一线索，必须随重建保留（2026-08-11 LME 做梦后 recall 全 0 实锤）
        merged_tags = list(dict.fromkeys(
            [*(old.tags if old else []), *(mf.tags.split(",") if mf.tags else [])]
        ))
        entry = IndexEntry(
            mem_id=mf.mem_id,
            summary=mf.summary,
            embedding_text=embedding_text,
            category=mf.label1 or mf.category,
            sub_category=mf.label2 or mf.sub_category,
            tags=merged_tags,
            source=mf.source,
            speaker=mf.speaker,
            original_date=mf.original_date,
            event_timestamp=mf.event_timestamp,
            base_weight=old.base_weight if old else 0.5,
            access_count=old.access_count if old else 0,
            utility_count=old.utility_count if old else 0.0,
            related_ids=list(old.related_ids) if old else [],
            created_at=old.created_at if old else now_str,
            last_access=old.last_access if old else now_str,
        )
        self._index.add([entry], documents=[embedding_text])

    # ── 优化版：合并 Step 1 + Step 2 ──

    def structure_and_link_memories(self, limit: int = None, top_k: int = 10) -> dict:
        """合并 Step 1 (结构化) + Step 2 (关联建立) 为一次 LLM 调用

        对每条 raw 记忆：
          1. 检索 top-k 语义近邻
          2. 一次 LLM 调用同时完成：结构化提取 + 关联判断

        Args:
            limit: 最多处理 N 条（None 表示全部）
            top_k: 检索多少个邻居用于关联判断

        Returns:
            {"structured": N, "links": N}
        """
        raw_memories = self._persist.get_by_status("raw")
        if limit:
            raw_memories = raw_memories[:limit]
        if not raw_memories:
            logger.info("No raw memories to process")
            return {"structured": 0, "links": 0}

        logger.info(f"Processing {len(raw_memories)} raw memories (structure + link)...")
        logger.info(f"Label1 set ({len(self._label1_set)}): {self._label1_set}")
        logger.info(f"Label2 set ({len(self._label2_set)}): {self._label2_set[:10]}...")

        count_struct = 0
        count_links = 0
        new_labels = {"label1": set(), "label2": set()}

        for i, mf in enumerate(raw_memories):
            success, num_links = self._structure_and_link_one(mf, top_k, new_labels)
            if success:
                count_struct += 1
                count_links += num_links
            if (i + 1) % 10 == 0 or (i + 1) == len(raw_memories):
                print(f"  进度: {i+1}/{len(raw_memories)} (结构化 {count_struct}, 关联 {count_links})", flush=True)

        # 扩展标签集
        for new_l1 in new_labels["label1"]:
            if new_l1 not in self._label1_set:
                self._label1_set.append(new_l1)
                logger.info(f"  🆕 New label1: {new_l1}")
        for new_l2 in new_labels["label2"]:
            if new_l2 not in self._label2_set:
                self._label2_set.append(new_l2)
                logger.info(f"  🆕 New label2: {new_l2}")

        if new_labels["label1"] or new_labels["label2"]:
            self._save_label_sets()

        logger.info(f"Structured {count_struct}, Links {count_links}")
        return {"structured": count_struct, "links": count_links}

    def _structure_and_link_one(self, mf: MemoryFile, top_k: int, new_labels: dict) -> Tuple[bool, int]:
        """一次 LLM 调用完成结构化 + 关联判断

        Returns:
            (是否成功, 建立的关联数量)
        """
        # 1. 检索语义近邻
        query = mf.raw_content[:300] if mf.raw_content else ""
        results = self._index.search(query, top_k=top_k + 1)

        # 排除自身
        candidates = [(e, s) for e, s in results if e.mem_id != mf.mem_id]
        # 关联白名单：LLM 返回的 related_ids 只允许落在候选集中（防幻觉 ID 写入图）
        valid_link_ids = {e.mem_id for e, _ in candidates[:top_k]}

        # 构建邻居文本
        neighbors_text = ""
        if candidates:
            neighbor_lines = []
            for e, _ in candidates[:top_k]:
                neighbor_lines.append(f"ID: {e.mem_id}\nSummary: {e.summary}")
            neighbors_text = "\n\n".join(neighbor_lines)
        else:
            neighbors_text = "(No semantic neighbors found)"

        # 2. 一次 LLM 调用：结构化 + 关联判断
        prompt = STRUCTURE_AND_LINK_PROMPT.format(
            raw_content=mf.raw_content,
            label1_set=", ".join(self._label1_set),
            label2_set=", ".join(self._label2_set),
            neighbors=neighbors_text,
            time_anchor=_time_anchor_text(mf),
        )
        output = self._call_llm(prompt, max_tokens=2500)
        data = self._parse_json(output)

        if not data:
            logger.warning(f"LLM 返回无效 JSON for {mf.mem_id}")
            return (False, 0)

        # 3. 提取结构化信息
        label1 = data.get("label1", "")
        label2 = data.get("label2", "")

        # 提取关联判断结果
        related_ids_data = data.get("related_ids", [])

        # 追踪新标签
        if data.get("label1_is_new") and label1:
            new_labels["label1"].add(label1)
        if data.get("label2_is_new") and label2:
            new_labels["label2"].add(label2)

        # 更新 MemoryFile（保留原始元数据标签）
        mf.summary = data.get("summary", mf.summary)
        mf.fact_entries = json.dumps(data.get("facts", []), ensure_ascii=False)
        mf.rel_entries = json.dumps(data.get("relations", []), ensure_ascii=False)
        mf.label1 = label1
        mf.label2 = label2
        mf.category = label1
        mf.sub_category = label2

        # 保留原始元数据标签（dia_id, speaker, session_id）并追加 LLM 生成的标签
        original_tags = [t.strip() for t in mf.tags.split(",") if t.strip()] if mf.tags else []
        llm_tags = data.get("tags", [])
        # 保留元数据标签（dia_id 格式 D*:*, speaker 名, session_*）
        metadata_tags = [t for t in original_tags if ":" in t or t.startswith("session_") or t in [mf.speaker]]
        combined_tags = list(set(metadata_tags + llm_tags))  # 去重
        mf.tags = ",".join(sorted(combined_tags))

        mf.keywords = json.dumps(data.get("keywords", []), ensure_ascii=False)
        mf.context_description = data.get("context_description", "")
        mf.temporal_context = data.get("temporal_context", "")
        mf.event_sequence = data.get("event_sequence", "")
        # LLM 返回空（不可考事件）时保留已有值——交互 turn 落盘时已填
        # 提及时间，清空会让它掉出时间轴
        mf.event_timestamp = data.get("event_timestamp", "") or mf.event_timestamp
        mf.status = "dreamed"

        # 使用锁保护 SQLite 和 ChromaDB 并发写入
        with self._db_lock:
            # 保存回 SQLite
            self._persist.save([mf])

            # 重建 ChromaDB 索引（先删后写避免重复，保留统计与关联）
            self._rebuild_index_entry(mf)

        # 4. 处理关联判断结果（白名单过滤：丢弃不在候选集中的幻觉 ID）
        related_ids_data = data.get("related_ids", [])
        link_ids = []
        for item in related_ids_data:
            if isinstance(item, dict) and item.get("related") and item.get("id"):
                if item["id"] in valid_link_ids:
                    link_ids.append(item["id"])
                else:
                    logger.warning(
                        f"Dropping hallucinated link id '{item['id']}' "
                        f"(not in candidate set) for {mf.mem_id}"
                    )

        if link_ids:
            # 使用锁保护 SQLite 和 ChromaDB 并发写入
            with self._db_lock:
                # 更新 ChromaDB 中的 related_ids
                entry = self._index.get_by_id(mf.mem_id)
                if entry:
                    existing = set(entry.related_ids)
                    existing.update(link_ids)
                    entry.related_ids = list(existing)
                    self._index.update_entry(entry)

                # 同步写入 SQLite
                mf.related_ids = ",".join(set(
                    (mf.related_ids.split(",") if mf.related_ids else []) + link_ids
                ))
                self._persist.save([mf])

        # 5. 处理矛盾取代：标记被新事实淘汰的旧记忆（白名单过滤，防幻觉 ID）
        supersede_ids = [
            sid for sid in data.get("supersede_ids", [])
            if isinstance(sid, str) and sid in valid_link_ids and sid != mf.mem_id
        ]
        for sid in supersede_ids:
            with self._db_lock:
                self._persist.mark_superseded(sid, mf.mem_id)
                old_entry = self._index.get_by_id(sid)
                if old_entry:
                    old_entry.superseded_by = mf.mem_id
                    self._index.update_entry(old_entry)
        if supersede_ids:
            logger.info(f"{mf.mem_id} superseded {len(supersede_ids)} old memories: {supersede_ids}")

        return (True, len(link_ids))

    # ── 并行优化版（async）──

    async def structure_and_link_memories_async(self, limit: int = None, top_k: int = 10, batch_size: int = 5) -> dict:
        """并行版：合并 Step 1 + Step 2，使用 asyncio 并行处理

        Args:
            limit: 最多处理 N 条
            top_k: 检索多少个邻居
            batch_size: 同时处理多少条（控制并发）

        Returns:
            {"structured": N, "links": N}
        """
        import asyncio

        raw_memories = self._persist.get_by_status("raw")
        if limit:
            raw_memories = raw_memories[:limit]
        if not raw_memories:
            return {"structured": 0, "links": 0}

        logger.info(f"Processing {len(raw_memories)} raw memories (async, batch_size={batch_size})...")

        count_struct = 0
        count_links = 0
        new_labels = {"label1": set(), "label2": set()}

        # 使用信号量控制并发
        semaphore = asyncio.Semaphore(batch_size)

        async def process_one(mf: MemoryFile):
            nonlocal count_struct, count_links
            async with semaphore:
                # 在线程池中执行同步的 LLM 调用
                success, num_links = await asyncio.get_event_loop().run_in_executor(
                    None, self._structure_and_link_one, mf, top_k, new_labels
                )
                if success:
                    count_struct += 1
                    count_links += num_links

        # 分批处理
        for i in range(0, len(raw_memories), batch_size):
            batch = raw_memories[i:i + batch_size]
            tasks = [process_one(mf) for mf in batch]
            await asyncio.gather(*tasks)
            print(f"  进度: {min(i + batch_size, len(raw_memories))}/{len(raw_memories)} (结构化 {count_struct}, 关联 {count_links})", flush=True)

        # 扩展标签集
        for new_l1 in new_labels["label1"]:
            if new_l1 not in self._label1_set:
                self._label1_set.append(new_l1)
        for new_l2 in new_labels["label2"]:
            if new_l2 not in self._label2_set:
                self._label2_set.append(new_l2)

        if new_labels["label1"] or new_labels["label2"]:
            self._save_label_sets()

        return {"structured": count_struct, "links": count_links}

    # ── Step 2: 关联建立 ──

    def generate_links(self, top_k: int = 10) -> int:
        """为 dreamed 记忆建立关联（A-MEM 式 link generation）

        Returns:
            建立关联的记忆对数
        """
        # 获取所有 dreamed 记忆
        dreamed = self._persist.get_by_status("dreamed")
        if not dreamed:
            return 0

        logger.info(f"Generating links for {len(dreamed)} dreamed memories...")
        link_count = 0

        for i, mf in enumerate(dreamed):
            links = self._link_one(mf, top_k)
            if links:
                link_count += len(links)
            if (i + 1) % 10 == 0 or (i + 1) == len(dreamed):
                print(f"  进度: {i+1}/{len(dreamed)} (关联 {link_count} 条)", flush=True)

        logger.info(f"Generated {link_count} links")
        return link_count

    def _link_one(self, mf: MemoryFile, top_k: int) -> List[dict]:
        """为单条记忆建立关联"""
        # 检索语义近邻
        query = mf.summary or mf.raw_content[:200]
        results = self._index.search(query, top_k=top_k + 1)

        # 排除自身
        candidates = [(e, s) for e, s in results if e.mem_id != mf.mem_id]
        if not candidates:
            return []

        # 构建候选列表
        candidate_text = "\n".join(
            f'ID: {e.mem_id}, Summary: {e.summary}'
            for e, _ in candidates[:top_k]
        )

        prompt = LINK_PROMPT.format(
            new_memory=f"Summary: {mf.summary}\nFacts: {mf.fact_entries}",
            candidates=candidate_text,
        )
        output = self._call_llm(prompt, max_tokens=400)
        data = self._parse_json(output)

        if not data or not data.get("links"):
            return []

        # 更新 related_ids（白名单过滤：丢弃不在候选集中的幻觉 ID）
        valid_ids = {e.mem_id for e, _ in candidates[:top_k]}
        new_links = [
            l for l in data["links"]
            if "id" in l and l["id"] in valid_ids
        ]
        link_ids = [l["id"] for l in new_links]

        if link_ids:
            # 使用锁保护 SQLite 和 ChromaDB 并发写入
            with self._db_lock:
                # 更新 ChromaDB 中的 related_ids
                entry = self._index.get_by_id(mf.mem_id)
                if entry:
                    existing = set(entry.related_ids)
                    existing.update(link_ids)
                    entry.related_ids = list(existing)
                    self._index.update_entry(entry)

                # 同步写入 SQLite
                mf.related_ids = ",".join(set(
                    (mf.related_ids.split(",") if mf.related_ids else []) + link_ids
                ))
                self._persist.save([mf])

        return new_links

    # ── Step 3: 记忆演化 ──

    def evolve_memories(self, min_related: int = 3) -> int:
        """记忆演化：用关联记忆的新信息丰富已有记忆

        对每条记忆，收集所有关联记忆的摘要，一次 LLM 调用判断是否需要丰富。

        Args:
            min_related: 最少关联数（太少的跳过，节省 LLM 调用）

        Returns:
            更新的记忆数量
        """
        dreamed = self._persist.get_by_status("dreamed")
        if not dreamed:
            return 0

        # 过滤：只处理有足够关联的记忆
        candidates = []
        for mf in dreamed:
            entry = self._index.get_by_id(mf.mem_id)
            if entry and len(entry.related_ids) >= min_related:
                candidates.append((mf, entry))

        if not candidates:
            logger.info("No memories with enough related_ids for evolution")
            return 0

        logger.info(f"Evolving {len(candidates)} memories (min_related={min_related})...")
        evolve_count = 0
        updated_ids = set()  # 防止级联更新

        for i, (mf, entry) in enumerate(candidates):
            if mf.mem_id in updated_ids:
                continue
            evolved = self._evolve_one(mf, entry)
            if evolved:
                evolve_count += 1
                updated_ids.add(mf.mem_id)
            if (i + 1) % 10 == 0 or (i + 1) == len(candidates):
                print(f"  进度: {i+1}/{len(candidates)} (演化 {evolve_count} 条)", flush=True)

        logger.info(f"Evolved {evolve_count}/{len(candidates)} memories")
        return evolve_count

    def _evolve_one(self, mf: MemoryFile, entry: IndexEntry) -> bool:
        """用关联记忆丰富目标记忆"""
        # 收集关联记忆的摘要（最多 10 条）
        related_summaries = []
        for rid in entry.related_ids[:10]:
            rel_mf = self._persist.get(rid)
            if rel_mf and rel_mf.summary:
                related_summaries.append(f"- {rel_mf.summary}")

        if not related_summaries:
            return False

        prompt = EVOLVE_PROMPT.format(
            summary=mf.summary,
            facts=mf.fact_entries or "[]",
            tags=mf.tags or "",
            related_summaries="\n".join(related_summaries),
        )
        output = self._call_llm(prompt, max_tokens=300)
        data = self._parse_json(output)

        if not data or not data.get("update"):
            return False

        # 更新 summary
        new_summary = data.get("summary", "")
        if new_summary and new_summary != mf.summary:
            mf.summary = new_summary

        # 追加新 facts
        new_facts = data.get("new_facts", [])
        if new_facts:
            existing_facts = mf.get_fact_entries()
            existing_facts.extend(new_facts[:2])
            mf.fact_entries = json.dumps(existing_facts, ensure_ascii=False)

        # 扩展 tags
        new_tags = data.get("new_tags", [])
        if new_tags:
            existing_tags = set(mf.tags.split(",")) if mf.tags else set()
            existing_tags.update(new_tags[:3])
            mf.tags = ",".join(sorted(existing_tags))

        # 同步 SQLite + ChromaDB（重建索引，保留统计与关联）
        self._persist.save([mf])
        self._rebuild_index_entry(mf)
        return True

    # ── Step 4: 簇融合（G-Memory 式 + A-MEM Box 聚类）──

    def fuse_clusters(self, min_cluster_size: int = 4, box_similarity: float = 0.75) -> int:
        """发现记忆簇并融合为 L3 语义记忆

        方法（结合 G-Memory + A-MEM）:
        1. 按 label2 分组（类似 G-Memory 的 Query Graph 按任务分组）
        2. 组内按 context_description 相似度聚类为 Box（A-MEM 的 Box 概念）
        3. 每个 Box 内 LLM 对比分析不同记忆的异同 → 提取模式/教训（G-Memory 式 Insight 提取）

        Args:
            min_cluster_size: 每组最少记忆数
            box_similarity: Box 聚类的余弦相似度阈值

        Returns:
            创建的 L3 语义记忆数量
        """
        # 获取所有 dreamed/settled 记忆
        all_memories = (
            self._persist.get_by_status("dreamed") +
            self._persist.get_by_status("settled")
        )
        # 排除已有的 L3 记忆
        all_memories = [mf for mf in all_memories if mf.label1 != "semantic"]

        if not all_memories:
            return 0

        # Step 1: 按 label2 分组
        groups = {}
        for mf in all_memories:
            label2 = mf.label2 or "other"
            if label2 not in groups:
                groups[label2] = []
            groups[label2].append(mf)

        logger.info(f"Found {len(groups)} label2 groups")

        # Step 2+3: 组内 Box 聚类 + 融合
        fuse_count = 0
        total_boxes = 0

        for label2, memories in groups.items():
            if len(memories) < min_cluster_size:
                continue

            # Box 聚类：按 context_description 相似度分组
            boxes = self._cluster_into_boxes(memories, box_similarity)
            total_boxes += len(boxes)

            for box in boxes:
                if len(box) < min_cluster_size:
                    continue
                fused = self._fuse_one_box(box, label2)
                if fused:
                    fuse_count += 1

            print(f"  [{label2}] {len(memories)} 条记忆 → {len(boxes)} 个 Box → 融合中...", flush=True)

        print(f"  总计: {len(groups)} 组, {total_boxes} 个 Box, {fuse_count} 条 L3", flush=True)
        logger.info(f"Fused {fuse_count} boxes into L3 semantic memories")
        return fuse_count

    async def fuse_clusters_async(self, min_cluster_size: int = 4, box_similarity: float = 0.75, batch_size: int = 5) -> int:
        """并行版：发现记忆簇并融合为 L3 语义记忆

        Args:
            min_cluster_size: 每组最少记忆数
            box_similarity: Box 聚类的余弦相似度阈值
            batch_size: 同时处理多少个 Box（控制并发）

        Returns:
            创建的 L3 语义记忆数量
        """
        import asyncio

        # 获取所有 dreamed/settled 记忆
        all_memories = (
            self._persist.get_by_status("dreamed") +
            self._persist.get_by_status("settled")
        )
        # 排除已有的 L3 记忆
        all_memories = [mf for mf in all_memories if mf.label1 != "semantic"]

        if not all_memories:
            return 0

        # Step 1: 按 label2 分组
        groups = {}
        for mf in all_memories:
            label2 = mf.label2 or "other"
            if label2 not in groups:
                groups[label2] = []
            groups[label2].append(mf)

        logger.info(f"Found {len(groups)} label2 groups")

        # Step 2: 组内 Box 聚类（同步，因为用 embedding 模型）
        all_boxes = []  # [(box_memories, label2), ...]
        for label2, memories in groups.items():
            if len(memories) < min_cluster_size:
                continue

            boxes = self._cluster_into_boxes(memories, box_similarity)
            for box in boxes:
                if len(box) >= min_cluster_size:
                    all_boxes.append((box, label2))

            print(f"  [{label2}] {len(memories)} 条记忆 → {len(boxes)} 个 Box", flush=True)

        if not all_boxes:
            return 0

        logger.info(f"Total {len(all_boxes)} boxes to fuse (async, batch_size={batch_size})")

        # Step 3: 并行融合所有 Box
        fuse_count = 0
        semaphore = asyncio.Semaphore(batch_size)

        async def fuse_one(box_memories, label2):
            nonlocal fuse_count
            async with semaphore:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self._fuse_one_box, box_memories, label2
                )
                if result:
                    fuse_count += 1

        # 一次性创建所有任务，semaphore 控制并发
        tasks = [fuse_one(box, label2) for box, label2 in all_boxes]
        await asyncio.gather(*tasks)

        print(f"  总计: {len(groups)} 组, {len(all_boxes)} 个 Box, {fuse_count} 条 L3", flush=True)
        logger.info(f"Fused {fuse_count} boxes into L3 semantic memories (async)")
        return fuse_count

    def _cluster_into_boxes(self, memories: List[MemoryFile], threshold: float) -> List[List[MemoryFile]]:
        """A-MEM 式 Box 聚类：按 context_description 相似度分组

        A-MEM: "Box" 描述相关记忆通过相似上下文描述（X_i）互联的机制。
        一条记忆可以同时存在于多个 Box 中（这里简化为每个记忆只在一个 Box 中）。
        """
        if not memories:
            return []

        # 延迟加载 embedding 模型
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            model_path = "emo/models/all-MiniLM-L6-v2"
            if os.path.isdir(model_path):
                self._embed_model = SentenceTransformer(model_path)
            else:
                self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        # 获取 context_description 的嵌入向量
        descriptions = []
        for mf in memories:
            desc = mf.context_description or mf.summary or ""
            descriptions.append(desc)

        # 用 embedding 模型编码
        embeddings = self._embed_model.encode(descriptions, normalize_embeddings=True)

        # 贪心聚类：按相似度分组
        boxes = []
        assigned = set()

        for i in range(len(memories)):
            if i in assigned:
                continue

            box = [memories[i]]
            assigned.add(i)

            for j in range(i + 1, len(memories)):
                if j in assigned:
                    continue
                sim = float(np.dot(embeddings[i], embeddings[j]))
                if sim >= threshold:
                    box.append(memories[j])
                    assigned.add(j)

            boxes.append(box)

        return boxes

    def _fuse_one_box(self, box_memories: List[MemoryFile], label2: str) -> bool:
        """G-Memory 式 Insight 提取：LLM 对比分析 Box 内记忆的异同 → 提取模式/教训

        G-Memory: "从成功/失败轨迹中学习教训（通过 LLM 对比分析）"
        适配：从 Box 内多条记忆中对比分析，提取共同模式和差异
        """
        import hashlib
        member_ids = [mf.mem_id for mf in box_memories]
        box_hash = hashlib.md5(",".join(member_ids).encode()).hexdigest()[:8]
        fused_id = f"l3_{label2}_{box_hash}"
        member_ids_str = ",".join(member_ids)

        # 幂等回补（2026-08-13 多路召回改造）：簇已融合过则只补成员连接，不重复调 LLM
        existing = self._persist.get_batch([fused_id])
        if existing:
            mf = existing[0]
            if not mf.related_ids:
                mf.related_ids = member_ids_str
                with self._db_lock:
                    self._persist.save([mf])
                entry = self._index.get_by_id(fused_id)
                if entry is not None:
                    entry.related_ids = member_ids
                    self._index.update_entry(entry)
            return False

        # 构建对比分析文本
        memory_texts = []
        all_keywords = set()
        for i, mf in enumerate(box_memories[:10]):
            facts = mf.get_fact_entries()
            facts_str = "; ".join(facts[:3]) if facts else mf.summary
            memory_texts.append(f"Memory {i+1}: {facts_str}")
            all_keywords.update(mf.get_keywords())

        if len(memory_texts) < 3:
            return False

        # G-Memory 式 Insight 提取 prompt
        prompt = FUSE_PROMPT.format(memories="\n".join(memory_texts))
        insight_text = self._call_llm(prompt, max_tokens=100)

        if not insight_text or len(insight_text) < 10:
            return False

        fused_mf = MemoryFile(
            mem_id=fused_id,
            raw_content=f"L3 insight from {len(box_memories)} memories ({label2} box)",
            summary=insight_text.strip(),
            category="semantic",
            sub_category=label2,
            tags=",".join(list(all_keywords)[:10]),
            source="dreamed",
            label1="semantic",
            label2=label2,
            keywords=json.dumps(list(all_keywords)[:5], ensure_ascii=False),
            context_description=f"High-level insight about {label2} patterns",
            box_id=fused_id,
            status="settled",
            importance=0.8,
            related_ids=member_ids_str,  # 簇成员精确连接（多路召回 L3 通道用）
        )

        # 使用锁保护 SQLite 并发写入
        with self._db_lock:
            self._persist.save([fused_mf])

        # 写入 ChromaDB
        embedding_text = insight_text.strip()
        entry = IndexEntry(
            mem_id=fused_id,
            summary=insight_text.strip(),
            embedding_text=embedding_text,
            category="semantic",
            sub_category=label2,
            tags=list(all_keywords)[:10],
            source="dreamed",
            base_weight=0.8,
            related_ids=member_ids,
        )
        self._index.add([entry], documents=[embedding_text])
        return True

    # ── Step 5: 效用清理 ──

    def utility_cleanup(self, min_access: int = 3, utility_threshold: float = 0.3) -> int:
        """清理低效用记忆（降权而非删除）

        Args:
            min_access: 最少检索次数（低于此值不考虑清理）
            utility_threshold: 效用比阈值

        Returns:
            清理的记忆数量
        """
        all_entries = self._index.get_all_entries()
        total = len(all_entries)
        cleaned = 0
        skipped_no_access = 0

        # 守卫：没有任何效用数据时跳过（历史 bug：utility_count 从未被递增，
        # 此时 ratio 恒为 0，会把所有高频记忆误降权）
        total_utility = sum(e.utility_count for e in all_entries)
        if total_utility <= 0:
            logger.warning(
                "Utility cleanup skipped: no utility data recorded yet. "
                "Run online path (utility tracking) first."
            )
            print("  ⚠️ 跳过效用清理：尚无 utility 数据（先运行在线路径积累埋点）", flush=True)
            return 0

        for i, entry in enumerate(all_entries):
            if entry.access_count < min_access:
                skipped_no_access += 1
                continue

            utility_ratio = entry.utility_count / max(entry.access_count, 1)
            if utility_ratio < utility_threshold:
                entry.base_weight = max(entry.base_weight * 0.5, 0.01)
                self._index.update_entry(entry)
                cleaned += 1

            if (i + 1) % 100 == 0:
                print(f"  进度: {i+1}/{total}", flush=True)

        print(f"  跳过(无访问数据): {skipped_no_access}, 降权: {cleaned}", flush=True)
        logger.info(f"Utility cleanup: {cleaned} reduced, {skipped_no_access} skipped (no access data)")
        return cleaned

    # ── 完整做梦流程 ──

    def dream(self, use_async: bool = False, batch_size: int = 5, skip_decay: bool = False) -> dict:
        """执行完整做梦流程

        Args:
            use_async: 是否使用并行处理（减少 wall-clock 时间）
            batch_size: 并行处理的批次大小
            skip_decay: 跳过遗忘衰减（新系统/一次性评估场景，
                        衰减是为长时运行设计的）

        Returns:
            统计信息
        """
        logger.info("=" * 60)
        logger.info("🌙 Dream started")
        logger.info("=" * 60)

        stats = {}

        # Pre-step: 回写在线期间累积的访问/效用计数（读路径不写库，在此批量落盘）
        stats["flushed"] = self._index.flush_stats()

        # Step 0: 遗忘衰减
        if skip_decay:
            logger.info("\n--- Step 0: Weight decay (SKIPPED) ---")
            stats["decay"] = {"skipped": True}
        else:
            logger.info("\n--- Step 0: Weight decay (Ebbinghaus forgetting) ---")
            stats["decay"] = self.weight_decay()

        # Step 1+2: 结构化 + 关联建立（合并，减少 LLM 调用）
        logger.info("\n--- Step 1+2: Structuring + Linking (combined) ---")
        if use_async:
            import asyncio
            # asyncio.run 自建事件循环：后台线程（serve_persona.start_dream）
            # 没有默认循环，get_event_loop() 会直接抛 RuntimeError
            stats["struct_and_link"] = asyncio.run(
                self.structure_and_link_memories_async(batch_size=batch_size)
            )
        else:
            stats["struct_and_link"] = self.structure_and_link_memories()

        # Step 3: 记忆演化（已删除 - 依赖 Step 2 的关联判断，检索时自动返回关联记忆）
        logger.info("\n--- Step 3: Memory evolution (SKIPPED - handled by retrieval) ---")
        stats["evolved"] = 0

        # Step 4: 融合
        logger.info("\n--- Step 4: Fusing clusters ---")
        if use_async:
            stats["fused"] = asyncio.run(
                self.fuse_clusters_async(batch_size=batch_size)
            )
        else:
            stats["fused"] = self.fuse_clusters()

        # Step 5: 效用清理
        logger.info("\n--- Step 5: Utility cleanup ---")
        stats["cleaned"] = self.utility_cleanup()

        # Step 6: 标签分类器训练
        logger.info("\n--- Step 6: Training tag classifier ---")
        stats["tag_classifier"] = self.train_tag_classifier()

        # 标记已处理的记忆为 settled
        self._persist.mark_settled()

        logger.info("\n" + "=" * 60)
        logger.info(f"🌙 Dream complete: {stats}")
        logger.info("=" * 60)

        return stats

    # ── Step 6: 标签分类器 ──

    def train_tag_classifier(self) -> bool:
        """训练标签分类器

        从已结构化的记忆中收集 (user_message → label1, label2) 训练对，
        训练 sklearn LogisticRegression 分类器。

        Returns:
            是否训练成功
        """
        from .tag_classifier import TagClassifier

        tc = TagClassifier(
            persist_store=self._persist,
            llm_client=self._llm,
            llm_model=self._model,
        )

        success = tc.train()

        if success:
            logger.info(f"  ✅ Tag classifier trained: {tc.stats()}")
        else:
            logger.warning("  ❌ Tag classifier training failed")

        return success
