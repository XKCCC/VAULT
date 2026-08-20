"""LoCoMo data loader — 将 LoCoMo 对话数据导入到 EMO 记忆系统

利用 LoCoMo 已有的丰富标注数据：
  - observation: 每个 session 的 per-speaker 观察（结构化事实）
  - event_summary: 每个 session 的事件描述
  - session_summary: 每个 session 的段落摘要

存储策略：
  - 每条对话 turn → 一条 turn 级记忆
  - 每个 session → 一条 session 级记忆（高层摘要）

Usage:
    from memory.locomo_loader import LoCoMoLoader
    loader = LoCoMoLoader(index_store, persistent_store)
    loader.load_conversation(conv_data, observation, session_summary, event_summary)
"""

import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from memory.schema import IndexEntry, MemoryFile

logger = logging.getLogger(__name__)


def _parse_locomo_datetime(date_time_str: str) -> str:
    """"1:56 pm on 8 May, 2023" → "2023-05-08 13:56:00"（解析失败返回 ""）

    LoCoMo 的 session date_time 是真实事件时间，导入时回填 event_timestamp——
    否则做梦 Step1 跳过预结构化记忆（status="dreamed"），时间轴永远补不上。
    """
    m = re.match(
        r"^(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+(\w+),?\s+(\d{4})$",
        (date_time_str or "").strip(),
    )
    if not m:
        return ""
    hh, mm, ap, day, mon_name, year = m.groups()
    months = {name: i for i, name in enumerate(
        ["January", "February", "March", "April", "May", "June", "July",
         "August", "September", "October", "November", "December"], start=1)}
    mo = months.get(mon_name)
    if not mo:
        return ""
    h = int(hh) % 12 + (12 if ap.lower() == "pm" else 0)
    try:
        return datetime(int(year), mo, int(day), h, int(mm), 0).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return ""


class LoCoMoLoader:
    """将 LoCoMo 对话导入 EMO 记忆系统"""

    def __init__(self, index_store, persistent_store, session_id_prefix: str = "locomo",
                 raw_turns: bool = False):
        self._index = index_store
        self._persist = persistent_store
        self._prefix = session_id_prefix
        self._raw_turns = raw_turns

    def load_conversation(
        self,
        conv_data: dict,
        observation: Optional[dict] = None,
        session_summary: Optional[dict] = None,
        event_summary: Optional[dict] = None,
        time_offset_days: int = 30,
    ) -> int:
        """导入一组完整对话

        Args:
            conv_data: conversation 字段 from locomo10.json
            observation: observation 字段（per-speaker per-session 观察）
            session_summary: session_summary 字段（per-session 段落摘要）
            event_summary: event_summary 字段（per-session 事件描述）
            time_offset_days: 将对话映射到多少天前开始

        Returns:
            导入的记忆条数
        """
        speakers = [conv_data["speaker_a"], conv_data["speaker_b"]]
        session_nums = sorted(
            int(k.split("_")[1])
            for k in conv_data
            if k.startswith("session_") and "date_time" not in k
        )

        now = datetime.now()
        base_date = now - timedelta(days=time_offset_days)
        total_sessions = len(session_nums)

        index_entries: List[IndexEntry] = []
        memory_files: List[MemoryFile] = []
        embedding_docs: List[str] = []

        for sess_idx, sess_num in enumerate(session_nums):
            session_key = f"session_{sess_num}"
            date_time_key = f"session_{sess_num}_date_time"
            date_time_str = conv_data.get(date_time_key, "")
            dialogs = conv_data.get(session_key, [])

            # 时间映射
            days_offset = (sess_idx / max(total_sessions - 1, 1)) * time_offset_days
            session_time = base_date + timedelta(days=days_offset)
            session_ts = session_time.strftime("%Y-%m-%d %H:%M:%S")
            # 真实事件时间（LoCoMo 自带日期）→ event_timestamp，时间轴可见
            event_ts = _parse_locomo_datetime(date_time_str)

            # ── 提取 session 级元数据 ──
            obs_key = f"session_{sess_num}_observation"
            sum_key = f"session_{sess_num}_summary"
            evt_key = f"events_session_{sess_num}"

            session_obs = (observation or {}).get(obs_key, {})
            session_sum = (session_summary or {}).get(sum_key, "")
            session_events = (event_summary or {}).get(evt_key, {})

            # 收集该 session 的所有 observations（用于 embedding 上下文）
            all_session_obs = []
            for spk in speakers:
                spk_obs = session_obs.get(spk, [])
                all_session_obs.extend([o[0] if isinstance(o, list) else o for o in spk_obs])

            session_obs_text = " ".join(all_session_obs[:5])  # 最多取 5 条避免太长

            # ── Session 级记忆（高层摘要）──
            if session_sum:
                sess_mem_id = f"{self._prefix}_session_{sess_num}"
                sess_summary = f"({date_time_str}) {session_sum}"
                sess_embedding = (
                    f"{date_time_str}. Session summary: {session_sum} "
                    f"Key observations: {session_obs_text}"
                )

                sess_entry = IndexEntry(
                    mem_id=sess_mem_id,
                    summary=sess_summary,
                    embedding_text=sess_embedding,
                    category="对话",
                    sub_category="session_summary",
                    tags=[f"session_{sess_num}"] + speakers,
                    source="locomo",
                    speaker=",".join(speakers),
                    original_date=date_time_str,
                    event_timestamp=event_ts,
                    base_weight=0.7,  # session 级记忆稍高权重
                    created_at=session_ts,
                    last_access=session_ts,
                )
                index_entries.append(sess_entry)
                embedding_docs.append(sess_embedding)

                # Session 级 MemoryFile
                sess_facts = json.dumps(all_session_obs, ensure_ascii=False)
                sess_mf = MemoryFile(
                    mem_id=sess_mem_id,
                    raw_content=f"SESSION: {sess_num}\nDATE: {date_time_str}\nSUMMARY: {session_sum}",
                    summary=sess_summary,
                    speaker=",".join(speakers),
                    category="对话",
                    sub_category="session_summary",
                    tags=",".join([f"session_{sess_num}"] + speakers),
                    source="locomo",
                    fact_entries=sess_facts,
                    session_summary=session_sum,
                    original_date=date_time_str,
                    timestamp=session_ts,
                    event_timestamp=event_ts,
                    session_id=f"session_{sess_num}",
                    importance=0.7,
                    status="dreamed",  # 预结构化，同 turn 级
                )
                memory_files.append(sess_mf)

            # ── Turn 级记忆 ──
            # 为该 turn 查找对应的 observation
            turn_obs_map = {}  # dia_id -> list of observations
            for spk in speakers:
                for obs_item in session_obs.get(spk, []):
                    if isinstance(obs_item, list) and len(obs_item) >= 2:
                        obs_text, obs_dia_id = obs_item[0], obs_item[1]
                        # dia_id 可能是列表：一条 observation 跨多个 turn
                        # （如 [...'Home Alone'..., ['D8:24','D8:26','D8:28']]），逐个挂载
                        dia_ids = obs_dia_id if isinstance(obs_dia_id, list) else [obs_dia_id]
                        for did in dia_ids:
                            turn_obs_map.setdefault(did, []).append(obs_text)

            # 为该 turn 查找对应的 event description
            turn_events = {}
            for spk in speakers:
                events = session_events.get(spk, [])
                if events:
                    # event_summary 不按 dia_id 分，整个 session 共享
                    turn_events[spk] = events

            for turn_idx, dialog in enumerate(dialogs):
                speaker = dialog["speaker"]
                text = dialog["text"]
                dia_id = dialog.get("dia_id", f"D{sess_num}:{turn_idx + 1}")

                mem_id = f"{self._prefix}_{dia_id}"

                # ── 获取该 turn 的 observations ──
                turn_observations = turn_obs_map.get(dia_id, [])

                # ── 生成 summary（语义摘要）──
                if turn_observations:
                    # 用 observation 作为 summary（已经是提炼过的事实）
                    summary = f"({date_time_str}) {'; '.join(turn_observations)}"
                else:
                    # 没有 observation 时用原文截断
                    summary = f"({date_time_str}) {speaker}: {text[:120]}"

                # ── 生成 embedding_text（用于向量检索，包含丰富上下文）──
                embedding_parts = [f"{date_time_str}. {speaker} said: \"{text}\""]
                if session_sum:
                    embedding_parts.append(f"Session context: {session_sum[:200]}")
                if turn_observations:
                    embedding_parts.append(f"Key facts: {'; '.join(turn_observations)}")
                if all_session_obs:
                    # 加入 session 级的 observation 上下文
                    embedding_parts.append(f"Session observations: {session_obs_text[:200]}")

                embedding_text = ". ".join(embedding_parts)

                # ── fact_entries（事实条目）──
                fact_list = turn_observations.copy()
                if not fact_list:
                    # 没有 observation 时，从原文提取简单事实
                    fact_list = [f"{speaker} said: {text[:100]}"]

                # ── rel_entries（关系条目）──
                other_speaker = speakers[1] if speaker == speakers[0] else speakers[0]
                rel_list = [f"{speaker} shared this with {other_speaker}"]

                # ── event_description ──
                speaker_events = turn_events.get(speaker, [])
                event_desc = "; ".join(speaker_events[:2]) if speaker_events else ""

                # ── sub_category（简单规则分类）──
                sub_cat = self._classify_turn(text, turn_observations)

                # ── IndexEntry ──
                entry = IndexEntry(
                    mem_id=mem_id,
                    summary=summary,
                    embedding_text=embedding_text,
                    category="对话",
                    sub_category=sub_cat,
                    tags=[speaker, dia_id, f"session_{sess_num}"],
                    source="locomo",
                    speaker=speaker,
                    original_date=date_time_str,
                    event_timestamp=event_ts,
                    base_weight=0.5,
                    created_at=session_ts,
                    last_access=session_ts,
                )
                index_entries.append(entry)
                embedding_docs.append(embedding_text)

                # ── MemoryFile ──
                raw_content = (
                    f"DATE: {date_time_str}\n"
                    f"SPEAKER: {speaker}\n"
                    f"DIA_ID: {dia_id}\n"
                    f"TEXT: {text}"
                )

                mf = MemoryFile(
                    mem_id=mem_id,
                    raw_content=raw_content,
                    summary=summary,
                    speaker=speaker,
                    category="对话",
                    sub_category=sub_cat,
                    tags=",".join([speaker, dia_id, f"session_{sess_num}"]),
                    source="locomo",
                    fact_entries=json.dumps(fact_list, ensure_ascii=False),
                    rel_entries=json.dumps(rel_list, ensure_ascii=False),
                    session_summary=session_sum,
                    event_description=event_desc,
                    original_date=date_time_str,
                    timestamp=session_ts,
                    event_timestamp=event_ts,
                    session_id=f"session_{sess_num}",
                    turn_index=turn_idx,
                    importance=0.5,
                    # LoCoMo 导入即预结构化（observation/event_summary 提炼），
                    # 标记 dreamed：做梦 Step 2 可直接建关联，Step 1 不会重复结构化。
                    # raw_turns=True 时 turn 标 raw——用于"LoCoMo 真做梦"实验
                    # （2026-08-14：让 Step1 结构化/supersede 真正跑在原始 turn 上，
                    # 仅影响新建库，不改既有库）
                    status="raw" if self._raw_turns else "dreamed",
                )
                memory_files.append(mf)

        # ── 批量写入 ──
        if index_entries:
            self._index.add(index_entries, documents=embedding_docs)
            self._persist.save(memory_files)

        count = len(index_entries)
        logger.info(
            f"Imported {count} memories ({len(session_nums)} sessions, "
            f"{speakers[0]} & {speakers[1]})"
        )
        return count

    @staticmethod
    def _classify_turn(text: str, observations: list) -> str:
        """简单规则分类 turn 的 sub_category"""
        combined = (text + " " + " ".join(observations)).lower()

        if any(w in combined for w in ["work", "job", "boss", "office", "meeting", "deadline"]):
            return "work"
        if any(w in combined for w in ["family", "mom", "dad", "grandma", "parent", "kids", "children"]):
            return "family"
        if any(w in combined for w in ["love", "date", "crush", "relationship", "boyfriend", "girlfriend"]):
            return "relationship"
        if any(w in combined for w in ["gym", "run", "walk", "exercise", "yoga", "health"]):
            return "health_fitness"
        if any(w in combined for w in ["food", "eat", "restaurant", "cook", "dinner", "lunch"]):
            return "food"
        if any(w in combined for w in ["sad", "happy", "angry", "anxious", "stressed", "excited", "worried"]):
            return "emotional"
        if any(w in combined for w in ["travel", "trip", "vacation", "flight", "hotel"]):
            return "travel"
        return "daily_life"

    @staticmethod
    def get_speaker_names(conv_data: dict) -> Tuple[str, str]:
        return conv_data["speaker_a"], conv_data["speaker_b"]

    @staticmethod
    def get_session_count(conv_data: dict) -> int:
        return sum(
            1 for k in conv_data
            if k.startswith("session_") and "date_time" not in k
        )
