"""LifeBench data loader — 将 LifeBench（LoCoMo 格式）数据导入 EMO 记忆系统

数据源：LifeBench-memory/life_bench_data/locomo_format/our_en.json（10 个虚拟用户，
每人约 364+ 个日级 session、1.5 万条手机痕迹 turn，总长 ~3.66M token）。

与 LoCoMo 的差异：
  - session 日期是 ISO 格式 "2023-10-01"（LoCoMo 是 "1:56 pm on 8 May, 2023"）
  - turn 的 dia_id 形如 "2025-03-12_push566"（手机痕迹条目：sms/call/photo/push/
    calendar/note/agent_chat/fitness_health）
  - observation / event_summary / session_summary 大多为空 → 只有 turn 级 + 可选
    session 级记忆；证据（QA 的 evidence 字段）按 dia_id 后缀映射统计召回
  - 规模约 30 倍于 LoCoMo，默认不做梦（Step1+2 是 1 次 LLM 调用/条），
    做梦消融用 --dream 另行触发
  - status="raw"：让 --dream Step1 真正执行结构化+相对时间解析
    （LoCoMo A0 报告遗留的 raw-turn 变体实验）；默认不做梦路径不受影响
    （index_store.search 不按 status 过滤）

Usage:
    from memory.lifebench_loader import LifeBenchLoader
    loader = LifeBenchLoader(index_store, persistent_store, session_id_prefix=uid)
    loader.load_conversation(sample["conversation"])
"""

import logging
import re
from datetime import datetime
from typing import List, Optional, Tuple

from memory.schema import IndexEntry, MemoryFile

logger = logging.getLogger(__name__)


def parse_lifebench_date(date_str: str) -> str:
    """"2023-10-01" / "2023-10-01 13:30:00" → "2023-10-01 00:00:00"（失败返回 ""）"""
    s = (date_str or "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    return ""


def evidence_to_dia_suffix(evidence: str) -> str:
    """QA evidence（"call9"）与 dia_id（"2025-05-19_call1465" / "..._agent_chat231_1"）
    的映射口径：dia_id 去掉日期前缀后，与 evidence 完全相等，或去掉尾部子序号
    （_N）后相等。locomo_format 转换时过滤/拆分了部分条目（约 19% evidence 无
    对应 dia_id），这部分在召回统计中从分母剔除。
    """
    return evidence.strip()


class LifeBenchLoader:
    """将一个 LifeBench 用户（一年手机痕迹）导入 EMO 记忆系统"""

    def __init__(self, index_store, persistent_store, session_id_prefix: str = "life"):
        self._index = index_store
        self._persist = persistent_store
        self._prefix = session_id_prefix

    def load_conversation(self, conv_data: dict) -> int:
        speakers = [conv_data["speaker_a"], conv_data["speaker_b"]]
        session_nums = sorted(
            int(k.split("_")[1])
            for k in conv_data
            if k.startswith("session_") and "date_time" not in k
        )

        index_entries: List[IndexEntry] = []
        memory_files: List[MemoryFile] = []
        embedding_docs: List[str] = []

        for sess_num in session_nums:
            session_key = f"session_{sess_num}"
            date_str = conv_data.get(f"session_{sess_num}_date_time", "")
            dialogs = conv_data.get(session_key, [])

            event_ts = parse_lifebench_date(date_str)
            session_ts = event_ts or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # session 级摘要（本数据集基本为空；有才建 session 级记忆）
            sess_sum = conv_data.get(f"session_{sess_num}_summary", "")
            if sess_sum:
                sess_mem_id = f"{self._prefix}_session_{sess_num}"
                summary = f"({date_str}) {sess_sum}"
                index_entries.append(IndexEntry(
                    mem_id=sess_mem_id,
                    summary=summary,
                    embedding_text=f"{date_str}. Session summary: {sess_sum}",
                    category="对话",
                    sub_category="session_summary",
                    tags=[f"session_{sess_num}"] + speakers,
                    source="lifebench",
                    speaker=",".join(speakers),
                    original_date=date_str,
                    event_timestamp=event_ts,
                    base_weight=0.7,
                    created_at=session_ts,
                    last_access=session_ts,
                ))
                embedding_docs.append(f"{date_str}. Session summary: {sess_sum}")
                memory_files.append(MemoryFile(
                    mem_id=sess_mem_id,
                    raw_content=f"SESSION: {sess_num}\nDATE: {date_str}\nSUMMARY: {sess_sum}",
                    summary=summary,
                    speaker=",".join(speakers),
                    category="对话",
                    sub_category="session_summary",
                    tags=",".join([f"session_{sess_num}"] + speakers),
                    source="lifebench",
                    original_date=date_str,
                    timestamp=session_ts,
                    event_timestamp=event_ts,
                    session_id=f"session_{sess_num}",
                    importance=0.7,
                    status="raw",
                ))

            for turn_idx, dialog in enumerate(dialogs):
                speaker = dialog["speaker"]
                text = dialog["text"]
                dia_id = dialog.get("dia_id", f"{date_str}_t{turn_idx}")
                # 痕迹类型（sms/call/photo/...），从 dia_id 提取做 sub_category
                m = re.match(r"^\d{4}-\d{2}-\d{2}_([a-z_]+?)\d", dia_id)
                trace_type = m.group(1) if m else "trace"

                mem_id = f"{self._prefix}_{dia_id}"
                summary = f"({date_str}) {speaker}: {text[:200]}"
                embedding_text = f"{date_str}. {speaker}: \"{text}\""

                tags = [speaker, dia_id, f"session_{sess_num}", trace_type]
                index_entries.append(IndexEntry(
                    mem_id=mem_id,
                    summary=summary,
                    embedding_text=embedding_text,
                    category="对话",
                    sub_category=trace_type,
                    tags=tags,
                    source="lifebench",
                    speaker=speaker,
                    original_date=date_str,
                    event_timestamp=event_ts,
                    base_weight=0.5,
                    created_at=session_ts,
                    last_access=session_ts,
                ))
                embedding_docs.append(embedding_text)

                memory_files.append(MemoryFile(
                    mem_id=mem_id,
                    raw_content=(
                        f"DATE: {date_str}\n"
                        f"SPEAKER: {speaker}\n"
                        f"DIA_ID: {dia_id}\n"
                        f"TEXT: {text}"
                    ),
                    summary=summary,
                    speaker=speaker,
                    category="对话",
                    sub_category=trace_type,
                    tags=",".join(tags),
                    source="lifebench",
                    fact_entries="[]",
                    original_date=date_str,
                    timestamp=session_ts,
                    event_timestamp=event_ts,
                    session_id=f"session_{sess_num}",
                    turn_index=turn_idx,
                    importance=0.5,
                    status="raw",
                ))

        if index_entries:
            self._index.add(index_entries, documents=embedding_docs)
            self._persist.save(memory_files)

        logger.info(
            f"Imported {len(index_entries)} memories "
            f"({len(session_nums)} sessions) for {self._prefix}"
        )
        return len(index_entries)

    @staticmethod
    def get_last_session_date(conv_data: dict) -> str:
        """该用户最后一个 session 的日期——作为评测时的时间锚点 "now" """
        session_nums = [
            int(k.split("_")[1])
            for k in conv_data
            if k.startswith("session_") and "date_time" not in k
        ]
        if not session_nums:
            return ""
        return conv_data.get(f"session_{max(session_nums)}_date_time", "")

    @staticmethod
    def match_evidence(evidence: List[str], dia_ids: set) -> Tuple[dict, set]:
        """evidence → dia_id 映射

        Returns:
            (mapped, unmapped)：mapped = {evidence: {对应 dia_id, ...}}
            （agent_chat 拆条后一个 evidence 可对应多个 dia_id，命中其一即算召回）；
            unmapped = 无法映射的 evidence 集合（应从召回分母剔除）。
        """
        # 预建后缀索引：trace 基名（去日期前缀、去尾部 _N）→ dia_ids
        base_map = {}
        for d in dia_ids:
            m = re.match(r"^\d{4}-\d{2}-\d{2}_(.+)$", d)
            if not m:
                continue
            base = m.group(1)
            base_map.setdefault(base, set()).add(d)
            stripped = re.sub(r"_\d+$", "", base)
            if stripped != base:
                base_map.setdefault(stripped, set()).add(d)

        mapped, unmapped = {}, set()
        for ev in evidence:
            ev = evidence_to_dia_suffix(ev)
            if ev in dia_ids:
                mapped[ev] = {ev}
            elif ev in base_map:
                mapped[ev] = base_map[ev]
            else:
                unmapped.add(ev)
        return mapped, unmapped
