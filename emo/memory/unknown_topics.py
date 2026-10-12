"""UnknownTopics — 未知话题队列

在线时: 分类器预测置信度低于阈值 → 话题加入队列 → 虚拟人走"不知道"路径
做梦时: 取出队列中的话题 → 用 Skill 搜索互联网/知识库 → 生成新记忆 → 拓展知识
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class UnknownTopics:
    """未知话题队列管理器"""

    def __init__(self, filepath: str = "emo/memory/models/unknown_topics.json"):
        self._path = Path(filepath)
        self._topics: List[Dict] = []
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            with open(self._path, "r", encoding="utf-8") as f:
                self._topics = json.load(f)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._topics, f, indent=2, ensure_ascii=False)

    def add(self, query: str, label1_top: str = "", label2_top: str = "",
            label1_conf: float = 0.0, label2_conf: float = 0.0) -> None:
        """记录一个未知话题"""
        # 去重（相似 query 不重复添加）
        for existing in self._topics:
            if existing["query"] == query:
                existing["count"] += 1
                existing["last_seen"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self._save()
                return

        entry = {
            "query": query,
            "label1_top": label1_top,
            "label1_conf": round(label1_conf, 3),
            "label2_top": label2_top,
            "label2_conf": round(label2_conf, 3),
            "count": 1,
            "first_seen": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_seen": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "resolved": False,
        }
        self._topics.append(entry)
        self._save()
        logger.info(f"📝 Unknown topic recorded: '{query}'")

    def get_unresolved(self) -> List[Dict]:
        """获取所有未解决的未知话题"""
        return [t for t in self._topics if not t.get("resolved")]

    def mark_resolved(self, query: str) -> None:
        """标记某个话题为已解决（做梦拓展知识后调用）"""
        for topic in self._topics:
            if topic["query"] == query:
                topic["resolved"] = True
                topic["resolved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save()

    def get_all(self) -> List[Dict]:
        return self._topics.copy()

    def count_unresolved(self) -> int:
        return len(self.get_unresolved())

    def stats(self) -> dict:
        total = len(self._topics)
        unresolved = self.count_unresolved()
        # 按被问次数排序的 top 5
        top = sorted(self._topics, key=lambda x: x["count"], reverse=True)[:5]
        return {
            "total": total,
            "unresolved": unresolved,
            "top_unknown": [(t["query"], t["count"]) for t in top],
        }
