"""时间轴解析 — 把查询中的相对时间表达解析为绝对日期区间（年月日全量）

为什么需要:
  chatbot 会运行多年，"上周末 / last weekend / in July / 去年" 这类表达
  在不同日期指向不同区间，纯语义检索无法区分（几百条 weekend 记忆互相淹没）。
  本模块在查询时把相对时间表达解析为 [start, end) 半开区间
  （"YYYY-MM-DD HH:MM:SS" 字符串，与 MemoryFile.event_timestamp 字典序可比），
  供 SQLite 范围查询，与语义检索结果求并集。

设计原则:
  - 纯规则解析，零 LLM 调用
  - 保守：解析不出明确区间就返回 None（退回纯语义检索）
  - 年份必须显式落地：所有区间都带完整年月日，支持多年运行
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional, Tuple

_FMT = "%Y-%m-%d %H:%M:%S"

_EN_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

_CN_NUM = {
    "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}


def _num_token(s: str) -> Optional[int]:
    """数字 token → int：支持阿拉伯数字、中文数字（一~十）、英文 a/an"""
    s = s.strip().lower()
    if s.isdigit():
        return int(s)
    if s in ("a", "an"):
        return 1
    return _CN_NUM.get(s)


def _day(d: datetime) -> datetime:
    return d.replace(hour=0, minute=0, second=0, microsecond=0)


def _add_months(d: datetime, months: int) -> datetime:
    """月份加减（年界安全，多年运行必须）"""
    total = d.year * 12 + (d.month - 1) + months
    year, month = total // 12, total % 12 + 1
    day = min(d.day, 28)  # 避免 2 月溢出，区间计算只用月初/月末
    return d.replace(year=year, month=month, day=day)


def _month_range(d: datetime) -> Tuple[datetime, datetime]:
    start = d.replace(day=1)
    return start, _add_months(start, 1)


def _year_range(d: datetime) -> Tuple[datetime, datetime]:
    return d.replace(month=1, day=1), d.replace(year=d.year + 1, month=1, day=1)


def _week_monday(d: datetime) -> datetime:
    return _day(d) - timedelta(days=d.weekday())


def _fmt_span(span: Tuple[datetime, datetime]) -> Tuple[str, str]:
    return span[0].strftime(_FMT), span[1].strftime(_FMT)


def resolve_temporal_range(
    query: str, now: Optional[datetime] = None
) -> Optional[Tuple[str, str]]:
    """把查询中的时间表达解析为 [start, end) 日期区间

    Args:
        query: 用户查询（中/英）
        now: 参考时刻（默认当前时间；测试可注入）

    Returns:
        (start, end) "YYYY-MM-DD HH:MM:SS" 半开区间；无时间意图返回 None
    """
    if not query:
        return None
    if now is None:
        now = datetime.now()
    today = _day(now)
    q = query.lower()

    # ── 明确日期字面量（最高优先级）──
    m = re.search(r"(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})日?", query)
    if m:
        y, mo, da = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            d = datetime(y, mo, da)
            return _fmt_span((d, d + timedelta(days=1)))
        except ValueError:
            pass

    m = re.search(
        r"\b(" + "|".join(_EN_MONTHS) + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b",
        q,
    )
    if m:
        mo, da, y = _EN_MONTHS[m.group(1)], int(m.group(2)), int(m.group(3))
        try:
            d = datetime(y, mo, da)
            return _fmt_span((d, d + timedelta(days=1)))
        except ValueError:
            pass

    # ── 亚天粒度：分钟 / 小时（"几分钟前/几小时前"类问题，精确到分钟）──
    # N 分钟前：±5 分钟窗口（点引用，口头表达本身边界模糊）
    m = (re.search(r"([0-9一二两三四五六七八九十]+)\s*分钟前", query)
         or re.search(r"\b(\d+|an?)\s+minutes?\s+ago\b", q))
    if m:
        n = _num_token(m.group(1))
        if n is not None:
            start = now - timedelta(minutes=n + 5)
            end = now - timedelta(minutes=max(n - 5, 0))
            return _fmt_span((start, end))

    if "半小时前" in query or re.search(r"\bhalf\s+an?\s+hour\s+ago\b", q):
        return _fmt_span((now - timedelta(minutes=35), now - timedelta(minutes=25)))

    # N 小时前：±30 分钟窗口
    m = (re.search(r"([0-9一二两三四五六七八九十]+)\s*个?小时前", query)
         or re.search(r"\b(\d+|an?)\s+hours?\s+ago\b", q))
    if m:
        n = _num_token(m.group(1))
        if n is not None:
            start = now - timedelta(minutes=n * 60 + 30)
            end = now - timedelta(minutes=max(n * 60 - 30, 0))
            return _fmt_span((start, end))

    # 模糊亚天
    if "几小时前" in query or re.search(r"\ba\s+few\s+hours\s+ago\b", q):
        return _fmt_span((now - timedelta(hours=6), now - timedelta(hours=1)))
    if "几分钟前" in query or re.search(r"\ba\s+few\s+minutes\s+ago\b", q):
        return _fmt_span((now - timedelta(minutes=15), now))
    if "刚才" in query or "刚刚" in query or re.search(r"\bjust\s+now\b", q):
        return _fmt_span((now - timedelta(minutes=10), now))

    # ── N 天/周/月/年前 ──
    m = re.search(r"(\d+)\s*(?:天|日)前", query) or re.search(r"\b(\d+)\s+days?\s+ago\b", q)
    if m:
        d = today - timedelta(days=int(m.group(1)))
        return _fmt_span((d, d + timedelta(days=1)))

    m = re.search(r"(\d+)\s*(?:周|礼拜|星期)前", query) or re.search(r"\b(\d+)\s+weeks?\s+ago\b", q)
    if m:
        monday = _week_monday(today - timedelta(weeks=int(m.group(1))))
        return _fmt_span((monday, monday + timedelta(days=7)))

    m = re.search(r"(\d+)\s*个?月前", query) or re.search(r"\b(\d+)\s+months?\s+ago\b", q)
    if m:
        return _fmt_span(_month_range(_add_months(today, -int(m.group(1)))))

    m = re.search(r"(\d+)\s*年前", query) or re.search(r"\b(\d+)\s+years?\s+ago\b", q)
    if m:
        y = today.year - int(m.group(1))
        return _fmt_span((datetime(y, 1, 1), datetime(y + 1, 1, 1)))

    # ── 日粒度相对词 ──
    if "大前天" in query:
        return _fmt_span((today - timedelta(days=3), today - timedelta(days=2)))
    if "前天" in query or "the day before yesterday" in q:
        return _fmt_span((today - timedelta(days=2), today - timedelta(days=1)))
    if "昨天" in query or re.search(r"\byesterday\b", q):
        return _fmt_span((today - timedelta(days=1), today))
    if "今天" in query or re.search(r"\btoday\b", q):
        return _fmt_span((today, today + timedelta(days=1)))
    if "明天" in query or re.search(r"\btomorrow\b", q):
        return _fmt_span((today + timedelta(days=1), today + timedelta(days=2)))
    if "后天" in query:
        return _fmt_span((today + timedelta(days=2), today + timedelta(days=3)))

    # ── 周 / 周末 ──
    monday = _week_monday(today)
    if "上周末" in query or "上个周末" in query or re.search(r"\blast\s+weekend\b", q):
        return _fmt_span((monday - timedelta(days=2), monday))
    if "下周末" in query or "下个周末" in query or re.search(r"\bnext\s+weekend\b", q):
        return _fmt_span((monday + timedelta(days=12), monday + timedelta(days=14)))
    if "这周末" in query or "这个周末" in query or re.search(r"\bthis\s+weekend\b", q):
        return _fmt_span((monday + timedelta(days=5), monday + timedelta(days=7)))
    if re.search(r"上(一?个)?(周|礼拜|星期)", query) or re.search(r"\blast\s+week\b", q):
        return _fmt_span((monday - timedelta(days=7), monday))
    if re.search(r"下(一?个)?(周|礼拜|星期)", query) or re.search(r"\bnext\s+week\b", q):
        return _fmt_span((monday + timedelta(days=7), monday + timedelta(days=14)))
    if re.search(r"(这|本)(一?个)?(周|礼拜|星期)", query) or re.search(r"\bthis\s+week\b", q):
        return _fmt_span((monday, monday + timedelta(days=7)))

    # ── 月 ──
    if "上个月" in query or re.search(r"\blast\s+month\b", q):
        return _fmt_span(_month_range(_add_months(today, -1)))
    if "下个月" in query or re.search(r"\bnext\s+month\b", q):
        return _fmt_span(_month_range(_add_months(today, 1)))
    if "这个月" in query or "本月" in query or re.search(r"\bthis\s+month\b", q):
        return _fmt_span(_month_range(today))

    # 具体月份：YYYY 年 M 月 / in <Month> [YYYY]
    m = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月", query)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        try:
            return _fmt_span(_month_range(datetime(y, mo, 1)))
        except ValueError:
            pass
    m = re.search(r"\bin\s+(" + "|".join(_EN_MONTHS) + r")(?:\s+(\d{4}))?\b", q)
    if m:
        mo = _EN_MONTHS[m.group(1)]
        y = int(m.group(2)) if m.group(2) else (today.year if mo <= today.month else today.year - 1)
        try:
            return _fmt_span(_month_range(datetime(y, mo, 1)))
        except ValueError:
            pass
    m = re.search(r"(?<!\d)(\d{1,2})\s*月份?", query)  # "7月"（无年份，取最近一次）
    if m and not re.search(r"\d+\s*个?月前", query):
        mo = int(m.group(1))
        if 1 <= mo <= 12:
            y = today.year if mo <= today.month else today.year - 1
            return _fmt_span(_month_range(datetime(y, mo, 1)))

    # ── 年 ──
    if "前年" in query:
        return _fmt_span((datetime(today.year - 2, 1, 1), datetime(today.year - 1, 1, 1)))
    if "去年" in query or re.search(r"\blast\s+year\b", q):
        return _fmt_span((datetime(today.year - 1, 1, 1), datetime(today.year, 1, 1)))
    if "明年" in query or re.search(r"\bnext\s+year\b", q):
        return _fmt_span((datetime(today.year + 1, 1, 1), datetime(today.year + 2, 1, 1)))
    if "今年" in query or re.search(r"\bthis\s+year\b", q):
        return _fmt_span((datetime(today.year, 1, 1), datetime(today.year + 1, 1, 1)))
    m = re.search(r"\bin\s+(\d{4})\b", q) or re.search(r"(\d{4})\s*年", query)
    if m:
        y = int(m.group(1))
        if 1900 < y < 2100:
            return _fmt_span((datetime(y, 1, 1), datetime(y + 1, 1, 1)))

    # ── 模糊近期 ──
    if "最近" in query or "这几天" in query or re.search(r"\b(recently|lately|these days)\b", q):
        return _fmt_span((today - timedelta(days=14), today + timedelta(days=1)))
    if "前几天" in query or re.search(r"\bthe other day\b", q):
        return _fmt_span((today - timedelta(days=7), today))

    return None


def weekday_name(date_str: str) -> str:
    """"YYYY-MM-DD" 或 "YYYY-MM-DD HH:MM:SS" → 星期几（英文）"""
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d")
        return d.strftime("%A")
    except (ValueError, TypeError):
        return ""
