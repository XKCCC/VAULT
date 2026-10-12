#!/usr/bin/env python3
"""temporal_resolver.py — TReMu 式确定性相对日期解析器。

职责切分：LLM 只负责"认出相对时间表达式"（语言理解），本模块负责"那到底是哪天"
（确定性计算，全库同一把尺）。

API:
    resolve(mention: date, expr: str) -> Resolved | None
        Resolved = ("point", date) | ("range", (start, end))
        约定：
        - last Friday = 最近一个周五（不含 mention 当日）
        - last week   = 上一个完整自然周（周一~周日）
        - this X      = 本自然周内的 X
        - next X      = last X 之后的下一个

覆盖：last/next/this + weekday/week/month/year、N units ago、N units later/before/after、
yesterday/tomorrow/day before/after、in <Month> <Year>、bare year、
中文：昨天/前天/明天/后天/大后天/上周/这周/下周/上个月/下个月/去年/明年/上周五/下周一/N天前/N周前/N个月前/N年前。
"""
from __future__ import annotations

import re
from datetime import date, timedelta

WD = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
      "friday": 4, "saturday": 5, "sunday": 6}
CN_WD = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
       "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
       "一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6,
       "七": 7, "八": 8, "九": 9, "十": 10}
CN_UNIT = {"天": "days", "日": "days", "周": "weeks", "星期": "weeks",
           "个月": "months", "月": "months", "年": "years"}
EN_UNIT = {"day": "days", "week": "weeks", "month": "months", "year": "years"}
MONTHS = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
          "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
          "december": 12}


def _last_wd(d: date, wd: int) -> date:
    delta = (d.weekday() - wd) % 7 or 7
    return d - timedelta(days=delta)


def _next_wd(d: date, wd: int) -> date:
    delta = (wd - d.weekday()) % 7 or 7
    return d + timedelta(days=delta)


def _this_wd(d: date, wd: int) -> date:
    mon = d - timedelta(days=d.weekday())
    return mon + timedelta(days=wd)


def _week_of(d: date) -> tuple[date, date]:
    mon = d - timedelta(days=d.weekday())
    return mon, mon + timedelta(days=6)


def _month_of(y: int, m: int) -> tuple[date, date]:
    s = date(y, m, 1)
    e = date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1)
    return s, e


def _shift(d: date, n: int, unit: str, sign: int):
    if unit == "days":
        return "point", d + sign * timedelta(days=n)
    if unit == "weeks":
        return "point", d + sign * timedelta(weeks=n)
    if unit == "months":
        m = d.month - 1 + sign * n
        y = d.year + m // 12
        m = m % 12 + 1
        day = min(d.day, [31, 29 if y % 4 == 0 and (y % 100 != 0 or y % 400 == 0) else 28,
                          31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1])
        return "point", date(y, m, day)
    if unit == "years":
        try:
            return "point", d.replace(year=d.year + sign * n)
        except ValueError:  # 2/29
            return "point", d.replace(year=d.year + sign * n, day=28)
    return None


def resolve(mention: date, expr: str):
    """相对表达 → 绝对日期。返回 ("point", date) 或 ("range", (s, e))；无法解析 None。"""
    if not expr:
        return None
    e = " ".join(expr.strip().lower().replace("，", " ").split())
    e = re.sub(r"^(the|a|an)\s+", "", e)

    # ── 中文 ──
    cn = {"昨天": -1, "前天": -2, "大前天": -3, "明天": 1, "后天": 2, "大后天": 3,
          "今天": 0, "今日": 0}
    if e in cn:
        return "point", mention + timedelta(days=cn[e])
    if e in ("上周", "上星期", "上个星期", "上一周"):
        return "range", _week_of(mention - timedelta(days=7))
    if e in ("这周", "本周", "这周", "这星期", "本星期"):
        return "range", _week_of(mention)
    if e in ("下周", "下星期", "下个星期", "下一周"):
        return "range", _week_of(mention + timedelta(days=7))
    if e in ("上个月", "上月"):
        return "range", _month_of(mention.year + (mention.month == 1),
                                  12 if mention.month == 1 else mention.month - 1)
    if e in ("这个月", "本月"):
        return "range", _month_of(mention.year, mention.month)
    if e in ("下个月", "下月"):
        return "range", _month_of(mention.year + (mention.month == 12),
                                  1 if mention.month == 12 else mention.month + 1)
    if e == "去年":
        return "range", (date(mention.year - 1, 1, 1), date(mention.year - 1, 12, 31))
    if e == "今年":
        return "range", (date(mention.year, 1, 1), date(mention.year, 12, 31))
    if e == "明年":
        return "range", (date(mention.year + 1, 1, 1), date(mention.year + 1, 12, 31))
    m = re.fullmatch(r"(上|下|这)周([一二三四五六日天])", e)
    if m:
        wd = CN_WD[m.group(2)]
        return "point", {"上": _last_wd, "下": _next_wd, "这": _this_wd}[m.group(1)](mention, wd)
    m = re.fullmatch(r"([一两二三四五六七八九十\d]+)\s*(天|日|周|星期|个月|月|年)(以)?(前|后)", e)
    if m:
        n = NUM.get(m.group(1), None) or int(m.group(1))
        return _shift(mention, n, CN_UNIT[m.group(2)], -1 if m.group(4) == "前" else 1)

    # ── 英文 ──
    if e in ("yesterday",):
        return "point", mention - timedelta(days=1)
    if e in ("tomorrow",):
        return "point", mention + timedelta(days=1)
    if e in ("day before yesterday", "the day before yesterday"):
        return "point", mention - timedelta(days=2)
    if e in ("day after tomorrow", "the day after tomorrow"):
        return "point", mention + timedelta(days=2)
    m = re.fullmatch(r"(last|next|this)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", e)
    if m:
        wd = WD[m.group(2)]
        return "point", {"last": _last_wd, "next": _next_wd, "this": _this_wd}[m.group(1)](mention, wd)
    if e in ("last week", "past week", "previous week"):
        return "range", _week_of(mention - timedelta(days=7))
    if e in ("this week", "current week"):
        return "range", _week_of(mention)
    if e in ("next week",):
        return "range", _week_of(mention + timedelta(days=7))
    if e in ("last month", "past month", "previous month"):
        return "range", _month_of(mention.year + (mention.month == 1),
                                  12 if mention.month == 1 else mention.month - 1)
    if e in ("this month", "current month"):
        return "range", _month_of(mention.year, mention.month)
    if e in ("next month",):
        return "range", _month_of(mention.year + (mention.month == 12),
                                  1 if mention.month == 12 else mention.month + 1)
    if e in ("last year", "past year", "previous year"):
        return "range", (date(mention.year - 1, 1, 1), date(mention.year - 1, 12, 31))
    if e in ("this year", "current year"):
        return "range", (date(mention.year, 1, 1), date(mention.year, 12, 31))
    if e in ("next year",):
        return "range", (date(mention.year + 1, 1, 1), date(mention.year + 1, 12, 31))
    m = re.fullmatch(r"(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
                     r"(day|week|month|year)s?\s+ago", e)
    if m:
        n = NUM.get(m.group(1), None) or int(m.group(1))
        return _shift(mention, n, EN_UNIT[m.group(2)], -1)
    m = re.fullmatch(r"(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
                     r"(day|week|month|year)s?\s+(later|after|before|earlier)", e)
    if m:
        n = NUM.get(m.group(1), None) or int(m.group(1))
        sign = 1 if m.group(3) in ("later", "after") else -1
        return _shift(mention, n, EN_UNIT[m.group(2)], sign)
    m = re.fullmatch(r"(?:in\s+)?(january|february|march|april|may|june|july|august|"
                     r"september|october|november|december)(?:\s+(\d{4}))?", e)
    if m:
        y = int(m.group(2)) if m.group(2) else mention.year
        return "range", _month_of(y, MONTHS[m.group(1)])
    m = re.fullmatch(r"(?:in\s+)?(\d{4})", e)
    if m:
        y = int(m.group(1))
        return "range", (date(y, 1, 1), date(y, 12, 31))
    return None


def fmt(res) -> str:
    """Resolved → gold 风格绝对日期字符串。"""
    if res is None:
        return ""
    kind, v = res
    if kind == "point":
        return f"{v.day} {['January','February','March','April','May','June','July','August','September','October','November','December'][v.month-1]} {v.year}"
    s, e = v
    if s.year == e.year and s.month == e.month:
        return f"From {s.day} {['January','February','March','April','May','June','July','August','September','October','November','December'][s.month-1]} {s.year} to {e.day} {['January','February','March','April','May','June','July','August','September','October','November','December'][e.month-1]} {e.year}"
    f = lambda d: f"{d.day} {['January','February','March','April','May','June','July','August','September','October','November','December'][d.month-1]} {d.year}"
    return f"From {f(s)} to {f(e)}"


if __name__ == "__main__":
    cases = [
        (date(2023, 10, 22), "last Friday", ("point", date(2023, 10, 20))),   # 周日→最近周五
        (date(2023, 10, 20), "last Friday", ("point", date(2023, 10, 13))),   # 周五当日→上周
        (date(2023, 6, 27), "last week", ("range", (date(2023, 6, 19), date(2023, 6, 25)))),
        (date(2023, 5, 24), "a few days before", None),
        (date(2023, 7, 8), "two weeks ago", ("point", date(2023, 6, 24))),
        (date(2023, 7, 8), "2 days later", ("point", date(2023, 7, 10))),
        (date(2023, 3, 4), "6 weeks ago", ("point", date(2023, 1, 21))),
        (date(2023, 8, 22), "上个月", ("range", (date(2023, 7, 1), date(2023, 7, 31)))),
        (date(2023, 8, 22), "上周五", ("point", date(2023, 8, 18))),
        (date(2023, 1, 15), "last year", ("range", (date(2022, 1, 1), date(2022, 12, 31)))),
        (date(2023, 5, 10), "May 2023", ("range", (date(2023, 5, 1), date(2023, 5, 31)))),
        (date(2023, 12, 31), "next month", ("range", (date(2024, 1, 1), date(2024, 1, 31)))),
        (date(2023, 11, 10), "十天前", ("point", date(2023, 10, 31))),
    ]
    bad = 0
    for mention, expr, want in cases:
        got = resolve(mention, expr)
        ok = got == want
        bad += not ok
        print(f"{'✅' if ok else '✗'} {mention} + {expr!r:22s} → {got} (want {want})")
    print(f"\n{len(cases)-bad}/{len(cases)} 通过")
