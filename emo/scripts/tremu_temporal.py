#!/usr/bin/env python3
"""TReMu 正版读侧臂（2026-10-08）：LLM 按题现写 Python → 沙盒执行 → LLM 组装答案。

与错版（固定 resolver 出绝对日期）的区别：代码是中间推理依据，答案由 LLM 用自然语言
组装（保 gold 相对短语面）。组件：gen code（qwen3.7-max）→ safe_exec（受限命名空间，
datetime/dateutil+weekRange 辅助函数）→ compose（qwen3.7-max）。
"""
import datetime as _dt
import re
import sys
from datetime import date, timedelta


def weekRange(t):
    """t 所在自然周（周一~周日）。t: date 或可解析字符串。"""
    if isinstance(t, str):
        from dateutil import parser as _dp

        t = _dp.parse(t, fuzzy=True).date()
    mon = t - timedelta(days=t.weekday())
    return mon, mon + timedelta(days=6)


def lastWeekday(d, wd):
    """最近一个周 wd（0=周一…4=周五，不含 d 当日）。"""
    delta = (d.weekday() - wd) % 7 or 7
    return d - timedelta(days=delta)


def nextWeekday(d, wd):
    delta = (wd - d.weekday()) % 7 or 7
    return d + timedelta(days=delta)


def monthRange(y, m):
    s = date(y, m, 1)
    e = date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1)
    return s, e


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    allowed = {"datetime", "dateutil", "dateutil.relativedelta", "dateutil.parser",
               "math", "re", "calendar"}
    if name in allowed or any(name.startswith(a + ".") for a in allowed):
        return __import__(name, globals, locals, fromlist, level)
    raise ImportError(f"module {name} not allowed")


_SAFE_GLOBALS = {
    "__builtins__": {
        "print": print, "range": range, "len": len, "str": str, "int": int, "float": float,
        "bool": bool, "list": list, "dict": dict, "set": set, "tuple": tuple, "sorted": sorted,
        "min": min, "max": max, "sum": sum, "abs": abs, "round": round, "enumerate": enumerate,
        "zip": zip, "isinstance": isinstance, "repr": repr, "format": format,
        "__import__": _safe_import,
    },
    "date": date, "timedelta": timedelta, "datetime": _dt,
    "weekRange": weekRange, "lastWeekday": lastWeekday,
    "nextWeekday": nextWeekday, "monthRange": monthRange,
}


def safe_exec(code: str):
    """受限执行 LLM 生成的日期计算代码，返回 stdout（失败返回 None 与错误）。"""
    from contextlib import redirect_stdout
    from io import StringIO

    from dateutil import parser, relativedelta

    g = dict(_SAFE_GLOBALS)
    g["parser"] = parser
    g["relativedelta"] = relativedelta.relativedelta
    buf = StringIO()
    try:
        with redirect_stdout(buf):
            exec(code, g)  # noqa: S102
        return buf.getvalue().strip(), None
    except Exception as e:  # noqa: BLE001
        return buf.getvalue().strip(), f"{type(e).__name__}: {e}"


CODE_GEN_PROMPT = """You are solving a temporal reasoning question about a past conversation.
Write Python code to compute the answer, using the dates in the retrieved memories.

Available in the namespace (already imported, use directly):
- datetime, date, timedelta
- parser.parse(s, fuzzy=True)  # parse memory date strings like "9:55 am on 22 October, 2023"
- relativedelta  # dateutil.relativedelta (e.g. relativedelta(weeks=1), relativedelta(weekday=FR))
- weekRange(t)  # -> (monday, sunday) of t's week; t may be a date or date string
- monthRange(y, m)  # -> (first, last) of month
- lastWeekday(d, wd) / nextWeekday(d, wd)  # wd: 0=Mon..4=Fri..6=Sun

Rules:
- Parse each memory's parenthesized date with parser.parse(..., fuzzy=True).date() if needed.
- Compute step by step with clear variable names.
- Finally print() the computed answer value(s) (a date, a range, or a duration).
Output ONLY the code, no markdown fences, no explanations.

Question: {question}

Retrieved memories:
{lines}

Code:"""

COMPOSE_PROMPT = """Based on the retrieved memories and the computation result below, answer the temporal question concisely.

ANSWER FORM RULES (critical — the grader requires matching the answer's time granularity and forbids converting between relative and absolute time):
- Answer in RELATIVE-ANCHORED or period form, mirroring how conversational memories describe time. Do NOT answer with a bare absolute date when the event spans a period.
- Match the event's natural granularity:
  • specific day → "20 October 2023" or "the Friday before 22 October 2023"
  • week → "the week before 9 June 2023" or "the week of 19–25 June 2023" or "From 19 June 2023 to 25 June 2023"
  • month → "June 2023"; year → "2022"
  • weekend(s) → "two weekends before 17 July 2023"
- When the memory expresses the event with a relative phrase (e.g. "last Friday", "two weeks ago"), answer in the SAME relative-anchored style: "the {{period}} before {{the memory's own date}}" — e.g. memory dated 9 June says "last week" → answer "the week before 9 June 2023".
- For "how long" questions: answer with the duration itself (e.g. "10 years", "about 3 months", "4 years").
- If the memories do not contain the answer, say the information is not mentioned.

Question: {question}

Computation result (from executed code):
{result}

Answer:"""


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*([\s\S]*?)```", text)
    return (m.group(1) if m else text).strip()


async def tremu_answer(client, call_llm, sem, question, lines, log_tag=""):
    """TReMu 三步：gen code → exec → compose。返回 (answer, dbg)。
    exec 空/错时 answer 返回 "__FALLBACK__"（调用方回退常规答题，防过度拒答）。"""
    numbered = "\n".join(f"{i+1}. {l}" for i, l in enumerate(lines))
    code_text = await call_llm(
        CODE_GEN_PROMPT.format(question=question, lines=numbered), f"{log_tag}code", 512)
    code = extract_code(code_text)
    result, err = safe_exec(code)
    if err:  # 执行失败重生成一次（论文同策略）
        retry_prompt = (f"The following code failed with error: {err}\n\n{code}\n\n"
                        f"Fix it. Output ONLY the corrected code.\n\nQuestion: {question}")
        code_text = await call_llm(retry_prompt, f"{log_tag}code_fix", 512)
        code = extract_code(code_text)
        result, err = safe_exec(code)
    result_str = (result or "") + (f"  [error: {err}]" if err else "")
    dbg = {"code": code[:400], "exec": result_str[:200], "err": err}
    if not result.strip():  # 执行无产出 → 回退常规答题（v2 防过度拒答）
        return "__FALLBACK__", dbg
    answer = await call_llm(
        COMPOSE_PROMPT.format(question=question, result=result_str),
        f"{log_tag}compose", 128)
    return answer.strip(), dbg


if __name__ == "__main__":
    demo = """from datetime import date
anchor = parser.parse("9:55 am on 22 October, 2023", fuzzy=True).date()
passed = lastWeekday(anchor, 4)
print(passed)"""
    out, err = safe_exec(demo)
    print(f"demo exec: out={out} err={err}")
    assert out == "2023-10-20", out
    demo2 = """start = parser.parse("12 March 2020").date()
end = start + timedelta(weeks=1) - timedelta(days=1)
print(start, end)"""
    out, err = safe_exec(demo2)
    print(f"demo2 exec: out={out} err={err}")
    assert "2020-03-12 2020-03-18" in out, out
    print("safe_exec 自测通过")
