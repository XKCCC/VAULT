"""LLM 客户端与调用封装：temp 0、关思考、重试 4 次、可选成本打点（EMO_COST_LOG）。"""
import asyncio
import json
import os
import re
import time

from openai import AsyncOpenAI

from vault2lib.constants import DASHSCOPE_BASE

_COST_FH = None


def log_cost(stage, model, usage, lat):
    global _COST_FH
    if _COST_FH is None:
        p = os.environ.get("EMO_COST_LOG")
        if not p:
            return
        _COST_FH = open(p, "a")
    _COST_FH.write(json.dumps({"stage": stage, "model": model,
                               "pt": usage.get("prompt_tokens"), "ct": usage.get("completion_tokens"),
                               "lat": round(lat, 2), "ts": time.time()}, ensure_ascii=False) + "\n")
    _COST_FH.flush()


async def call_llm(client, sem, model, prompt, stage, max_tokens=128):
    async with sem:
        for attempt in range(4):
            try:
                t0 = time.time()
                r = await client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": prompt}],
                    temperature=0, max_tokens=max_tokens,
                    extra_body={"enable_thinking": False})
                log_cost(stage, model,
                         r.usage.model_dump() if hasattr(r.usage, "model_dump") else dict(r.usage),
                         time.time() - t0)
                return r.choices[0].message.content.strip()
            except Exception:  # noqa: BLE001
                if attempt == 3:
                    return ""
                await asyncio.sleep(3 * (attempt + 1))


def parse_json_obj(text):
    m = re.search(r"\{[\s\S]*\}", text or "")
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return {}


def make_client():
    return AsyncOpenAI(api_key=os.environ["DASHSCOPE_API_KEY"],
                       base_url=DASHSCOPE_BASE, timeout=180)
