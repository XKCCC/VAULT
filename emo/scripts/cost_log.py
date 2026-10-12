"""API 成本打点：设 EMO_COST_LOG=<jsonl 路径> 后，每次 LLM 调用落一行。

记录字段：ts（unix 秒）、src（调用方标识）、model（响应回传的真实模型名）、
pt/ct（prompt/completion tokens，来自 response.usage）、sec（调用耗时）。
多进程/协程安全：单行 append（O_APPEND）。任何异常都不影响主流程。
"""
import json
import os
import time

_PATH = os.environ.get("EMO_COST_LOG")


def log_cost(src, model, resp, elapsed):
    if not _PATH:
        return
    try:
        u = getattr(resp, "usage", None)
        rec = {
            "ts": round(time.time(), 3),
            "src": src,
            "model": getattr(resp, "model", None) or model,
            "pt": getattr(u, "prompt_tokens", None),
            "ct": getattr(u, "completion_tokens", None),
            "sec": round(elapsed, 3),
        }
        with open(_PATH, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass
