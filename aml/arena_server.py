#!/usr/bin/env python3
"""MemoryArena 记忆服务端（ZexueHe/MemoryArena 的 client 协议）。

三端点（memory/client.py MemoryClient 契约）：
  POST /memory/initialize   {user_id, memory_system_name} → 建 per-user 库
  POST /memory/add          {user_id, memory_system_name, chunk} → 同步可检索
  POST /memory/wrap_user_prompt {user_id, memory_system_name, question} → {"prompt": 记忆上下文+任务}
底层复用 AMLMemoryAdapter（per-user 库 + CSR 排序 Search + 后台做梦/迁移）。
arena 的 add 无 session 字段 → 每 user 单 session（"arena"），session_merge 按 turn 序拼段仍有效。

用法：
  python aml/arena_server.py --port 8000
config 里 memory.base_url 指向本服务、memory_system_name 任意（本服务忽略，固定 VAULT2）。
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from aml.adapter import AMLMemoryAdapter  # noqa: E402
from aml.config import AMLConfig  # noqa: E402

SESSION = "arena"
WRAP_TOP_K = 20


class InitReq(BaseModel):
    user_id: str
    memory_system_name: str = "vault2"


class AddReq(BaseModel):
    user_id: str
    memory_system_name: str = "vault2"
    chunk: str


class WrapReq(BaseModel):
    user_id: str
    memory_system_name: str = "vault2"
    question: str


def create_app() -> FastAPI:
    adapter = AMLMemoryAdapter(AMLConfig())
    app = FastAPI(title="VAULT2 MemoryArena memory server")

    @app.post("/memory/initialize")
    def initialize(req: InitReq):
        adapter._get_stores(req.user_id)  # noqa: SLF001  建库即初始化
        return {"status": "ok", "user_id": req.user_id}

    @app.post("/memory/add")
    def add(req: AddReq):
        n = adapter.add(req.user_id, SESSION, [{"role": "user", "content": req.chunk}])
        return {"status": "ok", "ingested": n}

    @app.post("/memory/wrap_user_prompt")
    def wrap(req: WrapReq):
        hits = adapter.search(req.user_id, req.question, WRAP_TOP_K)
        if hits:
            ctx = "\n".join(f"- {h['content']}" for h in hits)
            prompt = (f"Memories from previous sessions (use only what is relevant):\n{ctx}\n\n"
                      f"Current task:\n{req.question}")
        else:
            prompt = req.question
        return {"prompt": prompt}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


def main():
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
