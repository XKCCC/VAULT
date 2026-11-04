#!/usr/bin/env python3
"""AML Add/Search HTTP 服务（FastAPI）。

用法：
  AML_API_KEY=<token> python aml/server.py --port 8102
鉴权：Token/Bearer/X-Api-Key 三式；AML_API_KEY 留空=公开 smoke 模式（仅限联调）。
"""
import argparse
import json
import os
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, Header, HTTPException, Request  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from aml.adapter import AMLMemoryAdapter  # noqa: E402
from aml.config import AMLConfig  # noqa: E402


class AddRequest(BaseModel):
    request_id: str
    messages: list
    user_id: str
    session_id: str


class SearchRequest(BaseModel):
    query: str
    user_id: str
    top_k: int = 10
    options: list | None = None


_SEARCH_SEM = threading.Semaphore(int(os.environ.get("AML_SEARCH_CONC", "4")))

def create_app() -> FastAPI:
    cfg = AMLConfig()
    adapter = AMLMemoryAdapter(cfg)
    app = FastAPI(title="VAULT2 AML Adapter")

    def check_auth(request: Request, x_api_key: str | None):
        if not cfg.api_key:
            return
        auth = request.headers.get("authorization", "")
        token = auth.removeprefix("Bearer ").removeprefix("Token ").strip()
        if token != cfg.api_key and x_api_key != cfg.api_key:
            raise HTTPException(status_code=401, detail="invalid memory system key")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post(cfg.add_path)
    def add(req: AddRequest, request: Request, x_api_key: str | None = Header(None)):
        check_auth(request, x_api_key)
        n = adapter.add(req.user_id, req.session_id, req.messages, request_id=req.request_id)
        resp = {"success": True, "request_id": req.request_id,
                "user_id": req.user_id, "session_id": req.session_id}
        if n < 0:
            resp["dedup"] = True
        return resp

    @app.post(cfg.search_path)
    def search(req: SearchRequest, request: Request, x_api_key: str | None = Header(None)):
        check_auth(request, x_api_key)
        _SEARCH_SEM.acquire()
        try:
            if os.environ.get("AML_STREAM_HEADERS", "1") == "1":
                from fastapi.responses import StreamingResponse

                def gen():
                    yield json.dumps({"data": adapter.search(req.user_id, req.query, req.top_k)},
                                     ensure_ascii=False).encode("utf-8")

                return StreamingResponse(gen(), media_type="application/json")
            return {"data": adapter.search(req.user_id, req.query, req.top_k)}
        finally:
            _SEARCH_SEM.release()

    if os.environ.get("AML_DEBUG_SLOWTEST") == "1":
        @app.get("/slowtest")
        async def slowtest(request: Request, sec: int = 60,
                           x_api_key: str | None = Header(None)):
            check_auth(request, x_api_key)
            import asyncio as _aio
            from fastapi.responses import StreamingResponse

            async def gen():
                await _aio.sleep(sec)
                yield b'{"ok":true}'
            return StreamingResponse(gen(), media_type="application/json")

    return app


def main():
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8102)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    import logging as _lg
    _lg.basicConfig(level=_lg.INFO)
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
