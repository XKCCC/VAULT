"""AML Add/Search HTTP 服务（FastAPI）。

契约要点（agentmemories.ai/api-guide）：
- POST Add：{request_id, messages[{role,content,timestamp?}], user_id, session_id}
  → 200 {success:true, request_id, user_id, session_id}，返回前必须可检索
- POST Search：{query, options?, user_id, top_k}（正式评测 top_k=100）
  → 200 {data:[{id, content, score?, created_at?}]}，按相关性排序
- GET /health 无需鉴权；鉴权支持 Token/Bearer/X-Api-Key（AML_API_KEY 配置，空=公开 smoke 模式）

用法：
  python emo/aml/server.py            # 默认 0.0.0.0:8000
  AML_API_KEY=xxx python emo/aml/server.py --port 8000
"""
import argparse
import sys
from pathlib import Path

EMO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EMO_DIR))
sys.path.insert(0, str(EMO_DIR / "scripts"))

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


def create_app() -> FastAPI:
    cfg = AMLConfig()
    adapter = AMLMemoryAdapter(cfg)
    app = FastAPI(title="VAULT AML Adapter")

    def check_auth(request: Request, x_api_key: str | None):
        if not cfg.api_key:  # 公开 smoke 模式
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
        n = adapter.add(req.user_id, req.session_id, req.messages)
        return {
            "success": True,
            "request_id": req.request_id,
            "user_id": req.user_id,
            "session_id": req.session_id,
            "ingested": n,
        }

    @app.post(cfg.search_path)
    def search(req: SearchRequest, request: Request, x_api_key: str | None = Header(None)):
        check_auth(request, x_api_key)
        data = adapter.search(req.user_id, req.query, req.top_k)
        return {"data": data}

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
