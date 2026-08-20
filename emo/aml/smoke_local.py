"""本地端到端自测（不烧 AML 的每小时 1 次 smoke 配额）：

  python emo/aml/smoke_local.py

流程：TestClient 起服务 → /add 写入一段小对话 → /search 验证可检索 →
提示做梦队列行为。通过后再去 AML 平台跑真 smoke。
"""
import sys
import time
from pathlib import Path

EMO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EMO_DIR))
sys.path.insert(0, str(EMO_DIR / "scripts"))

from fastapi.testclient import TestClient  # noqa: E402

from aml.server import create_app  # noqa: E402

MESSAGES = [
    {"role": "user", "content": "我 2023 年 5 月 8 号开始学钢琴了，老师姓王。"},
    {"role": "assistant", "content": "太好了！每周练几次？"},
    {"role": "user", "content": "每周三次，周二周四周六晚上。"},
    {"role": "user", "content": "对了，我最喜欢吃的是云南菜，特别是汽锅鸡。"},
]


def main():
    client = TestClient(create_app())
    uid = "local-smoke-user"

    r = client.get("/health")
    assert r.status_code == 200, f"health 失败: {r.status_code}"
    print("✓ /health")

    r = client.post(
        "/add",
        json={
            "request_id": "smoke-1",
            "user_id": uid,
            "session_id": "s1",
            "messages": MESSAGES,
        },
    )
    body = r.json()
    assert r.status_code == 200 and body.get("success") is True, f"add 失败: {body}"
    assert body["request_id"] == "smoke-1" and body["user_id"] == uid
    print(f"✓ /add 写入 {body.get('ingested')} 条（契约字段一致）")

    for q, expect in [("我什么时候开始学钢琴的", "钢琴"), ("我喜欢吃什么菜", "汽锅鸡")]:
        r = client.post("/search", json={"query": q, "user_id": uid, "top_k": 5})
        data = r.json().get("data", [])
        assert r.status_code == 200 and isinstance(data, list), f"search 失败: {r.text[:200]}"
        hit = any(expect in d["content"] for d in data)
        assert all(d.get("id") and d.get("content") for d in data), "data 项缺 id/content"
        print(f"{'✓' if hit else '✗'} /search {q!r} → {len(data)} 条, 命中期望内容: {hit}")
        assert hit, "期望内容未命中"

    print("✓ 本地自测通过（raw 立即可检索已验证）")
    print("提示：做梦队列在写入静默 ~15s 后自动触发；正式评测前确认做梦日志无异常。")


if __name__ == "__main__":
    main()
