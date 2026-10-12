#!/usr/bin/env python3
"""VAULT2 写侧建库入口（qwen3.7-max 代际，两阶段固定配方，断点可续）。

Phase 1 做梦：Step1 蒸馏 / Step2 关联边+supersede / Step4 event_timestamp；已做梦条目自动跳过。
Phase 2 话题迁移：Pass1 主题发现 → Pass2 归派 → Pass3 簇文档+时间线；已有 topic_cluster 的库跳过。
"""
import argparse
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from vault2lib.constants import EMBED  # noqa: E402

MODEL_LITELLM = "dashscope/qwen3.7-max"   # eval_locomo 走 litellm 命名
MODEL_PLAIN = "qwen3.7-max"               # csr_topic_migrate 走 dashscope 兼容端点


def phase1_dream(out_dir: Path):
    cmd = [sys.executable, str(ROOT / "emo" / "scripts" / "eval_locomo.py"),
           "--model", MODEL_LITELLM, "--prompt-version", "v3", "--rerank", "--graph-expand",
           "--dream", "--dream-steps", "1,2,4", "--dream-batch", "32", "--raw-turns",
           "--limit", "1",
           "--embed_model", EMBED,
           "--chroma-dir", str(out_dir.parent / (out_dir.name + "_chroma")),
           "--sqlite-dir", str(out_dir),
           "--out-file", str(ROOT / "outputs" / f"{out_dir.name}_build_eval.json")]
    print("[vault2] Phase1 做梦:", " ".join(cmd[-8:]), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def phase2_migrate(out_dir: Path):
    for db in sorted(out_dir.glob("*.db")):
        con = sqlite3.connect(str(db))
        try:
            n = con.execute("SELECT COUNT(*) FROM topic_cluster").fetchone()[0]
        except sqlite3.OperationalError:
            n = 0
        con.close()
        if n:
            print(f"[vault2] 跳过已迁移: {db.stem} ({n} 簇)", flush=True)
            continue
        cmd = [sys.executable, str(ROOT / "emo" / "scripts" / "csr_topic_migrate.py"),
               db.stem, "--dir", str(out_dir), "--model", MODEL_PLAIN]
        print("[vault2] Phase2 迁移:", db.stem, flush=True)
        subprocess.run(cmd, check=True, cwd=str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="sqlite 库目录（新建或续跑）")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs").mkdir(exist_ok=True)
    phase1_dream(out)
    phase2_migrate(out)
    print("[vault2] 建库完成:", out, flush=True)


if __name__ == "__main__":
    main()
