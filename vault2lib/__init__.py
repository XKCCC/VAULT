"""VAULT2 — 冻结 headline 记忆架构的独立 repo（2026-10-12）。

读侧流水线在 vault2lib.*；写侧建库链路为 emo/ 下快照脚本，入口 run_build.py。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "emo"), str(ROOT / "emo" / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
