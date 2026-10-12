#!/usr/bin/env python3
"""VAULT2 读侧入口：答题，零机制开关（只有 I/O 参数）。"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vault2lib.answer import main  # noqa: E402

if __name__ == "__main__":
    asyncio.run(main())
