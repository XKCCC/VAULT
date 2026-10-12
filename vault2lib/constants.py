"""冻结常数与环境派生路径。改任何一个冻结值 = 新实验，不是 vault2。"""
import os
import re
from pathlib import Path

from vault2lib import ROOT

MODELS_DIR = Path(os.environ.get("VAULT2_MODELS_DIR", str(ROOT / "emo" / "models")))
EMBED = str(MODELS_DIR / "bge-m3")
RERANKER = str(MODELS_DIR / "bge-reranker-v2-m3")
DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
READER_MODEL = "qwen3.7-max"
AUX_MODEL = "qwen3.7-max"

N_CLUSTERS = 6        # round-1 话题簇数
TOP_N = 20            # 终池大小
N_FALLBACK = 15       # 稠密兜底条数
RRF_K = 60            # RRF 常数
RRF_CAND = 30         # 原文决胜候选数
GAPS_MAX = 3          # 单题最多诊断缺口数
SUPP_MAX = 2          # 单缺口最多补充条数
TRUNC_DETECT = 240    # 检测器输入截断
TRUNC_DIAG = 200      # 诊断输入截断

DUR_Q_RE = re.compile(r"how (long|many|much)|duration|lapsed|between", re.I)
