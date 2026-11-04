# VAULT2

VAULT headline 记忆架构的**冻结独立实现**（2026-10-12 定稿）。零外部代码依赖、零实验开关；
三个用途：同门跑实验 / 论文开源 / AML（Agent Memory Leaderboard）Add·Search 对接层（`aml/`，接入中）。

研究消融不在此 repo——开关坟场在母项目 `emo/scripts/csr_ab.py`；本 repo 只含实测验证的机制。

## 架构

**写侧（离线，qwen3.7-max）**：原文逐字落库 → 做梦 Step1 结构化蒸馏（创建即冻结）→
Step2 关联边 + supersede 版本治理 → Step4 event_timestamp（相对日期确定性解算，写侧冻结）→
话题迁移 Pass1-3（簇文档 = 叙事 + 时间线；时间线日期权威 = event_timestamp）。

**读侧（在线，`vault2lib/`）**：
1. CSR 两级检索：话题文档 top6 选簇 → 簇内成员 + 稠密兜底 25 → bge-reranker top30 →
   簇内原文决胜 RRF(k=60) → top20；
2. session 合并（同 session 命中按原 turn 序拼段，修复跨 turn 组装）；
3. 答题前判缺补充（diagnose → 双路混合召回 RRF → 单缺口选 ≤2，总缺口 ≤3）；
4. 证据条件检测（INFER_DETECT 看召回池、零 benchmark 标签）：
   temporal → 时间线不注入 + duration 题注入 TReMu 确定性代码参考值；infer → 推理 add-on；
5. 统一答题 prompt（机理级八条款、零-shot 无示例）→ reader qwen3.7-max。

## 目录

```
run_build.py        写侧建库入口（两阶段固定配方，断点可续）
run_answer.py       读侧答题入口（产出 refined 兼容 predictions）
vault2lib/          读侧流水线：constants / prompts / llm / retrieval / temporal / answer
emo/memory/         记忆栈快照（做梦流水线 / 存储层 / 话题迁移依赖）
emo/scripts/        建库链路脚本快照（eval_locomo / csr_topic_migrate / bench_utils / …）
emo/models/         权重目录（不进 git，见内层 README；VAULT2_MODELS_DIR 可改址）
locomo/data/        LoCoMo 原始对话（locomo10.json，公开数据集）
tools/              本地裁判服务配方（Qwen3-14B 关思考）
```

## 快速开始

```bash
pip install -r requirements.txt          # torch 按本机 CUDA 装（验证版 2.6.0+cu124）
export DASHSCOPE_API_KEY=...             # 读写两侧 LLM 均走 dashscope 兼容端点
# 权重：见 emo/models/README.md（bge-m3 / bge-reranker-v2-m3；Qwen3-14B 仅判分用）

python run_build.py --out /path/to/lib           # 建库（qwen3.7-max 代际）
python run_answer.py --db /path/to/lib \
    --questions questions.jsonl --out preds.jsonl [--concurrency 8]
```

判分（LoCoMo refined 官方可比尺）：Qwen3-14B 权重 + 官方 judge prompt，**必须关思考**
（vLLM 默认开思考会把尺收紧约 5 分）；本地服务配方 `tools/serve_judge_qwen3_14b.sh`。
协议只认权重身份，云端/本地部署均可。

可选环境变量：`VAULT2_MODELS_DIR`（权重根）、`EMO_COST_LOG=<jsonl>`（成本打点）。

一键复现（建库 → 答题 → 官方可比尺判分，断点可续；题集 `locomo/data/refined_questions.jsonl` 随 repo）：

```bash
bash tools/repro_locomo.sh            # 默认库 libs/locomo_repro
```

## 冻结常数（改任何一个 = 新实验，不是 vault2）

| 常数 | 值 | | 常数 | 值 |
|---|---|---|---|---|
| round-1 簇数 | 6 | | 判缺缺口上限 | 3 |
| 终池 top_n | 20 | | 单缺口补充上限 | 2 |
| 稠密兜底 | 25 | | 检测截断 | 240 字符 |
| RRF k | 60 | | 诊断截断 | 200 字符 |
| 原文决胜候选 | 30 | | reader/aux 模型 | qwen3.7-max |

变更记录：2026-10-15 稠密兜底 15→25（消融采纳：同尺 +1.01、翻 51/37、阶梯单调；topk 维持 20、combo 不采纳）；headline 更新至该常数同尺 80.32。2026-11-04 AML 对接层上生产版（幂等 Add / 契约对齐 / 追赶调度 / 检索缓存）。

## 休眠区（代码在快照里保留，入口不启用）

实体注册表与全局保送（cat1 -1.4）、chain 串联聚合（-0.22）、标签路由三分支 prompt
（被证据条件统一答题取代，+2.24）、cat1-contract / 引用先行 CoT / 计算锚注入（均证伪）、
temporal 题时间线注入（悖论：对统一 prompt 是负债）、图遍历 / L3 / 演化 / HyDE / MMR。

## headline 成绩（LoCoMo refined 官方可比尺）

- LoCoMo refined：**80.32**（cat1 62.4 / cat2 82.9 / cat3 76.5 / cat4 84.4，n=1382 同尺）；
  vs dense 锚 v2.7 75.83：+4.49；vs MemoraX SOTA 82.65：-2.33
