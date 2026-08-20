# VAULT × AML 适配层

把 VAULT 记忆系统包装成 AML（Agent Memory Leaderboard）协议要求的 Add/Search 服务。

## 协议要点（2026-08-19 核对的 api-guide）

| 约束 | 内容 |
|---|---|
| Add | POST 同步返回，返回前内容必须可检索；`user_id` 是唯一隔离域；>20 条消息/2000 词平台自动分段多次调用 |
| Search | POST；正式评测 `top_k` 固定 **100**；返回 `{data:[{id,content,score?,created_at?}]}` 按相关性排序 |
| 内部模型 | full 评测**强制 gpt-4o-mini**（含我们的做梦 LLM）；开发自测用 qwen-plus |
| 配额 | smoke 每小时 1 次、full 每 3 个月 1 次——先用 `smoke_local.py` 自测，别浪费配额 |
| 鉴权 | Token / Bearer / X-Api-Key 三选一；`AML_API_KEY` 留空=公开 smoke 的 none 模式 |

## 架构

```
AML 平台 ──POST /add──→ 原始条目持久化(status=raw) + 立即建索引（同步，契约要求）
                          └→ 后台做梦队列：结构化+链接+supersede（写入静默 15s 后触发）
AML 平台 ──POST /search─→ bge-m3 检索 + CE 精排 + 图扩展 → 按 mem_id 回取 raw_content 全文
```

- 每 `user_id` 一座独立库（`AML_ROOT/chroma|sqlite/<uid>`），严禁跨域共享
- Search 返回 **raw_content 全文**（全文保真注入是论文核心主张，勿改成摘要）
- 答题与裁判由 AML 平台固定，与适配层无关

## 运行

```bash
# 开发自测（qwen-plus）
export DASHSCOPE_API_KEY=...      # 必需（默认 AML_LLM_KEY_ENV 指向它）
python emo/aml/smoke_local.py     # 本地端到端自测，先跑这个

# 正式配置（gpt-4o-mini，full 前置清单要求）
export AML_LLM_BASE=https://api.naga.ac/v1
export AML_LLM_MODEL=gpt-4o-mini
export AML_LLM_KEY_ENV=OPENAI_API_KEY   # 配合 set -a; source .env.openai_official; set +a
export AML_API_KEY=<我们签发给平台的 Memory System Key>
export AML_DREAM_BATCH=32               # GPU 机；CPU 机器留 8
python emo/aml/server.py --port 8000    # 需要公网可达（生产 HTTPS）
```

## 部署注意

- 服务须在提交后 **30 天内稳定可访问**；GPU 机跑（做梦 + 检索嵌入需要）
- 平台 Add 并发 16–64、Search 并发 16–256；做梦队列可能滞后于写入高峰，
  若评测方反映早期 Search 命中 raw 内容属预期（raw 立即可检索，做梦态随后升级）
- 正式 full 之前先跑 1 次平台 smoke 验证鉴权与格式
