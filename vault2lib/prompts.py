"""冻结 prompt 族：统一答题（机理级八条款、零-shot 无示例）+ 推理 add-on + 证据条件检测 + 判缺/选补。"""

UNIFIED_ANSWER_PROMPT = (
    "You are answering a question about a past conversation based on the retrieved memories above.\n"
    "Each memory begins with its date in parentheses, e.g. \"(1:56 pm on 8 May, 2021)\".\n\n"
    "Instructions:\n"
    "1. Base your answer ONLY on the memories above. Use exact words from them when possible.\n"
    "2. If memories conflict, the most recent memory wins.\n"
    "3. TIME RULES (apply whenever the answer involves time):\n"
    "   - answer at the same granularity as the memory (day, month, year, week-span, or duration); "
    "never invent finer precision than the memory gives;\n"
    "   - point granularity: if the memory states the time directly, use it as written; if it uses a "
    "relative expression, resolve it against that memory's own date at the same granularity "
    "(\"last year\" in a 2021 memory → \"2020\"; \"last Friday\" in a 14 September 2021 memory → "
    "\"10 September 2021\");\n"
    "   - week/weekend spans: keep the span anchored to the memory's date (\"the week before "
    "21 March 2021\") or give the explicit range; never collapse a span to a single day;\n"
    "   - distinguish when an event happened from when it was discussed, planned, or announced; "
    "answer the time the question asks for;\n"
    "   - make the time value explicit; never answer with a bare relative expression.\n"
    "4. LISTS: include every item that directly answers the question; exclude merely related "
    "background; when two people or objects have different values, keep the ownership.\n"
    "5. COUNTS: count the distinct items or events evidenced and give that exact number; do not "
    "hedge with \"at least\" or \"about\".\n"
    "6. Preserve exact names, numbers, amounts, frequencies, and qualifiers; do not blend similar "
    "events from nearby dates or sessions.\n"
    "7. Answer with a short phrase; no explanation, no full sentences, no opening like \"Based on "
    "the memories\".\n"
    "8. If no memory directly states the answer, give your best answer from the closest relevant "
    "memories; say Unknown only when nothing relates at all.\n\n"
    "{ref}"
    "Question: {question}\n"
    "Answer:"
)

REASON_ADDON = (
    "\nThe memories above do not state the answer directly. First quote the most relevant memory "
    "fact(s) in one line, then reason in 1-3 sentences combining them with common sense. A "
    "well-reasoned guess is better than a refusal. End with a final line exactly \"Answer: X\" "
    "where X is a short phrase.\n"
)

INFER_DETECT_PROMPT = (
    "You are checking what the retrieved memories support for a question.\n"
    "Reply JSON only: {{\"infer\": true|false, \"temporal\": true|false}}\n"
    "- infer=false: some memory directly states the answer (a name, item, date, count, or yes/no).\n"
    "- infer=true: the memories contain related facts but the answer must be combined from several "
    "facts or completed with common sense (likelihood, explanation, or a yes/no not directly stated). "
    "When unsure between the two, answer true.\n"
    "- temporal=true only if the requested answer is a time, duration, or temporal order.\n\n"
    "Question: {question}\n\n"
    "Memories:\n{lines}"
)

DIAG_PROMPT = """你在做记忆问答的证据缺口诊断。给定一个问题和从记忆库检索到的证据条目（编号 1..N）。

任务：
1. 把问题拆成回答它所需的条件（实体+方面+时间要求）。
2. 逐条件判断证据是否已明确建立。只把"证据不足或缺失"的条件列出；全部覆盖输出空列表，宁缺毋滥。
3. 给每个缺口附 aka：该对象在记忆中可能出现的别名/同义说法（≤3 个）。

输出 JSON（只输出 JSON）：
{{"gaps": [{{"entity": "...", "facet": "...", "time": "previous|current|any", "need": "一句话", "aka": ["..."]}}]}}

问题：{question}

证据条目：
{lines}"""

CHAIN_PROMPT = """你在记忆库的相关候选中定位证据。下面是混合检索捞回的候选记忆条目（编号 1..M，按时间排；标注[已被后续更新]表示该陈述已被更新信息取代）。

待补条件：{entity} 的 {facet}（时间要求：{time}）。
问题：{question}

选出能回答该条件的条目编号（最多 2 条）。没有能回答的就输出空列表。
输出 JSON（只输出 JSON）：{{"selected": [编号, ...]}}

条目：
{chain_lines}"""
