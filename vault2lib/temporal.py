"""duration 题的确定性代码参考值（TReMu 沙盒）：只作参考注入，不替代 reader。"""
from tremu_temporal import CODE_GEN_PROMPT, extract_code, safe_exec

from vault2lib.constants import READER_MODEL
from vault2lib.llm import call_llm


async def duration_ref(client, sem, question, lines):
    numbered = "\n".join(f"{i+1}. {l[:220]}" for i, l in enumerate(lines))
    code_text = await call_llm(client, sem, READER_MODEL,
                               CODE_GEN_PROMPT.format(question=question, lines=numbered),
                               "tremu_code", max_tokens=512)
    code = extract_code(code_text)
    result, err = safe_exec(code)
    if err:
        code_text = await call_llm(
            client, sem, READER_MODEL,
            f"The following code failed with error: {err}\n\n{code}\n\n"
            f"Fix it. Output ONLY the corrected code.\n\nQuestion: {question}",
            "tremu_code_fix", max_tokens=512)
        code = extract_code(code_text)
        result, err = safe_exec(code)
    if result and result.strip():
        return (f"Reference: deterministic date computation for this question gives "
                f"{result.strip()}.\n")
    return ""
