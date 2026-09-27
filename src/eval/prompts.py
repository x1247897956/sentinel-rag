"""Prompt 版本集中管理。

prompt_version 会随每次问答落库（runs.prompt_version）——否则指标变化无法归因。
改动此处任何字符串都必须提升版本号，否则等于悄悄改了评测条件。
"""

PROMPT_VERSION = "answer-zh-v1.0"

ANSWER_SYSTEM = (
    "你是安全知识库检索助手。你只能依据下面提供的片段作答，"
    "不得使用片段之外的知识，不得编造事实、编号或版本号。\n"
    "每个结论后面必须附上来源片段的 chunk_id，格式为 [chunk_id]，例如 [CVE-2024-3094#1]。\n"
    "如果给出的片段不足以回答问题，只回答：知识库中没有依据，不能回答该问题。\n"
    "回答用中文，简洁分点，不要复述题目。"
)

ANSWER_USER_TEMPLATE = """问题：{question}

可用片段（共 {n} 段，每段开头方括号里是它的 chunk_id）：
{context}

请依据上述片段作答，并在每个结论后标注 [chunk_id]。"""

REFUSAL_TEXT = "知识库中没有依据，不能回答该问题。"

# ---------------- LLM 判分 ----------------

JUDGE_PROMPT_VERSION = "judge-zh-v1.0"

JUDGE_SYSTEM = (
    "你是严格的事实核查员，只输出 JSON，不输出任何解释文字。"
)

JUDGE_USER_TEMPLATE = """给定一个安全知识库问答的标准答案要点与模型回答，你只做机械比对，不做推理补充。

标准答案要点（必须逐条判断模型回答是否覆盖，覆盖=语义等价即可，不要求字面相同）：
{points}

模型回答：
{answer}

只输出如下 JSON：
{{"covered": [true, false, ...], "uncovered_points": ["未覆盖的要点原文"], "reason": "一句话说明判断依据", "faithful": true}}"""


def build_answer_messages(question: str, context_blocks: list[str]) -> list[dict[str, str]]:
    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "（无可用片段）"
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {
            "role": "user",
            "content": ANSWER_USER_TEMPLATE.format(
                question=question, n=len(context_blocks), context=context
            ),
        },
    ]


def build_judge_messages(points: list[str], answer: str) -> list[dict[str, str]]:
    pts = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(points))
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": JUDGE_USER_TEMPLATE.format(points=pts, answer=answer)},
    ]
