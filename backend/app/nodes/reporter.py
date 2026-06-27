import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm import _get_llm
from app.state import ResearchState
from app.utils.report_qa import evaluate_report

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一名专业的研究报告撰写助手。

## 任务
根据用户的研究问题和收集到的事实信息，撰写一份结构清晰、内容详实的研究报告。

## 输出格式
请直接输出 Markdown 格式的报告，不要包含额外的解释。

## 质量要求
1. 报告应包含标题、摘要、正文（分章节）和结论
2. 正文应充分引用和整合提供的事实信息
3. 语言专业、客观、逻辑清晰
4. 使用 Markdown 格式排版（标题、列表、加粗等）"""

HUMAN_PROMPT_TEMPLATE = """## 研究问题
{query}

## 收集到的事实信息
{facts}"""


async def reporter_node(state: ResearchState) -> dict:
    query = state.get("query", "")
    facts = state.get("facts", [])

    if not query:
        logger.warning("reporter_node: query 为空，跳过报告生成")
        return {"report_md": "# Report\n\nNo query provided."}

    facts_text = "\n".join(
        f"- {f.get('content', '')}" for f in facts
    ) if facts else "暂无收集到事实信息。"

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=HUMAN_PROMPT_TEMPLATE.format(query=query, facts=facts_text)),
    ]

    try:
        response = await _get_llm().ainvoke(messages)
    except Exception as e:
        logger.error("reporter_node: LLM 调用失败", exc_info=e)
        return {"report_md": "# Report\n\nFailed to generate report due to an error."}

    report_md = response.content if response.content else "# Report\n\nFailed to generate report."
    print(f"📄 生成的研究报告（前200字）：{report_md[:200]}")

    scores = evaluate_report(
        query=state["query"],
        report_md=report_md,
        facts=state["facts"],
    )

    if scores["quality_score"] < 5:
        logger.warning("⚠️ 报告质量偏低，建议人工复核")
    return {"report_md": report_md, "qa_score": scores["quality_score"], "qa_suggestions": scores["suggestions"]}
