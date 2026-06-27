# app/nodes/refine.py
from app.state import ResearchState
from app.utils.report_qa import evaluate_report
from app.llm import _get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------- Prompt ----------
REFINE_PROMPT = ChatPromptTemplate.from_template("""
你是一位专业的研究报告编辑。

你的任务是根据用户给出的“质量建议”，对报告进行**局部优化与重写**。

【用户原始问题】
{query}

【质量评分与建议】
- 总分：{score}
- 建议：
{suggestions}

【已有事实（仅参考，不要编造）】
{facts}

【当前报告（待优化）】
{report}

请按以下要求修改：
1. 必须逐条回应“质量建议”
2. 必须使用已有事实，不得编造新数据
3. 保持 Markdown 结构（## 标题）
4. 不要改变整体结论，只优化表达和结构

请直接输出优化后的完整报告：
""")

# ---------- Node ----------
async def refine_node(state: ResearchState) -> dict:
    """
    Refine 节点：基于 QA 建议，对报告做局部重写
    """
    llm = _get_llm()
    chain = REFINE_PROMPT | llm | StrOutputParser()

    # 1. 准备 Facts（只给摘要，避免太长）
    facts_summary = "\n".join(
        f"- {f.get('task', '')}: {f.get('content', '')[:120]}..."
        for f in state.get("facts", [])
    )

    # 2. 调用 LLM 重写
    refined_report = await chain.ainvoke({
        "query": state.get("query"),
        "score": state.get("qa_score", 0),
        "suggestions": "\n".join(state.get("qa_suggestions", [])),
        "facts": facts_summary or "无可用事实",
        "report": state.get("report_md", "")
    })

    # 3. 再次 QA 评分
    qa_result = evaluate_report(
        query=state["query"],
        report_md=refined_report,
        facts=state["facts"]
    )

    return {
        "report_md": refined_report,
        "qa_score": qa_result["quality_score"],
        "qa_suggestions": qa_result["suggestions"],
        "refine_count": state.get("refine_count", 0) + 1,
        "next": "reflect"  # 回到 reflect 再判断一次
    }