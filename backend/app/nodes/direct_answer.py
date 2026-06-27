# app/nodes/direct_answer.py
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.llm import _get_llm
from app.state import ResearchState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
你是一个友好的AI助手。请根据用户输入给出简洁、友好的回复。
如果是打招呼或闲聊，请礼貌回应。
如果是简单查询，请直接回答。
回复要简洁，不超过50字。
"""


async def direct_answer_node(state: ResearchState) -> dict:
    user_input = state.get("query", "")
    logger.info("💬 直接回复模式，用户输入=%s", user_input)

    llm = _get_llm()

    try:
        response = await llm.ainvoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_input),
        ])
        answer_text = response.content if response.content else ""
        logger.info("✅ 直接回复内容: %s", answer_text[:100])
    except Exception as e:
        logger.error("❌ 直接回复失败: %s", e)
        answer_text = f"抱歉，我暂时无法回复您的问题。错误: {e}"
        response = AIMessage(content=answer_text)

    return {
        "messages": [response],
        "next": "end",
    }
