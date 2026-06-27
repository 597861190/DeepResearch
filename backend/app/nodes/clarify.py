# app/nodes/clarify.py
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm import _get_llm
from app.state import ResearchState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
你是一个意图澄清专家。用户的问题意图不明确，请礼貌地请用户提供更多信息或澄清他们的需求。
回复要简洁、友好，引导用户更清楚地表达问题。
不超过80字。
"""


async def clarify_node(state: ResearchState) -> dict:
    user_input = state.get("query", "")
    logger.info("❓ 意图不明确，进入澄清模式，用户输入=%s", user_input)

    llm = _get_llm()

    try:
        response = await llm.ainvoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_input),
        ])
        logger.info("✅ 澄清回复: %s", response.content[:100] if response.content else "")
    except Exception as e:
        logger.error("❌ 澄清回复失败: %s", e)
        response = None

    return {
        "messages": [response] if response else [],
        "next": "end",
    }
