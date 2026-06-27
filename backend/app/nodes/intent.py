# app/nodes/intent.py
import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm import _get_llm
from app.schema import IntentType
from app.state import ResearchState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
你是一个意图识别专家。请分析用户输入，将其分类为以下之一：
- greeting: 打招呼、感谢、闲聊。
- simple_query: 简单信息查询（如"今天几号"、"北京天气"）。
- deep_research: 复杂研究分析（如"分析光伏行业"、"对比竞品"）。
- ambiguous: 意图不明，需要用户澄清。

请只返回 JSON 格式：{"intent": "deep_research"}
"""


async def intent_node(state: ResearchState) -> dict:
    user_input = state.get("query", "")
    logger.info("🔍 意图识别开始，用户输入=%s", user_input)

    llm = _get_llm()

    try:
        response = await llm.ainvoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_input),
        ])
        result = json.loads(response.content)
        intent_str = result.get("intent", IntentType.AMBIGUOUS.value)
        intent = next(
            (it for it in IntentType if it.value == intent_str),
            IntentType.AMBIGUOUS,
        )
    except Exception as e:
        logger.warning("⚠️ 意图识别失败，默认 ambiguous，error=%s", e)
        intent = IntentType.AMBIGUOUS

    logger.info("🎯 意图识别结果: %s", intent.value)
    return {
        "intent": intent.value,
        "messages": [HumanMessage(content=user_input)],
    }
