# app/nodes/retrieval.py
import logging

from app.retrieval.vector_store import search_facts
from app.state import ResearchState

logger = logging.getLogger(__name__)


async def retrieval_node(state: ResearchState) -> dict:
    """
    从本地向量库中检索与 Query 相关的历史 Facts。
    在 planner 之前执行，为研究规划提供已有知识。
    """
    query = state.get("query", "")
    if not query:
        return {"private_context": ""}

    local_facts = search_facts(query, top_k=3)

    if local_facts:
        logger.info("🔍 检索到 %d 条本地记忆", len(local_facts))
        context = "\n\n".join([f["content"] for f in local_facts])
        return {"private_context": context}

    logger.debug("未检索到与 '%s' 相关的本地记忆", query[:30])
    return {"private_context": ""}
