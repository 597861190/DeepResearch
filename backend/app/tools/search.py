# app/tools/search.py
"""
网络搜索模块。

职责：
    封装 Tavily API 搜索逻辑，对外暴露 search_web()。
    通过 app.utils.cache 自动缓存结果。

数据类：
    Fact — 结构化的搜索返回结果，含 content/source/valid 等信息。
"""

from dataclasses import dataclass, field
import logging
import os

from tavily import TavilyClient

from app.utils.cache import search_cache

from app.retrieval.vector_store import upsert_fact

logger = logging.getLogger(__name__)

# ── 数据类 ────────────────────────────────────────────────────


@dataclass
class Fact:
    """一次搜索返回的结构化事实。

    Attributes:
        task:    搜索关键词（query）
        content: 搜索到的正文内容（多段文本拼接）
        source:  数据来源，默认 "tavily"
        valid:   是否有效（低质量 / 失败时为 False）
    """
    task: str
    content: str = ""
    source: str = "tavily"
    valid: bool = True


# ── 低质量判断 ────────────────────────────────────────────────


def is_low_quality_fact(content: str) -> bool:
    """判断搜索结果的文本内容是否低质量。

    参数:
        content: 待判断的文本（Fact.content）

    返回:
        True = 低质量（空 / 太短 / 含失败关键词）
        False = 高质量，可以采纳
    """
    if not content:
        return True

    text = content.strip()
    if len(text) < 50:
        return True

    fail_keywords = ("no relevant", "抱歉", "无法找到", "搜索失败")
    if any(kw in text.lower() for kw in fail_keywords):
        return True

    return False


# ── 实际搜索逻辑（被 cache 包裹） ──────────────────────────────


def _real_search(query: str) -> dict:
    """真实的 Tavily 搜索（没有缓存）。

    返回 dict 而非 Fact，因为 lru_cache 缓存 dict 更安全（不可变）。
    """
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    try:
        response = client.search(query=query, max_results=3)
        raw_results = response if isinstance(response, list) else response.get("results", [])

        if not raw_results:
            logger.warning("Tavily 返回结果为空（query=%s）", query)
            return {"task": query, "content": "", "source": "tavily", "valid": False}

        # 取每个结果的正文内容（content 字段），拼接为长文本
        contents = []
        for item in raw_results:
            if isinstance(item, dict):
                title = item.get("title", "")
                body = item.get("content", "")
                # 有正文就用 "标题: 正文" 格式，没正文就用标题
                if body:
                    contents.append(f"{title}: {body}" if title else body)
                elif title:
                    contents.append(title)

        full_text = "\n\n".join(contents) if contents else ""

        if is_low_quality_fact(full_text):
            return {"task": query, "content": "", "source": "tavily", "valid": False}
        
        upsert_fact(task=query, content=full_text, source="tavily")

        return {
            "task": query,
            "content": full_text,
            "source": "tavily",
            "valid": True,
        }

    except Exception as e:
        logger.error("Tavily 搜索异常（query=%s）: %s", query, e)
        return {"task": query, "content": f"搜索失败: {e}", "source": "tavily", "valid": False}


# ── 对外接口（自动走缓存） ──────────────────────────────────────


def search_web(query: str) -> dict:
    """
    对外暴露的统一搜索接口（自动走缓存）。

    相同 query 只会调用一次 Tavily，后续命中 lru_cache。

    返回:
        dict，含 task / content / source / valid 四个 key。
    """
    return search_cache(query)
