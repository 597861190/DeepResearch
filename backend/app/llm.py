"""
共享 LLM 实例模块

提供统一的 LLM 初始化与获取函数，避免各节点重复定义。
"""

import os
from typing import Optional

from langchain_openai import ChatOpenAI

_llm: Optional[ChatOpenAI] = None

# 默认 LLM 配置
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_API_BASE = "https://api.deepseek.com/v1"
DEFAULT_TIMEOUT = 60  # 秒


def _get_llm() -> ChatOpenAI:
    """获取或初始化 LLM 实例（单例模式）。"""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", DEFAULT_MODEL),
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            openai_api_base=os.getenv("LLM_API_BASE", DEFAULT_API_BASE),
            timeout=int(os.getenv("LLM_TIMEOUT", str(DEFAULT_TIMEOUT))),
            max_retries=2,
            request_timeout=int(os.getenv("LLM_TIMEOUT", str(DEFAULT_TIMEOUT))),
        )
    return _llm
