import json
import logging
import re
from typing import Union

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm import _get_llm
from app.state import ResearchState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一名专业的研究规划助手。

## 任务
根据用户提出的研究问题，生成一份具体、可执行的研究计划，将问题拆解为一系列独立的研究任务。

## 输出格式
请严格按以下 JSON 格式输出，不要包含任何额外的解释或标记：
{"plan": ["任务1的描述", "任务2的描述", "任务3的描述"]}

## 质量要求
1. 每个任务应具体明确，能够独立进行搜索或调研
2. 任务之间按逻辑递进排列，从基础信息到深入分析
3. 任务数量控制在 3~6 个
4. 除 JSON 外不要输出任何其他内容

## 示例
用户：我想了解 AI Agent 的发展现状
输出：{"plan": ["搜索 AI Agent 的定义和发展历史", "梳理主流 AI Agent 框架及其对比", "调研 AI Agent 在各行业的应用案例", "分析 AI Agent 当前的技术瓶颈与挑战", "总结 AI Agent 的未来发展趋势"]}"""

HUMAN_PROMPT_TEMPLATE = "用户的研究问题是：{query}"

# 如果向量库中有相关历史知识，追加到 prompt 中
HUMAN_PROMPT_WITH_CONTEXT = "用户的研究问题是：{query}\n\n## 本地知识库中的相关信息\n{context}"

MAX_CONTEXT_CHARS = 2000

MAX_PLAN_STEPS = 6


def _extract_text(content: Union[str, list, None]) -> Union[str, None]:
    """从 LLM 响应 content 中提取文本。"""
    if not content:
        return None

    if isinstance(content, str):
        return content

    # LangChain 多模态模式下 content 是 list[dict]
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p.strip() for p in parts if p.strip()) or None

    return None


def _parse_plan(text: str) -> list[str]:
    """将 LLM 返回的文本解析为计划列表，支持多种格式。

    自动处理以下情况:
    - 裸 JSON：{"plan": [...]} 或直接 [...]
    - markdown 代码块（带/不带语言标记）：```json ... ```
    - 代码块前后有额外文字说明
    """
    raw = text
    text = text.strip()

    # ── 步骤1：尝试从 markdown 代码块中提取 JSON ──
    # 匹配 ```json ... ``` 或 ``` ... ```，允许前后有说明文字
    code_block = re.search(
        r"```(?:json)?\s*\n(.*?)\n\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if code_block:
        text = code_block.group(1).strip()

    # ── 步骤2：JSON 解析 ──
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("⚠️ JSON 解析失败，原始回复=%r", raw[:300])
        return []

    # ── 步骤3：提取列表 ──
    # 情况 A：{"plan": [...]} 或 {"tasks": [...]} 等
    if isinstance(data, dict):
        # 优先查找常见的计划字段名
        for key in ("plan", "tasks", "steps", "items", "list"):
            val = data.get(key)
            if isinstance(val, list) and len(val) > 0:
                return _clean_plan_items(val)
        # 兜底：取第一个非空列表字段
        for key, val in data.items():
            if isinstance(val, list) and len(val) > 0:
                logger.info("_parse_plan: 使用备用字段 '%s' 作为计划", key)
                return _clean_plan_items(val)
        logger.warning("_parse_plan: dict 中未找到有效的列表字段，data=%r", data)
        return []

    # 情况 B：直接就是数组 ["任务1", "任务2", ...]
    if isinstance(data, list):
        return _clean_plan_items(data)

    logger.warning("_parse_plan: 未知的 JSON 结构，type=%s", type(data).__name__)
    return []


def _clean_plan_items(items: list) -> list[str]:
    """清洗计划列表：确保每一项都是非空字符串。"""
    cleaned = [str(item).strip() for item in items if item and str(item).strip()]
    return cleaned


async def planner_node(state: ResearchState) -> dict:
    query = state.get("query", "")
    private_ctx = state.get("private_context", "")
    if not query:
        logger.warning("planner_node: query 为空，跳过规划")
        return {"plan": [], "current_task_idx": 0}

    # 如果私有上下文非空，带入 prompt 辅助规划
    if private_ctx:
        truncated_ctx = private_ctx[:MAX_CONTEXT_CHARS]
        if len(private_ctx) > MAX_CONTEXT_CHARS:
            truncated_ctx += "\n\n...（内容过长已截断）"
        human_prompt = HUMAN_PROMPT_WITH_CONTEXT.format(
            query=query,
            context=truncated_ctx,
        )
        logger.info("📖 已携带 %d 字符本地知识给规划器", len(truncated_ctx))
    else:
        human_prompt = HUMAN_PROMPT_TEMPLATE.format(query=query)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_prompt),
    ]

    try:
        response = await _get_llm().ainvoke(messages)
    except Exception as e:
        logger.error("planner_node: LLM 调用失败", exc_info=e)
        return {"plan": [], "current_task_idx": 0}

    text = _extract_text(response.content)
    if text is None:
        logger.warning("planner_node: LLM 返回内容为空")
        return {"plan": [], "current_task_idx": 0}

    plan = _parse_plan(text)[:MAX_PLAN_STEPS]
    if not plan:
        logger.warning("planner_node: 解析后的 plan 为空列表")
        logger.debug("planner_node: LLM 原始回复=\n%s", text)  # 保留完整回复供排查
        return {"plan": [], "current_task_idx": 0}
    print(f"📝 生成的研究计划：{plan}")
    return {"plan": plan, "current_task_idx": 0}
