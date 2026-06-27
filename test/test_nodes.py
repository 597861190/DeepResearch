"""
节点单元测试 — 所有 LLM 调用被 mock，不依赖外部服务。
"""

import json
import sys
import os
from unittest.mock import AsyncMock, patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import pytest
from app.state import ResearchState

# 所有从 app.llm 导入 _get_llm 的模块列表，需要逐个 patch
_LLM_CLIENTS = [
    "app.nodes.intent._get_llm",
    "app.nodes.planner._get_llm",
    "app.nodes.reporter._get_llm",
    "app.nodes.direct_answer._get_llm",
    "app.nodes.clarify._get_llm",
    "app.nodes.refine._get_llm",
]


# ── Fixtures ──────────────────────────────────────────────────────


def make_state(**overrides) -> ResearchState:
    """创建基础 ResearchState，允许覆盖。"""
    defaults = dict(
        query="",
        intent="",
        messages=[],
        plan=[],
        current_task_idx=0,
        facts=[],
        reflect_count=0,
        next="continue",
        report_md="",
        completed_tasks=0,
        failed_tasks=[],
        avg_fact_length=0.0,
        private_context="",
        qa_score=0.0,
        qa_suggestions=[],
        refine_count=0,
    )
    defaults.update(overrides)
    return ResearchState(**defaults)


@pytest.fixture(autouse=True)
def mock_llm():
    """Mock 所有节点的 LLM 调用，每个测试自动生效。"""
    mock = AsyncMock()
    mock.ainvoke = AsyncMock()

    patchers = [patch(ref, return_value=mock) for ref in _LLM_CLIENTS]
    for p in patchers:
        p.start()

    yield mock

    for p in reversed(patchers):
        p.stop()


# ── intent_node ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_intent_node_deep_research(mock_llm):
    from app.nodes.intent import intent_node

    mock_llm.ainvoke.return_value = MagicMock(
        content='{"intent": "deep_research"}'
    )
    result = await intent_node(make_state(query="AI agent 的发展前景"))
    assert result["intent"] == "deep_research"


@pytest.mark.asyncio
async def test_intent_node_greeting(mock_llm):
    from app.nodes.intent import intent_node

    mock_llm.ainvoke.return_value = MagicMock(
        content='{"intent": "greeting"}'
    )
    result = await intent_node(make_state(query="你好"))
    assert result["intent"] == "greeting"


@pytest.mark.asyncio
async def test_intent_node_ambiguous_on_bad_json(mock_llm):
    from app.nodes.intent import intent_node

    mock_llm.ainvoke.return_value = MagicMock(
        content="not json at all"
    )
    result = await intent_node(make_state(query="???"))
    assert result["intent"] == "ambiguous"


@pytest.mark.asyncio
async def test_intent_node_ambiguous_on_llm_error(mock_llm):
    from app.nodes.intent import intent_node

    mock_llm.ainvoke.side_effect = Exception("API timeout")
    result = await intent_node(make_state(query="test"))
    assert result["intent"] == "ambiguous"


# ── reflect_node ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reflect_max_rounds():
    from app.nodes.reflect import reflect_node

    result = await reflect_node(make_state(reflect_count=2))  # +1 = 3
    assert result["next"] == "reporter"


@pytest.mark.asyncio
async def test_reflect_empty_plan():
    from app.nodes.reflect import reflect_node

    result = await reflect_node(make_state(plan=[]))
    assert result["next"] == "reporter"


@pytest.mark.asyncio
async def test_reflect_sufficient_facts():
    from app.nodes.reflect import reflect_node

    result = await reflect_node(make_state(
        plan=["t1", "t2"],
        facts=[{"content": "ok"}, {"content": "ok"}],
    ))
    assert result["next"] == "reporter"


@pytest.mark.asyncio
async def test_reflect_no_report_yet():
    from app.nodes.reflect import reflect_node

    result = await reflect_node(make_state(
        plan=["t1", "t2", "t3"],
        facts=[{"content": "ok"}],
        report_md="",
    ))
    assert result["next"] == "reporter", "应先出报告再 refine"


@pytest.mark.asyncio
async def test_reflect_score_pass():
    from app.nodes.reflect import reflect_node

    result = await reflect_node(make_state(
        plan=["t1", "t2", "t3"],
        facts=[{"content": "ok"}],
        report_md="# Report",
        qa_score=8.0,
    ))
    assert result["next"] == "reporter"


@pytest.mark.asyncio
async def test_reflect_refine_needed():
    from app.nodes.reflect import reflect_node

    result = await reflect_node(make_state(
        plan=["t1", "t2", "t3"],
        facts=[{"content": "ok"}],
        report_md="# Report",
        qa_score=5.0,
        refine_count=0,
    ))
    assert result["next"] == "refine"


@pytest.mark.asyncio
async def test_reflect_refine_exhausted():
    from app.nodes.reflect import reflect_node

    result = await reflect_node(make_state(
        plan=["t1", "t2", "t3"],
        facts=[{"content": "ok"}],
        report_md="# Report",
        qa_score=5.0,
        refine_count=2,
    ))
    assert result["next"] == "planner"


# ── planner_node ──────────────────────────────────────────────────


def _make_llm_response(content: str):
    """创建一个模拟的 LLM 响应。"""
    return MagicMock(content=content)


@pytest.mark.asyncio
async def test_planner_node_normal(mock_llm):
    from app.nodes.planner import planner_node

    plan_json = json.dumps({"plan": ["任务1", "任务2", "任务3"]})
    mock_llm.ainvoke.return_value = _make_llm_response(plan_json)

    result = await planner_node(make_state(query="AI 发展"))
    assert len(result["plan"]) == 3
    assert result["current_task_idx"] == 0


@pytest.mark.asyncio
async def test_planner_node_with_markdown_block(mock_llm):
    from app.nodes.planner import planner_node

    plan_json = '{"plan": ["任务1", "任务2"]}'
    mock_llm.ainvoke.return_value = _make_llm_response(
        f"```json\n{plan_json}\n```"
    )

    result = await planner_node(make_state(query="test"))
    assert len(result["plan"]) == 2


@pytest.mark.asyncio
async def test_planner_node_empty_query(mock_llm):
    from app.nodes.planner import planner_node

    result = await planner_node(make_state(query=""))
    assert result["plan"] == []


@pytest.mark.asyncio
async def test_planner_node_llm_error(mock_llm):
    from app.nodes.planner import planner_node

    mock_llm.ainvoke.side_effect = Exception("LLM error")
    result = await planner_node(make_state(query="test"))
    assert result["plan"] == []


@pytest.mark.asyncio
async def test_planner_node_with_context(mock_llm):
    from app.nodes.planner import planner_node

    mock_llm.ainvoke.return_value = _make_llm_response(
        '{"plan": ["task1"]}'
    )
    result = await planner_node(make_state(
        query="test",
        private_context="已有的研究：AI Agent 是..."
    ))
    assert len(result["plan"]) == 1


# ── reporter_node ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reporter_node_generates_report(mock_llm):
    from app.nodes.reporter import reporter_node

    mock_llm.ainvoke.return_value = _make_llm_response(
        "# 研究报告\n\n## 结论\n...",
    )
    result = await reporter_node(make_state(
        query="AI 发展",
        facts=[{"content": "AI 是..."}],
    ))
    assert "report_md" in result
    assert "qa_score" in result
    assert "qa_suggestions" in result


@pytest.mark.asyncio
async def test_reporter_node_empty_query(mock_llm):
    from app.nodes.reporter import reporter_node

    result = await reporter_node(make_state(query=""))
    assert result["report_md"] == "# Report\n\nNo query provided."


@pytest.mark.asyncio
async def test_reporter_node_llm_error(mock_llm):
    from app.nodes.reporter import reporter_node

    mock_llm.ainvoke.side_effect = Exception("API error")
    result = await reporter_node(make_state(query="test"))
    assert "Failed to generate" in result["report_md"]


# ── retrieval_node ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_retrieval_node_empty_query():
    from app.nodes.retrieval import retrieval_node

    result = await retrieval_node(make_state(query=""))
    assert result["private_context"] == ""


# ── direct_answer_node ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_direct_answer(mock_llm):
    from app.nodes.direct_answer import direct_answer_node

    mock_llm.ainvoke.return_value = _make_llm_response("你好！")
    result = await direct_answer_node(make_state(query="你好"))
    assert "messages" in result
    assert result["next"] == "end"


@pytest.mark.asyncio
async def test_direct_answer_error(mock_llm):
    from app.nodes.direct_answer import direct_answer_node

    mock_llm.ainvoke.side_effect = Exception("error")
    result = await direct_answer_node(make_state(query="hi"))
    assert result["next"] == "end"
    assert len(result["messages"]) > 0


# ── clarify_node ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_clarify_node(mock_llm):
    from app.nodes.clarify import clarify_node

    mock_llm.ainvoke.return_value = _make_llm_response("请具体说明")
    result = await clarify_node(make_state(query="苹果"))
    assert result["next"] == "end"
