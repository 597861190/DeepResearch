"""
FastAPI 研究服务入口。

提供 REST API 接口，包装 LangGraph 研究管线。
不修改业务逻辑（nodes / graph / tools），只增加 HTTP 层。

启动方式：
    cd backend && .venv/bin/uvicorn app.main:app --reload --port 8000

API：
    POST /research         提交研究问题（后台异步执行）
    GET  /research/{id}    查询研究结果/进度
    GET  /health           健康检查
"""

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.graph import app as research_graph
from app.state import ResearchState
from app.utils.summary import summarize_output as _summarize_output

load_dotenv()

logger = logging.getLogger("api")

# ── 内存任务存储 ──────────────────────────────────────────────
tasks_store: dict[str, dict] = {}

# ── 会话记忆存储（user_id + session_id → 历史消息） ────────────
sessions_store: dict[tuple[str, str], list] = {}
MAX_SESSION_HISTORY = 10  # 每个 session 最多保留 10 轮对话


def _save_session_memory(
    session_key: tuple[str, str],
    query: str,
    report_md: str,
    prev_history: list,
) -> None:
    """将本次研究的 query + report 追加到会话记忆。"""
    from langchain_core.messages import AIMessage, HumanMessage

    new_entries = [
        HumanMessage(content=f"用户问题：{query}"),
        AIMessage(content=report_md[:2000] if report_md else "（研究未能生成报告）"),
    ]
    updated = prev_history + new_entries
    # 限制历史长度，防止无限增长
    sessions_store[session_key] = updated[-MAX_SESSION_HISTORY * 2:]
    logger.info(
        "💬 会话记忆已更新（%s），共 %d 条消息",
        session_key,
        len(sessions_store[session_key]),
    )


# 已完成任务的保留时间（超过此时间后被清理）
TASK_RETENTION_SECONDS = 3600  # 1 小时
_CLEANUP_INTERVAL_SECONDS = 300  # 每 5 分钟清理一次


def _cleanup_old_tasks() -> None:
    """清理超过保留时间的已完成/失败任务。"""
    now = datetime.now()
    expired_keys = [
        rid
        for rid, t in tasks_store.items()
        if t["status"] in ("completed", "failed")
        and "created_at" in t
        and (now - datetime.fromisoformat(t["created_at"])).total_seconds()
        > TASK_RETENTION_SECONDS
    ]
    for rid in expired_keys:
        del tasks_store[rid]
    if expired_keys:
        logger.info("🧹 清理了 %d 个过期任务", len(expired_keys))


async def _periodic_cleanup() -> None:
    """定期清理过期任务的协程。"""
    try:
        while True:
            await asyncio.sleep(_CLEANUP_INTERVAL_SECONDS)
            _cleanup_old_tasks()
    except asyncio.CancelledError:
        pass


# ── 日志配置（在 lifespan 中注入 request_id） ─────────────────


def _inject_log_context(request_id: str, thread_id: str) -> None:
    """向当前线程的 LogRecord 注入 request_id / thread_id 字段。"""
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.request_id = request_id
        record.thread_id = thread_id
        return record

    logging.setLogRecordFactory(record_factory)

    # 更新已有 handler 的格式，显示 request_id
    fmt = "[%(asctime)s] %(levelname)-7s [%(request_id)s|%(thread_id)s] %(name)s | %(message)s"
    for handler in logging.getLogger().handlers:
        handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭时的生命周期钩子。"""
    # ── 启动时 ──────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("langgraph", "langchain", "httpx", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger.info("🔧 Deep Research API 启动完成")

    # 启动后台清理任务
    cleanup_task = asyncio.create_task(_periodic_cleanup(), name="cleanup")

    yield
    # ── 关闭时 ──────────────────────────────────────────────
    cleanup_task.cancel()
    logger.info("🛑 Deep Research API 关闭")


app = FastAPI(title="Deep Research API", version="1.0.0", lifespan=lifespan)


# ── 请求/响应模型 ─────────────────────────────────────────────


class ResearchRequest(BaseModel):
    query: str
    user_id: str = "anonymous"
    session_id: str = "default"
    thread_id: str = "default"


class ResearchResponse(BaseModel):
    request_id: str
    thread_id: str
    status: str  # running / completed / failed
    created_at: str
    result: Optional[dict] = None
    error: Optional[str] = None


# ── 健康检查 ──────────────────────────────────────────────────


@app.get("/health")
async def health():
    _cleanup_old_tasks()
    running = sum(1 for t in tasks_store.values() if t["status"] == "running")
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "active_tasks": running,
        "total_tasks": len(tasks_store),
    }


# ── 提交研究 ──────────────────────────────────────────────────


@app.post("/research", response_model=ResearchResponse)
async def start_research(req: ResearchRequest):
    request_id = str(uuid.uuid4())[:8]

    # 向此请求的日志注入上下文
    _inject_log_context(request_id, req.thread_id)

    task_record = {
        "request_id": request_id,
        "thread_id": req.thread_id,
        "query": req.query,
        "status": "running",
        "created_at": datetime.now().isoformat(),
        "result": None,
        "error": None,
    }
    tasks_store[request_id] = task_record

    logger.info("📩 收到研究请求: query=%s (user=%s, session=%s)",
                req.query, req.user_id, req.session_id)

    # 后台异步执行，不阻塞 HTTP 响应
    asyncio.create_task(_run_research(
        request_id, req.query, req.thread_id, req.user_id, req.session_id,
    ))

    return ResearchResponse(
        request_id=request_id,
        thread_id=req.thread_id,
        status="running",
        created_at=task_record["created_at"],
    )


# ── 查询结果 ──────────────────────────────────────────────────


@app.get("/research/{request_id}", response_model=ResearchResponse)
async def get_research(request_id: str):
    record = tasks_store.get(request_id)
    if not record:
        raise HTTPException(status_code=404, detail="request_id 不存在")

    return ResearchResponse(
        request_id=record["request_id"],
        thread_id=record["thread_id"],
        status=record["status"],
        created_at=record["created_at"],
        result=record["result"],
        error=record["error"],
    )


# ── 后台研究执行 ──────────────────────────────────────────────


async def _run_research(
    request_id: str,
    query: str,
    thread_id: str,
    user_id: str = "anonymous",
    session_id: str = "default",
) -> None:
    """后台执行 LangGraph 管线，结果写入 tasks_store。

    支持会话记忆：自动从 sessions_store 加载同 session 的历史消息，
    执行完成后将本次研究结果追加回会话记忆。
    """
    # 为后台任务注入日志上下文
    _inject_log_context(request_id, thread_id)

    # ── 加载会话历史（同一 user + session 的过往研究记录） ──
    session_key = (user_id, session_id)
    session_history = list(sessions_store.get(session_key, []))

    state = ResearchState(
        query=query,
        user_id=user_id,
        session_id=session_id,
        intent="",
        messages=session_history,  # 带上历史消息
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

    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"request_id": request_id},
    }

    try:
        final_state = dict(state)
        async for event in research_graph.astream(state, config=config):
            for node_name, output in event.items():
                if node_name == "__end__":
                    continue
                final_state.update(output)

                # 节点级别日志
                logger.info(
                    "▶ %s | %s",
                    node_name,
                    _summarize_output(node_name, output),
                )

        tasks_store[request_id].update({
            "status": "completed",
            "result": {
                "report_md": final_state.get("report_md", ""),
                "facts_count": len(final_state.get("facts", [])),
                "reflect_count": final_state.get("reflect_count", 0),
                "completed_tasks": final_state.get("completed_tasks", 0),
                "intent": final_state.get("intent", ""),
            },
        })

        logger.info(
            "✅ 研究完成: facts=%d, reflect=%d 轮",
            len(final_state.get("facts", [])),
            final_state.get("reflect_count", 0),
        )

        # ── 将会话更新写入 sessions_store（持久化对话记忆） ──
        report_md = final_state.get("report_md", "")
        _save_session_memory(
            session_key=session_key,
            query=query,
            report_md=report_md,
            prev_history=session_history,
        )

    except Exception as e:
        logger.error("❌ 研究失败: %s", e)
        tasks_store[request_id].update({
            "status": "failed",
            "error": str(e),
        })
