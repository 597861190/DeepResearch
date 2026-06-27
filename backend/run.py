"""
研究管线入口脚本

启动方式：
    cd backend && .venv/bin/python3 run.py

流程说明：
    1. 加载 .env 环境变量（API Key 等）
    2. 初始化 ResearchState（研究问题、空计划、空事实等）
    3. 以流式方式（astream）逐节点驱动 LangGraph 管线：
       intent → planner → executor → reflect →（循环或 reporter）
    4. 打印每个节点的执行摘要和最终报告
"""

import asyncio
import logging
import os
import uuid

from dotenv import load_dotenv

from app.graph import app
from app.state import ResearchState
from app.utils.summary import summarize_output as _summarize_output

logger = logging.getLogger("run")

# 加载 .env 文件中的环境变量（如 DEEPSEEK_API_KEY）
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))


def _setup_logging(request_id: str, thread_id: str) -> None:
    """
    配置全局日志格式，使每条日志自动附带 request_id / thread_id。

    使用 setLogRecordFactory 向所有日志记录注入上下文字段，
    各模块（intent/planner/executor 等）的 logger.info() 自动携带。
    """
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.request_id = request_id
        record.thread_id = thread_id
        return record

    logging.setLogRecordFactory(record_factory)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)-7s [%(request_id)s|%(thread_id)s] %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,  # force=True 覆盖之前可能已调用的 basicConfig
    )

    # 降低第三方库的日志噪音
    for noisy in ("langgraph", "langchain", "httpx", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger.info("🔧 日志上下文已注入: request_id=%s, thread_id=%s", request_id, thread_id)


async def main():
    REQUEST_ID = str(uuid.uuid4())[:8]
    THREAD_ID = "agent_dev_001"

    # ── 日志配置（在 main 内执行，以便注入 request_id / thread_id） ──
    _setup_logging(REQUEST_ID, THREAD_ID)

    logger.info("=" * 50)
    logger.info("🚀 开始执行研究管线")
    logger.info(f"🆔 request_id={REQUEST_ID}")
    logger.info(f"🧵 thread_id={THREAD_ID}")
    logger.info(f"📌 研究问题: agent的发展前景")
    logger.info("=" * 50)

    state = ResearchState(
        query="agent的发展前景",
        user_id="cli",
        session_id="cli_session",
        intent="",
        private_context="",
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
        qa_score=0.0,
        qa_suggestions=[],
        refine_count=0,
    )

    final_state = dict(state)

    config = {
        "configurable": {
            "thread_id": THREAD_ID
        },
        "metadata": {
            "request_id": REQUEST_ID
        }
    }

    async for event in app.astream(state, config=config):
        for node_name, output in event.items():
            if node_name == "__end__":
                continue
            final_state.update(output)
            logger.info("▶ %s | %s", node_name, _summarize_output(node_name, output))

    logger.info("=" * 50)
    logger.info("✅ 管线执行完成")

    report_md = final_state.get("report_md", "")
    logger.info("📋 最终报告:")
    for line in report_md.strip().split("\n"):
        logger.info("   %s", line)


if __name__ == "__main__":
    asyncio.run(main())
