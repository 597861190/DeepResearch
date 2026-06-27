import asyncio
import logging
from datetime import datetime

from app.state import ResearchState
from app.tools.search import is_low_quality_fact, search_web

logger = logging.getLogger(__name__)

# 搜索配置
MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 1


async def safe_search(task: str, task_index: int, total_tasks: int) -> dict:
    """
    带重试机制的搜索封装。一次只搜索一个任务。

    流程：
        1. 打印搜索开始日志（含任务进度）
        2. 执行搜索（失败自动重试），返回 Fact 对象
        3. 打印搜索结果摘要
        4. 将 Fact 转为 dict 返回（与 state schema 兼容）
    """
    logger.info("───── 搜索任务 %d/%d ─────", task_index, total_tasks)
    logger.info("🔍 搜索关键词: %s", task)

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result: dict = await asyncio.to_thread(search_web, task)

            # 打印搜索结果摘要
            content = result.get("content", "")
            source = result.get("source", "")
            valid = result.get("valid", False)

            if content:
                line_count = content.count("\n") + 1
                preview = content[:200].replace("\n", " ")
                logger.info("✅ 搜索成功（来源=%s），内容长度=%d 字符，%d 行", source, len(content), line_count)
                logger.info("  预览: %s...", preview)
            else:
                logger.warning("⚠️ 搜索返回空结果（query=%s）", task)

            # 组装事实条目（转为 dict，与 state 兼容）
            fact = {
                "task": task,
                "content": content,
                "source": source,
                "valid": valid,
                "timestamp": datetime.now().isoformat(),
            }
            return fact

        except Exception as e:
            last_error = e
            logger.warning(
                "⚠️ 搜索失败（尝试 %d/%d）: %s",
                attempt,
                MAX_RETRIES,
                e,
            )
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY_SECONDS)

    # 所有重试都失败
    logger.error("❌ 搜索最终失败（已重试 %d 次）: %s", MAX_RETRIES, task)
    return {
        "task": task,
        "content": f"搜索失败: {last_error}" if last_error else "",
        "source": "tavily",
        "valid": False,
        "timestamp": datetime.now().isoformat(),
    }


async def executor_node(state: ResearchState) -> dict:
    idx = state.get("current_task_idx", 0)
    plan = state.get("plan", [])

    logger.info("=" * 50)
    logger.info("📋 当前状态：计划共 %d 项，已完成 %d 项", len(plan), idx)
    logger.info("🧠 已收集事实：%d 条", len(state.get("facts", [])))
    logger.info("=" * 50)

    # 已全部完成 → 直接结束
    if idx >= len(plan):
        logger.info("🎉 所有任务已完成，准备进入报告阶段")
        return {
            "next": "done",
            "current_task_idx": idx,
            "facts": state.get("facts", []),
        }

    # 取出所有未完成的任务
    remaining_tasks = plan[idx:]
    total = len(plan)
    logger.info("⚡ 并行搜索 %d 个剩余任务...", len(remaining_tasks))

    # 全部并行执行
    tasks_with_index = [
        (task, idx + i + 1)
        for i, task in enumerate(remaining_tasks)
    ]
    results = await asyncio.gather(*[
        safe_search(task, task_idx, total)
        for task, task_idx in tasks_with_index
    ])

    # 聚合所有结果
    current_facts = list(state.get("facts", []))
    failed_tasks = list(state.get("failed_tasks", []))
    completed_tasks = state.get("completed_tasks", 0)

    for task, fact in zip(remaining_tasks, results):
        content = fact.get("content", "")
        if not fact.get("valid") or is_low_quality_fact(content):
            logger.warning("Executor: 任务 '%s' 未获取到有效信息", task)
            failed_tasks.append(task)
        else:
            current_facts.append(fact)
            completed_tasks += 1
            logger.info(
                "Executor: 任务 '%s' 成功获取信息（%d 字符）",
                task,
                len(content),
            )

    # 计算平均事实长度
    total_length = sum(
        len(f.get("content", "")) for f in current_facts
    )
    avg_fact_length = total_length / len(current_facts) if current_facts else 0.0

    logger.info("📊 总计: %d/%d 任务完成，%d 项失败", completed_tasks, len(plan), len(failed_tasks))

    return {
        "facts": current_facts,
        "current_task_idx": len(plan),  # 全部完成
        "next": "continue",
        "reflect_count": state.get("reflect_count", 0),
        "completed_tasks": completed_tasks,
        "failed_tasks": failed_tasks,
        "avg_fact_length": avg_fact_length,
    }
