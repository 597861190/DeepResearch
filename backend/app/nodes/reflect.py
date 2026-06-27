import logging
from app.state import ResearchState

logger = logging.getLogger(__name__)
MAX_REFLECT_ROUNDS = 3


async def reflect_node(state: ResearchState) -> dict:
    reflect_count = state.get("reflect_count", 0) + 1
    refine_count = state.get("refine_count", 0)
    facts = state.get("facts", [])
    plan = state.get("plan", [])
    score = state.get("qa_score", 0)
    has_report = bool(state.get("report_md", ""))

    valid_facts = [f for f in facts if f.get("content")]

    # 0️⃣ 强制终止
    if reflect_count >= MAX_REFLECT_ROUNDS:
        logger.info("Reflect: 🛑 达到最大轮次，强制结束")
        return {"next": "reporter", "reflect_count": reflect_count}

    # 1️⃣ 计划为空 → 无法执行，直接结束
    if not plan:
        logger.info("Reflect: 计划为空，直接结束")
        return {"next": "reporter", "reflect_count": reflect_count}

    # 2️⃣ 数据足够直接出报告
    if len(valid_facts) >= len(plan):
        logger.info("Reflect: ✅ 所有任务已完成，准备生成报告")
        return {"next": "reporter", "reflect_count": reflect_count}

    # 3️⃣ 还没有报告初稿 → 先让 reporter 生成，不做 refine
    if not has_report:
        logger.info("Reflect: 尚未生成报告初稿，先生成再评估")
        return {"next": "reporter", "reflect_count": reflect_count}

    # 4️⃣ 质量判断（有报告之后才走 refine 流程）
    if score >= 7:
        logger.info("Reflect: ✅ 质量评分 %.1f，直接交卷", score)
        return {"next": "reporter", "reflect_count": reflect_count}

    if refine_count >= 2:
        logger.info("Reflect: 🔄 精炼 %d 次仍不达标，调整研究方向", refine_count)
        return {"next": "planner", "reflect_count": reflect_count}

    logger.info("Reflect: 🔧 质量评分 %.1f，局部重写（第 %d 次精炼）", score, refine_count)
    return {"next": "refine", "reflect_count": reflect_count}
