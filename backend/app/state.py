from typing import List, Literal, Optional
from typing_extensions import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ResearchState(TypedDict):
    # ── 用户与会话标识（用于跨请求关联） ──
    # 用户 ID，同一用户的多次研究可关联
    user_id: str
    # 当前会话 ID，用于多轮对话上下文
    session_id: str
    # 研究问题
    query: str
    # 意图分类结果
    intent: str
    # 消息历史
    messages: Annotated[List[BaseMessage], add_messages]
    # 研究计划：由 planner 生成的任务列表
    plan: List[str]
    # 当前执行到第几个任务
    current_task_idx: int
    # 从 Qdrant 向量库检索到的历史上下文
    private_context: str
    # 收集到的事实列表（每个元素是 dict，含 task/content/timestamp/status）
    facts: List[dict]
    # 反思轮次计数（避免死循环）
    reflect_count: int
    # 下一步指令
    next: Literal["continue", "replan", "done", "end", "reporter", "refine", "planner"]
    # 最终生成的报告
    report_md: str
    # ── 执行统计指标（由 executor 更新，reflect 读取判断） ──
    # 已完成任务数（成功获取有效信息的）
    completed_tasks: int
    # 失败任务列表
    failed_tasks: List[str]
    # 已收集事实的平均长度（用于质量评估）
    avg_fact_length: float
    # ── 报告质量评估（由 reporter / refine 更新） ──
    # 综合质量评分（0-10）
    qa_score: float
    # 评分改进建议列表
    qa_suggestions: list
    # 精炼轮次计数
    refine_count: int
