from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver

from app.schema import IntentType
from app.state import ResearchState
from app.nodes.intent import intent_node
from app.nodes.direct_answer import direct_answer_node
from app.nodes.clarify import clarify_node
from app.nodes.planner import planner_node
from app.nodes.executor import executor_node
from app.nodes.reflect import reflect_node
from app.nodes.reporter import reporter_node
from app.nodes.retrieval import retrieval_node
from app.nodes.refine import refine_node


def route_intent(state: ResearchState) -> str:
    intent = state.get("intent", IntentType.DEEP_RESEARCH.value)
    if intent in (IntentType.GREETING.value, IntentType.SIMPLE_QUERY.value):
        return "direct_answer"
    elif intent == IntentType.AMBIGUOUS.value:
        return "clarify"
    # deep_research 先走 retrieval（向量库检索），再到 planner
    return "retrieval"


def route_reflect(state: ResearchState) -> str:
    n = state["next"]
    if n == "replan" or n == "planner":
        return "planner"
    elif n == "refine":
        return "refine"
    return "reporter"


builder = StateGraph(ResearchState)

builder.add_node("intent", intent_node)
builder.add_node("direct_answer", direct_answer_node)
builder.add_node("clarify", clarify_node)
builder.add_node("planner", planner_node)
builder.add_node("retrieval", retrieval_node)
builder.add_node("executor", executor_node)
builder.add_node("reflect", reflect_node)
builder.add_node("refine", refine_node)  # reflect 和 refine 共用同一个节点函数，根据 state["next"] 判断下一步
builder.add_node("reporter", reporter_node)

builder.add_edge(START, "intent")

builder.add_conditional_edges("intent", route_intent, {
    "direct_answer": "direct_answer",
    "clarify": "clarify",
    "retrieval": "retrieval",
})

builder.add_edge("retrieval", "planner")

builder.add_edge("planner", "executor")
builder.add_edge("executor", "reflect")

builder.add_conditional_edges("reflect", route_reflect, {
    "planner": "planner",
    "reporter": "reporter",
    "refine": "refine", #局部重写
})
builder.add_edge("refine", "reflect") # refine 结束后继续回到 reflect 评估

app = builder.compile()
