"""
节点输出摘要工具 — 将节点的输出字典压缩为一行可读文本。
"""


def summarize_output(node: str, output: dict) -> str:
    """将节点的输出字典压缩为一行可读的摘要文本，用于日志展示。

    参数：
        node:   节点名称（如 "planner", "executor"）
        output: 节点返回的状态变更字典

    返回：
        一行字符串，如 'plan=(6 项) | idx=0'
    """
    parts = []

    if "next" in output:
        parts.append(f"next={output['next']}")
    if "plan" in output and output["plan"]:
        n = len(output["plan"])
        parts.append(f"plan=({n} 项)")
    if "facts" in output and output["facts"]:
        parts.append(f"facts=({len(output['facts'])} 条)")
    if "report_md" in output:
        md = output["report_md"]
        preview = md.strip()[:60].replace("\n", " ")
        if len(md) > 60:
            parts.append(f'report="{preview}..."')
        else:
            parts.append(f'report="{md.strip()}"')
    if "reflect_count" in output:
        parts.append(f"reflect_count={output['reflect_count']}")
    if "current_task_idx" in output and output["current_task_idx"] is not None:
        parts.append(f"idx={output['current_task_idx']}")
    if "completed_tasks" in output:
        parts.append(f"done={output['completed_tasks']}")
    if "avg_fact_length" in output:
        parts.append(f"avg_len={output['avg_fact_length']:.0f}")
    if "failed_tasks" in output and output["failed_tasks"]:
        parts.append(f"failed={len(output['failed_tasks'])}")
    if "intent" in output and output["intent"]:
        parts.append(f"intent={output['intent']}")
    if "messages" in output and output["messages"]:
        parts.append(f"msg={len(output['messages'])}条")

    return " | ".join(parts) if parts else "(no fields changed)"
