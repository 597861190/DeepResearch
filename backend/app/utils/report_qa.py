import re

def evaluate_report(
    query: str,
    report_md: str,
    facts: list[dict],
) -> dict:
    report = report_md or ""
    report_l = report.lower()

    # 1. 相关度
    relevance = 0
    query_words = set(query.lower().split())
    report_words = set(report_l.split())
    overlap = len(query_words & report_words) / max(len(query_words), 1)
    if overlap > 0.3:
        relevance += 4
    if len(report) > 600:
        relevance += 3
    if re.search(r"conclusion|总结|展望", report_l):
        relevance += 3

    # 2. 事实覆盖率
    factuality = 0
    for f in facts:
        snippet = f.get("content", "")[:40]
        if snippet and snippet in report:
            factuality += 1
        elif f.get("task") and f["task"].lower() in report_l:
            factuality += 1
    factuality = min(factuality, 10)

    # 3. 结构连贯性
    sections = re.findall(r"^##\s+.+", report, re.MULTILINE)
    coherence = min(10, len(sections) * 2)

    # 4. 综合分
    quality_score = round((relevance + factuality + coherence) / 3, 1)

    # 5. 建议
    suggestions = []
    if relevance < 7:
        suggestions.append("报告与研究问题相关性不足，建议更紧扣主题。")
    if factuality < 7:
        suggestions.append("报告未能充分引用已收集的事实，建议加强事实支撑。")
    if coherence < 7:
        suggestions.append("报告结构不够清晰，建议使用更多二级标题划分内容。")

    return {
        "quality_score": quality_score,
        "relevance_score": relevance,
        "factuality_score": factuality,
        "coherence_score": coherence,
        "suggestions": suggestions,
    }