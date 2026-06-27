# app/schema.py
from enum import Enum

class IntentType(str, Enum):
    GREETING = "greeting"           # 打招呼
    SIMPLE_QUERY = "simple_query"   # 简单查询（一句话能答）
    DEEP_RESEARCH = "deep_research" # 深度研究（需要拆解）
    AMBIGUOUS = "ambiguous"         # 不清楚