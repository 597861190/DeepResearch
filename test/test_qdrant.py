#!/usr/bin/env python3
"""测试 Qdrant 向量数据库连接"""

import requests

# 1. 配置 Qdrant 的本地地址
QDRANT_URL = "http://localhost:6333"

# 2. 测试连接（获取集群信息）
try:
    response = requests.get(f"{QDRANT_URL}/collections")
    print("✅ Qdrant 连接成功！")
    print("当前已有的集合:", response.json())
except Exception as e:
    print("❌ 连接失败:", e)
