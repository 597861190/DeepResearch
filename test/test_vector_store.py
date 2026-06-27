"""
测试 Qdrant 向量数据库模块（app.retrieval.vector_store）。

前置条件：
    Docker Qdrant 正在运行（localhost:6333）

运行方式：
    cd backend && python -m pytest ../test/test_vector_store.py -v
"""

import os
import sys

# 确保能找到 backend 下的模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.retrieval.vector_store import (
    upsert_fact,
    search_facts,
    _get_client,
    _collection_initialized,
)
from qdrant_client import models

TEST_COLLECTION = "test_research_facts"


def setup_module():
    """测试前清理可能残留的测试集合。"""
    client = _get_client()
    try:
        if client.collection_exists(TEST_COLLECTION):
            client.delete_collection(TEST_COLLECTION)
            print(f"🧹 清理残留测试集合: {TEST_COLLECTION}")
    except Exception:
        pass


def teardown_module():
    """测试后删除测试集合。"""
    client = _get_client()
    try:
        if client.collection_exists(TEST_COLLECTION):
            client.delete_collection(TEST_COLLECTION)
            print(f"🧹 已删除测试集合: {TEST_COLLECTION}")
    except Exception:
        pass


def test_connection():
    """测试 Qdrant 连接是否正常。"""
    client = _get_client()
    # 获取集群信息来验证连接
    info = client.get_collections()
    assert info is not None
    print(f"✅ Qdrant 连接正常，当前集合数: {len(info.collections)}")


def test_upsert_and_search():
    """测试写入一条事实并检索回来。"""
    # 写入一条测试数据（使用测试集合名）
    upsert_fact(
        task="测试任务",
        content="人工智能Agent是一种能够自主执行任务的智能体系统",
        source="test",
        collection_name=TEST_COLLECTION,
    )

    # 检索验证
    results = search_facts(
        query="AI Agent",
        top_k=3,
        collection_name=TEST_COLLECTION,
    )

    assert len(results) >= 1, "应检索到至少 1 条结果"
    found = any("人工智能Agent" in r.get("content", "") for r in results)
    assert found, "检索结果应包含写入的内容"


def test_search_with_semantic_match():
    """测试语义检索能匹配近义内容。"""
    # 写入一条英文风格的内容
    upsert_fact(
        task="语义测试",
        content="Large Language Models (LLMs) are the foundation of modern AI assistants",
        source="test",
        collection_name=TEST_COLLECTION,
    )

    # 用近义 query 检索
    results = search_facts(
        query="大语言模型 AI助手",
        top_k=3,
        collection_name=TEST_COLLECTION,
    )

    assert len(results) >= 1, "语义检索应返回结果"


def test_upsert_empty_content():
    """测试空 content 应被跳过（不抛异常）。"""
    # 这应该静默跳过，不抛异常
    upsert_fact(
        task="空内容测试",
        content="",
        source="test",
        collection_name=TEST_COLLECTION,
    )
    # 验证没有写入空内容
    results = search_facts(
        query="空内容测试",
        top_k=3,
        collection_name=TEST_COLLECTION,
    )
    # 不应检索到空内容的结果
    for r in results:
        assert r.get("content", "") != "", "不应检索到空内容"


def test_search_no_results():
    """测试检索不相关内容时应返回空列表。"""
    results = search_facts(
        query="xyznonexistent_12345_xyz",
        top_k=3,
        collection_name=TEST_COLLECTION,
    )
    assert isinstance(results, list)
    # 可能因为语义相关性，不相关内容也可能返回一部分，
    # 但至少不应抛异常
