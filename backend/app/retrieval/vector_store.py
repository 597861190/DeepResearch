"""
Qdrant 向量数据库封装模块。

职责：
    连接本地 Qdrant，提供事实（Fact）的向量存储和语义检索能力。
    使用 SentenceTransformer 将文本转为向量后存入 / 检索。

使用方式（惰性初始化，第一次调用时自动连接）：
    from app.retrieval.vector_store import upsert_fact, search_facts

    upsert_fact("task", "content", "source")
    results = search_facts("query", top_k=3)
"""

import logging
import os
import uuid
from typing import Optional

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ── 默认配置（可通过环境变量覆盖） ──────────────────────────────

DEFAULT_QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
DEFAULT_QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
DEFAULT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "research_facts")
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 固定维度

# ── 模块级懒加载状态 ───────────────────────────────────────────

_client: Optional[QdrantClient] = None
_model: Optional[SentenceTransformer] = None
_collection_initialized: bool = False


def _get_client() -> QdrantClient:
    """惰性获取 Qdrant 客户端（首次调用时创建连接）。"""
    global _client
    if _client is None:
        _client = QdrantClient(
            host=DEFAULT_QDRANT_HOST,
            port=DEFAULT_QDRANT_PORT,
            prefer_grpc=False,
            check_compatibility=False,
        )
        logger.info(
            "🔗 已连接 Qdrant: %s:%s",
            DEFAULT_QDRANT_HOST,
            DEFAULT_QDRANT_PORT,
        )
    return _client


def _get_model() -> SentenceTransformer:
    """惰性加载 Embedding 模型（首次调用时自动下载）。"""
    global _model
    if _model is None:
        logger.info("🧠 加载 Embedding 模型: %s ...", DEFAULT_EMBED_MODEL)
        _model = SentenceTransformer(DEFAULT_EMBED_MODEL)
        logger.info("✅ Embedding 模型加载完成")
    return _model


def _ensure_collection(collection_name: str = DEFAULT_COLLECTION_NAME) -> None:
    """确保 Qdrant 集合存在（不存在则创建），仅执行一次。

    使用 collection_exists 检查 + create_collection 创建，
    避免 recreate_collection 意外删除已有数据。
    """
    global _collection_initialized
    if _collection_initialized:
        return

    client = _get_client()
    try:
        if client.collection_exists(collection_name):
            logger.info("✅ 集合 '%s' 已存在", collection_name)
        else:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=EMBEDDING_DIM,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info("✅ 集合 '%s' 创建成功", collection_name)
        _collection_initialized = True
    except Exception as e:
        logger.error("❌ 集合 '%s' 初始化失败: %s", collection_name, e)
        raise


def init_collection(collection_name: str = DEFAULT_COLLECTION_NAME) -> None:
    """对外暴露的集合初始化接口，可被 main.py 等显式调用。"""
    _ensure_collection(collection_name)


def upsert_fact(
    task: str,
    content: str,
    source: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> None:
    """存储一条事实到向量库。

    参数:
        task:    搜索关键词 / 任务名
        content: 事实正文
        source:  来源（如 "tavily"）
    """
    if not content:
        logger.debug("upsert_fact: content 为空，跳过")
        return

    _ensure_collection(collection_name)
    model = _get_model()
    client = _get_client()

    try:
        vector = model.encode(content).tolist()
        point_id = str(uuid.uuid4())

        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "task": task,
                        "content": content,
                        "source": source,
                    },
                )
            ],
        )
        logger.info("📥 已存入 Fact: %s（%d 字符）", task, len(content))
    except Exception as e:
        logger.error("❌ 写入 Fact 失败（task=%s）: %s", task, e)


def search_facts(
    query: str,
    top_k: int = 3,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> list:
    """根据语义相似度检索相关 Facts。

    参数:
        query:  检索文本
        top_k:  返回 top-K 条结果

    返回:
        list[dict]，每个 dict 含 task / content / source 字段
    """
    _ensure_collection(collection_name)
    model = _get_model()
    client = _get_client()

    try:
        vector = model.encode(query).tolist()
        resp = client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=top_k,
            with_payload=True,
        )
        results = [hit.payload for hit in resp.points if hit.payload]
        if results:
            logger.info("🔍 检索到 %d 条相关记忆", len(results))
        else:
            logger.debug("🔍 未检索到相关记忆")
        return results
    except Exception as e:
        logger.error("❌ 检索失败（query=%s）: %s", query[:50], e)
        return []
