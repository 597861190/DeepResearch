"""
FastAPI API 测试 — 验证 HTTP 端点的正确行为。
不依赖外部服务（graph 执行被 mock）。
"""

import sys
import os
from unittest.mock import AsyncMock, patch
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import pytest
from fastapi.testclient import TestClient

# 必须在 main 导入之前 mock graph.astream，避免真正执行管线


async def _empty_astream(*args, **kwargs):
    """空 async generator — 不产生任何事件，管线立即结束。"""
    return
    yield  # pragma: no cover — 'yield' 使函数成为 async generator


from app.graph import app as research_graph

research_graph.astream = _empty_astream


@pytest.fixture(autouse=True)
def _reset_tasks_store():
    """每个测试前重置 tasks_store。"""
    from app.main import tasks_store
    tasks_store.clear()
    yield


@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)


class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "timestamp" in data
        assert "active_tasks" in data


class TestResearch:
    def test_start_research_returns_request_id(self, client):
        resp = client.post(
            "/research",
            json={"query": "AI 发展前景", "thread_id": "test-001"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "running"
        assert data["request_id"]
        assert data["thread_id"] == "test-001"
        assert "created_at" in data

    def test_start_research_default_thread_id(self, client):
        resp = client.post(
            "/research",
            json={"query": "test"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["thread_id"] == "default"

    def test_get_research_returns_task(self, client):
        # 先创建一个任务
        create_resp = client.post("/research", json={"query": "test"})
        rid = create_resp.json()["request_id"]

        # 查询
        resp = client.get(f"/research/{rid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["request_id"] == rid
        assert data["status"] in ("running", "completed")

    def test_get_nonexistent_research_returns_404(self, client):
        resp = client.get("/research/nonexistent")
        assert resp.status_code == 404

    def test_research_task_count_on_health(self, client):
        """验证创建任务后 health 能反映活跃任务数。"""
        client.post("/research", json={"query": "q1"})
        client.post("/research", json={"query": "q2"})
        resp = client.get("/health")
        assert resp.json()["total_tasks"] >= 2

    def test_get_research_result_shape(self, client):
        """验证返回结果的字段完整性。"""
        resp = client.post("/research", json={"query": "test"})
        rid = resp.json()["request_id"]

        get_resp = client.get(f"/research/{rid}")
        data = get_resp.json()
        assert set(data.keys()) == {
            "request_id", "thread_id", "status",
            "created_at", "result", "error",
        }
