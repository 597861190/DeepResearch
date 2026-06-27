# app/utils/cache.py
"""
搜索结果缓存层。

职责：
    提供进程中唯一的缓存装饰器，避免重复搜索相同 query。
    所有搜索请求通过此模块走缓存，减少 Tavily API 调用。

用法（在 search.py 中）：
    search_web(query) → search_cache(query) → 命中缓存 or _real_search(query)
"""

from functools import lru_cache

# 缓存容量
CACHE_MAXSIZE = 256


def _get_search_func():
    """
    惰性导入 _real_search，避免循环依赖：
    cache.py ←import→ search.py 会形成循环，
    因此在函数首次被调用时才导入 search 模块。
    """
    from app.tools.search import _real_search  # lazy import
    return _real_search


@lru_cache(maxsize=CACHE_MAXSIZE)
def search_cache(query: str) -> dict:
    """
    带缓存的搜索接口。

    参数:
        query: 搜索关键词

    返回:
        dict，含 task / content / source / valid 等字段
    """
    search_func = _get_search_func()
    return search_func(query)


def cache_info():
    """返回缓存命中统计，便于监控。"""
    return search_cache.cache_info()


def cache_clear():
    """清空缓存。"""
    search_cache.cache_clear()
