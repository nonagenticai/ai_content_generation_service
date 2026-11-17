"""Redis client for ai_content_generation_service caching.

Thin re-export from w2v_common. The ``RedisClient`` class, singleton helpers,
and lifecycle functions live in the shared library; this module preserves
the ``src.cache.redis_client`` import path.
"""

from w2v_common.cache.redis_client import (
    RedisClient as RedisClient,
)
from w2v_common.cache.redis_client import (
    close_redis as close_redis,
)
from w2v_common.cache.redis_client import (
    get_redis_client as get_redis_client,
)
from w2v_common.cache.redis_client import (
    init_redis as init_redis,
)

__all__ = ["RedisClient", "close_redis", "get_redis_client", "init_redis"]
