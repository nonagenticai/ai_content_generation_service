"""Unit tests for error classification (Phase 2)."""

from __future__ import annotations

from src.services.errors import is_retriable_error


def test_timeout_error_is_retriable():
    assert is_retriable_error(TimeoutError("timed out")) is True


def test_asyncio_timeout_is_retriable():
    assert is_retriable_error(TimeoutError()) is True


def test_runtime_error_with_timeout_is_retriable():
    assert is_retriable_error(RuntimeError("Connection timeout")) is True


def test_runtime_error_with_rate_limit_is_retriable():
    assert is_retriable_error(RuntimeError("rate limit exceeded")) is True


def test_runtime_error_generic_is_not_retriable():
    assert is_retriable_error(RuntimeError("Something else")) is False


def test_value_error_is_not_retriable():
    assert is_retriable_error(ValueError("bad input")) is False


def test_type_error_is_not_retriable():
    assert is_retriable_error(TypeError()) is False


def test_key_error_is_not_retriable():
    assert is_retriable_error(KeyError("missing")) is False


def test_httpx_timeout_is_retriable():
    try:
        import httpx

        exc = httpx.TimeoutException("read timed out")
        assert is_retriable_error(exc) is True
    except ImportError:
        pass


def test_httpx_429_is_retriable():
    try:
        import httpx

        response = httpx.Response(429, request=httpx.Request("POST", "https://example.com"))
        exc = httpx.HTTPStatusError("rate limited", request=response.request, response=response)
        assert is_retriable_error(exc) is True
    except ImportError:
        pass


def test_httpx_500_is_retriable():
    try:
        import httpx

        response = httpx.Response(500, request=httpx.Request("POST", "https://example.com"))
        exc = httpx.HTTPStatusError("server error", request=response.request, response=response)
        assert is_retriable_error(exc) is True
    except ImportError:
        pass


def test_httpx_502_is_retriable():
    try:
        import httpx

        response = httpx.Response(502, request=httpx.Request("POST", "https://example.com"))
        exc = httpx.HTTPStatusError("bad gateway", request=response.request, response=response)
        assert is_retriable_error(exc) is True
    except ImportError:
        pass


def test_httpx_400_is_not_retriable():
    try:
        import httpx

        response = httpx.Response(400, request=httpx.Request("POST", "https://example.com"))
        exc = httpx.HTTPStatusError("bad request", request=response.request, response=response)
        assert is_retriable_error(exc) is False
    except ImportError:
        pass


def test_httpx_401_is_not_retriable():
    try:
        import httpx

        response = httpx.Response(401, request=httpx.Request("POST", "https://example.com"))
        exc = httpx.HTTPStatusError("unauthorized", request=response.request, response=response)
        assert is_retriable_error(exc) is False
    except ImportError:
        pass


def test_httpx_404_is_not_retriable():
    try:
        import httpx

        response = httpx.Response(404, request=httpx.Request("POST", "https://example.com"))
        exc = httpx.HTTPStatusError("not found", request=response.request, response=response)
        assert is_retriable_error(exc) is False
    except ImportError:
        pass


def test_provider_error_retriable():
    from src.services.exceptions import ProviderError

    exc = ProviderError("timeout", provider="runway", retriable=True)
    assert is_retriable_error(exc) is True


def test_provider_error_not_retriable():
    from src.services.exceptions import ProviderError

    exc = ProviderError("bad request", provider="runway", retriable=False)
    assert is_retriable_error(exc) is False
