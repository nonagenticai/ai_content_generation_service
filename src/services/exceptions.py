"""Custom exception hierarchy for ai_content_generation_service."""

from __future__ import annotations


class ServiceError(Exception):
    """Base exception. Guarantees non-empty str() via repr() fallback."""

    def __init__(self, message: str = "", *, cause: Exception | None = None) -> None:
        if not message and cause is not None:
            message = f"{type(cause).__name__}: {cause!r}"
        super().__init__(message)
        self.__cause__ = cause


class ProviderError(ServiceError):
    """Provider failure with retriable flag."""

    def __init__(
        self,
        message: str = "",
        *,
        provider: str = "",
        retriable: bool = False,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message, cause=cause)
        self.provider = provider
        self.retriable = retriable


class ProviderTimeoutError(ProviderError):
    """Provider timed out (retriable)."""

    def __init__(
        self,
        message: str = "",
        *,
        provider: str = "",
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message, provider=provider, retriable=True, cause=cause)


class ProviderRateLimitError(ProviderError):
    """Provider rate limited (retriable)."""

    def __init__(
        self,
        message: str = "",
        *,
        provider: str = "",
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message, provider=provider, retriable=True, cause=cause)


class BudgetExceededError(ServiceError):
    """Budget limit exceeded."""


class JobNotFoundError(ServiceError):
    """Job not found in the database."""


class BatchNotFoundError(ServiceError):
    """Batch not found."""


class StorageError(ServiceError):
    """Storage operation failed."""


class CallbackDeliveryError(ServiceError):
    """Callback delivery failed after all retries."""


def format_error(exc: Exception) -> str:
    """Format an exception message, never returning an empty string."""
    msg = str(exc)
    return msg if msg else f"{type(exc).__name__}: {exc!r}"
