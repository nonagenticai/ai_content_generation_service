"""Racing MinIO bucket-create must not fail service startup.

Post the w2v_common storage-facade migration (#62), ``StorageService.initialize``
swallows any startup probe/create failure (the bucket is created on first
write), so two replicas racing the bucket-create — one of which loses with a
``BucketAlreadyOwnedByYou`` / ``BucketAlreadyExists`` — both start cleanly
instead of one crash-looping.
"""

from __future__ import annotations

import asyncio
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.storage_service import StorageService


class _FakeBucketExistsError(Exception):
    """Stand-in for minio's BucketAlreadyOwnedByYou/BucketAlreadyExists."""


def _minio_storage(raising: bool) -> StorageService:
    with patch("src.services.storage_service.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(
            storage=MagicMock(
                local_path=tempfile.mkdtemp(),
                backend="minio",
                minio_endpoint="minio.test:8389",
                minio_access_key="k",
                minio_secret_key="s",  # pragma: allowlist secret
                minio_bucket="content-generation-service",
                minio_secure=False,
            )
        )
        svc = StorageService()
    svc._backend = "minio"
    facade = MagicMock()
    facade.bucket = "content-generation-service"
    facade.initialize = AsyncMock(
        side_effect=_FakeBucketExistsError("bucket already owned by you") if raising else None
    )
    svc._storage = facade
    return svc


async def test_racing_bucket_create_does_not_fail_startup():
    """A concurrent create that loses the race is swallowed, not re-raised."""
    svc = _minio_storage(raising=True)
    # Must NOT raise even though the facade create raised a bucket-exists error.
    await svc.initialize()


async def test_concurrent_initialize_all_succeed():
    """Two replicas initializing at once both start cleanly."""
    a = _minio_storage(raising=True)
    b = _minio_storage(raising=False)
    # gather with no exception surfaced from either.
    await asyncio.gather(a.initialize(), b.initialize())
