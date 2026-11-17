"""Unit tests for StorageService's MinIO-backend delegation to the shared
``w2v_common.storage.StorageService`` facade.

The wrapper no longer speaks to ``minio`` directly — it forwards to the facade.
These tests build a wrapper on the MinIO backend and monkeypatch the facade's
async primitives (``put_bytes`` / ``get_bytes`` / ``presign_get`` /
``delete_object`` / ``check_health`` / ``initialize``) to assert the wrapper
forwards with the ``content_generation/{job_id}/{filename}`` key layout and the
expected content types, that ``presign_url`` now awaits ``presign_get``, and that
a boot-time MinIO failure is swallowed (anti-crashloop).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _minio_settings(**overrides: str | bool | int | None) -> SimpleNamespace:
    """Build a settings-like namespace configured for the MinIO backend."""
    defaults: dict[str, str | bool | int | None] = {
        "backend": "minio",
        "local_path": "/tmp/test-storage",
        "minio_endpoint": "localhost:9000",
        "minio_access_key": "minioaccess",
        "minio_secret_key": "miniosecret",  # pragma: allowlist secret
        "minio_bucket": "test-bucket",
        "minio_secure": False,
        "minio_external_endpoint": None,
        "presign_expiry_seconds": 3600,
    }
    defaults.update(overrides)
    return SimpleNamespace(storage=SimpleNamespace(**defaults))


def _make_service(**overrides: str | bool | int | None):
    from src.services.storage_service import StorageService

    with patch("src.services.storage_service.get_settings") as mock_settings:
        mock_settings.return_value = _minio_settings(**overrides)
        return StorageService()


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


async def test_minio_constructor_builds_facade() -> None:
    """backend='minio' constructs a delegated facade without touching the network."""
    svc = _make_service()
    assert svc._backend == "minio"
    assert svc._storage is not None
    # The facade defers client creation to initialize(); nothing connected yet.
    assert svc._storage.backend == "minio"


# ---------------------------------------------------------------------------
# store_video / store_image -> put_bytes
# ---------------------------------------------------------------------------


async def test_store_video_delegates_to_put_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    """store_video forwards the content_generation key + video/mp4 content type."""
    svc = _make_service()
    seen: dict[str, str] = {}

    async def _put(key: str, data: bytes, *, content_type: str = "") -> str:
        seen["key"] = key
        seen["content_type"] = content_type
        return key

    monkeypatch.setattr(svc._storage, "put_bytes", _put)
    result = await svc.store_video("job-1", b"data", "clip.mp4")

    assert result == "content_generation/job-1/clip.mp4"
    assert seen["key"] == "content_generation/job-1/clip.mp4"
    assert seen["content_type"] == "video/mp4"


async def test_store_image_delegates_to_put_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    """store_image forwards the content_generation key + the given content type."""
    svc = _make_service()
    seen: dict[str, str] = {}

    async def _put(key: str, data: bytes, *, content_type: str = "") -> str:
        seen["key"] = key
        seen["content_type"] = content_type
        return key

    monkeypatch.setattr(svc._storage, "put_bytes", _put)
    result = await svc.store_image("job-9", b"png", "concept.png", content_type="image/png")

    assert result == "content_generation/job-9/concept.png"
    assert seen["key"] == "content_generation/job-9/concept.png"
    assert seen["content_type"] == "image/png"


# ---------------------------------------------------------------------------
# upload_to_minio
# ---------------------------------------------------------------------------


async def test_upload_to_minio_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """upload_to_minio forwards to put_bytes and returns the MinIO key on minio backend."""
    svc = _make_service()
    seen: dict[str, str] = {}

    async def _put(key: str, data: bytes, *, content_type: str = "") -> str:
        seen["key"] = key
        seen["content_type"] = content_type
        return key

    monkeypatch.setattr(svc._storage, "put_bytes", _put)
    result = await svc.upload_to_minio("job-7", b"payload", "video.mp4")

    assert result == "content_generation/job-7/video.mp4"
    assert seen["key"] == "content_generation/job-7/video.mp4"
    assert seen["content_type"] == "video/mp4"


# ---------------------------------------------------------------------------
# retrieve
# ---------------------------------------------------------------------------


async def test_retrieve_minio_key_delegates_to_get_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """retrieve() with a relative key on the minio backend forwards the bare key."""
    svc = _make_service()
    seen: dict[str, str] = {}

    async def _get(ref: str) -> bytes | None:
        seen["ref"] = ref
        return b"minio-video-data"

    monkeypatch.setattr(svc._storage, "get_bytes", _get)
    data = await svc.retrieve("content_generation/job-1/clip.mp4")

    assert data == b"minio-video-data"
    assert seen["ref"] == "content_generation/job-1/clip.mp4"


async def test_retrieve_minio_missing_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """retrieve() returns None when the facade reports the object absent."""
    svc = _make_service()

    async def _get(ref: str) -> bytes | None:
        return None

    monkeypatch.setattr(svc._storage, "get_bytes", _get)
    assert await svc.retrieve("content_generation/job-1/gone.mp4") is None


# ---------------------------------------------------------------------------
# presign_url -> presign_get (now async)
# ---------------------------------------------------------------------------


async def test_presign_url_awaits_presign_get(monkeypatch: pytest.MonkeyPatch) -> None:
    """presign_url awaits the facade presign_get and forwards the expiry."""
    svc = _make_service()
    seen: dict[str, object] = {}

    async def _presign(key: str, *, expires: int | None = None, bucket: str | None = None) -> str:
        seen["key"] = key
        seen["expires"] = expires
        return "https://minio/presigned-url"

    monkeypatch.setattr(svc._storage, "presign_get", _presign)
    url = await svc.presign_url("content_generation/job-1/clip.mp4", expires_s=7200)

    assert url == "https://minio/presigned-url"
    assert seen["key"] == "content_generation/job-1/clip.mp4"
    assert seen["expires"] == 7200


async def test_presign_url_returns_none_when_facade_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """presign_url returns None when the facade cannot presign."""
    svc = _make_service()

    async def _presign(key: str, *, expires: int | None = None, bucket: str | None = None) -> None:
        return None

    monkeypatch.setattr(svc._storage, "presign_get", _presign)
    assert await svc.presign_url("some/key") is None


# ---------------------------------------------------------------------------
# delete_from_minio -> delete_object
# ---------------------------------------------------------------------------


async def test_delete_from_minio_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    """delete_from_minio forwards the key to the facade delete_object."""
    svc = _make_service()
    seen: dict[str, str] = {}

    async def _del(key: str, *, bucket: str | None = None) -> bool:
        seen["key"] = key
        return True

    monkeypatch.setattr(svc._storage, "delete_object", _del)
    result = await svc.delete_from_minio("content_generation/job-1/clip.mp4")

    assert result is True
    assert seen["key"] == "content_generation/job-1/clip.mp4"


async def test_delete_from_minio_returns_false_on_facade_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """delete_from_minio propagates the facade's False on a storage error."""
    svc = _make_service()

    async def _del(key: str, *, bucket: str | None = None) -> bool:
        return False

    monkeypatch.setattr(svc._storage, "delete_object", _del)
    assert await svc.delete_from_minio("bad/key") is False


# ---------------------------------------------------------------------------
# delete_local (local disk + facade delete_object + parent prune)
# ---------------------------------------------------------------------------


async def test_delete_local_removes_file_and_empty_parent() -> None:
    """delete_local deletes the file (via the facade) and prunes the empty parent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # A local backend so the facade actually unlinks the on-disk file.
        svc = _make_service(backend="local", local_path=tmpdir)

        sub = Path(tmpdir) / "content_generation" / "job-99"
        sub.mkdir(parents=True)
        target = sub / "clip.mp4"
        target.write_bytes(b"data")

        result = await svc.delete_local(str(target))

        assert result is True
        assert not target.exists()
        assert not sub.exists()  # empty parent removed


async def test_delete_local_file_not_found() -> None:
    """delete_local returns False for a path outside/absent from the storage root."""
    with tempfile.TemporaryDirectory() as tmpdir:
        svc = _make_service(backend="local", local_path=tmpdir)
        assert await svc.delete_local("/nonexistent/path/file.mp4") is False


# ---------------------------------------------------------------------------
# health_check -> check_health
# ---------------------------------------------------------------------------


async def test_health_check_delegates_true(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _make_service()

    async def _health() -> bool:
        return True

    monkeypatch.setattr(svc._storage, "check_health", _health)
    assert await svc.health_check() is True


async def test_health_check_delegates_false(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _make_service()

    async def _health() -> bool:
        return False

    monkeypatch.setattr(svc._storage, "check_health", _health)
    assert await svc.health_check() is False


# ---------------------------------------------------------------------------
# initialize -> facade.initialize (+ anti-crashloop)
# ---------------------------------------------------------------------------


async def test_initialize_delegates_to_facade(monkeypatch: pytest.MonkeyPatch) -> None:
    """initialize() forwards to the facade on the minio backend."""
    svc = _make_service()
    called = {"init": False}

    async def _init() -> None:
        called["init"] = True

    monkeypatch.setattr(svc._storage, "initialize", _init)
    await svc.initialize()
    assert called["init"] is True


async def test_initialize_tolerates_facade_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-crashloop: a MinIO blip at boot must be swallowed, never raised."""
    svc = _make_service()

    async def _boom() -> None:
        raise ConnectionError("minio unreachable at boot")

    monkeypatch.setattr(svc._storage, "initialize", _boom)
    # Must NOT raise.
    await svc.initialize()
