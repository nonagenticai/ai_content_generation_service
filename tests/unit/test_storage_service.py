"""Unit tests for StorageService with local and MinIO backends."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch


@patch("src.services.storage_service.get_settings")
async def test_store_and_retrieve_local(mock_settings):
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_settings.return_value = MagicMock(
            storage=MagicMock(
                local_path=tmpdir,
                backend="local",
                minio_endpoint="",
                minio_access_key="",
                minio_secret_key="",
                minio_bucket="test",
                minio_secure=False,
            )
        )
        from src.services.storage_service import StorageService

        svc = StorageService()
        path = await svc.store_video("job1", b"video-data", "clip.mp4")
        assert "job1" in path
        assert Path(path).exists()

        data = await svc.retrieve(path)
        assert data == b"video-data"


@patch("src.services.storage_service.get_settings")
async def test_store_image_local(mock_settings):
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_settings.return_value = MagicMock(
            storage=MagicMock(
                local_path=tmpdir,
                backend="local",
                minio_endpoint="",
                minio_access_key="",
                minio_secret_key="",
                minio_bucket="test",
                minio_secure=False,
            )
        )
        from src.services.storage_service import StorageService

        svc = StorageService()
        path = await svc.store_image("job1", b"image-data", "concept.png", content_type="image/png")
        assert "job1" in path
        assert Path(path).exists()

        data = await svc.retrieve(path)
        assert data == b"image-data"


@patch("src.services.storage_service.get_settings")
async def test_retrieve_missing_local(mock_settings):
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_settings.return_value = MagicMock(
            storage=MagicMock(
                local_path=tmpdir,
                backend="local",
                minio_endpoint="",
                minio_access_key="",
                minio_secret_key="",
                minio_bucket="test",
                minio_secure=False,
            )
        )
        from src.services.storage_service import StorageService

        svc = StorageService()
        result = await svc.retrieve("/nonexistent/path.mp4")
        assert result is None


@patch("src.services.storage_service.get_settings")
async def test_get_job_dir(mock_settings):
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_settings.return_value = MagicMock(
            storage=MagicMock(
                local_path=tmpdir,
                backend="local",
                minio_endpoint="",
                minio_access_key="",
                minio_secret_key="",
                minio_bucket="test",
                minio_secure=False,
            )
        )
        from src.services.storage_service import StorageService

        svc = StorageService()
        job_dir = svc.get_job_dir("test-job-id")
        assert job_dir.exists()
        assert job_dir.name == "test-job-id"


@patch("src.services.storage_service.get_settings")
async def test_initialize_creates_directory(mock_settings):
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "subdir" / "storage"
        mock_settings.return_value = MagicMock(
            storage=MagicMock(
                local_path=str(target),
                backend="local",
                minio_endpoint="",
                minio_access_key="",
                minio_secret_key="",
                minio_bucket="test",
                minio_secure=False,
            )
        )
        from src.services.storage_service import StorageService

        svc = StorageService()
        await svc.initialize()
        assert target.exists()


@patch("src.services.storage_service.get_settings")
async def test_health_check_local(mock_settings):
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_settings.return_value = MagicMock(
            storage=MagicMock(
                local_path=tmpdir,
                backend="local",
                minio_endpoint="",
                minio_access_key="",
                minio_secret_key="",
                minio_bucket="test",
                minio_secure=False,
            )
        )
        from src.services.storage_service import StorageService

        svc = StorageService()
        assert await svc.health_check() is True


@patch("src.services.storage_service.get_settings")
async def test_upload_to_minio_returns_none_when_local(mock_settings):
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_settings.return_value = MagicMock(
            storage=MagicMock(
                local_path=tmpdir,
                backend="local",
                minio_endpoint="",
                minio_access_key="",
                minio_secret_key="",
                minio_bucket="test",
                minio_secure=False,
            )
        )
        from src.services.storage_service import StorageService

        svc = StorageService()
        result = await svc.upload_to_minio("job1", b"data", "clip.mp4")
        assert result is None


@patch("src.services.storage_service.get_settings")
async def test_presign_url_returns_none_without_minio(mock_settings):
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_settings.return_value = MagicMock(
            storage=MagicMock(
                local_path=tmpdir,
                backend="local",
                minio_endpoint="",
                minio_access_key="",
                minio_secret_key="",
                minio_bucket="test",
                minio_secure=False,
            )
        )
        from src.services.storage_service import StorageService

        svc = StorageService()
        assert await svc.presign_url("some/key") is None


def _local_storage(local_path: str):
    from src.services.storage_service import StorageService

    with patch("src.services.storage_service.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(
            storage=MagicMock(
                local_path=local_path,
                backend="local",
                minio_endpoint="",
                minio_access_key="",
                minio_secret_key="",
                minio_bucket="test",
                minio_secure=False,
            )
        )
        return StorageService()


async def test_retrieve_rejects_absolute_outside_root():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "root"
        root.mkdir()
        svc = _local_storage(str(root))
        outside = Path(tmpdir) / "secret.bin"
        outside.write_bytes(b"sensitive")
        assert await svc.retrieve(str(outside)) is None
        assert outside.read_bytes() == b"sensitive"


async def test_retrieve_rejects_traversal():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "root"
        root.mkdir()
        svc = _local_storage(str(root))
        outside = Path(tmpdir) / "secret.bin"
        outside.write_bytes(b"sensitive")
        assert await svc.retrieve("../secret.bin") is None
        assert outside.read_bytes() == b"sensitive"


async def test_delete_local_rejects_outside_root():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "root"
        root.mkdir()
        svc = _local_storage(str(root))
        outside = Path(tmpdir) / "secret.bin"
        outside.write_bytes(b"sensitive")
        assert await svc.delete_local(str(outside)) is False
        assert outside.exists()


async def test_delete_local_legitimate_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        svc = _local_storage(tmpdir)
        path = await svc.store_video("job-1", b"data", "clip.mp4")
        assert Path(path).exists()
        assert await svc.delete_local(path) is True
        assert not Path(path).exists()
